from collections import OrderedDict
from typing import Tuple, Union, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from backbone import QuickGELU, LayerNorm
import math


def _init_he_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
class RawCrossAttention(nn.Module):
    def __init__(
        self,
        d_query:int,
        d_key:int, 
        d_value: int,
        embed_dim: int,
        num_heads: int,
        dropout: float=0.0,
        batch_first:bool=True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) phải chia hết cho num_heads ({num_heads})"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.batch_first = batch_first
        
        self.q_proj = nn.Linear(d_query, embed_dim)
        self.k_proj = nn.Linear(d_key, embed_dim)
        self.v_proj = nn.Linear(d_value, embed_dim)
        
        self.out_proj = nn.Linear(d_query, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    )-> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            
        B, L_q, _ = query.size()
        _, L_kv, _ = key.size()
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        q = q.view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if attn_mask is not None:
            scores = scores + attn_mask
        
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(B, 1, 1, L_kv)
            scores = scores.masked_fill(key_padding_mask, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights_dropped, v)
        attn_output = attn_output.transpose(1,2).contiguous().view(B, L_q, self.embed_dim)
        
        attn_output = self.out_proj(attn_output)
        
        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)
        return attn_output, (attn_weights if need_weights else None)
    
class CrossAttentionBlock(nn.Module):
    def __init__(
        self, 
        d_query: int, 
        d_kv: int,
        embed_dim: int, 
        num_heads: int, 
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU()
    ):
        super().__init__()
        
        self.cross_attn = RawCrossAttention(
            d_query=d_query,
            d_key=d_kv,
            d_value=d_kv,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.ln_q = nn.LayerNorm(d_query)
        self.ln_kv = nn.LayerNorm(d_kv)
        
        self.out_proj = nn.Linear(embed_dim, d_query) if embed_dim != d_query else nn.Identity()

        ffn_hidden = ffn_dim if ffn_dim is not None else embed_dim * 4
        self.ln_ffn = nn.LayerNorm(d_query)
        self.ffn = nn.Sequential(OrderedDict([
            ("linear_1", nn.Linear(d_query, ffn_hidden)),
            ("act", activation),
            ("dropout_1", nn.Dropout(dropout)),
            
            ("linear_2", nn.Linear(ffn_hidden, d_query)),
            ("dropout_2", nn.Dropout(dropout))
        ]))
        
    def forward(
        self, 
        query: torch.Tensor, 
        kv_source: torch.Tensor, 
        kv_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        norm_q = self.ln_q(query)
        norm_kv = self.ln_kv(kv_source)
        
        attn_out, _ = self.cross_attn(
            query=norm_q,
            key=norm_kv,
            value=norm_kv,
            key_padding_mask=kv_padding_mask,
            need_weights=False,
        )
        attn_out = self.out_proj(attn_out)
        x = query + attn_out
        x = x + self.ffn(self.ln_ffn(x))
        
        return x
        
class stage1VL(nn.Module):
    def __init__(
        self,
        d_model_v: int,
        d_model_t: int, 
        embed_dim: int=512,
        num_heads: int=8,
        depth: int=2,
        dropout: float=0.1
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            CrossAttentionBlock(
                d_query=d_model_v,
                d_kv=d_model_t,
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(depth)
        ])
        self.apply(_init_he_weights)
        
    def forward(self, 
        image_features: torch.Tensor, 
        text_features: torch.Tensor, 
        text_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        is_2d = image_features.dim() == 2
        if is_2d:
            image_features = image_features.unsqueeze(1)
            
        fused_features = image_features
        for layer in self.layers:
            fused_features = layer(
                query=fused_features,
                kv_source=text_features,
                kv_padding_mask=text_padding_mask
            )

        return fused_features