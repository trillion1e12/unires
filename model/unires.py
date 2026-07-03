from typing import Dict
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPTextModel


class UniResImageEncoder(nn.Module):
    def __init__(self, image_encoder: CLIPVisionModel) -> None:
        super().__init__()
        self.image_encoder = image_encoder

    def forward(
        self,
        pixel_values: torch.Tensor,
        low_tokens: torch.Tensor,
        high_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # tokenize the image + positional embedding
        # (B,50,768)
        hidden_states = self.image_encoder.embeddings(pixel_values)
        batch_size, seq_len, _ = hidden_states.shape

        # insert low level group tokens
        # (B,64,768)
        low_tokens = low_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        # (B,114,768)
        hidden_states = torch.cat((hidden_states, low_tokens), dim=1)

        # norm
        hidden_states = self.image_encoder.pre_layrnorm(hidden_states)

        # find index of the middle of the transformer blocks
        # 12
        num_layers = len(self.image_encoder.encoder.layers)
        # 6
        mid_idx = num_layers // 2

        # first encoder half
        for encoder_layer in self.image_encoder.encoder.layers[:mid_idx]:
            hidden_states = encoder_layer(hidden_states, attention_mask=None)

        # take out low level group tokens
        low_tokens = hidden_states[:, seq_len:, :]
        # (B,50,768)
        hidden_states = hidden_states[:, :seq_len, :]

        # insert high level group tokens
        # (B,8,768)
        high_tokens = high_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        # (B,58,768)
        hidden_states = torch.cat((hidden_states, high_tokens), dim=1)

        # second encoder half
        for encoder_layer in self.image_encoder.encoder.layers[mid_idx:]:
            hidden_states = encoder_layer(hidden_states, attention_mask=None)

        # take out high level group tokens
        high_tokens = hidden_states[:, seq_len:, :]
        # (B,50,768)
        last_hidden_state = hidden_states[:, :seq_len, :]

        # CLS token
        # (B,768)
        pooled_output = last_hidden_state[:, 0, :]
        # (B,768)
        pooled_output = self.image_encoder.post_layernorm(pooled_output)

        return {
            "last_hidden_state": last_hidden_state,
            "pooler_output": pooled_output,
            "low_tokens": low_tokens,
            "high_tokens": high_tokens,
        }


class LanguageGuidedRegionFilter(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward) -> None:
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self, group_feat: torch.Tensor, text_feat: torch.Tensor
    ) -> torch.Tensor:
        # feature tensor shape stay at (B,L,512), L is number of group token (64 or 8)
        # cross attention
        normed_group = self.norm1(group_feat)
        fused_feat, _ = self.cross_attn(
            query=normed_group, key=text_feat, value=text_feat
        )
        fused_feat = group_feat + fused_feat

        # feed forward network
        normed_fused = self.norm2(fused_feat)
        ffn_out = self.ffn(normed_fused)
        ffn_out = fused_feat + ffn_out

        return ffn_out


class VLDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(
        self,
        image_feat: torch.Tensor,
        text_feat: torch.Tensor,
    ) -> torch.Tensor:
        # feature tensor shape stay at (B,50,512)
        # self attention
        normed_image = self.norm1(image_feat)
        image_feat_2, _ = self.self_attn(
            query=normed_image, key=normed_image, value=normed_image
        )
        image_feat_2 = image_feat + image_feat_2

        # cross attention
        normed_image_2 = self.norm2(image_feat_2)
        fused_feat, _ = self.cross_attn(
            query=normed_image_2, key=text_feat, value=text_feat
        )
        fused_feat = image_feat_2 + fused_feat

        # feed forward network
        normed_fused = self.norm3(fused_feat)
        ffn_out = self.ffn(normed_fused)
        ffn_out = fused_feat + ffn_out

        return ffn_out


class VLDecoder1(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [VLDecoderLayer(d_model, nhead, dim_feedforward) for _ in range(num_layers)]
        )

    def forward(
        self, image_feat: torch.Tensor, text_feat: torch.Tensor
    ) -> torch.Tensor:
        x = image_feat
        for layer in self.layers:
            x = layer(x, text_feat)

        return x


class VLDecoder2(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers) -> None:
        super().__init__()

        # transformer
        self.layers = nn.ModuleList(
            [VLDecoderLayer(d_model, nhead, dim_feedforward) for _ in range(num_layers)]
        )

        # upsample
        self.decoder = nn.Sequential(
            # (B,512,7,7) to (B,256,14,14)
            nn.ConvTranspose2d(d_model, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # (B,256,14,14) to (B,128,28, 28)
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # (B,128,28,28) to (B,64,56,56)
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # (B,64,56,56) to (B,32,112,112)
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # (B,32,112,112) to (B,16,224,224)
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # (B,16,224,224) to (B,1,224,224)
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(
        self, image_feat: torch.Tensor, group_feat: torch.Tensor
    ) -> torch.Tensor:
        # transformer blocks
        # (B,50,512)
        x = image_feat
        for layer in self.layers:
            # (B,50,512)
            x = layer(x, group_feat)

        # reshape for cnn
        # (B,49,512)
        x = x[:, 1:, :]
        batch_size, seq_len, d_model = x.shape

        # (B,512,49)
        x = x.transpose(1, 2)

        height = width = int(seq_len**0.5)
        # (B,512,7,7)
        x = x.reshape(batch_size, d_model, height, width)

        # upsample
        # (B,1,224,224)
        x = self.decoder(x)

        return x


class UniRes(nn.Module):
    def __init__(
        self,
        image_encoder: CLIPVisionModel,
        text_encoder: CLIPTextModel,
        num_low_tokens: int,
        num_high_tokens: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        # clip image and text encoder
        self.image_encoder = UniResImageEncoder(image_encoder)
        self.text_encoder = text_encoder

        image_dmodel = image_encoder.config.hidden_size
        text_dmodel = text_encoder.config.hidden_size

        self.low_tokens = nn.Parameter(torch.rand(num_low_tokens, image_dmodel))
        self.high_tokens = nn.Parameter(torch.rand(num_high_tokens, image_dmodel))

        self.image_projection = nn.Linear(image_dmodel, d_model)
        self.text_projection = nn.Linear(text_dmodel, d_model)

        self.lrf = LanguageGuidedRegionFilter(d_model, nhead, dim_feedforward)

        self.decoder1 = VLDecoder1(d_model, nhead, dim_feedforward, num_layers)
        self.decoder2 = VLDecoder2(d_model, nhead, dim_feedforward, num_layers)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # clip image
        image_out = self.image_encoder(pixel_values, self.low_tokens, self.high_tokens)
        # (B,50,768)
        image_feat = image_out["last_hidden_state"]
        # (B,64,768)
        low_tokens = image_out["low_tokens"]
        # (B,8,768)
        high_tokens = image_out["high_tokens"]
        # (B,50,512)
        image_feat = self.image_projection(image_feat)
        # (B,64,512)
        low_tokens = self.image_projection(low_tokens)
        # (B,8,512)
        high_tokens = self.image_projection(high_tokens)

        # clip text
        text_out = self.text_encoder(input_ids, attention_mask)
        # (B,S,512)
        text_feat = text_out["last_hidden_state"]
        # (B,S,512)
        text_feat = self.text_projection(text_feat)

        # language guided region filter (LRF)
        # (B,64,512)
        low_feat = self.lrf(low_tokens, text_feat)
        # (B,8,512)
        high_feat = self.lrf(high_tokens, text_feat)
        # (B,72,512)
        group_feat = torch.concat((low_feat, high_feat), dim=1)

        # stage 1 and stage 2 decoder
        # (B,50,512)
        fused_feat = self.decoder1(image_feat, text_feat)
        # (B,224,224)
        mask = self.decoder2(fused_feat, group_feat)

        return mask
