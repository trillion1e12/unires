from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from backbone import QuickGELU, LayerNorm

import math


def _make_group_norm(num_channels: int) -> nn.GroupNorm:
    max_groups = min(32, num_channels)
    for num_groups in range(max_groups, 0, -1):
        if num_channels % num_groups == 0:
            return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    return nn.GroupNorm(num_groups=1, num_channels=num_channels)


def _init_he_weights(module: nn.Module) -> None:
    if isinstance(module, ScratchConv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(module.bias)
    elif isinstance(module, ScratchTransposedConv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.GroupNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class ResidualUpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int,
    ):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.main = nn.Sequential(
            ScratchTransposedConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=scale_factor,
                stride=scale_factor,
            ),
            _make_group_norm(out_channels),
            nn.GELU(),
        )
        self.skip = nn.Sequential(
            ScratchConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            _make_group_norm(out_channels),
        ) if (in_channels != out_channels or scale_factor != 1) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_upsampled = self.upsample(x)
        main_out = self.main(x)
        skip_out = self.skip(x_upsampled)
        return F.gelu(main_out + skip_out)

class ScratchBilinearResize2d(nn.Module):
    def __init__(self, target_size: Tuple[int, int]):
        super().__init__()
        self.target_size = target_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H_in, W_in = x.shape
        H_out, W_out = self.target_size
        
        if H_in == H_out and W_in == W_out:
            return x
        
        y_indices = torch.linspace(0, H_in - 1, steps=H_out, device=x.device)
        x_indices = torch.linspace(0, W_in - 1, steps=W_out, device=x.device)
        
        y0 = torch.floor(y_indices).long()
        y1 = torch.clamp(y0 + 1, max=H_in -1)
        
        x0 = torch.floor(x_indices).long()
        x1 = torch.clamp(x0 + 1, max=W_in-1)
        
        dy = (y_indices - y0).view(1, 1, -1, 1)
        dx = (x_indices - x0).view(1, 1, 1, -1)

        v00 = x[:, :, y0][..., x0]
        v01 = x[:, :, y0][..., x1]
        v10 = x[:, :, y1][..., x0]
        v11 = x[:, :, y1][..., x1]

        w0 = v00 * (1 - dx) + v01 * dx
        w1 = v10 * (1 - dx) + v11 * dx

        out = w0 * (1 - dy) + w1 * dy

        return out

class ScratchTransposedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int, 
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        B, C_in, H, W = x.shape
        K = self.kernel_size
        
        # X: (B, C_in, H, W) -> (B, H, W, C_in)
        x_permuted = x.permute(0,2,3,1).contiguous()
        # W: (in_channels, out_channels, K, K) -> (in_channels, out_channels * K * K)
        w_flat = self.weight.view(self.in_channels, -1)
        
        # Result: (B, H, W, out_channels * K * K)
        out = torch.matmul(x_permuted, w_flat)
        out = out.view(B, H, W, self.out_channels, K, K)
        
        out = out.permute(0, 3, 1, 4, 2, 5).contiguous()
        out = out.view(B, self.out_channels, H*K, W*K)
        
        if self.padding > 0:
            P = self.padding
            out = out[:, :, P:-P, P:-P]
        out = out + self.bias.view(1, -1, 1, 1)
        return out

class ScratchConv2d(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        kernel_size: int=1,
    ):                
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C_in, H, W = x.shape
        K = self.kernel_size
         
        H_out = H - K + 1
        W_out = W - K + 1
        
        # (out_channels, in_channels * K * K)
        weight_flat = self.weight.view(self.out_channels, -1)
        
        # Kết quả: shape (B, in_channels, H_out, W_out, K, K)
        patches = x.unfold(2, K, 1).unfold(3, K, 1)
        
        # (B, H_out, W_out, in_channels * K * K)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(B, H_out, W_out, -1)

        out = torch.matmul(patches, weight_flat.t())
        
        out = out.permute(0, 3, 1, 2) + self.bias.view(1, -1, 1, 1)
        return out
class stage2VL(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int, 
        upsample_configs: list[Tuple[int, int]],
        target_size: Tuple[int, int] = (224, 224)
    ):
        super().__init__()
        self.upsample_blocks = nn.ModuleList()
        current_channels = in_channels
        
        for scale_factor, out_channels in upsample_configs:
            block = ResidualUpsampleBlock(
                in_channels=current_channels,
                out_channels=out_channels,
                scale_factor=scale_factor,
            )
            self.upsample_blocks.append(block)
            current_channels = out_channels
            
        self.mask_proj = ScratchConv2d(
            in_channels=current_channels,
            out_channels=num_classes,
            kernel_size=1
        )
        self.mask_norm = _make_group_norm(num_classes)
        
        self.resize_layer = ScratchBilinearResize2d(target_size=target_size)
        self.apply(_init_he_weights)
    
    def forward(self, x: torch.Tensor, spatial_shape: Union[Tuple[int, int], None] = None) -> torch.Tensor:
        B, N, C = x.shape
        
        if spatial_shape is not None:
            H, W = spatial_shape
            assert H * W == N, f"Kích thước HxW ({H * W}) không khớp với N ({N})"
        else:
            H = int(math.sqrt(N))
            W = H
            assert H * W == N, "Giá trị N không phải là số chính phương.@@@@@@@"
            
        x_spatial = x.transpose(1, 2).contiguous().view(B, C, H, W)
        
        out = x_spatial
        for block in self.upsample_blocks:
            out = block(out)
            
        mask_logits = self.mask_proj(out)
        mask_logits = self.mask_norm(mask_logits)
        
        out_resized = self.resize_layer(mask_logits)
        
        return out_resized
    


if __name__ == "__main__":
    in_channels_64 = 64
    num_classes = 10 

    # Cấu hình scale_factor: x2 (->20x20), x2 (->40x40), x2 (->80x80), x2 (->160x160)
    configs = [
        (2, 32), 
        (2, 16), 
        (2, 16), 
        (2, 16) 
    ]

    model = stage2VL(
        in_channels=in_channels_64, 
        num_classes=num_classes, 
        upsample_configs=configs,
        target_size=(224, 224)
    )

    # Chạy thử 
    x_input = torch.randn(8, 100, 64) # Batch = 8
    mask_output = model(x_input)
    print(mask_output.shape)