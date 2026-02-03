"""
Cost Map CNN Encoder for L2 Layer

Encodes 2D semantic cost maps from L3 (VLM) into latent vectors
for conditioning the conditional flow matching model.

Input: 2D Cost Map (H, W) representing semantic risk distribution
Output: Latent vector e_map for conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CostMapEncoder(nn.Module):
    """
    CNN-based encoder for 2D semantic cost maps.
    
    Converts spatial cost distributions into compact latent representations
    that can be used as conditioning signals for trajectory generation.
    
    Architecture follows standard ConvNet design with:
    - Progressive downsampling via strided convolutions
    - Residual connections for gradient flow
    - Global average pooling for spatial invariance
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        base_channels: int = 32,
        latent_dim: int = 256,
        num_downsample_layers: int = 4,
        use_batch_norm: bool = True,
    ):
        """
        Args:
            input_channels: Number of input channels (1 for grayscale cost map)
            base_channels: Base number of channels (doubles at each downsample)
            latent_dim: Dimension of output latent vector
            num_downsample_layers: Number of downsampling stages
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        
        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        
        # Downsampling layers with residual connections
        self.down_layers = nn.ModuleList()
        in_ch = base_channels
        
        for i in range(num_downsample_layers):
            out_ch = base_channels * (2 ** (i + 1))
            self.down_layers.append(
                ResidualBlock2D(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    stride=2,
                    use_batch_norm=use_batch_norm,
                )
            )
            in_ch = out_ch
        
        # Global pooling and projection to latent space
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.output_proj = nn.Sequential(
            nn.Linear(in_ch, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )
    
    def forward(self, cost_map: torch.Tensor) -> torch.Tensor:
        """
        Encode 2D cost map to latent vector.
        
        Args:
            cost_map: [B, C, H, W] cost map tensor
            
        Returns:
            Latent encoding [B, latent_dim]
        """
        # Initial projection
        h = self.input_proj(cost_map)
        
        # Progressive downsampling
        for layer in self.down_layers:
            h = layer(h)
        
        # Global pooling to fixed-size representation
        h = self.global_pool(h)  # [B, C, 1, 1]
        h = h.flatten(1)  # [B, C]
        
        # Project to latent space
        latent = self.output_proj(h)  # [B, latent_dim]
        
        return latent


class ResidualBlock2D(nn.Module):
    """2D Residual block with optional downsampling."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=not use_batch_norm
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=not use_batch_norm
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity(),
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = F.relu(out, inplace=True)
        
        return out


class MultiScaleCostMapEncoder(nn.Module):
    """
    Multi-scale cost map encoder with feature pyramid.
    
    Captures both fine-grained local details and coarse global structure
    by extracting features at multiple resolutions.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        base_channels: int = 32,
        latent_dim: int = 256,
        num_scales: int = 3,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        
        self.num_scales = num_scales
        self.latent_dim = latent_dim
        
        # Calculate per-scale latent dimensions to ensure sum equals latent_dim
        # This handles cases where latent_dim is not divisible by num_scales
        per_scale_dim = latent_dim // num_scales
        remainder = latent_dim % num_scales
        
        # Encoder for each scale
        self.scale_encoders = nn.ModuleList()
        for i in range(num_scales):
            # Last encoder gets the remainder to ensure total equals latent_dim
            scale_latent_dim = per_scale_dim + (remainder if i == num_scales - 1 else 0)
            self.scale_encoders.append(
                CostMapEncoder(
                    input_channels=input_channels,
                    base_channels=base_channels,
                    latent_dim=scale_latent_dim,
                    num_downsample_layers=i + 2,
                    use_batch_norm=use_batch_norm,
                )
            )
        
        # Fusion layer - input dimension is the sum of all scale dimensions (should equal latent_dim)
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )
    
    def forward(self, cost_map: torch.Tensor) -> torch.Tensor:
        """
        Multi-scale encoding.
        
        Args:
            cost_map: [B, C, H, W]
            
        Returns:
            Fused latent encoding [B, latent_dim]
        """
        # Encode at each scale
        scale_features = []
        for encoder in self.scale_encoders:
            features = encoder(cost_map)
            scale_features.append(features)
        
        # Concatenate multi-scale features
        fused = torch.cat(scale_features, dim=-1)  # [B, latent_dim]
        
        # Final fusion
        output = self.fusion(fused)
        
        return output


def create_cost_map_encoder(
    encoder_type: str = "single_scale",
    **kwargs,
) -> nn.Module:
    """
    Factory function for cost map encoders.
    
    Args:
        encoder_type: "single_scale" or "multi_scale"
        **kwargs: Additional arguments for encoder
        
    Returns:
        Cost map encoder module
    """
    if encoder_type == "single_scale":
        return CostMapEncoder(**kwargs)
    elif encoder_type == "multi_scale":
        return MultiScaleCostMapEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
