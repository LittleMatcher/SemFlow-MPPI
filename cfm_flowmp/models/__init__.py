"""
FlowMP Network Architectures

Contains:
- FlowMPTransformer: Main conditional vector field prediction network
- FlowMPUNet1D: 1D U-Net baseline for comparison
- GaussianFourierProjection: Time embedding
- AdaLN: Adaptive Layer Normalization for conditioning
"""

from .transformer import FlowMPTransformer, create_flowmp_transformer
from .unet_1d import FlowMPUNet1D, create_flowmp_unet1d
from .embeddings import (
    GaussianFourierProjection,
    SinusoidalPositionalEncoding,
    ConditionEncoder,
    AdaLN,
    FiLM,
)

__all__ = [
    "FlowMPTransformer",
    "create_flowmp_transformer",
    "FlowMPUNet1D",
    "create_flowmp_unet1d",
    "GaussianFourierProjection",
    "SinusoidalPositionalEncoding", 
    "ConditionEncoder",
    "AdaLN",
    "FiLM",
]
