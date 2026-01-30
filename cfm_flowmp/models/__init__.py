"""
FlowMP Network Architectures

Contains:
- FlowMPTransformer: Main conditional vector field prediction network
- FlowMPUNet1D: 1D U-Net baseline for comparison
- L2SafetyCFM: Safety-Embedded CFM for three-tier architecture
- CostMapEncoder: CNN encoder for semantic cost maps
- GaussianFourierProjection: Time embedding
- AdaLN: Adaptive Layer Normalization for conditioning
"""

from .transformer import FlowMPTransformer, create_flowmp_transformer
from .unet_1d import FlowMPUNet1D, create_flowmp_unet1d
from .l2_safety_cfm import L2SafetyCFM, L2Config, create_l2_safety_cfm
from .cost_map_encoder import (
    CostMapEncoder,
    MultiScaleCostMapEncoder,
    create_cost_map_encoder,
)
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
    "L2SafetyCFM",
    "L2Config",
    "create_l2_safety_cfm",
    "CostMapEncoder",
    "MultiScaleCostMapEncoder",
    "create_cost_map_encoder",
    "GaussianFourierProjection",
    "SinusoidalPositionalEncoding", 
    "ConditionEncoder",
    "AdaLN",
    "FiLM",
]
