"""
CFM FlowMP Training Module

Contains:
- FlowMatchingLoss: Loss computation for flow matching
- FlowMatchingConfig: Config for flow matching
- FlowInterpolator: Constructs interpolation paths
- CFMTrainer: Main training class
- TrainerConfig: Trainer configuration
"""

from .flow_matching import FlowMatchingLoss, FlowMatchingConfig, FlowInterpolator
from .trainer import CFMTrainer, TrainerConfig

__all__ = [
    "FlowMatchingLoss",
    "FlowMatchingConfig",
    "FlowInterpolator",
    "CFMTrainer",
    "TrainerConfig",
]
