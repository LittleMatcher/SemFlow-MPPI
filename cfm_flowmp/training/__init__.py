"""
CFM FlowMP Training Module

Contains:
- FlowMatchingLoss: Loss computation for flow matching
- FlowInterpolator: Constructs interpolation paths
- CFMTrainer: Main training class
"""

from .flow_matching import FlowMatchingLoss, FlowInterpolator
from .trainer import CFMTrainer

__all__ = [
    "FlowMatchingLoss",
    "FlowInterpolator", 
    "CFMTrainer",
]
