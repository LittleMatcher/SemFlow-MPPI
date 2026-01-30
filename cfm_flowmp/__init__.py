"""
CFM FlowMP: Conditional Flow Matching for Trajectory Planning

A PyTorch implementation of the FlowMP architecture for learning
trajectory generation using Conditional Flow Matching.

Key Components:
- Transformer-based conditional vector field prediction
- Gaussian Fourier time embedding with AdaLN conditioning
- RK4 ODE solver for trajectory generation
- B-spline smoothing for physical consistency
"""

__version__ = "0.1.0"
__author__ = "CFM FlowMP Team"

from .models import FlowMPTransformer
from .training import CFMTrainer, FlowMatchingLoss
from .inference import TrajectoryGenerator, RK4Solver

__all__ = [
    "FlowMPTransformer",
    "CFMTrainer", 
    "FlowMatchingLoss",
    "TrajectoryGenerator",
    "RK4Solver",
]
