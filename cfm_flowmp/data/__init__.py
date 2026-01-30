"""
CFM FlowMP Data Module

Contains:
- TrajectoryDataset: Dataset for expert trajectories
- Data utilities and transforms
"""

from .dataset import TrajectoryDataset, create_dataloader, SyntheticTrajectoryDataset

__all__ = [
    "TrajectoryDataset",
    "SyntheticTrajectoryDataset",
    "create_dataloader",
]
