"""
CFM FlowMP Data Module

Contains:
- TrajectoryDataset: Dataset for expert trajectories
- SyntheticTrajectoryDataset: Synthetic trajectory generation
- MockL3Dataset: Mock L3 (VLM) data generator for L2 training
- FlowMPEnvDataset: Load generated env data from .npz (flow_mp style)
- create_dataloader / create_l2_dataloaders / split_dataset
"""

from .dataset import (
    TrajectoryDataset,
    SyntheticTrajectoryDataset,
    MockL3Dataset,
    FlowMPEnvDataset,
    create_dataloader,
    create_l2_dataloaders,
    split_dataset,
)

# HDF5 dataset utilities (参考 mpd-splines-public 的设计)
try:
    from .hdf5_dataset import (
        HDF5TrajectoryDataset,
        save_trajectories_to_hdf5,
        load_trajectories_from_hdf5,
    )
    __all__ = [
        "TrajectoryDataset",
        "SyntheticTrajectoryDataset",
        "MockL3Dataset",
        "FlowMPEnvDataset",
        "create_dataloader",
        "create_l2_dataloaders",
        "split_dataset",
        "HDF5TrajectoryDataset",
        "save_trajectories_to_hdf5",
        "load_trajectories_from_hdf5",
    ]
except ImportError:
    __all__ = [
        "TrajectoryDataset",
        "SyntheticTrajectoryDataset",
        "MockL3Dataset",
        "FlowMPEnvDataset",
        "create_dataloader",
        "create_l2_dataloaders",
        "split_dataset",
    ]
