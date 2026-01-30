"""
Dataset Classes for CFM FlowMP Training

Provides dataset implementations for loading expert trajectories:
- TrajectoryDataset: Load trajectories from files
- SyntheticTrajectoryDataset: Generate synthetic trajectories for testing
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path


class TrajectoryDataset(Dataset):
    """
    Dataset for loading expert trajectory data.
    
    Expected data format:
    - positions: [N, T, D] array of position trajectories
    - velocities: [N, T, D] array of velocity trajectories  
    - accelerations: [N, T, D] array of acceleration trajectories
    
    Supports loading from:
    - NumPy .npy/.npz files
    - HDF5 files
    - Pickle files
    """
    
    def __init__(
        self,
        data_path: str = None,
        positions: np.ndarray = None,
        velocities: np.ndarray = None,
        accelerations: np.ndarray = None,
        normalize: bool = True,
        compute_derivatives: bool = True,
        dt: float = 0.1,
    ):
        """
        Initialize trajectory dataset.
        
        Args:
            data_path: Path to data file
            positions: Pre-loaded position data [N, T, D]
            velocities: Pre-loaded velocity data [N, T, D]
            accelerations: Pre-loaded acceleration data [N, T, D]
            normalize: Whether to normalize trajectories
            compute_derivatives: Compute vel/acc from positions if not provided
            dt: Time step for derivative computation
        """
        self.normalize = normalize
        self.dt = dt
        
        # Load data
        if data_path is not None:
            data = self._load_data(data_path)
            positions = data.get('positions', positions)
            velocities = data.get('velocities', velocities)
            accelerations = data.get('accelerations', accelerations)
        
        if positions is None:
            raise ValueError("positions must be provided either via data_path or directly")
        
        # Convert to torch tensors
        self.positions = torch.tensor(positions, dtype=torch.float32)
        
        # Compute or load velocities
        if velocities is not None:
            self.velocities = torch.tensor(velocities, dtype=torch.float32)
        elif compute_derivatives:
            self.velocities = self._compute_velocity(self.positions)
        else:
            self.velocities = torch.zeros_like(self.positions)
        
        # Compute or load accelerations
        if accelerations is not None:
            self.accelerations = torch.tensor(accelerations, dtype=torch.float32)
        elif compute_derivatives:
            self.accelerations = self._compute_acceleration(self.velocities)
        else:
            self.accelerations = torch.zeros_like(self.positions)
        
        # Compute normalization statistics
        self.pos_mean = None
        self.pos_std = None
        self.vel_mean = None
        self.vel_std = None
        self.acc_mean = None
        self.acc_std = None
        
        if normalize:
            self._compute_normalization_stats()
    
    def _load_data(self, path: str) -> Dict[str, np.ndarray]:
        """Load data from file."""
        path = Path(path)
        
        if path.suffix == '.npy':
            # Single array file (assumes positions only)
            return {'positions': np.load(path)}
        
        elif path.suffix == '.npz':
            data = np.load(path)
            return {key: data[key] for key in data.files}
        
        elif path.suffix in ['.h5', '.hdf5']:
            import h5py
            data = {}
            with h5py.File(path, 'r') as f:
                for key in f.keys():
                    data[key] = f[key][:]
            return data
        
        elif path.suffix == '.pkl':
            import pickle
            with open(path, 'rb') as f:
                return pickle.load(f)
        
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def _compute_velocity(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute velocity from positions using finite differences."""
        # Central difference for interior points
        velocity = torch.zeros_like(positions)
        velocity[:, 1:-1] = (positions[:, 2:] - positions[:, :-2]) / (2 * self.dt)
        
        # Forward/backward difference for endpoints
        velocity[:, 0] = (positions[:, 1] - positions[:, 0]) / self.dt
        velocity[:, -1] = (positions[:, -1] - positions[:, -2]) / self.dt
        
        return velocity
    
    def _compute_acceleration(self, velocities: torch.Tensor) -> torch.Tensor:
        """Compute acceleration from velocities using finite differences."""
        return self._compute_velocity(velocities)  # Same operation on velocities
    
    def _compute_normalization_stats(self):
        """Compute mean and std for normalization."""
        self.pos_mean = self.positions.mean(dim=(0, 1))
        self.pos_std = self.positions.std(dim=(0, 1)).clamp(min=1e-6)
        
        self.vel_mean = self.velocities.mean(dim=(0, 1))
        self.vel_std = self.velocities.std(dim=(0, 1)).clamp(min=1e-6)
        
        self.acc_mean = self.accelerations.mean(dim=(0, 1))
        self.acc_std = self.accelerations.std(dim=(0, 1)).clamp(min=1e-6)
    
    def _normalize(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Normalize tensor."""
        return (x - mean) / std
    
    def _denormalize(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Denormalize tensor."""
        return x * std + mean
    
    def __len__(self) -> int:
        return len(self.positions)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a trajectory sample.
        
        Returns:
            Dictionary containing:
                - positions: [T, D] position trajectory
                - velocities: [T, D] velocity trajectory
                - accelerations: [T, D] acceleration trajectory
                - start_pos: [D] starting position
                - goal_pos: [D] goal position
                - start_vel: [D] starting velocity
        """
        positions = self.positions[idx]
        velocities = self.velocities[idx]
        accelerations = self.accelerations[idx]
        
        # Normalize if enabled
        if self.normalize and self.pos_mean is not None:
            positions = self._normalize(positions, self.pos_mean, self.pos_std)
            velocities = self._normalize(velocities, self.vel_mean, self.vel_std)
            accelerations = self._normalize(accelerations, self.acc_mean, self.acc_std)
        
        # Extract start and goal
        start_pos = positions[0]
        goal_pos = positions[-1]
        start_vel = velocities[0]
        
        return {
            'positions': positions,
            'velocities': velocities,
            'accelerations': accelerations,
            'start_pos': start_pos,
            'goal_pos': goal_pos,
            'start_vel': start_vel,
        }
    
    def get_normalization_stats(self) -> Dict[str, torch.Tensor]:
        """Get normalization statistics for use during inference."""
        return {
            'pos_mean': self.pos_mean,
            'pos_std': self.pos_std,
            'vel_mean': self.vel_mean,
            'vel_std': self.vel_std,
            'acc_mean': self.acc_mean,
            'acc_std': self.acc_std,
        }


class SyntheticTrajectoryDataset(Dataset):
    """
    Synthetic trajectory dataset for testing and development.
    
    Generates smooth trajectories using various methods:
    - Bezier curves
    - Polynomial trajectories
    - Sinusoidal motions
    """
    
    def __init__(
        self,
        num_trajectories: int = 1000,
        seq_len: int = 64,
        state_dim: int = 2,
        trajectory_type: str = "bezier",  # "bezier", "polynomial", "sine"
        noise_std: float = 0.01,
        seed: int = 42,
    ):
        """
        Initialize synthetic dataset.
        
        Args:
            num_trajectories: Number of trajectories to generate
            seq_len: Trajectory length
            state_dim: Dimension of state space (typically 2 for 2D)
            trajectory_type: Type of trajectory generation
            noise_std: Standard deviation of noise to add
            seed: Random seed
        """
        super().__init__()
        
        self.num_trajectories = num_trajectories
        self.seq_len = seq_len
        self.state_dim = state_dim
        self.noise_std = noise_std
        
        # Set seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Generate trajectories
        self.positions, self.velocities, self.accelerations = \
            self._generate_trajectories(trajectory_type)
    
    def _generate_trajectories(
        self, 
        trajectory_type: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate synthetic trajectories."""
        
        if trajectory_type == "bezier":
            return self._generate_bezier_trajectories()
        elif trajectory_type == "polynomial":
            return self._generate_polynomial_trajectories()
        elif trajectory_type == "sine":
            return self._generate_sine_trajectories()
        else:
            raise ValueError(f"Unknown trajectory type: {trajectory_type}")
    
    def _generate_bezier_trajectories(self) -> Tuple[torch.Tensor, ...]:
        """Generate trajectories using Bezier curves."""
        positions = []
        velocities = []
        accelerations = []
        
        t = np.linspace(0, 1, self.seq_len)
        
        for _ in range(self.num_trajectories):
            # Random control points (4 points for cubic Bezier)
            num_control = 4
            control_points = np.random.randn(num_control, self.state_dim) * 2
            
            # De Casteljau algorithm for Bezier curve
            pos = np.zeros((self.seq_len, self.state_dim))
            for i, ti in enumerate(t):
                points = control_points.copy()
                for r in range(1, num_control):
                    for j in range(num_control - r):
                        points[j] = (1 - ti) * points[j] + ti * points[j + 1]
                pos[i] = points[0]
            
            # Compute derivatives numerically
            dt = 1.0 / (self.seq_len - 1)
            vel = np.gradient(pos, dt, axis=0)
            acc = np.gradient(vel, dt, axis=0)
            
            # Add noise
            pos += np.random.randn(*pos.shape) * self.noise_std
            
            positions.append(pos)
            velocities.append(vel)
            accelerations.append(acc)
        
        return (
            torch.tensor(np.stack(positions), dtype=torch.float32),
            torch.tensor(np.stack(velocities), dtype=torch.float32),
            torch.tensor(np.stack(accelerations), dtype=torch.float32),
        )
    
    def _generate_polynomial_trajectories(self) -> Tuple[torch.Tensor, ...]:
        """Generate trajectories using polynomial curves."""
        positions = []
        velocities = []
        accelerations = []
        
        t = np.linspace(0, 1, self.seq_len)
        
        for _ in range(self.num_trajectories):
            # Random polynomial coefficients (degree 5)
            degree = 5
            coeffs = np.random.randn(degree + 1, self.state_dim) * 0.5
            
            # Evaluate polynomial
            pos = np.zeros((self.seq_len, self.state_dim))
            for i in range(degree + 1):
                pos += coeffs[i] * t[:, None] ** i
            
            # Compute derivatives
            dt = 1.0 / (self.seq_len - 1)
            vel = np.gradient(pos, dt, axis=0)
            acc = np.gradient(vel, dt, axis=0)
            
            # Add noise
            pos += np.random.randn(*pos.shape) * self.noise_std
            
            positions.append(pos)
            velocities.append(vel)
            accelerations.append(acc)
        
        return (
            torch.tensor(np.stack(positions), dtype=torch.float32),
            torch.tensor(np.stack(velocities), dtype=torch.float32),
            torch.tensor(np.stack(accelerations), dtype=torch.float32),
        )
    
    def _generate_sine_trajectories(self) -> Tuple[torch.Tensor, ...]:
        """Generate trajectories using sinusoidal motions."""
        positions = []
        velocities = []
        accelerations = []
        
        t = np.linspace(0, 2 * np.pi, self.seq_len)
        
        for _ in range(self.num_trajectories):
            # Random frequencies and phases
            num_harmonics = 3
            freqs = np.random.uniform(0.5, 2.0, (num_harmonics, self.state_dim))
            phases = np.random.uniform(0, 2 * np.pi, (num_harmonics, self.state_dim))
            amps = np.random.uniform(0.5, 1.5, (num_harmonics, self.state_dim))
            
            # Sum of sinusoids
            pos = np.zeros((self.seq_len, self.state_dim))
            for h in range(num_harmonics):
                pos += amps[h] * np.sin(freqs[h] * t[:, None] + phases[h])
            
            # Analytical derivatives
            vel = np.zeros((self.seq_len, self.state_dim))
            acc = np.zeros((self.seq_len, self.state_dim))
            for h in range(num_harmonics):
                vel += amps[h] * freqs[h] * np.cos(freqs[h] * t[:, None] + phases[h])
                acc -= amps[h] * freqs[h]**2 * np.sin(freqs[h] * t[:, None] + phases[h])
            
            # Add noise
            pos += np.random.randn(*pos.shape) * self.noise_std
            
            positions.append(pos)
            velocities.append(vel)
            accelerations.append(acc)
        
        return (
            torch.tensor(np.stack(positions), dtype=torch.float32),
            torch.tensor(np.stack(velocities), dtype=torch.float32),
            torch.tensor(np.stack(accelerations), dtype=torch.float32),
        )
    
    def __len__(self) -> int:
        return self.num_trajectories
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        positions = self.positions[idx]
        velocities = self.velocities[idx]
        accelerations = self.accelerations[idx]
        
        return {
            'positions': positions,
            'velocities': velocities,
            'accelerations': accelerations,
            'start_pos': positions[0],
            'goal_pos': positions[-1],
            'start_vel': velocities[0],
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for trajectory data.
    
    Args:
        dataset: Trajectory dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: Full dataset
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        seed: Random seed
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    from torch.utils.data import random_split
    
    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size
    
    generator = torch.Generator().manual_seed(seed)
    
    return random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )
