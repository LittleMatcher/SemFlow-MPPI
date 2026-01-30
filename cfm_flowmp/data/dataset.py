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


# ============================================================================
# Mock L3 Data Generator for L2 Training
# ============================================================================

class MockL3Dataset(Dataset):
    """
    Mock L3 (VLM) Dataset for L2 Layer Training.
    
    Generates synthetic cost maps with obstacles and expert trajectories
    that avoid these obstacles using A* pathfinding.
    
    This allows L2 layer training without waiting for real VLM integration.
    
    Features:
    - Generates 64x64 cost maps with Gaussian blob obstacles
    - Creates expert trajectories using A* that avoid obstacles
    - Supports style weights [w_safe, w_fast] for controllability
    - Validates that L2 can learn to avoid obstacles and respond to styles
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        map_size: int = 64,
        seq_len: int = 64,
        num_obstacles_range: Tuple[int, int] = (2, 5),
        obstacle_sigma_range: Tuple[float, float] = (0.05, 0.15),
        min_obstacle_distance: float = 0.2,
        state_dim: int = 2,
        style_mode: str = "random",  # "random", "safe", "fast", "balanced"
        seed: Optional[int] = None,
    ):
        """
        Initialize Mock L3 Dataset.
        
        Args:
            num_samples: Number of samples in dataset
            map_size: Size of cost map (map_size x map_size)
            seq_len: Trajectory sequence length
            num_obstacles_range: (min, max) number of obstacles per map
            obstacle_sigma_range: (min, max) obstacle size (in normalized coords)
            min_obstacle_distance: Minimum distance from obstacles to start/goal
            state_dim: State dimension (2 for 2D)
            style_mode: How to generate style weights
                - "random": Random weights
                - "safe": Always safe style [1.0, 0.0]
                - "fast": Always fast style [0.0, 1.0]
                - "balanced": Always balanced [0.5, 0.5]
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.map_size = map_size
        self.seq_len = seq_len
        self.num_obstacles_range = num_obstacles_range
        self.obstacle_sigma_range = obstacle_sigma_range
        self.min_obstacle_distance = min_obstacle_distance
        self.state_dim = state_dim
        self.style_mode = style_mode
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Generate one synthetic sample.
        
        Returns:
            Dictionary containing:
                - cost_map: [1, H, W] cost map with obstacles
                - start_state: [6] current state [px, py, vx, vy, ax, ay]
                - goal_state: [4] goal state [px, py, vx, vy]
                - style_weights: [2] [w_safe, w_fast]
                - trajectory: [T, 2] expert trajectory positions
                - velocities: [T, 2] expert trajectory velocities
                - accelerations: [T, 2] expert trajectory accelerations
        """
        # Generate valid start and goal points
        start, goal, cost_map = self._generate_map_and_points()
        
        # Generate expert trajectory using A*
        trajectory_positions = self._generate_expert_trajectory(
            start, goal, cost_map
        )
        
        # Compute velocities and accelerations
        velocities = self._compute_velocities(trajectory_positions)
        accelerations = self._compute_accelerations(velocities)
        
        # Generate style weights
        style_weights = self._generate_style_weights()
        
        # Prepare start state [px, py, vx, vy, ax, ay]
        start_state = np.concatenate([
            trajectory_positions[0],  # [px, py]
            velocities[0],            # [vx, vy]
            accelerations[0],         # [ax, ay]
        ])
        
        # Prepare goal state [px, py, vx, vy]
        goal_state = np.concatenate([
            trajectory_positions[-1],  # [px, py]
            velocities[-1],            # [vx, vy]
        ])
        
        return {
            'cost_map': torch.from_numpy(cost_map).float(),
            'start_state': torch.from_numpy(start_state).float(),
            'goal_state': torch.from_numpy(goal_state).float(),
            'style_weights': torch.from_numpy(style_weights).float(),
            'positions': torch.from_numpy(trajectory_positions).float(),
            'velocities': torch.from_numpy(velocities).float(),
            'accelerations': torch.from_numpy(accelerations).float(),
        }
    
    def _generate_map_and_points(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate cost map with obstacles and valid start/goal points.
        
        Returns:
            start: [2] start position
            goal: [2] goal position
            cost_map: [1, H, W] cost map
        """
        max_attempts = 100
        
        for attempt in range(max_attempts):
            # Generate cost map
            cost_map = np.zeros((1, self.map_size, self.map_size), dtype=np.float32)
            
            # Generate random number of obstacles
            num_obstacles = np.random.randint(*self.num_obstacles_range)
            
            obstacle_centers = []
            for _ in range(num_obstacles):
                # Random obstacle position (avoid edges)
                obs_center = np.random.uniform(0.15, 0.85, size=2)
                obstacle_centers.append(obs_center)
                
                # Random obstacle size
                sigma = np.random.uniform(*self.obstacle_sigma_range)
                
                # Draw Gaussian obstacle
                self._draw_gaussian_obstacle(cost_map, obs_center, sigma)
            
            # Generate start and goal points
            start = np.random.uniform(0.1, 0.9, size=2)
            goal = np.random.uniform(0.1, 0.9, size=2)
            
            # Check if start and goal are valid (far from obstacles)
            start_valid = all(
                np.linalg.norm(start - obs) > self.min_obstacle_distance
                for obs in obstacle_centers
            )
            goal_valid = all(
                np.linalg.norm(goal - obs) > self.min_obstacle_distance
                for obs in obstacle_centers
            )
            
            # Check distance between start and goal
            distance_valid = np.linalg.norm(goal - start) > 0.3
            
            if start_valid and goal_valid and distance_valid:
                return start, goal, cost_map
        
        # Fallback: no obstacles
        cost_map = np.zeros((1, self.map_size, self.map_size), dtype=np.float32)
        start = np.array([0.2, 0.2])
        goal = np.array([0.8, 0.8])
        return start, goal, cost_map
    
    def _draw_gaussian_obstacle(
        self,
        cost_map: np.ndarray,
        center: np.ndarray,
        sigma: float,
    ) -> None:
        """
        Draw a Gaussian blob obstacle on the cost map.
        
        Args:
            cost_map: [1, H, W] cost map to modify in-place
            center: [2] obstacle center in normalized coords [0, 1]
            sigma: Gaussian standard deviation
        """
        H, W = cost_map.shape[1:]
        
        # Convert normalized coords to pixel coords
        cx, cy = int(center[0] * W), int(center[1] * H)
        
        # Compute Gaussian radius in pixels
        radius_pixels = int(sigma * W * 3)  # 3-sigma coverage
        
        # Create meshgrid
        y, x = np.ogrid[-radius_pixels:radius_pixels+1, -radius_pixels:radius_pixels+1]
        
        # Gaussian formula
        sigma_pixels = sigma * W
        gaussian = np.exp(-(x**2 + y**2) / (2 * sigma_pixels**2))
        
        # Clip to map bounds
        y_start = max(0, cy - radius_pixels)
        y_end = min(H, cy + radius_pixels + 1)
        x_start = max(0, cx - radius_pixels)
        x_end = min(W, cx + radius_pixels + 1)
        
        # Compute slice indices for gaussian
        gy_start = max(0, -cy + radius_pixels)
        gy_end = gy_start + (y_end - y_start)
        gx_start = max(0, -cx + radius_pixels)
        gx_end = gx_start + (x_end - x_start)
        
        # Draw (max with existing to handle overlaps)
        cost_map[0, y_start:y_end, x_start:x_end] = np.maximum(
            cost_map[0, y_start:y_end, x_start:x_end],
            gaussian[gy_start:gy_end, gx_start:gx_end]
        )
    
    def _generate_expert_trajectory(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        cost_map: np.ndarray,
    ) -> np.ndarray:
        """
        Generate expert trajectory using A* pathfinding.
        
        Args:
            start: [2] start position in [0, 1]
            goal: [2] goal position in [0, 1]
            cost_map: [1, H, W] cost map
            
        Returns:
            trajectory: [T, 2] trajectory positions
        """
        # Run A* to get waypoints
        waypoints = self._astar_pathfinding(start, goal, cost_map[0])
        
        # Interpolate to desired sequence length
        trajectory = self._interpolate_path(waypoints, self.seq_len)
        
        return trajectory
    
    def _astar_pathfinding(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        cost_map: np.ndarray,
    ) -> np.ndarray:
        """
        A* pathfinding on grid with cost map.
        
        Args:
            start: [2] start in normalized coords
            goal: [2] goal in normalized coords
            cost_map: [H, W] cost map
            
        Returns:
            waypoints: [N, 2] path waypoints in normalized coords
        """
        import heapq
        
        H, W = cost_map.shape
        
        # Convert to grid coordinates
        start_grid = (int(start[1] * H), int(start[0] * W))
        goal_grid = (int(goal[1] * H), int(goal[0] * W))
        
        # Clip to bounds
        start_grid = (np.clip(start_grid[0], 0, H-1), np.clip(start_grid[1], 0, W-1))
        goal_grid = (np.clip(goal_grid[0], 0, H-1), np.clip(goal_grid[1], 0, W-1))
        
        # A* search
        open_set = [(0, start_grid)]
        came_from = {}
        g_score = {start_grid: 0}
        
        def heuristic(a, b):
            return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal_grid:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                
                # Convert to normalized coords
                waypoints = np.array([
                    [c[1] / W, c[0] / H] for c in path
                ])
                return waypoints
            
            # Explore neighbors (8-connected)
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                neighbor = (current[0] + dy, current[1] + dx)
                
                # Check bounds
                if not (0 <= neighbor[0] < H and 0 <= neighbor[1] < W):
                    continue
                
                # Cost includes distance + obstacle penalty
                move_cost = np.sqrt(dy**2 + dx**2)
                obstacle_cost = cost_map[neighbor[0], neighbor[1]] * 10.0  # Penalty
                tentative_g = g_score[current] + move_cost + obstacle_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score, neighbor))
        
        # Fallback: straight line
        return np.array([start, goal])
    
    def _interpolate_path(
        self,
        waypoints: np.ndarray,
        num_points: int,
    ) -> np.ndarray:
        """
        Interpolate waypoints to fixed number of points.
        
        Args:
            waypoints: [N, 2] waypoints
            num_points: Desired number of points
            
        Returns:
            trajectory: [num_points, 2]
        """
        if len(waypoints) < 2:
            # Fallback: repeat single point
            return np.tile(waypoints[0], (num_points, 1))
        
        # Compute cumulative distance along path
        distances = np.cumsum(np.r_[0, np.linalg.norm(np.diff(waypoints, axis=0), axis=1)])
        
        # Interpolate
        t_original = distances / distances[-1]  # Normalize to [0, 1]
        t_new = np.linspace(0, 1, num_points)
        
        trajectory = np.zeros((num_points, 2))
        trajectory[:, 0] = np.interp(t_new, t_original, waypoints[:, 0])
        trajectory[:, 1] = np.interp(t_new, t_original, waypoints[:, 1])
        
        return trajectory
    
    def _compute_velocities(self, positions: np.ndarray) -> np.ndarray:
        """Compute velocities from positions using finite differences."""
        velocities = np.zeros_like(positions)
        
        # Forward difference for first point
        velocities[0] = positions[1] - positions[0]
        
        # Central difference for middle points
        velocities[1:-1] = (positions[2:] - positions[:-2]) / 2.0
        
        # Backward difference for last point
        velocities[-1] = positions[-1] - positions[-2]
        
        return velocities
    
    def _compute_accelerations(self, velocities: np.ndarray) -> np.ndarray:
        """Compute accelerations from velocities using finite differences."""
        accelerations = np.zeros_like(velocities)
        
        # Forward difference for first point
        accelerations[0] = velocities[1] - velocities[0]
        
        # Central difference for middle points
        accelerations[1:-1] = (velocities[2:] - velocities[:-2]) / 2.0
        
        # Backward difference for last point
        accelerations[-1] = velocities[-1] - velocities[-2]
        
        return accelerations
    
    def _generate_style_weights(self) -> np.ndarray:
        """
        Generate style weights [w_safe, w_fast].
        
        Returns:
            weights: [2] normalized weights
        """
        if self.style_mode == "random":
            # Random weights that sum to 1
            weights = np.random.dirichlet([1.0, 1.0])
        elif self.style_mode == "safe":
            weights = np.array([1.0, 0.0])
        elif self.style_mode == "fast":
            weights = np.array([0.0, 1.0])
        elif self.style_mode == "balanced":
            weights = np.array([0.5, 0.5])
        else:
            raise ValueError(f"Unknown style_mode: {self.style_mode}")
        
        return weights.astype(np.float32)


# ============================================================================
# FlowMP 生成数据加载（参照 flow_mp-main 数据格式）
# ============================================================================

class FlowMPEnvDataset(Dataset):
    """
    从 generate_env_trajs_cfm.py 生成的 .npz 加载 L2 训练数据。
    
    与 MockL3Dataset 返回格式一致，便于复用 train_l2_mock / validate_l2。
    
    期望 .npz 包含:
        positions [N, T, 2], velocities [N, T, 2], accelerations [N, T, 2]
        cost_maps [N, 1, H, W], start_states [N, 6], goal_states [N, 4]
        style_weights [N, 2]
    """
    
    def __init__(
        self,
        data_path: str,
        normalize: bool = False,
    ):
        """
        Args:
            data_path: 指向 data.npz 的路径（或包含 data.npz 的目录）
            normalize: 是否对轨迹做归一化（默认 False，生成数据已在 [0,1]）
        """
        path = Path(data_path)
        if path.is_dir():
            path = path / "data.npz"
        if not path.exists():
            raise FileNotFoundError(f"FlowMPEnvDataset: not found {path}")
        
        data = np.load(path)
        self.positions = torch.from_numpy(data["positions"]).float()
        self.velocities = torch.from_numpy(data["velocities"]).float()
        self.accelerations = torch.from_numpy(data["accelerations"]).float()
        self.cost_maps = torch.from_numpy(data["cost_maps"]).float()
        self.start_states = torch.from_numpy(data["start_states"]).float()
        self.goal_states = torch.from_numpy(data["goal_states"]).float()
        self.style_weights = torch.from_numpy(data["style_weights"]).float()
        
        self.normalize = normalize
        self._len = len(self.positions)
    
    def __len__(self) -> int:
        return self._len
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            与 MockL3Dataset 相同 keys:
            cost_map [1, H, W], start_state [6], goal_state [4], style_weights [2]
            positions [T, 2], velocities [T, 2], accelerations [T, 2]
        """
        cost_map = self.cost_maps[idx]
        if cost_map.dim() == 3:
            pass
        else:
            cost_map = cost_map.unsqueeze(0)
        return {
            "cost_map": cost_map,
            "start_state": self.start_states[idx],
            "goal_state": self.goal_states[idx],
            "style_weights": self.style_weights[idx],
            "positions": self.positions[idx],
            "velocities": self.velocities[idx],
            "accelerations": self.accelerations[idx],
        }


def create_l2_dataloaders(
    data_source: str = "mock",
    data_dir: Optional[str] = None,
    num_train: int = 5000,
    num_val: int = 500,
    batch_size: int = 32,
    map_size: int = 64,
    seq_len: int = 64,
    style_mode: str = "random",
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    创建 L2 训练/验证 DataLoader。
    
    Args:
        data_source: "mock" 使用 MockL3Dataset；"generated" 使用 FlowMPEnvDataset
        data_dir: 生成数据目录（data_source=generated 时指向含 data.npz 的目录）
        num_train / num_val: mock 时样本数；generated 时从 data 中按 train_ratio 划分
        batch_size, map_size, seq_len, style_mode: mock 专用
        train_ratio: generated 时训练集比例
        seed: 随机种子
        
    Returns:
        train_loader, val_loader
    """
    if data_source == "mock":
        train_ds = MockL3Dataset(
            num_samples=num_train,
            map_size=map_size,
            seq_len=seq_len,
            style_mode=style_mode,
            seed=seed,
        )
        val_ds = MockL3Dataset(
            num_samples=num_val,
            map_size=map_size,
            seq_len=seq_len,
            style_mode=style_mode,
            seed=seed + 1,
        )
    elif data_source == "generated":
        if not data_dir:
            raise ValueError("data_dir required when data_source='generated'")
        full_ds = FlowMPEnvDataset(data_path=data_dir)
        n = len(full_ds)
        n_train = int(n * train_ratio)
        n_val = n - n_train
        from torch.utils.data import random_split
        train_ds, val_ds = random_split(
            full_ds,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(seed),
        )
    else:
        raise ValueError(f"data_source must be 'mock' or 'generated', got {data_source}")
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if data_source == "mock" else 4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if data_source == "mock" else 2,
        pin_memory=True,
    )
    return train_loader, val_loader
