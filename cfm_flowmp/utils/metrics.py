"""
Metrics for Trajectory Evaluation

Functions for computing:
- Trajectory quality metrics
- Physical consistency metrics
- Comparison metrics
"""

import torch
import numpy as np
from typing import Dict, Optional, Union


def compute_metrics(
    generated: Dict[str, torch.Tensor],
    target: Optional[Dict[str, torch.Tensor]] = None,
    obstacles: Optional[list] = None,
    dt: float = 0.1,
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for generated trajectories.
    
    Args:
        generated: Generated trajectory dictionary with 'positions', 'velocities', 'accelerations'
        target: Optional ground truth dictionary
        obstacles: Optional list of obstacles for collision checking
        dt: Time step for derivative computations
        
    Returns:
        Dictionary of metric values
    """
    metrics = {}
    
    positions = generated['positions']
    velocities = generated['velocities']
    accelerations = generated['accelerations']
    
    # Ensure 3D: [B, T, D]
    if positions.dim() == 2:
        positions = positions.unsqueeze(0)
        velocities = velocities.unsqueeze(0)
        accelerations = accelerations.unsqueeze(0)
    
    B, T, D = positions.shape
    
    # ============ Smoothness Metrics ============
    
    # Path length
    path_lengths = compute_path_length(positions)
    metrics['path_length_mean'] = path_lengths.mean().item()
    metrics['path_length_std'] = path_lengths.std().item()
    
    # Jerk (rate of change of acceleration)
    jerk = compute_jerk(accelerations, dt)
    metrics['jerk_mean'] = jerk.mean().item()
    metrics['jerk_max'] = jerk.max().item()
    
    # Curvature
    curvature = compute_curvature(positions, velocities)
    metrics['curvature_mean'] = curvature.mean().item()
    metrics['curvature_max'] = curvature.max().item()
    
    # Velocity smoothness (variation)
    vel_smoothness = compute_velocity_smoothness(velocities)
    metrics['velocity_smoothness'] = vel_smoothness.mean().item()
    
    # ============ Physical Consistency ============
    
    # Check if derivatives are consistent
    consistency_error = compute_derivative_consistency(positions, velocities, dt)
    metrics['derivative_consistency_error'] = consistency_error.mean().item()
    
    # ============ Goal-Reaching Metrics ============
    
    if target is not None:
        target_positions = target['positions']
        if target_positions.dim() == 2:
            target_positions = target_positions.unsqueeze(0)
        
        # Position error
        pos_error = (positions - target_positions).norm(dim=-1)
        metrics['position_mse'] = pos_error.pow(2).mean().item()
        metrics['position_mae'] = pos_error.mean().item()
        
        # Goal reaching error
        goal_error = (positions[:, -1] - target_positions[:, -1]).norm(dim=-1)
        metrics['goal_reaching_error'] = goal_error.mean().item()
        
        # Frechet distance
        frechet_dist = compute_frechet_distance(positions, target_positions)
        metrics['frechet_distance'] = frechet_dist.mean().item()
    
    # ============ Obstacle Metrics ============
    
    if obstacles is not None:
        collision_rate = compute_collision_rate(positions, obstacles)
        metrics['collision_rate'] = collision_rate
        
        min_clearance = compute_min_clearance(positions, obstacles)
        metrics['min_clearance'] = min_clearance
    
    return metrics


def compute_path_length(positions: torch.Tensor) -> torch.Tensor:
    """
    Compute total path length for each trajectory.
    
    Args:
        positions: [B, T, D] position trajectories
        
    Returns:
        [B] path lengths
    """
    diffs = positions[:, 1:] - positions[:, :-1]
    segment_lengths = diffs.norm(dim=-1)
    return segment_lengths.sum(dim=-1)


def compute_jerk(accelerations: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
    """
    Compute jerk (derivative of acceleration) magnitude.
    
    Args:
        accelerations: [B, T, D] acceleration trajectories
        dt: Time step
        
    Returns:
        [B, T-1] jerk magnitudes
    """
    jerk = (accelerations[:, 1:] - accelerations[:, :-1]) / dt
    return jerk.norm(dim=-1)


def compute_curvature(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute trajectory curvature.
    
    Curvature κ = |v × a| / |v|^3 (for 2D: κ = (vx*ay - vy*ax) / |v|^3)
    
    Args:
        positions: [B, T, D] positions
        velocities: [B, T, D] velocities
        eps: Small value for numerical stability
        
    Returns:
        [B, T] curvature values
    """
    # Compute acceleration from velocity
    accelerations = torch.zeros_like(velocities)
    accelerations[:, 1:-1] = (velocities[:, 2:] - velocities[:, :-2]) / 2
    accelerations[:, 0] = velocities[:, 1] - velocities[:, 0]
    accelerations[:, -1] = velocities[:, -1] - velocities[:, -2]
    
    # For 2D: cross product gives scalar
    if positions.shape[-1] == 2:
        cross = velocities[..., 0] * accelerations[..., 1] - velocities[..., 1] * accelerations[..., 0]
        vel_norm = velocities.norm(dim=-1).clamp(min=eps)
        curvature = cross.abs() / (vel_norm ** 3)
    else:
        # For 3D: use vector cross product
        cross = torch.cross(velocities, accelerations, dim=-1)
        vel_norm = velocities.norm(dim=-1, keepdim=True).clamp(min=eps)
        curvature = cross.norm(dim=-1) / (vel_norm.squeeze(-1) ** 3)
    
    return curvature


def compute_velocity_smoothness(velocities: torch.Tensor) -> torch.Tensor:
    """
    Compute velocity smoothness as inverse of variation.
    
    Args:
        velocities: [B, T, D] velocity trajectories
        
    Returns:
        [B] smoothness scores (higher is smoother)
    """
    vel_diff = velocities[:, 1:] - velocities[:, :-1]
    variation = vel_diff.norm(dim=-1).mean(dim=-1)
    return 1.0 / (variation + 1e-6)


def compute_derivative_consistency(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    dt: float = 0.1,
) -> torch.Tensor:
    """
    Check if velocity is consistent with position derivative.
    
    Args:
        positions: [B, T, D] positions
        velocities: [B, T, D] velocities
        dt: Time step
        
    Returns:
        [B, T-2] consistency errors
    """
    # Numerical derivative of positions
    pos_deriv = (positions[:, 2:] - positions[:, :-2]) / (2 * dt)
    
    # Compare with actual velocities (middle points)
    vel_mid = velocities[:, 1:-1]
    
    error = (pos_deriv - vel_mid).norm(dim=-1)
    return error


def compute_frechet_distance(
    trajectory1: torch.Tensor,
    trajectory2: torch.Tensor,
) -> torch.Tensor:
    """
    Compute discrete Frechet distance between trajectories.
    
    Args:
        trajectory1: [B, T1, D] first trajectory
        trajectory2: [B, T2, D] second trajectory
        
    Returns:
        [B] Frechet distances
    """
    B, T1, D = trajectory1.shape
    T2 = trajectory2.shape[1]
    
    distances = []
    
    for b in range(B):
        traj1 = trajectory1[b]  # [T1, D]
        traj2 = trajectory2[b]  # [T2, D]
        
        # Dynamic programming for Frechet distance
        ca = torch.full((T1, T2), float('inf'), device=trajectory1.device)
        
        ca[0, 0] = (traj1[0] - traj2[0]).norm()
        
        for i in range(1, T1):
            ca[i, 0] = max(ca[i-1, 0].item(), (traj1[i] - traj2[0]).norm().item())
        
        for j in range(1, T2):
            ca[0, j] = max(ca[0, j-1].item(), (traj1[0] - traj2[j]).norm().item())
        
        for i in range(1, T1):
            for j in range(1, T2):
                point_dist = (traj1[i] - traj2[j]).norm()
                ca[i, j] = max(point_dist, min(ca[i-1, j], ca[i-1, j-1], ca[i, j-1]))
        
        distances.append(ca[T1-1, T2-1])
    
    return torch.stack(distances)


def compute_collision_rate(
    positions: torch.Tensor,
    obstacles: list,
) -> float:
    """
    Compute rate of trajectories that collide with obstacles.
    
    Args:
        positions: [B, T, D] position trajectories
        obstacles: List of obstacle dictionaries
        
    Returns:
        Collision rate (0 to 1)
    """
    B, T, D = positions.shape
    positions_np = positions.detach().cpu().numpy()
    
    collisions = 0
    
    for b in range(B):
        traj = positions_np[b]
        has_collision = False
        
        for obs in obstacles:
            if obs['type'] == 'circle':
                center = np.array(obs['center'])
                radius = obs['radius']
                
                distances = np.linalg.norm(traj - center, axis=-1)
                if np.any(distances < radius):
                    has_collision = True
                    break
            
            elif obs['type'] == 'rectangle':
                corner = np.array(obs['corner'])
                width = obs['width']
                height = obs['height']
                
                inside_x = (traj[:, 0] >= corner[0]) & (traj[:, 0] <= corner[0] + width)
                inside_y = (traj[:, 1] >= corner[1]) & (traj[:, 1] <= corner[1] + height)
                
                if np.any(inside_x & inside_y):
                    has_collision = True
                    break
        
        if has_collision:
            collisions += 1
    
    return collisions / B


def compute_min_clearance(
    positions: torch.Tensor,
    obstacles: list,
) -> float:
    """
    Compute minimum clearance from obstacles across all trajectories.
    
    Args:
        positions: [B, T, D] position trajectories
        obstacles: List of obstacle dictionaries
        
    Returns:
        Minimum clearance distance
    """
    positions_np = positions.detach().cpu().numpy()
    min_clearance = float('inf')
    
    for b in range(positions.shape[0]):
        traj = positions_np[b]
        
        for obs in obstacles:
            if obs['type'] == 'circle':
                center = np.array(obs['center'])
                radius = obs['radius']
                
                distances = np.linalg.norm(traj - center, axis=-1)
                clearance = distances.min() - radius
                min_clearance = min(min_clearance, clearance)
            
            elif obs['type'] == 'rectangle':
                # Distance to rectangle (simplified)
                corner = np.array(obs['corner'])
                center = corner + np.array([obs['width']/2, obs['height']/2])
                
                distances = np.linalg.norm(traj - center, axis=-1)
                clearance = distances.min() - max(obs['width'], obs['height']) / 2
                min_clearance = min(min_clearance, clearance)
    
    return min_clearance


def compute_success_rate(
    positions: torch.Tensor,
    goal_positions: torch.Tensor,
    threshold: float = 0.1,
) -> float:
    """
    Compute rate of trajectories that reach the goal.
    
    Args:
        positions: [B, T, D] position trajectories
        goal_positions: [B, D] goal positions
        threshold: Distance threshold for success
        
    Returns:
        Success rate (0 to 1)
    """
    final_positions = positions[:, -1]  # [B, D]
    distances = (final_positions - goal_positions).norm(dim=-1)  # [B]
    
    successes = (distances < threshold).float().sum()
    return (successes / positions.shape[0]).item()
