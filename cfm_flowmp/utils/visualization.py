"""
Visualization Utilities for CFM FlowMP

Functions for visualizing:
- Generated trajectories
- Flow fields
- Training progress
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def visualize_trajectory(
    positions: Union[torch.Tensor, np.ndarray],
    velocities: Optional[Union[torch.Tensor, np.ndarray]] = None,
    start_pos: Optional[Union[torch.Tensor, np.ndarray]] = None,
    goal_pos: Optional[Union[torch.Tensor, np.ndarray]] = None,
    obstacles: Optional[List] = None,
    title: str = "Generated Trajectory",
    figsize: tuple = (10, 10),
    show_velocity: bool = True,
    velocity_scale: float = 0.1,
    colormap: str = "viridis",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize a 2D trajectory.
    
    Args:
        positions: Trajectory positions [T, 2] or [B, T, 2]
        velocities: Optional velocity vectors [T, 2] or [B, T, 2]
        start_pos: Starting position [2]
        goal_pos: Goal position [2]
        obstacles: List of obstacle specifications
        title: Plot title
        figsize: Figure size
        show_velocity: Whether to show velocity arrows
        velocity_scale: Scale factor for velocity arrows
        colormap: Colormap for trajectory coloring
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    # Convert to numpy
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().numpy()
    if velocities is not None and isinstance(velocities, torch.Tensor):
        velocities = velocities.detach().cpu().numpy()
    
    # Handle batch dimension
    if positions.ndim == 3:
        # Multiple trajectories
        fig, ax = plt.subplots(figsize=figsize)
        
        for i in range(positions.shape[0]):
            _plot_single_trajectory(
                ax, positions[i],
                velocities[i] if velocities is not None else None,
                colormap=colormap,
                alpha=0.7,
                velocity_scale=velocity_scale,
                show_velocity=show_velocity and i == 0,  # Show velocity for first only
                label=f"Trajectory {i+1}" if positions.shape[0] <= 5 else None,
            )
    else:
        # Single trajectory
        fig, ax = plt.subplots(figsize=figsize)
        _plot_single_trajectory(
            ax, positions, velocities,
            colormap=colormap,
            velocity_scale=velocity_scale,
            show_velocity=show_velocity,
        )
    
    # Plot start and goal
    if start_pos is not None:
        if isinstance(start_pos, torch.Tensor):
            start_pos = start_pos.detach().cpu().numpy()
        ax.plot(start_pos[0], start_pos[1], 'go', markersize=15, label='Start', zorder=10)
    
    if goal_pos is not None:
        if isinstance(goal_pos, torch.Tensor):
            goal_pos = goal_pos.detach().cpu().numpy()
        ax.plot(goal_pos[0], goal_pos[1], 'r*', markersize=20, label='Goal', zorder=10)
    
    # Plot obstacles
    if obstacles is not None:
        for obs in obstacles:
            if obs['type'] == 'circle':
                circle = plt.Circle(
                    obs['center'], obs['radius'],
                    color='gray', alpha=0.5
                )
                ax.add_patch(circle)
            elif obs['type'] == 'rectangle':
                rect = plt.Rectangle(
                    obs['corner'], obs['width'], obs['height'],
                    color='gray', alpha=0.5
                )
                ax.add_patch(rect)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def _plot_single_trajectory(
    ax,
    positions: np.ndarray,
    velocities: Optional[np.ndarray] = None,
    colormap: str = "viridis",
    alpha: float = 1.0,
    velocity_scale: float = 0.1,
    show_velocity: bool = True,
    label: Optional[str] = None,
):
    """Plot a single trajectory with color-coded time."""
    T = len(positions)
    
    # Create colored line segments
    points = positions.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Color by time
    norm = plt.Normalize(0, T - 1)
    lc = LineCollection(segments, cmap=colormap, norm=norm, alpha=alpha)
    lc.set_array(np.arange(T - 1))
    lc.set_linewidth(2)
    
    ax.add_collection(lc)
    
    # Add colorbar
    # plt.colorbar(lc, ax=ax, label='Time step')
    
    # Plot velocity arrows
    if show_velocity and velocities is not None:
        # Subsample for clarity
        step = max(1, T // 10)
        for i in range(0, T, step):
            ax.arrow(
                positions[i, 0], positions[i, 1],
                velocities[i, 0] * velocity_scale,
                velocities[i, 1] * velocity_scale,
                head_width=0.05, head_length=0.02,
                fc='red', ec='red', alpha=0.5
            )
    
    # Auto-scale axes
    ax.autoscale()


def visualize_flow_field(
    model,
    t: float,
    start_pos: torch.Tensor,
    goal_pos: torch.Tensor,
    grid_range: tuple = (-3, 3),
    grid_size: int = 20,
    seq_len: int = 64,
    figsize: tuple = (12, 10),
    title: str = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize the learned flow field at a specific time.
    
    Args:
        model: Trained FlowMP model
        t: Flow time (0 to 1)
        start_pos: Starting position [D]
        goal_pos: Goal position [D]
        grid_range: Range for grid (min, max)
        grid_size: Number of grid points per dimension
        seq_len: Sequence length for model input
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Create grid
    x = np.linspace(grid_range[0], grid_range[1], grid_size)
    y = np.linspace(grid_range[0], grid_range[1], grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Flatten grid points
    grid_points = np.stack([X.flatten(), Y.flatten()], axis=-1)
    num_points = grid_points.shape[0]
    
    # Create pseudo-trajectories (constant position)
    positions = torch.tensor(grid_points, dtype=torch.float32, device=device)
    positions = positions.unsqueeze(1).expand(-1, seq_len, -1)  # [num_points, seq_len, 2]
    
    # Create full state (pos, vel, acc) - use zeros for vel and acc
    x_t = torch.zeros(num_points, seq_len, 6, device=device)
    x_t[..., :2] = positions
    
    # Prepare conditions
    start_pos_batch = start_pos.unsqueeze(0).expand(num_points, -1).to(device)
    goal_pos_batch = goal_pos.unsqueeze(0).expand(num_points, -1).to(device)
    t_batch = torch.full((num_points,), t, device=device)
    
    # Get model predictions
    with torch.no_grad():
        output = model(
            x_t=x_t,
            t=t_batch,
            start_pos=start_pos_batch,
            goal_pos=goal_pos_batch,
        )
    
    # Extract velocity field (first 2 channels)
    # Average over sequence
    velocity_field = output[..., :2].mean(dim=1).cpu().numpy()  # [num_points, 2]
    
    # Reshape for quiver plot
    U = velocity_field[:, 0].reshape(grid_size, grid_size)
    V = velocity_field[:, 1].reshape(grid_size, grid_size)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Velocity magnitude for coloring
    magnitude = np.sqrt(U**2 + V**2)
    
    # Quiver plot
    quiver = ax.quiver(X, Y, U, V, magnitude, cmap='coolwarm', alpha=0.8)
    plt.colorbar(quiver, ax=ax, label='Velocity Magnitude')
    
    # Streamlines
    ax.streamplot(X, Y, U, V, color='gray', linewidth=0.5, density=1.5, alpha=0.5)
    
    # Plot start and goal
    start_np = start_pos.cpu().numpy()
    goal_np = goal_pos.cpu().numpy()
    ax.plot(start_np[0], start_np[1], 'go', markersize=15, label='Start', zorder=10)
    ax.plot(goal_np[0], goal_np[1], 'r*', markersize=20, label='Goal', zorder=10)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title or f'Flow Field at t={t:.2f}')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    figsize: tuple = (15, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training curves.
    
    Args:
        history: Dictionary of metric histories
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    num_metrics = len(history)
    fig, axes = plt.subplots(1, min(num_metrics, 4), figsize=figsize)
    
    if num_metrics == 1:
        axes = [axes]
    
    for ax, (name, values) in zip(axes, history.items()):
        ax.plot(values)
        ax.set_xlabel('Step')
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_trajectory_animation(
    trajectories: List[np.ndarray],
    interval: int = 50,
    figsize: tuple = (10, 10),
    save_path: Optional[str] = None,
):
    """
    Create an animation of trajectory generation.
    
    Args:
        trajectories: List of trajectory snapshots during generation
        interval: Animation interval in milliseconds
        figsize: Figure size
        save_path: Path to save animation (as gif)
    """
    from matplotlib.animation import FuncAnimation
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine plot limits
    all_positions = np.concatenate(trajectories, axis=0)
    margin = 0.5
    ax.set_xlim(all_positions[:, :, 0].min() - margin, all_positions[:, :, 0].max() + margin)
    ax.set_ylim(all_positions[:, :, 1].min() - margin, all_positions[:, :, 1].max() + margin)
    
    line, = ax.plot([], [], 'b-', linewidth=2)
    point, = ax.plot([], [], 'ro', markersize=10)
    
    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point
    
    def animate(i):
        traj = trajectories[i][0]  # First trajectory in batch
        line.set_data(traj[:, 0], traj[:, 1])
        point.set_data([traj[-1, 0]], [traj[-1, 1]])
        ax.set_title(f'Generation Step {i+1}/{len(trajectories)}')
        return line, point
    
    anim = FuncAnimation(
        fig, animate, init_func=init,
        frames=len(trajectories), interval=interval, blit=True
    )
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=1000//interval)
    
    return anim
