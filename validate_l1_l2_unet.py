"""
Validation Script for L1+L2 Integration with Unet

This script tests the L1+L2 integration using Unet model:
1. Loads L2 model (Unet) trained on generated trajectories and heatmaps
2. Tests L1 reactive control layer
3. Validates warm-start mechanism effectiveness

Usage:
    python validate_l1_l2_unet.py --checkpoint checkpoints_l2_unet/final_model.pt --data_dir traj_data/cfm_env
    python validate_l1_l2_unet.py --checkpoint checkpoints_l2_unet/final_model.pt --data_dir traj_data/cfm_env --test_warm_start
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from typing import Dict, List, Optional

from cfm_flowmp.models import create_l2_safety_cfm, L2Config
from cfm_flowmp.data import FlowMPEnvDataset
from cfm_flowmp.inference import L1ReactiveController, L1Config


def parse_args():
    parser = argparse.ArgumentParser(description="Validate L1+L2 integration with Unet")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to L2 model checkpoint')
    parser.add_argument('--data_dir', type=str, default='traj_data/cfm_env',
                        help='Directory containing data.npz with trajectories and heatmaps')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Sample index from dataset')
    parser.add_argument('--num_samples', type=int, default=64,
                        help='Number of trajectory samples to generate (L2 output)')
    parser.add_argument('--l1_iterations', type=int, default=10,
                        help='Number of L1 MPPI iterations')
    parser.add_argument('--test_warm_start', action='store_true',
                        help='Test warm-start mechanism (compare with/without)')
    parser.add_argument('--num_steps', type=int, default=10,
                        help='Number of planning steps for warm-start test')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model_type', type=str, default=None,
                        choices=['transformer', 'unet1d'],
                        help='Model type (auto-detect if not specified)')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save visualization plots')
    parser.add_argument('--output_dir', type=str, default='validation_results',
                        help='Directory to save results')
    return parser.parse_args()


def detect_model_type(checkpoint_path, device):
    """Detect model type from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Check for Unet keys
    has_unet_keys = any('down_blocks' in k or 'up_blocks' in k for k in state_dict.keys())
    # Check for Transformer keys
    has_transformer_keys = any('blocks.' in k and 'attn' in k for k in state_dict.keys())
    
    if has_unet_keys:
        return 'unet1d'
    elif has_transformer_keys:
        return 'transformer'
    else:
        # Default to transformer if unclear
        return 'transformer'


def load_model(checkpoint_path, device, model_type=None):
    """Load trained L2 model (auto-detect or use specified type)."""
    print(f"Loading L2 model from {checkpoint_path}...")
    
    # Auto-detect model type if not specified
    if model_type is None:
        model_type = detect_model_type(checkpoint_path, device)
        print(f"  Auto-detected model type: {model_type}")
    else:
        print(f"  Using specified model type: {model_type}")
    
    # Create model based on detected/specified type
    if model_type == 'unet1d':
        model = create_l2_safety_cfm(
            model_type='unet1d',
            state_dim=2,
            max_seq_len=64,
            hidden_dim=256,
            num_layers=8,
            num_heads=8,
            cost_map_channels=1,
            cost_map_latent_dim=256,
            cost_map_encoder_type='single_scale',
            use_style_conditioning=True,
            style_dim=3,  # [w_safety, w_energy, w_smooth]
            use_8step_schedule=True,
            unet_base_channels=128,
            unet_channel_mults=(1, 2, 4, 8),
        )
    else:  # transformer
        model = create_l2_safety_cfm(
            model_type='transformer',
            state_dim=2,
            max_seq_len=64,
            hidden_dim=256,
            num_layers=8,
            num_heads=8,
            cost_map_channels=1,
            cost_map_latent_dim=256,
            cost_map_encoder_type='single_scale',
            use_style_conditioning=True,
            style_dim=3,  # [w_safety, w_energy, w_smooth]
            use_8step_schedule=True,
        )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    
    print(f"L2 model ({model_type}) loaded successfully!")
    return model


def load_data_sample(data_dir, sample_idx, device):
    """Load a sample from the generated dataset."""
    print(f"Loading data sample {sample_idx} from {data_dir}...")
    
    dataset = FlowMPEnvDataset(data_dir)
    if sample_idx >= len(dataset):
        print(f"Warning: sample_idx {sample_idx} >= dataset size {len(dataset)}, using 0")
        sample_idx = 0
    
    sample = dataset[sample_idx]
    
    # Extract data
    cost_map = sample['cost_map'].to(device)  # [1, H, W] or [H, W]
    if cost_map.dim() == 2:
        cost_map = cost_map.unsqueeze(0)
    if cost_map.dim() == 3:
        cost_map = cost_map.unsqueeze(0)  # [1, 1, H, W]
    
    start_state = sample['start_state'].to(device)  # [6]
    goal_state = sample['goal_state'].to(device)  # [4]
    style_weights = sample.get('style_weights', None)
    if style_weights is None:
        style_weights = torch.tensor([[0.5, 0.5]], device=device, dtype=torch.float32)  # [1, 2]
    else:
        style_weights = style_weights.to(device)  # Ensure on correct device
    if style_weights.dim() == 1:
        style_weights = style_weights.unsqueeze(0)
    
    # Get ground truth trajectory for comparison
    positions = sample.get('positions', None)
    if positions is not None:
        positions = positions.to(device)
    
    print(f"  Cost map shape: {cost_map.shape}")
    print(f"  Start state: {start_state.cpu().numpy()}")
    print(f"  Goal state: {goal_state.cpu().numpy()}")
    print(f"  Style weights: {style_weights.cpu().numpy()}")
    
    return {
        'cost_map': cost_map,
        'start_state': start_state,
        'goal_state': goal_state,
        'style_weights': style_weights,
        'ground_truth_positions': positions,
    }


def visualize_second_order_dynamics(
    positions: np.ndarray,  # [N, T, 2]
    velocities: np.ndarray,  # [N, T, 2]
    accelerations: np.ndarray,  # [N, T, 2]
    start_state: torch.Tensor,
    goal_state: torch.Tensor,
    output_dir: Path,
    num_trajectories_to_plot: int = 10,
):
    """
    Visualize second-order dynamics: position, velocity, and acceleration over time.
    
    Args:
        positions: Position trajectories [N, T, 2]
        velocities: Velocity trajectories [N, T, 2]
        accelerations: Acceleration trajectories [N, T, 2]
        start_state: Start state tensor
        goal_state: Goal state tensor
        output_dir: Output directory for saving plots
        num_trajectories_to_plot: Number of trajectories to plot (for clarity)
    """
    N, T, D = positions.shape
    time_steps = np.linspace(0, 1, T)
    
    # Select a subset of trajectories for visualization
    indices = np.linspace(0, N - 1, num_trajectories_to_plot, dtype=int)
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # ============ Row 1: Position trajectories ============
    # X position over time
    ax1 = fig.add_subplot(gs[0, 0])
    for idx in indices:
        ax1.plot(time_steps, positions[idx, :, 0], alpha=0.3, linewidth=1)
    ax1.axhline(y=start_state[0].item(), color='g', linestyle='--', linewidth=2, label='Start X')
    ax1.axhline(y=goal_state[0].item(), color='r', linestyle='--', linewidth=2, label='Goal X')
    ax1.set_xlabel('Time (normalized)')
    ax1.set_ylabel('X Position')
    ax1.set_title('X Position vs Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Y position over time
    ax2 = fig.add_subplot(gs[0, 1])
    for idx in indices:
        ax2.plot(time_steps, positions[idx, :, 1], alpha=0.3, linewidth=1)
    ax2.axhline(y=start_state[1].item(), color='g', linestyle='--', linewidth=2, label='Start Y')
    ax2.axhline(y=goal_state[1].item(), color='r', linestyle='--', linewidth=2, label='Goal Y')
    ax2.set_xlabel('Time (normalized)')
    ax2.set_ylabel('Y Position')
    ax2.set_title('Y Position vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Position trajectory in 2D space
    ax3 = fig.add_subplot(gs[0, 2])
    for idx in indices:
        ax3.plot(positions[idx, :, 0], positions[idx, :, 1], alpha=0.3, linewidth=1)
    start_pos = start_state[:2].cpu().numpy()
    goal_pos = goal_state[:2].cpu().numpy()
    ax3.plot(start_pos[0], start_pos[1], 'go', markersize=12, label='Start', zorder=10)
    ax3.plot(goal_pos[0], goal_pos[1], 'r*', markersize=15, label='Goal', zorder=10)
    ax3.set_xlabel('X Position')
    ax3.set_ylabel('Y Position')
    ax3.set_title('2D Position Trajectories')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')
    
    # ============ Row 2: Velocity trajectories ============
    # X velocity over time
    ax4 = fig.add_subplot(gs[1, 0])
    for idx in indices:
        ax4.plot(time_steps, velocities[idx, :, 0], alpha=0.3, linewidth=1)
    if start_state.shape[0] >= 4:
        ax4.axhline(y=start_state[2].item(), color='g', linestyle='--', linewidth=2, label='Start Vx')
    if goal_state.shape[0] >= 3:
        ax4.axhline(y=goal_state[2].item() if goal_state.shape[0] > 2 else 0, 
                   color='r', linestyle='--', linewidth=2, label='Goal Vx')
    ax4.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    ax4.set_xlabel('Time (normalized)')
    ax4.set_ylabel('X Velocity')
    ax4.set_title('X Velocity vs Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Y velocity over time
    ax5 = fig.add_subplot(gs[1, 1])
    for idx in indices:
        ax5.plot(time_steps, velocities[idx, :, 1], alpha=0.3, linewidth=1)
    if start_state.shape[0] >= 4:
        ax5.axhline(y=start_state[3].item(), color='g', linestyle='--', linewidth=2, label='Start Vy')
    if goal_state.shape[0] >= 4:
        ax5.axhline(y=goal_state[3].item() if goal_state.shape[0] > 3 else 0, 
                   color='r', linestyle='--', linewidth=2, label='Goal Vy')
    ax5.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    ax5.set_xlabel('Time (normalized)')
    ax5.set_ylabel('Y Velocity')
    ax5.set_title('Y Velocity vs Time')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Velocity magnitude over time
    ax6 = fig.add_subplot(gs[1, 2])
    for idx in indices:
        vel_mag = np.linalg.norm(velocities[idx], axis=1)
        ax6.plot(time_steps, vel_mag, alpha=0.3, linewidth=1)
    ax6.set_xlabel('Time (normalized)')
    ax6.set_ylabel('Velocity Magnitude')
    ax6.set_title('Velocity Magnitude vs Time')
    ax6.grid(True, alpha=0.3)
    
    # ============ Row 3: Acceleration trajectories ============
    # X acceleration over time
    ax7 = fig.add_subplot(gs[2, 0])
    for idx in indices:
        ax7.plot(time_steps, accelerations[idx, :, 0], alpha=0.3, linewidth=1)
    if start_state.shape[0] >= 6:
        ax7.axhline(y=start_state[4].item(), color='g', linestyle='--', linewidth=2, label='Start Ax')
    ax7.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    ax7.set_xlabel('Time (normalized)')
    ax7.set_ylabel('X Acceleration')
    ax7.set_title('X Acceleration vs Time')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Y acceleration over time
    ax8 = fig.add_subplot(gs[2, 1])
    for idx in indices:
        ax8.plot(time_steps, accelerations[idx, :, 1], alpha=0.3, linewidth=1)
    if start_state.shape[0] >= 6:
        ax8.axhline(y=start_state[5].item(), color='g', linestyle='--', linewidth=2, label='Start Ay')
    ax8.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    ax8.set_xlabel('Time (normalized)')
    ax8.set_ylabel('Y Acceleration')
    ax8.set_title('Y Acceleration vs Time')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Acceleration magnitude over time
    ax9 = fig.add_subplot(gs[2, 2])
    for idx in indices:
        acc_mag = np.linalg.norm(accelerations[idx], axis=1)
        ax9.plot(time_steps, acc_mag, alpha=0.3, linewidth=1)
    ax9.set_xlabel('Time (normalized)')
    ax9.set_ylabel('Acceleration Magnitude')
    ax9.set_title('Acceleration Magnitude vs Time')
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle('L2 Second-Order Dynamics Visualization', fontsize=16, fontweight='bold')
    
    save_path = output_dir / 'l2_second_order_dynamics.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved second-order dynamics visualization to {save_path}")
    plt.close()


def test_l1_l2_integration(model, data_sample, num_samples, l1_iterations, device, output_dir=None):
    """
    Test L1+L2 integration: L2 generates anchors, L1 optimizes.
    
    Args:
        output_dir: Output directory for saving results (default: args.output_dir)
    """
    print("\n" + "=" * 80)
    print("Test: L1+L2 Integration")
    print("=" * 80)
    
    cost_map = data_sample['cost_map']
    start_state = data_sample['start_state']
    goal_state = data_sample['goal_state']
    style_weights = data_sample['style_weights']
    
    # ============ Step 1: L2 generates trajectory anchors ============
    print("\n[Step 1] L2 generating trajectory anchors...")
    t_l2_start = time.time()
    
    with torch.no_grad():
        # Extract positions from start/goal states
        start_pos = start_state[:2].unsqueeze(0)  # [1, 2]
        goal_pos = goal_state[:2].unsqueeze(0)  # [1, 2]
        start_vel = start_state[2:4].unsqueeze(0) if start_state.shape[0] >= 4 else None
        
        # Generate trajectories using L2
        l2_result = model.generate_trajectory_anchors(
            cost_map=cost_map,
            x_curr=start_state.unsqueeze(0),  # [1, 6]
            x_goal=goal_state.unsqueeze(0),  # [1, 4]
            w_style=style_weights,
            num_samples=num_samples,
        )
    
    t_l2 = time.time() - t_l2_start
    
    # Extract all dynamics components
    l2_trajectories = l2_result['trajectories'].cpu().numpy()  # [N, T, 2] positions
    l2_velocities = l2_result['velocities'].cpu().numpy()  # [N, T, 2] velocities
    l2_accelerations = l2_result['accelerations'].cpu().numpy()  # [N, T, 2] accelerations
    l2_full_states = l2_result['full_states'].cpu().numpy()  # [N, T, 6] [p, v, a]
    
    print(f"  Generated {len(l2_trajectories)} trajectory anchors")
    print(f"  L2 generation time: {t_l2*1000:.2f} ms")
    
    # Save L2 trajectory cluster
    if output_dir is None:
        # Try to use global args if available
        try:
            output_dir = Path(args.output_dir)
        except NameError:
            output_dir = Path('validation_results')
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    l2_data = {
        'trajectories': l2_trajectories,  # [N, T, 2] positions
        'velocities': l2_velocities,  # [N, T, 2] velocities
        'accelerations': l2_accelerations,  # [N, T, 2] accelerations
        'full_states': l2_full_states,  # [N, T, 6] [p, v, a]
        'num_samples': len(l2_trajectories),
        'time_horizon': 5.0,  # Default time horizon
        'start_state': start_state.cpu().numpy(),
        'goal_state': goal_state.cpu().numpy(),
    }
    
    l2_save_path = output_dir / 'l2_trajectory_cluster.npz'
    np.savez(l2_save_path, **l2_data)
    print(f"  ✓ Saved L2 trajectory cluster to {l2_save_path}")
    
    # Visualize second-order dynamics
    visualize_second_order_dynamics(
        l2_trajectories, l2_velocities, l2_accelerations,
        start_state, goal_state, output_dir
    )
    
    # ============ Step 2: L1 optimizes ============
    print("\n[Step 2] L1 optimizing trajectories...")
    
    
    # Create semantic cost function from cost map
    def semantic_cost_fn(positions: torch.Tensor) -> torch.Tensor:
        """Compute semantic cost from cost map."""
        B, T, D = positions.shape
        costs = torch.zeros(B, device=positions.device)
        
        cost_map_np = cost_map[0, 0].cpu().numpy()
        map_size = cost_map_np.shape[0]
        
        for b in range(B):
            for t in range(T):
                pos = positions[b, t].cpu().numpy()
                # Convert normalized position to map indices
                x_idx = int(np.clip(pos[0] * (map_size - 1), 0, map_size - 1))
                y_idx = int(np.clip(pos[1] * (map_size - 1), 0, map_size - 1))
                costs[b] += cost_map_np[y_idx, x_idx]
        
        return costs
    
    # Set PyTorch to use all available CPU cores for L1 layer
    num_cpu_cores = os.cpu_count()
    if num_cpu_cores is not None:
        torch.set_num_threads(num_cpu_cores)
        print(f"  Using {num_cpu_cores} CPU cores for L1 layer")
    
    # Create L1 controller with semantic cost function
    l1_config = L1Config(
        n_samples_per_mode=100,
        n_timesteps=l2_trajectories.shape[1],
        time_horizon=5.0,
        tube_radius=0.5,
        w_semantic=1.0,
        w_tube=10.0,
        w_energy=0.1,
        temperature=1.0,
        use_warm_start=False,  # First test without warm-start
        device=device,
    )
    
    l1_controller = L1ReactiveController(config=l1_config, semantic_fn=semantic_cost_fn)
    
    # Initialize L1 from L2 output
    t_l1_start = time.time()
    l1_controller.initialize_from_l2_output(l2_result)
    
    # Optimize
    l1_result = l1_controller.optimize(n_iterations=l1_iterations)
    t_l1 = time.time() - t_l1_start
    
    optimal_control = l1_controller.get_next_control(
        l2_result, n_iterations=l1_iterations
    )
    
    print(f"  L1 optimization time: {t_l1*1000:.2f} ms")
    print(f"  Best cost: {l1_result['best_cost']:.4f}")
    print(f"  Mean cost: {l1_result['mean_cost']:.4f}")
    print(f"  Best mode: {l1_result['best_mode']}")
    
    optimal_traj = optimal_control.cpu().numpy()  # [T, 2]
    
    # ============ Visualization ============
    if args.save_plots:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: L2 anchors
        ax1 = axes[0]
        cost_map_np = cost_map[0, 0].cpu().numpy()
        ax1.imshow(cost_map_np, extent=[0, 1, 0, 1], origin='lower', cmap='Reds', alpha=0.5)
        
        # Plot L2 trajectories
        for traj in l2_trajectories[:20]:  # Plot subset
            ax1.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.2, linewidth=0.5)
        
        # Plot start and goal
        start_pos_np = start_state[:2].cpu().numpy()
        goal_pos_np = goal_state[:2].cpu().numpy()
        ax1.plot(start_pos_np[0], start_pos_np[1], 'go', markersize=12, label='Start')
        ax1.plot(goal_pos_np[0], goal_pos_np[1], 'r*', markersize=15, label='Goal')
        
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title(f'L2 Trajectory Anchors ({len(l2_trajectories)} samples)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: L1 optimal trajectory
        ax2 = axes[1]
        ax2.imshow(cost_map_np, extent=[0, 1, 0, 1], origin='lower', cmap='Reds', alpha=0.5)
        
        # Plot L1 optimal
        ax2.plot(optimal_traj[:, 0], optimal_traj[:, 1], 'g-', linewidth=3, label='L1 Optimal', alpha=0.8)
        
        # Plot best L2 trajectory for comparison
        best_l2_idx = l1_result['best_mode']
        if best_l2_idx < len(l2_trajectories):
            ax2.plot(l2_trajectories[best_l2_idx, :, 0], 
                    l2_trajectories[best_l2_idx, :, 1], 
                    'b--', linewidth=2, label='Best L2 Anchor', alpha=0.6)
        
        # Plot start and goal
        ax2.plot(start_pos_np[0], start_pos_np[1], 'go', markersize=12, label='Start')
        ax2.plot(goal_pos_np[0], goal_pos_np[1], 'r*', markersize=15, label='Goal')
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title(f'L1 Optimized Trajectory (Cost: {l1_result["best_cost"]:.4f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('L1+L2 Integration Test', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = Path(args.output_dir) / 'l1_l2_integration.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Plot saved to {save_path}")
        plt.close()
    
    return {
        'l2_trajectories': l2_trajectories,
        'l1_optimal': optimal_traj,
        'l1_result': l1_result,
        't_l2': t_l2,
        't_l1': t_l1,
    }


def test_warm_start(model, data_sample, num_samples, l1_iterations, num_steps, device):
    """
    Test warm-start mechanism: compare with/without warm-start.
    """
    print("\n" + "=" * 80)
    print("Test: Warm-Start Mechanism")
    print("=" * 80)
    
    # Set PyTorch to use all available CPU cores for L1 layer
    num_cpu_cores = os.cpu_count()
    if num_cpu_cores is not None:
        torch.set_num_threads(num_cpu_cores)
        print(f"  Using {num_cpu_cores} CPU cores for L1 layer")
    
    cost_map = data_sample['cost_map']
    start_state = data_sample['start_state']
    goal_state = data_sample['goal_state']
    style_weights = data_sample['style_weights']
    
    results_without = []
    results_with = []
    
    # Create semantic cost function
    def semantic_cost_fn(positions: torch.Tensor) -> torch.Tensor:
        """Compute semantic cost from cost map."""
        B, T, D = positions.shape
        costs = torch.zeros(B, device=positions.device)
        
        cost_map_np = cost_map[0, 0].cpu().numpy()
        map_size = cost_map_np.shape[0]
        
        for b in range(B):
            for t in range(T):
                pos = positions[b, t].cpu().numpy()
                x_idx = int(np.clip(pos[0] * (map_size - 1), 0, map_size - 1))
                y_idx = int(np.clip(pos[1] * (map_size - 1), 0, map_size - 1))
                costs[b] += cost_map_np[y_idx, x_idx]
        
        return costs
    
    # ============ Test WITHOUT warm-start ============
    print("\n[Test 1] Without Warm-Start...")
    
    current_pos = start_state[:2].clone()
    current_vel = start_state[2:4].clone() if start_state.shape[0] >= 4 else None
    previous_optimal = None  # No warm-start
    
    for step in range(num_steps):
        print(f"  Step {step+1}/{num_steps}...", end=' ')
        
        # Prepare current state
        if current_vel is not None:
            x_curr = torch.cat([current_pos, current_vel, torch.zeros(2, device=device)]).unsqueeze(0)
        else:
            x_curr = torch.cat([current_pos, torch.zeros(4, device=device)]).unsqueeze(0)
        
        # Generate trajectory (no warm-start: always use random noise)
        with torch.no_grad():
            l2_result = model.generate_trajectory_anchors(
                cost_map=cost_map,
                x_curr=x_curr,
                x_goal=goal_state.unsqueeze(0),
                w_style=style_weights,
                num_samples=num_samples,
            )
        
        # L1 optimize
        l1_config = L1Config(
            n_samples_per_mode=50,  # Fewer samples for speed
            n_timesteps=l2_result['trajectories'].shape[1],
            use_warm_start=False,
            device=device,
        )
        l1_controller = L1ReactiveController(config=l1_config, semantic_fn=semantic_cost_fn)
        l1_controller.initialize_from_l2_output(l2_result)
        l1_result = l1_controller.optimize(n_iterations=l1_iterations)
        optimal_control = l1_controller.get_next_control(l2_result, n_iterations=l1_iterations)
        
        # Execute first step
        if optimal_control.shape[0] > 0:
            current_pos = optimal_control[0, :2].clone()
            if optimal_control.shape[1] >= 4:
                current_vel = optimal_control[0, 2:4].clone()
        
        results_without.append({
            'cost': l1_result['best_cost'],
            'position': current_pos.cpu().numpy(),
        })
        
        print(f"cost={l1_result['best_cost']:.4f}")
    
    # ============ Test WITH warm-start ============
    print("\n[Test 2] With Warm-Start...")
    
    current_pos = start_state[:2].clone()
    current_vel = start_state[2:4].clone() if start_state.shape[0] >= 4 else None
    previous_optimal = None  # Will store previous optimal trajectory for warm-start
    
    for step in range(num_steps):
        print(f"  Step {step+1}/{num_steps}...", end=' ')
        
        # Prepare current state
        if current_vel is not None:
            x_curr = torch.cat([current_pos, current_vel, torch.zeros(2, device=device)]).unsqueeze(0)
        else:
            x_curr = torch.cat([current_pos, torch.zeros(4, device=device)]).unsqueeze(0)
        
        # Generate trajectory (with warm-start: modify model's initial noise)
        # Note: This is a simplified warm-start - in full implementation,
        # we would modify the model's generate_trajectory_anchors to accept warm_start_prior
        with torch.no_grad():
            l2_result = model.generate_trajectory_anchors(
                cost_map=cost_map,
                x_curr=x_curr,
                x_goal=goal_state.unsqueeze(0),
                w_style=style_weights,
                num_samples=num_samples,
            )
        
        # L1 optimize
        l1_config = L1Config(
            n_samples_per_mode=50,
            n_timesteps=l2_result['trajectories'].shape[1],
            use_warm_start=True,
            warm_start_noise_scale=0.1,
            device=device,
        )
        l1_controller = L1ReactiveController(config=l1_config, semantic_fn=semantic_cost_fn)
        
        # Set warm-start state if available
        if previous_optimal is not None:
            l1_controller.previous_optimal_control = previous_optimal
        
        l1_controller.initialize_from_l2_output(l2_result)
        l1_result = l1_controller.optimize(n_iterations=l1_iterations)
        optimal_control = l1_controller.get_next_control(l2_result, n_iterations=l1_iterations)
        
        # Store for next iteration (warm-start)
        previous_optimal = optimal_control.clone()
        
        # Execute first step
        if optimal_control.shape[0] > 0:
            current_pos = optimal_control[0, :2].clone()
            if optimal_control.shape[1] >= 4:
                current_vel = optimal_control[0, 2:4].clone()
        
        results_with.append({
            'cost': l1_result['best_cost'],
            'position': current_pos.cpu().numpy(),
        })
        
        print(f"cost={l1_result['best_cost']:.4f}")
    
    # ============ Compare results ============
    print("\n" + "=" * 80)
    print("Warm-Start Comparison Results")
    print("=" * 80)
    
    avg_cost_without = np.mean([r['cost'] for r in results_without])
    avg_cost_with = np.mean([r['cost'] for r in results_with])
    
    # Compute path smoothness (jerk)
    def compute_smoothness(positions):
        if len(positions) < 3:
            return 0.0
        positions = np.array(positions)
        velocities = np.diff(positions, axis=0)
        accelerations = np.diff(velocities, axis=0)
        jerks = np.diff(accelerations, axis=0)
        return np.mean(np.linalg.norm(jerks, axis=1))
    
    smoothness_without = compute_smoothness([r['position'] for r in results_without])
    smoothness_with = compute_smoothness([r['position'] for r in results_with])
    
    # Compute path length
    def compute_path_length(positions):
        positions = np.array(positions)
        if len(positions) < 2:
            return 0.0
        diffs = np.diff(positions, axis=0)
        return np.sum(np.linalg.norm(diffs, axis=1))
    
    path_length_without = compute_path_length([r['position'] for r in results_without])
    path_length_with = compute_path_length([r['position'] for r in results_with])
    
    print(f"\nAverage Cost:")
    print(f"  Without warm-start: {avg_cost_without:.4f}")
    print(f"  With warm-start:    {avg_cost_with:.4f}")
    print(f"  Improvement:        {(1 - avg_cost_with/avg_cost_without)*100:.1f}%")
    
    print(f"\nPath Smoothness (lower is better):")
    print(f"  Without warm-start: {smoothness_without:.6f}")
    print(f"  With warm-start:    {smoothness_with:.6f}")
    print(f"  Improvement:        {(1 - smoothness_with/smoothness_without)*100:.1f}%")
    
    print(f"\nPath Length:")
    print(f"  Without warm-start: {path_length_without:.4f}")
    print(f"  With warm-start:    {path_length_with:.4f}")
    print(f"  Improvement:        {(1 - path_length_with/path_length_without)*100:.1f}%")
    
    # Visualization
    if args.save_plots:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        cost_map_np = cost_map[0, 0].cpu().numpy()
        
        # Plot without warm-start
        ax1 = axes[0]
        ax1.imshow(cost_map_np, extent=[0, 1, 0, 1], origin='lower', cmap='Reds', alpha=0.5)
        positions_without = np.array([r['position'] for r in results_without])
        ax1.plot(positions_without[:, 0], positions_without[:, 1], 'b-o', 
                linewidth=2, markersize=6, label='Path', alpha=0.7)
        ax1.plot(start_state[0].item(), start_state[1].item(), 'go', markersize=12, label='Start')
        ax1.plot(goal_state[0].item(), goal_state[1].item(), 'r*', markersize=15, label='Goal')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title(f'Without Warm-Start\nAvg Cost: {avg_cost_without:.4f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot with warm-start
        ax2 = axes[1]
        ax2.imshow(cost_map_np, extent=[0, 1, 0, 1], origin='lower', cmap='Reds', alpha=0.5)
        positions_with = np.array([r['position'] for r in results_with])
        ax2.plot(positions_with[:, 0], positions_with[:, 1], 'g-o', 
                linewidth=2, markersize=6, label='Path', alpha=0.7)
        ax2.plot(start_state[0].item(), start_state[1].item(), 'go', markersize=12, label='Start')
        ax2.plot(goal_state[0].item(), goal_state[1].item(), 'r*', markersize=15, label='Goal')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title(f'With Warm-Start\nAvg Cost: {avg_cost_with:.4f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Warm-Start Mechanism Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = Path(args.output_dir) / 'warm_start_comparison.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Plot saved to {save_path}")
        plt.close()
    
    return {
        'without': results_without,
        'with': results_with,
        'avg_cost_without': avg_cost_without,
        'avg_cost_with': avg_cost_with,
        'smoothness_without': smoothness_without,
        'smoothness_with': smoothness_with,
        'path_length_without': path_length_without,
        'path_length_with': path_length_with,
    }


def main():
    global args
    args = parse_args()
    
    # Set PyTorch to use all available CPU cores for L1 layer
    num_cpu_cores = os.cpu_count()
    if num_cpu_cores is not None:
        torch.set_num_threads(num_cpu_cores)
        print(f"PyTorch CPU threads set to: {num_cpu_cores} (all available cores)")
    
    print("=" * 80)
    print("L1+L2 Integration Validation (Unet)")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data directory: {args.data_dir}")
    print("=" * 80)
    
    # Load model
    model = load_model(args.checkpoint, args.device, model_type=args.model_type)
    
    # Load data sample
    data_sample = load_data_sample(args.data_dir, args.sample_idx, args.device)
    
    # Test L1+L2 integration
    integration_results = test_l1_l2_integration(
        model, data_sample, args.num_samples, args.l1_iterations, args.device, args.output_dir
    )
    
    # Test warm-start if requested
    if args.test_warm_start:
        warm_start_results = test_warm_start(
            model, data_sample, args.num_samples, args.l1_iterations, 
            args.num_steps, args.device
        )
    
    print("\n" + "=" * 80)
    print("Validation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

