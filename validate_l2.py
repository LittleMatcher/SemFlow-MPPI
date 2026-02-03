"""
Validation Experiments for L2 Layer

Test A: Geometric Constraint Response
Test B: Style Controllability

Usage:
    python validate_l2.py --checkpoint checkpoints_l2/best_model.pt --test_type all
    python validate_l2.py --checkpoint checkpoints_l2/best_model.pt --test_type geometric
    python validate_l2.py --checkpoint checkpoints_l2/best_model.pt --test_type style
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from cfm_flowmp.models import create_l2_safety_cfm, L2Config
from cfm_flowmp.data import FlowMPEnvDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Validate L2 layer")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test_type', type=str, default='all',
                        choices=['all', 'geometric', 'style'],
                        help='Type of test to run')
    parser.add_argument('--num_samples', type=int, default=64,
                        help='Number of trajectory samples to generate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save visualization plots')
    parser.add_argument('--output_dir', type=str, default='validation_results',
                        help='Directory to save results')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Optional: use cost_map from generated data (dir with data.npz)')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Sample index when using --data_dir')
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """Load trained L2 model."""
    print(f"Loading model from {checkpoint_path}...")
    
    # Create model (same config as train_l2_mock)
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
        use_bspline_smoothing=True,  # Enable B-spline smoothing for smooth trajectories
        bspline_degree=3,
        bspline_num_control_points=20,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    return model


def create_wall_cost_map(map_size=64):
    """
    Create cost map with a vertical wall in the center.
    
    Returns:
        cost_map: [1, 1, H, W] tensor
    """
    cost_map = np.zeros((1, 1, map_size, map_size), dtype=np.float32)
    
    # Vertical wall in center (from y=0.2 to y=0.8)
    wall_x = map_size // 2
    wall_y_start = int(0.2 * map_size)
    wall_y_end = int(0.8 * map_size)
    
    # Make wall 3 pixels wide for visibility
    cost_map[0, 0, wall_y_start:wall_y_end, wall_x-1:wall_x+2] = 1.0
    
    return torch.from_numpy(cost_map)


def create_obstacle_cost_map(map_size=64):
    """
    Create cost map with multiple Gaussian obstacles.
    
    Returns:
        cost_map: [1, 1, H, W] tensor
    """
    cost_map = np.zeros((1, 1, map_size, map_size), dtype=np.float32)
    
    # Add 3 obstacles
    obstacles = [
        ([0.3, 0.5], 0.1),  # (center, sigma)
        ([0.5, 0.3], 0.08),
        ([0.7, 0.6], 0.12),
    ]
    
    for center, sigma in obstacles:
        cx, cy = int(center[0] * map_size), int(center[1] * map_size)
        sigma_pixels = sigma * map_size
        radius_pixels = int(3 * sigma_pixels)
        
        y, x = np.ogrid[-radius_pixels:radius_pixels+1, -radius_pixels:radius_pixels+1]
        gaussian = np.exp(-(x**2 + y**2) / (2 * sigma_pixels**2))
        
        y_start = max(0, cy - radius_pixels)
        y_end = min(map_size, cy + radius_pixels + 1)
        x_start = max(0, cx - radius_pixels)
        x_end = min(map_size, cx + radius_pixels + 1)
        
        gy_start = max(0, -cy + radius_pixels)
        gy_end = gy_start + (y_end - y_start)
        gx_start = max(0, -cx + radius_pixels)
        gx_end = gx_start + (x_end - x_start)
        
        cost_map[0, 0, y_start:y_end, x_start:x_end] = np.maximum(
            cost_map[0, 0, y_start:y_end, x_start:x_end],
            gaussian[gy_start:gy_end, gx_start:gx_end]
        )
    
    return torch.from_numpy(cost_map)


def test_geometric_compliance(model, device, num_samples, save_plots, output_dir,
                              cost_map_from_data=None):
    """
    Test A: Geometric Constraint Response
    
    Check if trajectories avoid obstacles (wall in center).
    If cost_map_from_data is provided [1,1,H,W], use it instead of wall map.
    """
    print("\n" + "=" * 80)
    print("Test A: Geometric Constraint Response")
    print("=" * 80)
    
    if cost_map_from_data is not None:
        cost_map = cost_map_from_data.to(device)
        if cost_map.dim() == 3:
            cost_map = cost_map.unsqueeze(0)
    else:
        cost_map = create_wall_cost_map().to(device)
    
    # Setup start and goal (on opposite sides of wall)
    start_pos = np.array([0.2, 0.5])
    goal_pos = np.array([0.8, 0.5])
    
    # Create states
    x_curr = torch.tensor([
        [start_pos[0], start_pos[1], 0.0, 0.0, 0.0, 0.0]  # [px, py, vx, vy, ax, ay]
    ], dtype=torch.float32, device=device)
    
    x_goal = torch.tensor([
        [goal_pos[0], goal_pos[1], 0.0, 0.0]  # [px, py, vx, vy]
    ], dtype=torch.float32, device=device)
    
    # Balanced style
    w_style = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)
    
    # Generate trajectories
    print(f"Generating {num_samples} trajectories...")
    with torch.no_grad():
        result = model.generate_trajectory_anchors(
            cost_map=cost_map,
            x_curr=x_curr,
            x_goal=x_goal,
            w_style=w_style,
            num_samples=num_samples,
        )
    
    trajectories = result['trajectories'].cpu().numpy()  # [N, T, 2]
    
    # Check if trajectories pass through wall
    wall_x = 0.5
    wall_y_range = (0.2, 0.8)
    
    wall_violations = 0
    for traj in trajectories:
        # Check if trajectory crosses x=0.5 in the wall region
        for i in range(len(traj) - 1):
            p1, p2 = traj[i], traj[i+1]
            # Check if trajectory segment crosses wall
            if (p1[0] < wall_x < p2[0] or p2[0] < wall_x < p1[0]):
                # Interpolate y coordinate at wall_x
                alpha = (wall_x - p1[0]) / (p2[0] - p1[0] + 1e-8)
                y_at_wall = p1[1] + alpha * (p2[1] - p1[1])
                
                if wall_y_range[0] < y_at_wall < wall_y_range[1]:
                    wall_violations += 1
                    break
    
    compliance_rate = (num_samples - wall_violations) / num_samples * 100
    
    print(f"\nResults:")
    print(f"  Total trajectories: {num_samples}")
    print(f"  Wall violations: {wall_violations}")
    print(f"  Compliance rate: {compliance_rate:.1f}%")
    
    if compliance_rate > 90:
        print("  ✓ PASS: Model successfully avoids obstacles")
    else:
        print("  ✗ FAIL: Model does not reliably avoid obstacles")
    
    # Visualization
    if save_plots:
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot cost map
        ax.imshow(cost_map[0, 0].cpu().numpy(), extent=[0, 1, 0, 1], 
                 origin='lower', cmap='Reds', alpha=0.5)
        
        # Plot all trajectories
        for traj in trajectories:
            ax.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.3, linewidth=1)
        
        # Plot start and goal
        ax.plot(start_pos[0], start_pos[1], 'go', markersize=15, label='Start')
        ax.plot(goal_pos[0], goal_pos[1], 'r*', markersize=20, label='Goal')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Test A: Geometric Compliance ({compliance_rate:.1f}% pass rate)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        save_path = Path(output_dir) / 'test_a_geometric_compliance.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Plot saved to {save_path}")
        plt.close()
    
    return {
        'compliance_rate': compliance_rate,
        'violations': wall_violations,
        'total': num_samples,
    }


def test_style_controllability(model, device, num_samples, save_plots, output_dir,
                                cost_map_from_data=None):
    """
    Test B: Style Controllability
    
    Check if model responds to different style weights.
    If cost_map_from_data is provided [1,1,H,W], use it instead of default obstacles.
    """
    print("\n" + "=" * 80)
    print("Test B: Style Controllability")
    print("=" * 80)
    
    if cost_map_from_data is not None:
        cost_map = cost_map_from_data.to(device)
        if cost_map.dim() == 3:
            cost_map = cost_map.unsqueeze(0)
    else:
        cost_map = create_obstacle_cost_map().to(device)
    
    # Setup start and goal
    start_pos = np.array([0.1, 0.1])
    goal_pos = np.array([0.9, 0.9])
    
    x_curr = torch.tensor([
        [start_pos[0], start_pos[1], 0.0, 0.0, 0.0, 0.0]
    ], dtype=torch.float32, device=device)
    
    x_goal = torch.tensor([
        [goal_pos[0], goal_pos[1], 0.0, 0.0]
    ], dtype=torch.float32, device=device)
    
    # Test different styles
    styles = {
        'Safe (Conservative)': torch.tensor([[1.0, 0.0]], dtype=torch.float32, device=device),
        'Balanced': torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device),
        'Fast (Aggressive)': torch.tensor([[0.0, 1.0]], dtype=torch.float32, device=device),
    }
    
    results = {}
    all_trajectories = {}
    
    for style_name, w_style in styles.items():
        print(f"\nTesting {style_name} style...")
        
        with torch.no_grad():
            result = model.generate_trajectory_anchors(
                cost_map=cost_map,
                x_curr=x_curr,
                x_goal=x_goal,
                w_style=w_style,
                num_samples=num_samples,
            )
        
        trajectories = result['trajectories'].cpu().numpy()  # [N, T, 2]
        all_trajectories[style_name] = trajectories
        
        # Compute metrics
        # 1. Average minimum distance to obstacles
        min_distances = []
        for traj in trajectories:
            # For each point, compute distance to nearest obstacle
            traj_min_dists = []
            for point in traj:
                # Sample cost map at point
                x_idx = int(point[0] * 63)
                y_idx = int(point[1] * 63)
                x_idx = np.clip(x_idx, 0, 63)
                y_idx = np.clip(y_idx, 0, 63)
                
                # Find distance to nearest obstacle (cost > 0.5)
                cost_val = cost_map[0, 0, y_idx, x_idx].cpu().numpy()
                
                if cost_val > 0.1:
                    traj_min_dists.append(0.0)  # Inside obstacle
                else:
                    # Approximate distance (simplified)
                    traj_min_dists.append(1.0 - cost_val)
            
            if traj_min_dists:
                min_distances.append(np.mean(traj_min_dists))
        
        avg_clearance = np.mean(min_distances)
        
        # 2. Path length
        path_lengths = []
        for traj in trajectories:
            length = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
            path_lengths.append(length)
        
        avg_length = np.mean(path_lengths)
        
        # 3. Smoothness (curvature)
        smoothness_scores = []
        for traj in trajectories:
            if len(traj) > 2:
                # Compute angles between consecutive segments
                segments = np.diff(traj, axis=0)
                angles = []
                for i in range(len(segments) - 1):
                    v1, v2 = segments[i], segments[i+1]
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                    angle = np.arccos(np.clip(cos_angle, -1, 1))
                    angles.append(angle)
                smoothness_scores.append(np.mean(angles))
        
        avg_smoothness = np.mean(smoothness_scores) if smoothness_scores else 0.0
        
        results[style_name] = {
            'clearance': avg_clearance,
            'path_length': avg_length,
            'smoothness': avg_smoothness,
        }
        
        print(f"  Avg clearance: {avg_clearance:.3f}")
        print(f"  Avg path length: {avg_length:.3f}")
        print(f"  Avg smoothness: {avg_smoothness:.3f}")
    
    # Check controllability
    print(f"\n{'='*80}")
    print("Controllability Analysis:")
    print(f"{'='*80}")
    
    safe_clearance = results['Safe (Conservative)']['clearance']
    fast_clearance = results['Fast (Aggressive)']['clearance']
    
    if safe_clearance > fast_clearance * 1.1:
        print("✓ PASS: Safe mode maintains larger clearance")
    else:
        print("✗ FAIL: Safe mode does not increase clearance significantly")
    
    safe_length = results['Safe (Conservative)']['path_length']
    fast_length = results['Fast (Aggressive)']['path_length']
    
    if fast_length < safe_length * 0.95:
        print("✓ PASS: Fast mode produces shorter paths")
    else:
        print("✗ FAIL: Fast mode does not reduce path length significantly")
    
    # Visualization
    if save_plots:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, (style_name, trajectories) in enumerate(all_trajectories.items()):
            ax = axes[idx]
            
            # Plot cost map
            ax.imshow(cost_map[0, 0].cpu().numpy(), extent=[0, 1, 0, 1],
                     origin='lower', cmap='Reds', alpha=0.5)
            
            # Plot trajectories
            for traj in trajectories[:20]:  # Plot subset for clarity
                ax.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.4, linewidth=1)
            
            # Plot start and goal
            ax.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start')
            ax.plot(goal_pos[0], goal_pos[1], 'r*', markersize=15, label='Goal')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'{style_name}\nClearance: {results[style_name]["clearance"]:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Test B: Style Controllability', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = Path(output_dir) / 'test_b_style_controllability.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Plot saved to {save_path}")
        plt.close()
    
    return results


def main():
    args = parse_args()
    
    print("=" * 80)
    print("L2 Layer Validation Experiments")
    print("=" * 80)
    
    # Load model
    model = load_model(args.checkpoint, args.device)
    
    # Optional: load cost_map from generated data
    cost_map_geom = cost_map_style = None
    if args.data_dir:
        from pathlib import Path
        data_path = Path(args.data_dir)
        if data_path.is_dir():
            data_path = data_path / "data.npz"
        if data_path.exists():
            ds = FlowMPEnvDataset(str(data_path))
            idx = min(args.sample_idx, len(ds) - 1)
            sample = ds[idx]
            cm = sample["cost_map"]
            cost_map_geom = cm.unsqueeze(0) if cm.dim() == 3 else cm
            cost_map_style = cost_map_geom
            print(f"Using cost_map from generated data (sample_idx={idx})")
    
    # Run tests
    if args.test_type in ['all', 'geometric']:
        geometric_results = test_geometric_compliance(
            model, args.device, args.num_samples, args.save_plots, args.output_dir,
            cost_map_from_data=cost_map_geom,
        )
    
    if args.test_type in ['all', 'style']:
        style_results = test_style_controllability(
            model, args.device, args.num_samples, args.save_plots, args.output_dir,
            cost_map_from_data=cost_map_style,
        )
    
    print("\n" + "=" * 80)
    print("Validation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
