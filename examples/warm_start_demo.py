#!/usr/bin/env python3
"""
Warm-Start (On-Policy) Demonstration for SemFlow-MPPI

This demo shows how the L2 CFM can work like On-Policy RL by maintaining
temporal continuity through warm-starting.

Concept:
--------
1. At time t, MPPI optimizes and produces optimal trajectory u*_t
2. At time t+1, u*_t is shifted forward to create prior ũ_t+1
3. CFM starts from noised ũ_t+1 instead of pure Gaussian noise
4. This creates "policy continuation" - decisions build on previous steps

This is analogous to On-Policy RL where the policy at t+1 builds upon
the policy at t, rather than starting from scratch each time.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cfm_flowmp.models import create_flowmp_transformer
from cfm_flowmp.inference import TrajectoryGenerator, GeneratorConfig


def simulate_navigation_sequence(
    model: torch.nn.Module,
    start_pos: torch.Tensor,
    goal_pos: torch.Tensor,
    num_steps: int = 10,
    enable_warm_start: bool = True,
    device: str = "cpu",
):
    """
    Simulate a navigation sequence with/without warm-start.
    
    This mimics a real deployment scenario where the robot:
    1. Generates a trajectory using L2 CFM
    2. L1 MPPI optimizes it (simulated here)
    3. Executes one step
    4. Repeats from the new position
    
    Args:
        model: Trained FlowMP model
        start_pos: Starting position [1, 2]
        goal_pos: Goal position [1, 2]
        num_steps: Number of planning cycles
        enable_warm_start: Whether to use warm-start
        device: Device to use
        
    Returns:
        Dictionary with trajectory history
    """
    # Create generator with/without warm-start
    config = GeneratorConfig(
        solver_type="rk4",
        use_8step_schedule=True,
        seq_len=64,
        state_dim=2,
        num_samples=1,
        enable_warm_start=enable_warm_start,
        warm_start_noise_scale=0.1,
        warm_start_shift_mode="predict",
    )
    
    generator = TrajectoryGenerator(model, config)
    
    # Track execution
    executed_positions = [start_pos[0].cpu().numpy().copy()]
    all_planned_trajectories = []
    generation_times = []
    
    current_pos = start_pos.clone()
    
    for step in range(num_steps):
        print(f"\n{'='*60}")
        print(f"Step {step+1}/{num_steps}")
        print(f"  Current position: {current_pos[0].cpu().numpy()}")
        print(f"  Goal position: {goal_pos[0].cpu().numpy()}")
        
        # Check if reached goal
        dist_to_goal = torch.norm(current_pos - goal_pos).item()
        if dist_to_goal < 0.1:
            print(f"  ✓ Reached goal! (distance: {dist_to_goal:.4f})")
            break
        
        # Generate trajectory (L2 CFM)
        import time
        start_time = time.time()
        
        with torch.no_grad():
            result = generator.generate(
                start_pos=current_pos,
                goal_pos=goal_pos,
                return_raw=True,
            )
        
        generation_time = time.time() - start_time
        generation_times.append(generation_time)
        
        positions = result['positions'][0]  # [T, 2]
        
        # Store planned trajectory
        all_planned_trajectories.append(positions.cpu().numpy())
        
        print(f"  Generated trajectory in {generation_time*1000:.2f}ms")
        print(f"  Warm-start active: {generator.warm_start_cache is not None}")
        
        # === Simulate MPPI optimization (L1) ===
        # In real deployment, L1 MPPI would:
        # 1. Use this trajectory as an anchor
        # 2. Sample perturbations around it
        # 3. Evaluate costs and find optimal control
        # 4. Execute the first control action
        #
        # Here we simulate by just executing the planned trajectory
        
        # Update warm-start cache with "optimal" trajectory
        # In reality, this would be the MPPI-optimized trajectory
        if enable_warm_start:
            generator.update_warm_start_cache(result)
        
        # Execute one step (move along planned trajectory)
        # In reality, this would be the first control from MPPI
        execution_step = min(5, len(positions))  # Execute ~5 steps
        current_pos = positions[execution_step:execution_step+1].to(device)
        executed_positions.append(current_pos[0].cpu().numpy().copy())
        
        print(f"  Executed to: {current_pos[0].cpu().numpy()}")
    
    return {
        'executed_positions': np.array(executed_positions),
        'planned_trajectories': all_planned_trajectories,
        'generation_times': generation_times,
        'reached_goal': dist_to_goal < 0.1,
    }


def visualize_comparison(
    results_with_ws: Dict,
    results_without_ws: Dict,
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
    save_path: str = "warm_start_comparison.png",
):
    """Visualize comparison between warm-start and no warm-start."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Executed trajectories comparison
    ax = axes[0, 0]
    exec_with = results_with_ws['executed_positions']
    exec_without = results_without_ws['executed_positions']
    
    ax.plot(exec_without[:, 0], exec_without[:, 1], 
            'b-o', label='Without Warm-Start', alpha=0.6, markersize=8)
    ax.plot(exec_with[:, 0], exec_with[:, 1], 
            'r-o', label='With Warm-Start', alpha=0.6, markersize=8)
    ax.scatter(start_pos[0], start_pos[1], c='green', s=200, 
              marker='*', label='Start', zorder=5)
    ax.scatter(goal_pos[0], goal_pos[1], c='orange', s=200, 
              marker='*', label='Goal', zorder=5)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Executed Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 2: Planned trajectories (with warm-start)
    ax = axes[0, 1]
    for i, traj in enumerate(results_with_ws['planned_trajectories']):
        alpha = 0.3 + 0.5 * (i / len(results_with_ws['planned_trajectories']))
        ax.plot(traj[:, 0], traj[:, 1], alpha=alpha, linewidth=1)
    ax.plot(exec_with[:, 0], exec_with[:, 1], 
            'r-o', label='Executed', linewidth=2, markersize=6)
    ax.scatter(start_pos[0], start_pos[1], c='green', s=200, marker='*', zorder=5)
    ax.scatter(goal_pos[0], goal_pos[1], c='orange', s=200, marker='*', zorder=5)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Planned Trajectories (With Warm-Start)')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 3: Planned trajectories (without warm-start)
    ax = axes[1, 0]
    for i, traj in enumerate(results_without_ws['planned_trajectories']):
        alpha = 0.3 + 0.5 * (i / len(results_without_ws['planned_trajectories']))
        ax.plot(traj[:, 0], traj[:, 1], alpha=alpha, linewidth=1)
    ax.plot(exec_without[:, 0], exec_without[:, 1], 
            'b-o', label='Executed', linewidth=2, markersize=6)
    ax.scatter(start_pos[0], start_pos[1], c='green', s=200, marker='*', zorder=5)
    ax.scatter(goal_pos[0], goal_pos[1], c='orange', s=200, marker='*', zorder=5)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Planned Trajectories (Without Warm-Start)')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 4: Generation time comparison
    ax = axes[1, 1]
    steps_with = np.arange(len(results_with_ws['generation_times']))
    steps_without = np.arange(len(results_without_ws['generation_times']))
    times_with = np.array(results_with_ws['generation_times']) * 1000
    times_without = np.array(results_without_ws['generation_times']) * 1000
    
    ax.plot(steps_without, times_without, 'b-o', label='Without Warm-Start', alpha=0.7)
    ax.plot(steps_with, times_with, 'r-o', label='With Warm-Start', alpha=0.7)
    ax.axhline(times_without.mean(), color='b', linestyle='--', 
               label=f'Avg w/o: {times_without.mean():.1f}ms')
    ax.axhline(times_with.mean(), color='r', linestyle='--', 
               label=f'Avg w/: {times_with.mean():.1f}ms')
    ax.set_xlabel('Planning Step')
    ax.set_ylabel('Generation Time (ms)')
    ax.set_title('Computational Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to: {save_path}")


def print_statistics(results_with_ws: Dict, results_without_ws: Dict):
    """Print comparison statistics."""
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    # Path efficiency
    exec_with = results_with_ws['executed_positions']
    exec_without = results_without_ws['executed_positions']
    
    path_len_with = np.sum(np.linalg.norm(np.diff(exec_with, axis=0), axis=1))
    path_len_without = np.sum(np.linalg.norm(np.diff(exec_without, axis=0), axis=1))
    
    print(f"\n📏 Path Length:")
    print(f"  Without Warm-Start: {path_len_without:.3f}")
    print(f"  With Warm-Start:    {path_len_with:.3f}")
    print(f"  Improvement:        {(path_len_without-path_len_with)/path_len_without*100:.1f}%")
    
    # Smoothness (acceleration changes)
    def compute_smoothness(positions):
        if len(positions) < 3:
            return 0.0
        vel = np.diff(positions, axis=0)
        acc = np.diff(vel, axis=0)
        jerk = np.diff(acc, axis=0)
        return np.mean(np.linalg.norm(jerk, axis=1))
    
    smooth_with = compute_smoothness(exec_with)
    smooth_without = compute_smoothness(exec_without)
    
    print(f"\n🌊 Smoothness (avg jerk):")
    print(f"  Without Warm-Start: {smooth_without:.4f}")
    print(f"  With Warm-Start:    {smooth_with:.4f}")
    print(f"  Improvement:        {(smooth_without-smooth_with)/smooth_without*100:.1f}%")
    
    # Timing
    times_with = np.array(results_with_ws['generation_times']) * 1000
    times_without = np.array(results_without_ws['generation_times']) * 1000
    
    print(f"\n⏱️  Generation Time:")
    print(f"  Without Warm-Start: {times_without.mean():.2f} ± {times_without.std():.2f} ms")
    print(f"  With Warm-Start:    {times_with.mean():.2f} ± {times_with.std():.2f} ms")
    
    # Steps to goal
    print(f"\n🎯 Steps to Goal:")
    print(f"  Without Warm-Start: {len(exec_without)}")
    print(f"  With Warm-Start:    {len(exec_with)}")
    
    print("\n" + "="*60)


def main():
    print("="*60)
    print("Warm-Start (On-Policy) Demo for SemFlow-MPPI")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # ============ Create a simple model for demo ============
    print("\n[1] Creating model...")
    model = create_flowmp_transformer(variant="tiny")
    model = model.to(device)
    model.eval()
    print("  ✓ Model created (random weights - for demo purposes)")
    
    # ============ Define navigation task ============
    print("\n[2] Setting up navigation task...")
    start_pos = torch.tensor([[0.0, 0.0]], device=device)
    goal_pos = torch.tensor([[2.0, 2.0]], device=device)
    print(f"  Start: {start_pos[0].cpu().numpy()}")
    print(f"  Goal:  {goal_pos[0].cpu().numpy()}")
    
    # ============ Run WITHOUT warm-start ============
    print("\n[3] Running WITHOUT warm-start (standard CFM)...")
    print("  Each step starts from pure Gaussian noise N(0, I)")
    results_without = simulate_navigation_sequence(
        model=model,
        start_pos=start_pos,
        goal_pos=goal_pos,
        num_steps=10,
        enable_warm_start=False,
        device=device,
    )
    
    # ============ Run WITH warm-start ============
    print("\n[4] Running WITH warm-start (On-Policy style)...")
    print("  Each step builds on previous trajectory (temporal continuity)")
    results_with = simulate_navigation_sequence(
        model=model,
        start_pos=start_pos,
        goal_pos=goal_pos,
        num_steps=10,
        enable_warm_start=True,
        device=device,
    )
    
    # ============ Analyze and visualize ============
    print("\n[5] Analyzing results...")
    print_statistics(results_with, results_without)
    
    print("\n[6] Creating visualization...")
    visualize_comparison(
        results_with,
        results_without,
        start_pos[0].cpu().numpy(),
        goal_pos[0].cpu().numpy(),
        save_path="warm_start_comparison.png",
    )
    
    # ============ Summary ============
    print("\n" + "="*60)
    print("WARM-START MECHANISM SUMMARY")
    print("="*60)
    print("""
The warm-start mechanism implements "On-Policy" behavior:

1. ⏭️  SHIFT: At time t, take optimal trajectory u*_t
           Shift forward: [u1, u2, ..., uT] → [u2, u3, ..., uT, uT]
           
2. 🎲 NOISE: Add exploration noise: ũ_t+1 = shifted(u*_t) + ε
           Noise scale controls exploration vs exploitation
           
3. 🚀 INIT: Start CFM ODE from ũ_t+1 instead of N(0, I)
           This provides strong prior from previous solution
           
4. 🔄 LOOP: CFM refines ũ_t+1 → better trajectory
           MPPI optimizes → new u*_t+1
           Cache u*_t+1 for next step

Benefits:
  ✓ Temporal continuity (like On-Policy RL)
  ✓ Smoother trajectories
  ✓ Faster convergence
  ✓ Better sample efficiency
  ✓ More consistent behavior
    """)
    print("="*60)
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    main()
