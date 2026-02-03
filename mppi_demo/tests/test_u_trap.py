"""
Test MPPI with U-Trap Scenario
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scenarios.u_trap import create_u_trap_scenario
from mppi_core import (
    CollisionCost, SmoothnessCost, GoalCost, CompositeCost,
    PathLengthCost, TurnCost, MPPI_BSpline, Visualizer
)


def run_mppi_u_trap(temperature=1.0, n_samples=200, n_iterations=120,
                    show_animation=True, save_plots=True):
    """
    Run MPPI on U-trap scenario
    
    Args:
        temperature: MPPI temperature parameter (lambda)
                    Lower = more exploitation, Higher = more exploration
        n_samples: Number of trajectory samples
        n_iterations: Number of optimization iterations
        show_animation: Whether to create animation
        save_plots: Whether to save plots
    """
    print("=" * 60)
    print("MPPI with B-Spline Trajectories - U-Trap Scenario")
    print("=" * 60)
    print(f"Temperature (λ): {temperature}")
    print(f"Samples (K): {n_samples}")
    print(f"Iterations: {n_iterations}")
    print()
    
    # Create environment
    env, start, goal, bounds = create_u_trap_scenario()
    
    # Robot parameters
    robot_radius = 0.2
    
    # Create cost function
    collision_cost = CollisionCost(
        env=env,
        robot_radius=robot_radius,
        epsilon=0.15,
        weight=100.0,
        use_hard_constraint=True,
        hard_penalty=1e6
    )
    
    smoothness_cost = SmoothnessCost(
        penalize='acceleration',
        weight=0.5
    )
    
    goal_cost = GoalCost(
        goal=goal,
        weight=80.0
    )
    
    path_length_cost = PathLengthCost(
        weight=20.0
    )
    
    dt = 0.1
    turn_cost = TurnCost(
        weight=8.0,
        method='angle_diff',
        dt=dt,
        use_sharp_turn_penalty=True,
        sharp_turn_threshold=None
    )
    
    cost_function = CompositeCost([
        collision_cost,
        smoothness_cost,
        goal_cost,
        path_length_cost,
        turn_cost
    ])
    
    # Create MPPI optimizer
    mppi = MPPI_BSpline(
        cost_function=cost_function,
        n_samples=n_samples,
        n_control_points=12,
        bspline_degree=3,
        time_horizon=5.0,
        n_timesteps=50,
        temperature=temperature,
        noise_std=0.5,
        bounds=bounds
    )
    
    # Run optimization
    print("Running MPPI optimization...")
    result = mppi.optimize(
        start=start,
        goal=goal,
        n_iterations=n_iterations,
        verbose=True
    )
    
    print()
    print("Optimization complete!")
    print(f"Final iteration cost: {result['final_cost']:.2f}")
    print(f"Best cost (all iterations): {result['best_cost_all_time']:.2f} (iteration {result['best_iteration']})")
    if result['best_iteration'] != len(result['cost_history']) - 1:
        print(f"  Note: Best trajectory found at iteration {result['best_iteration']}, not the final one")
    print()
    
    # Visualization
    vis = Visualizer(env, figsize=(12, 10))
    
    # 1. Plot final trajectory
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 10))
    vis.plot_environment(ax1)
    vis.plot_trajectory(
        result['trajectory'],
        ax=ax1,
        color='blue',
        linewidth=3,
        label=f'MPPI Solution (Iteration {result["best_iteration"]})',
        show_control_points=True,
        control_points=result['best_control_points_all_time']
    )
    ax1.set_title('MPPI Solution - U-Trap Scenario', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    
    if save_plots:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        fig1.savefig(os.path.join(output_dir, 'u_trap_solution.png'),
                    dpi=150, bbox_inches='tight')
        print(f"Saved: outputs/u_trap_solution.png")
    
    # 2. Plot with SDF visualization
    fig2, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    vis.visualize_sdf(ax=axes[0], robot_radius=robot_radius)
    axes[0].plot(result['trajectory'][:, 0], result['trajectory'][:, 1],
                color='blue', linewidth=3, label='MPPI Solution')
    axes[0].scatter(start[0], start[1], color='green', s=200, marker='o',
                   edgecolor='black', linewidth=2, zorder=5)
    axes[0].scatter(goal[0], goal[1], color='red', s=200, marker='*',
                   edgecolor='black', linewidth=2, zorder=5)
    axes[0].legend()
    axes[0].set_title('Signed Distance Field', fontsize=14)
    
    final_info = result['info_history'][-1]
    vis.plot_environment(axes[1])
    vis.plot_weighted_trajectories(
        final_info['all_trajectories'][:50],
        final_info['weights'][:50],
        ax=axes[1]
    )
    vis.plot_trajectory(
        result['trajectory'],
        ax=axes[1],
        color='blue',
        linewidth=3,
        label=f'Best Trajectory (Iteration {result["best_iteration"]})'
    )
    axes[1].set_title('Final Iteration Samples (colored by weight)', fontsize=14)
    axes[1].legend()
    
    if save_plots:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        fig2.savefig(os.path.join(output_dir, 'u_trap_analysis.png'),
                    dpi=150, bbox_inches='tight')
        print(f"Saved: outputs/u_trap_analysis.png")
    
    if show_animation:
        print("\nCreating animation...")
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        anim = vis.create_animation(
            result['info_history'],
            save_path=os.path.join(output_dir, 'u_trap_animation.gif'),
            show_samples=False,
            fps=10
        )
        print(f"Animation created: outputs/u_trap_animation.gif")
    
    plt.close('all')
    
    return result


if __name__ == "__main__":
    result = run_mppi_u_trap(
        temperature=1.0,
        n_samples=200,
        n_iterations=120,
        show_animation=True,
        save_plots=True
    )
    
    print("\n" + "=" * 60)
    print("U-Trap Scenario Complete!")
    print("=" * 60)

