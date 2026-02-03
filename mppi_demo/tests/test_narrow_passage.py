"""
Test MPPI with Narrow Passage Scenario
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scenarios.narrow_passage import create_narrow_passage_scenario
from mppi_core import (
    CollisionCost, SmoothnessCost, GoalCost, CompositeCost,
    MPPI_BSpline, Visualizer
)


def run_mppi_narrow_passage(temperature=0.8, n_samples=250, n_iterations=150,
                            passage_width=0.8, show_animation=True, save_plots=True):
    """
    Run MPPI on narrow passage scenario
    
    Args:
        temperature: MPPI temperature parameter
        n_samples: Number of trajectory samples
        n_iterations: Number of optimization iterations
        passage_width: Width of the narrow passage
        show_animation: Whether to create animation
        save_plots: Whether to save plots
    """
    print("=" * 60)
    print("MPPI with B-Spline Trajectories - Narrow Passage Scenario")
    print("=" * 60)
    print(f"Passage Width: {passage_width:.2f}m")
    print(f"Temperature (λ): {temperature}")
    print(f"Samples (K): {n_samples}")
    print(f"Iterations: {n_iterations}")
    print()
    
    # Create environment
    env, start, goal, bounds = create_narrow_passage_scenario(passage_width=passage_width)
    
    print(f"Environment: {len(env.obstacles)} obstacles")
    print(f"Start: ({start[0]:.1f}, {start[1]:.1f})")
    print(f"Goal: ({goal[0]:.1f}, {goal[1]:.1f})")
    print()
    
    # Robot parameters
    robot_radius = 0.2
    
    # Cost functions
    collision_cost = CollisionCost(
        env=env,
        robot_radius=robot_radius,
        epsilon=0.1,
        weight=150.0,
        use_hard_constraint=True,
        hard_penalty=1e6
    )
    
    smoothness_cost = SmoothnessCost(
        penalize='acceleration',
        weight=1.0
    )
    
    goal_cost = GoalCost(
        goal=goal,
        weight=30.0
    )
    
    cost_function = CompositeCost([
        collision_cost,
        smoothness_cost,
        goal_cost
    ])
    
    # Create MPPI optimizer
    mppi = MPPI_BSpline(
        cost_function=cost_function,
        n_samples=n_samples,
        n_control_points=15,
        bspline_degree=3,
        time_horizon=6.0,
        n_timesteps=60,
        temperature=temperature,
        noise_std=0.3,
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
    
    # Check if solution is feasible
    if result['best_cost_all_time'] > 1e5:
        print("⚠ Warning: High cost detected. Path may have collisions or other issues.")
        print("Suggestions: Increase n_iterations, n_samples, or temperature")
    else:
        print("✓ Solution found with reasonable cost")
    print()
    
    # Visualization
    vis = Visualizer(env, figsize=(14, 8))
    
    # 1. Plot final trajectory
    fig1, ax1 = plt.subplots(1, 1, figsize=(14, 8))
    vis.plot_environment(ax1)
    vis.plot_trajectory(
        result['trajectory'],
        ax=ax1,
        color='blue',
        linewidth=3,
        label='MPPI Solution',
        show_control_points=True,
        control_points=result['control_points']
    )
    
    ax1.annotate('Narrow\nPassage', xy=(4, 0), fontsize=14,
                ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax1.set_title(f'MPPI Solution - Narrow Passage (width={passage_width:.2f}m)',
                 fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    
    if save_plots:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        fig1.savefig(os.path.join(output_dir, 'narrow_passage_solution.png'),
                    dpi=150, bbox_inches='tight')
        print(f"Saved: outputs/narrow_passage_solution.png")
    
    # 2. Show evolution over iterations
    n_plots = 6
    step = max(1, n_iterations // n_plots)
    iterations_to_show = [i * step for i in range(n_plots)]
    iterations_to_show.append(n_iterations - 1)
    iterations_to_show = sorted(list(set(iterations_to_show)))
    iterations_to_show = [i for i in iterations_to_show if i < n_iterations]
    
    n_cols = 3
    n_rows = (len(iterations_to_show) + n_cols - 1) // n_cols
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, iter_idx in enumerate(iterations_to_show):
        if iter_idx < len(result['info_history']):
            info = result['info_history'][iter_idx]
            
            vis.plot_environment(axes[i])
            vis.plot_multiple_trajectories(
                info['all_trajectories'][:30],
                ax=axes[i],
                alpha=0.2
            )
            vis.plot_trajectory(
                info['best_trajectory'],
                ax=axes[i],
                color='blue',
                linewidth=2
            )
            axes[i].set_title(f"Iteration {iter_idx}, Cost: {info['best_cost']:.1f}",
                            fontsize=12, fontweight='bold')
    
    for i in range(len(iterations_to_show), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_plots:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        fig2.savefig(os.path.join(output_dir, 'narrow_passage_evolution.png'),
                    dpi=150, bbox_inches='tight')
        print(f"Saved: outputs/narrow_passage_evolution.png")
    
    # 3. Detailed analysis
    fig3, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    vis.plot_cost_history(
        result['cost_history'], 
        ax=axes[0],
        best_iteration=result['best_iteration'],
        best_cost=result['best_cost_all_time']
    )
    
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
        color='darkblue',
        linewidth=3,
        label=f'Best Trajectory (Iteration {result["best_iteration"]})'
    )
    axes[1].set_title('Final Samples (colored by weight)', fontsize=14, fontweight='bold')
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_plots:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        fig3.savefig(os.path.join(output_dir, 'narrow_passage_analysis.png'),
                    dpi=150, bbox_inches='tight')
        print(f"Saved: outputs/narrow_passage_analysis.png")
    
    if show_animation:
        print("\nCreating animation...")
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        anim = vis.create_animation(
            result['info_history'],
            save_path=os.path.join(output_dir, 'narrow_passage_animation.gif'),
            show_samples=False,
            fps=10
        )
        print(f"Animation created: outputs/narrow_passage_animation.gif")
    
    plt.close('all')
    
    return result


if __name__ == "__main__":
    result = run_mppi_narrow_passage(
        temperature=0.8,
        n_samples=250,
        n_iterations=150,
        passage_width=0.8,
        show_animation=True,
        save_plots=True
    )
    
    print("\n" + "=" * 60)
    print("Narrow Passage Scenario Complete!")
    print("=" * 60)

