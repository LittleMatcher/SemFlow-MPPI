"""
Test MPPI with Christmas Market Scenario
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scenarios.christmas_market import create_christmas_market_environment
from mppi_core import (
    CollisionCost, SmoothnessCost, GoalCost, CompositeCost,
    GoalApproachCost, PathLengthCost, ReferencePathCost, TerminalVelocityCost,
    TurnCost, CrowdDensityCost, BoundaryConstraintCost, MPPI_BSpline, Visualizer
)
from rrt_star_path_planner import RRTStarPlanner


def run_mppi_christmas_market(temperature=1.0, n_samples=400, n_iterations=250,
                              show_animation=True, save_plots=True):
    """
    Run MPPI on Christmas Market scenario
    
    Args:
        temperature: MPPI temperature parameter (lambda)
        n_samples: Number of trajectory samples
        n_iterations: Number of optimization iterations
        show_animation: Whether to create animation
        save_plots: Whether to save plots
    """
    print("=" * 60)
    print("MPPI with B-Spline Trajectories - Christmas Market Scenario")
    print("=" * 60)
    print(f"Temperature (λ): {temperature}")
    print(f"Samples (K): {n_samples}")
    print(f"Iterations: {n_iterations}")
    print()
    
    # Create environment
    variant = "v2"
    env, start, goal, bounds, crowd_regions = create_christmas_market_environment(variant=variant)
    print(f"Scenario variant: {variant}")
    
    print(f"Environment: {len(env.obstacles)} obstacles")
    print(f"Start: ({start[0]:.1f}, {start[1]:.1f})")
    print(f"Goal: ({goal[0]:.1f}, {goal[1]:.1f})")
    print(f"Crowd regions: {len(crowd_regions)}")
    for region in crowd_regions:
        print(f"  - {region.name}: density {region.density_multiplier}x")
    print()
    
    # Robot parameters
    robot_radius = 0.01  # Slightly larger for safety in crowded environment

    # Build a coarse guide path using RRT*
    def _densify_path(path: np.ndarray, spacing: float = 0.4) -> np.ndarray:
        if path is None or len(path) < 2:
            return path
        dense = [path[0]]
        for i in range(len(path) - 1):
            seg = path[i + 1] - path[i]
            length = np.linalg.norm(seg)
            if length < 1e-6:
                continue
            steps = max(1, int(np.ceil(length / spacing)))
            for j in range(1, steps + 1):
                alpha = min(j / steps, 1.0)
                dense.append(path[i] + alpha * seg)
        return np.array(dense)

    planner = RRTStarPlanner(
        env, 
        resolution=0.1,
        robot_radius=0.35,
        step_size=0.8,
        goal_sample_rate=0.10,
        max_iterations=3000,
        search_radius=2.0,
        crowd_regions=crowd_regions,
        # Bias RRT* sampling toward low-density regions and penalize high-density edges.
        # This makes the reference path more consistent with the MPPI crowd-density cost.
        density_bias_strength=1.2,
        density_edge_weight=2.0,
        density_edge_samples=12,
        sample_candidates=16,
    )
    print("Using RRT* for reference path generation...")
    guide_path, planner_info = planner.plan(start, goal)
    if guide_path is None:
        print("⚠️  Warning: RRT* planner failed, fallback to straight-line guide path")
        guide_path = np.vstack([start, goal])
    else:
        print(f"RRT* path length: {planner_info['path_length']:.2f}m with {len(guide_path)} waypoints")
        if 'path_cost' in planner_info:
            print(f"RRT* density-aware path cost: {planner_info['path_cost']:.2f}")
        print(f"RRT* iterations: {planner_info['iterations']}, nodes: {planner_info['nodes']}")
    
    # Simplify reference path to only 4 points
    if len(guide_path) > 4:
        # Sample 4 points uniformly along the path
        indices = np.linspace(0, len(guide_path) - 1, 4, dtype=int)
        guide_path = guide_path[indices]
        print(f"Reference path simplified to 4 points: start -> 2 intermediate -> goal")
    else:
        print(f"Reference path kept at {len(guide_path)} points")
    
    # Cost functions
    collision_cost = CollisionCost(
        env=env,
        robot_radius=robot_radius,
        epsilon=0.2,  # Moderate safety margin to avoid grazing obstacles
        weight=120.0,  # Stronger weight to discourage collisions
        use_hard_constraint=True,
        hard_penalty=1e6  # Very large penalty so any collision is effectively infeasible
    )
    
    smoothness_cost = SmoothnessCost(
        penalize='acceleration',
        weight=0.5
    )
    
    goal_cost = GoalCost(
        goal=goal,
        weight=100.0
    )

    # Prevent "orbiting" around the goal: penalize moving away from goal once approaching
    goal_approach_cost = GoalApproachCost(
        goal=goal,
        weight=20.0,  # Reduced to allow detours around crowds
        power=2.0
    )
    
    path_length_cost = PathLengthCost(
        weight=2.0  # Reduced to allow longer detours around crowded areas
    )
    
    dt = 0.1
    turn_cost = TurnCost(
        weight=5.0,  # Moderate weight to discourage but not prohibit sharp turns
        method='angle_diff',
        dt=dt,
        use_sharp_turn_penalty=True,  # Enable penalty for very sharp turns
        max_angular_change=0.5,  # Moderate limit (1 rad ≈ 57°)
        sharp_turn_threshold=0.4  # Penalize turns above 0.8 rad (≈ 46°)
    )

    reference_cost = ReferencePathCost(
        reference_path=guide_path,
        weight=2.0,  # Reduced to allow deviation for crowd avoidance
        progress_weight=1.0,
        backtrack_weight=40.0,
        lateral_power=2.0
    )

    # Encourage slowing down near the end of the horizon (reduces end spirals)
    terminal_vel_cost = TerminalVelocityCost(
        weight=15.0,
        last_fraction=0.25
    )
    
    # 人流密度代价：避开高人流区域
    crowd_density_cost = CrowdDensityCost(
        crowd_regions=crowd_regions,
        weight=100.0  # 大幅提高权重，优先避开高密度区域
    )
    
    # 边界约束：严格限制不能超出地图边界
    boundary_cost = BoundaryConstraintCost(
        bounds=bounds,
        margin=0.5,  # 距离边界0.5m开始惩罚
        weight=200.0,  # 高权重确保严格遵守边界
        use_hard_constraint=True,  # 使用硬约束
        hard_penalty=1e6  # 超出边界给予极大惩罚
    )
    
    cost_function = CompositeCost([
        collision_cost,
        smoothness_cost,
        reference_cost,
        goal_cost,
        goal_approach_cost,
        path_length_cost,
        terminal_vel_cost,
        turn_cost,
        crowd_density_cost,  # 添加人流密度代价
        boundary_cost  # 添加边界约束
    ])
    
    # Create MPPI optimizer
    # 注意：由于B-Spline评估非常快（~0.04ms/样本），任务粒度较小
    # 使用4-8核比16核更高效，可以减少进程通信开销
    n_jobs = 2  # 2核心在400样本时效果最佳
    print(f"配置MPPI: n_samples={n_samples}, n_jobs={n_jobs}")
    print(f"  (任务粒度优化：4核比16核更高效，减少进程通信开销)")
    mppi = MPPI_BSpline(
        cost_function=cost_function,
        n_samples=n_samples,
        n_control_points=12,  # More DOF -> can discover more distinct routes
        bspline_degree=3,
        time_horizon=8.0,  # Longer horizon for complex paths
        n_timesteps=80,  # More timesteps for smoother paths
        temperature=temperature,
        noise_std=0.5,  # Larger local perturbations -> more exploration
        bounds=bounds,
        elite_ratio=0.15,  # Use more good samples -> keep diversity (less premature collapse)
        random_sample_ratio=0.12,  # Mix in global random samples to find new homotopies
        random_noise_std=1.2,
        n_jobs=n_jobs  # 使用4个CPU核心进行并行处理
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
    
    # Calculate crowd density statistics
    trajectory = result['trajectory']
    total_crowd_cost = 0.0
    region_distances = {region.name: 0.0 for region in crowd_regions}
    
    for i in range(len(trajectory) - 1):
        p1 = trajectory[i]
        p2 = trajectory[i + 1]
        segment_length = np.linalg.norm(p2 - p1)
        midpoint = (p1 + p2) / 2
        
        # Find which region this segment is in
        for region in crowd_regions:
            if region.contains_point(midpoint):
                region_distances[region.name] += segment_length
                total_crowd_cost += segment_length * region.density_multiplier
                break
    
    print("Path Crowd Density Analysis:")
    total_distance = sum(region_distances.values())
    for region in crowd_regions:
        dist = region_distances[region.name]
        pct = 100 * dist / total_distance if total_distance > 0 else 0
        print(f"  {region.name} ({region.density_multiplier}x): {dist:.2f}m ({pct:.1f}%)")
    
    avg_density = total_crowd_cost / total_distance if total_distance > 0 else 0
    print(f"  Total distance: {total_distance:.2f}m")
    print(f"  Weighted crowd cost: {total_crowd_cost:.2f}")
    print(f"  Average density factor: {avg_density:.2f}x")
    print()
    
    # Check if solution is feasible
    if result['best_cost_all_time'] > 1e5:
        print("⚠ Warning: High cost detected. Path may have collisions or other issues.")
        print("Suggestions: Increase n_iterations, n_samples, or adjust cost weights")
    else:
        print("✓ Solution found with reasonable cost")
    print()
    
    # Visualization
    vis = Visualizer(env, figsize=(14, 14))
    
    # 1. Plot final trajectory with crowd density regions
    fig1, ax1 = plt.subplots(1, 1, figsize=(14, 14))
    
    # Draw crowd density regions first (as background)
    for region in crowd_regions:
        x_min, x_max, y_min, y_max = region.x_min, region.x_max, region.y_min, region.y_max
        width = x_max - x_min
        height = y_max - y_min
        
        # Color based on density: red (high) to yellow (medium) to green (low)
        if region.density_multiplier >= 4.0:
            color = 'red'
            alpha = 0.15
        elif region.density_multiplier >= 2.0:
            color = 'orange'
            alpha = 0.12
        elif region.density_multiplier >= 1.5:
            color = 'yellow'
            alpha = 0.10
        else:
            color = 'lightgreen'
            alpha = 0.08
        
        rect = plt.Rectangle((x_min, y_min), width, height,
                            facecolor=color, alpha=alpha, edgecolor='gray',
                            linewidth=1, linestyle='--', zorder=1)
        ax1.add_patch(rect)
        
        # Add density label
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        ax1.text(cx, cy, f'{region.name}\n{region.density_multiplier}x',
                ha='center', va='center', fontsize=9, color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    vis.plot_environment(ax1)
    
    # Plot reference path (RRT* guide path)
    if guide_path is not None and len(guide_path) > 0:
        ax1.plot(guide_path[:, 0], guide_path[:, 1], 
                color='cyan', linewidth=2, linestyle='--', alpha=0.7,
                label='Reference Path (RRT*)', zorder=2)
    
    vis.plot_trajectory(
        result['trajectory'],
        ax=ax1,
        color='blue',
        linewidth=3,
        label=f'MPPI Solution (Iteration {result["best_iteration"]})',
        show_control_points=True,
        control_points=result['best_control_points_all_time']
    )
    ax1.scatter(start[0], start[1], color='green', s=300, marker='o',
               edgecolor='black', linewidth=2, zorder=5, label='Start')
    ax1.scatter(goal[0], goal[1], color='red', s=300, marker='*',
               edgecolor='black', linewidth=2, zorder=5, label='Goal')
    ax1.set_title('MPPI Solution - Christmas Market with Crowd Density', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12, loc='upper right')
    
    if save_plots:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        fig1.savefig(os.path.join(output_dir, 'christmas_market_solution.png'),
                    dpi=150, bbox_inches='tight')
        print(f"Saved: outputs/christmas_market_solution.png")
    
    if show_animation:
        print("\nCreating animation...")
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        anim = vis.create_animation(
            result['info_history'],
            save_path=os.path.join(output_dir, 'christmas_market_animation.gif'),
            show_samples=False,  # Don't show all samples (too cluttered)
            fps=10
        )
        print(f"Animation created: outputs/christmas_market_animation.gif")
    
    # 2. Plot with SDF visualization
    fig2, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    vis.visualize_sdf(ax=axes[0], robot_radius=robot_radius)
    if guide_path is not None and len(guide_path) > 0:
        axes[0].plot(guide_path[:, 0], guide_path[:, 1],
                    color='cyan', linewidth=2, linestyle='--', alpha=0.7,
                    label='Reference Path (RRT*)')
    axes[0].plot(result['trajectory'][:, 0], result['trajectory'][:, 1],
                color='blue', linewidth=3, label='MPPI Solution')
    axes[0].scatter(start[0], start[1], color='green', s=300, marker='o',
                   edgecolor='black', linewidth=2, zorder=5)
    axes[0].scatter(goal[0], goal[1], color='red', s=300, marker='*',
                   edgecolor='black', linewidth=2, zorder=5)
    axes[0].legend(fontsize=11)
    axes[0].set_title('Signed Distance Field', fontsize=14, fontweight='bold')
    
    final_info = result['info_history'][-1]
    vis.plot_environment(axes[1])
    if guide_path is not None and len(guide_path) > 0:
        axes[1].plot(guide_path[:, 0], guide_path[:, 1],
                    color='cyan', linewidth=2, linestyle='--', alpha=0.7,
                    label='Reference Path (RRT*)')
    vis.plot_weighted_trajectories(
        final_info['all_trajectories'][:30],
        final_info['weights'][:30],
        ax=axes[1]
    )
    vis.plot_trajectory(
        result['trajectory'],
        ax=axes[1],
        color='blue',
        linewidth=3,
        label=f'Best Trajectory (Iteration {result["best_iteration"]})'
    )
    axes[1].scatter(start[0], start[1], color='green', s=300, marker='o',
                   edgecolor='black', linewidth=2, zorder=5)
    axes[1].scatter(goal[0], goal[1], color='red', s=300, marker='*',
                   edgecolor='black', linewidth=2, zorder=5)
    axes[1].set_title('Final Iteration Samples (colored by weight)', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    
    plt.tight_layout()
    
    if save_plots:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        fig2.savefig(os.path.join(output_dir, 'christmas_market_analysis.png'),
                    dpi=150, bbox_inches='tight')
        print(f"Saved: outputs/christmas_market_analysis.png")
    
    # 3. Cost history
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
    vis.plot_cost_history(
        result['cost_history'],
        ax=ax3,
        best_iteration=result['best_iteration'],
        best_cost=result['best_cost_all_time']
    )
    
    if save_plots:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        fig3.savefig(os.path.join(output_dir, 'christmas_market_cost_history.png'),
                    dpi=150, bbox_inches='tight')
        print(f"Saved: outputs/christmas_market_cost_history.png")
    
    plt.close('all')
    
    return result


if __name__ == "__main__":
    result = run_mppi_christmas_market(
        temperature=1.4,
        n_samples=1200,
        n_iterations=400,
        show_animation=True,
        save_plots=True
    )
    
    print("\n" + "=" * 60)
    print("Christmas Market Scenario Complete!")
    print("=" * 60)

