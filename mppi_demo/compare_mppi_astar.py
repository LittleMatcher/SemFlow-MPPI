"""
Compare MPPI and A* algorithms

Run both algorithms on the same scenarios and visualize comparison results.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import time

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from mppi_core.environment_2d import Environment2D
from mppi_core.visualization import Visualizer
from astar_path_planner import AStarPlanner
from scenarios.u_trap import create_u_trap_scenario
from scenarios.narrow_passage import create_narrow_passage_scenario
from scenarios.christmas_market import create_christmas_market_environment
from tests.test_u_trap import run_mppi_u_trap


def compare_algorithms(env, start, goal, bounds, scenario_name, robot_radius=0.3):
    """Compare MPPI and A* on the same environment
    
    Args:
        env: Environment2D object
        start: Start position (2,)
        goal: Goal position (2,)
        bounds: Environment bounds
        scenario_name: Name of scenario for labeling
        robot_radius: Robot radius
    """
    print(f"\n{'='*60}")
    print(f"Comparing MPPI vs A*: {scenario_name}")
    print(f"{'='*60}\n")
    
    results = {}
    
    # === Run A* ===
    print("Running A* algorithm...")
    astar_start_time = time.time()
    
    astar_planner = AStarPlanner(
        env=env,
        resolution=0.15,  # Grid resolution
        robot_radius=robot_radius
    )
    
    astar_path, astar_info = astar_planner.plan(start, goal)
    astar_time = time.time() - astar_start_time
    
    if astar_path is not None:
        astar_length = astar_info['path_length']
        print(f"  ✓ A* found path (length: {astar_length:.2f}, time: {astar_time:.2f}s)")
        print(f"    Nodes expanded: {astar_info['nodes_expanded']}")
        results['astar'] = {
            'path': astar_path,
            'length': astar_length,
            'time': astar_time,
            'info': astar_info
        }
    else:
        print(f"  ✗ A* failed: {astar_info.get('error', 'Unknown')}")
        results['astar'] = None
    
    # === Run MPPI ===
    print("\nRunning MPPI algorithm...")
    mppi_start_time = time.time()
    
    # Import cost functions here to avoid circular imports
    from mppi_core.cost_functions import CollisionCost, SmoothnessCost, GoalCost, CompositeCost, PathLengthCost, TurnCost
    from mppi_core.mppi import MPPI_BSpline
    
    collision_cost = CollisionCost(
        env=env,
        robot_radius=robot_radius,
        epsilon=0.15,
        weight=100.0,
        use_hard_constraint=True,
        hard_penalty=1e6
    )
    
    smoothness_cost = SmoothnessCost(penalize='acceleration', weight=0.5)
    goal_cost = GoalCost(goal=goal, weight=80.0)
    path_length_cost = PathLengthCost(weight=20.0)
    
    dt = 0.1
    turn_cost = TurnCost(
        weight=8.0,
        method='angle_diff',
        dt=dt,
        use_sharp_turn_penalty=True
    )
    
    cost_function = CompositeCost([
        collision_cost,
        smoothness_cost,
        goal_cost,
        path_length_cost,
        turn_cost
    ])
    
    mppi = MPPI_BSpline(
        cost_function=cost_function,
        n_samples=200,
        n_control_points=12,
        bspline_degree=3,
        time_horizon=5.0,
        n_timesteps=50,
        temperature=1.0,
        noise_std=0.5,
        bounds=bounds
    )
    
    mppi_result = mppi.optimize(
        start=start,
        goal=goal,
        n_iterations=80,  # Reduced for faster comparison
        verbose=False
    )
    mppi_time = time.time() - mppi_start_time
    
    mppi_trajectory = mppi_result['trajectory']
    mppi_length = np.sum(np.linalg.norm(
        np.diff(mppi_trajectory, axis=0), axis=1
    ))
    
    print(f"  ✓ MPPI found path (length: {mppi_length:.2f}, time: {mppi_time:.2f}s)")
    print(f"    Best cost: {mppi_result['best_cost_all_time']:.2f}")
    print(f"    Iterations: {len(mppi_result['cost_history'])}")
    
    results['mppi'] = {
        'path': mppi_trajectory,
        'length': mppi_length,
        'time': mppi_time,
        'cost': mppi_result['best_cost_all_time'],
        'iterations': len(mppi_result['cost_history'])
    }
    
    # === Comparison ===
    print(f"\n{'='*60}")
    print("Comparison Summary:")
    print(f"{'='*60}")
    
    if results['astar'] is not None:
        print(f"A* Path Length:  {results['astar']['length']:.2f}")
        print(f"A* Time:         {results['astar']['time']:.2f}s")
    else:
        print(f"A*:              Failed")
    
    print(f"MPPI Path Length: {results['mppi']['length']:.2f}")
    print(f"MPPI Time:        {results['mppi']['time']:.2f}s")
    
    if results['astar'] is not None:
        length_diff = results['mppi']['length'] - results['astar']['length']
        length_diff_pct = (length_diff / results['astar']['length']) * 100
        print(f"\nLength Difference: {length_diff:+.2f} ({length_diff_pct:+.1f}%)")
        print(f"Time Ratio (MPPI/A*): {results['mppi']['time']/results['astar']['time']:.2f}x")
    
    return results


def visualize_comparison(env, start, goal, results, scenario_name, save_path):
    """Visualize comparison of MPPI and A* paths
    
    Args:
        env: Environment2D
        start: Start position
        goal: Goal position
        results: Dictionary with 'astar' and 'mppi' results
        scenario_name: Scenario name
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    vis = Visualizer(env, figsize=(10, 10))
    
    # Left: A* path
    vis.plot_environment(axes[0])
    if results['astar'] is not None:
        astar_path = results['astar']['path']
        axes[0].plot(astar_path[:, 0], astar_path[:, 1],
                    color='red', linewidth=3, label=f"A* (length: {results['astar']['length']:.2f})",
                    linestyle='-', marker='o', markersize=4, alpha=0.8)
    axes[0].scatter(start[0], start[1], color='green', s=300, marker='o',
                   edgecolor='black', linewidth=2, zorder=5, label='Start')
    axes[0].scatter(goal[0], goal[1], color='red', s=300, marker='*',
                   edgecolor='black', linewidth=2, zorder=5, label='Goal')
    axes[0].set_title(f'A* Path Planning - {scenario_name}', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].set_aspect('equal')
    
    # Right: MPPI path
    vis.plot_environment(axes[1])
    mppi_path = results['mppi']['path']
    axes[1].plot(mppi_path[:, 0], mppi_path[:, 1],
                color='blue', linewidth=3, label=f"MPPI (length: {results['mppi']['length']:.2f})",
                linestyle='-', alpha=0.8)
    axes[1].scatter(start[0], start[1], color='green', s=300, marker='o',
                   edgecolor='black', linewidth=2, zorder=5, label='Start')
    axes[1].scatter(goal[0], goal[1], color='red', s=300, marker='*',
                   edgecolor='black', linewidth=2, zorder=5, label='Goal')
    axes[1].set_title(f'MPPI Path Planning - {scenario_name}', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison: {save_path}")


def compare_u_trap():
    """Compare MPPI and A* on U-trap scenario"""
    env, start, goal, bounds = create_u_trap_scenario()
    results = compare_algorithms(env, start, goal, bounds, "U-Trap Scenario", robot_radius=0.2)
    
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'comparison_u_trap.png')
    visualize_comparison(env, start, goal, results, "U-Trap", save_path)
    
    return results


def compare_narrow_passage():
    """Compare MPPI and A* on narrow passage scenario"""
    env, start, goal, bounds = create_narrow_passage_scenario(passage_width=0.8)
    results = compare_algorithms(env, start, goal, bounds, "Narrow Passage Scenario", robot_radius=0.2)
    
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'comparison_narrow_passage.png')
    visualize_comparison(env, start, goal, results, "Narrow Passage", save_path)
    
    return results


def compare_christmas_market():
    """Compare MPPI and A* on Christmas Market scenario"""
    env, start, goal, bounds = create_christmas_market_environment()
    results = compare_algorithms(env, start, goal, bounds, "Christmas Market Scenario", robot_radius=0.3)
    
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'comparison_christmas_market.png')
    visualize_comparison(env, start, goal, results, "Christmas Market", save_path)
    
    return results


if __name__ == "__main__":
    print("=" * 70)
    print(" " * 15 + "MPPI vs A* Comparison")
    print("=" * 70)
    print("\nComparing path planning algorithms on multiple scenarios...\n")
    
    # Compare on different scenarios
    results_u = compare_u_trap()
    results_n = compare_narrow_passage()
    # results_c = compare_christmas_market()  # Commented out as it takes longer
    
    print("\n" + "=" * 70)
    print("Comparison Complete!")
    print("=" * 70)
    print("\nCheck the generated comparison images for visual results.")

