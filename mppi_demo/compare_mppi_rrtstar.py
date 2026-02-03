"""
Compare MPPI and RRT* algorithms on Christmas Market scenario

This script compares MPPI (Model Predictive Path Integral) with RRT*
(Rapidly-exploring Random Tree Star) path planning algorithms.
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
from rrt_star_path_planner import RRTStarPlanner
from scenarios.christmas_market import create_christmas_market_environment
from tests.test_christmas_market import run_mppi_christmas_market


def compare_algorithms(env, start, goal, bounds, scenario_name, robot_radius=0.3, crowd_regions=None):
    """Compare MPPI and RRT* on the same environment
    
    Args:
        env: Environment2D object
        start: Start position (2,)
        goal: Goal position (2,)
        bounds: Environment bounds
        scenario_name: Name of scenario for labeling
        robot_radius: Robot radius
    """
    print(f"\n{'='*60}")
    print(f"Comparing MPPI vs RRT*: {scenario_name}")
    print(f"{'='*60}\n")
    
    results = {}
    
    # === Run RRT* ===
    print("Running RRT* algorithm...")
    rrt_start_time = time.time()
    
    rrt_planner = RRTStarPlanner(
        env=env,
        resolution=0.15,  # Grid resolution
        robot_radius=robot_radius,
        step_size=0.4,  # Step size for tree expansion
        goal_sample_rate=0.1,  # 10% probability to sample goal
        max_iterations=3000,  # Maximum iterations
        search_radius=1.5,  # Rewiring radius
        crowd_regions=crowd_regions,
        density_bias_strength=1.2,
        density_edge_weight=2.0,
        density_edge_samples=12,
        sample_candidates=16,
    )
    
    rrt_path, rrt_info = rrt_planner.plan(start, goal)
    rrt_time = time.time() - rrt_start_time
    
    if rrt_path is not None:
        rrt_length = rrt_info['path_length']
        print(f"  ✓ RRT* found path (length: {rrt_length:.2f}, time: {rrt_time:.2f}s)")
        print(f"    Iterations: {rrt_info['iterations']}")
        print(f"    Nodes in tree: {rrt_info['nodes']}")
        results['rrt_star'] = {
            'path': rrt_path,
            'length': rrt_length,
            'time': rrt_time,
            'info': rrt_info,
            'planner': rrt_planner
        }
    else:
        print(f"  ✗ RRT* failed: {rrt_info.get('error', 'Unknown')}")
        results['rrt_star'] = None
    
    # === Run MPPI ===
    print("\nRunning MPPI algorithm...")
    mppi_start_time = time.time()
    
    # Run MPPI with optimized parameters (suppress output for cleaner comparison)
    result_mppi = run_mppi_christmas_market(
        temperature=0.6,  # Balanced exploration/exploitation
        n_samples=400,  # Increased for better exploration
        n_iterations=70,  # Increased for convergence
        show_animation=False,  # Don't create animation during comparison
        save_plots=False  # Don't save plots during comparison
    )
    mppi_time = time.time() - mppi_start_time
    
    if result_mppi is not None:
        mppi_path = result_mppi['trajectory']
        mppi_length = np.sum(np.linalg.norm(np.diff(mppi_path, axis=0), axis=1))
        print(f"  ✓ MPPI found path (length: {mppi_length:.2f}, time: {mppi_time:.2f}s)")
        print(f"    Iterations: {len(result_mppi['cost_history'])}")
        print(f"    Best cost: {result_mppi['best_cost_all_time']:.2f}")
        results['mppi'] = {
            'path': mppi_path,
            'length': mppi_length,
            'time': mppi_time,
            'info': result_mppi,
            'cost': result_mppi['best_cost_all_time']
        }
    else:
        print(f"  ✗ MPPI failed")
        results['mppi'] = None
    
    # === Comparison Summary ===
    print(f"\n{'='*60}")
    print("Comparison Summary:")
    print(f"{'='*60}")
    
    if results['rrt_star'] is not None:
        print(f"RRT*:")
        print(f"  Path length: {results['rrt_star']['length']:.4f}")
        print(f"  Planning time: {results['rrt_star']['time']:.2f}s")
        print(f"  Tree nodes: {results['rrt_star']['info']['nodes']}")
        print(f"  Iterations: {results['rrt_star']['info']['iterations']}")
    
    if results['mppi'] is not None:
        print(f"\nMPPI:")
        print(f"  Path length: {results['mppi']['length']:.4f}")
        print(f"  Planning time: {results['mppi']['time']:.2f}s")
        print(f"  Iterations: {len(results['mppi']['info']['cost_history'])}")
        print(f"  Best cost: {results['mppi']['cost']:.2f}")
    
    if results['rrt_star'] is not None and results['mppi'] is not None:
        length_ratio = results['mppi']['length'] / results['rrt_star']['length']
        time_ratio = results['mppi']['time'] / results['rrt_star']['time']
        
        print(f"\nComparison:")
        print(f"  Length ratio (MPPI/RRT*): {length_ratio:.3f}")
        print(f"  Time ratio (MPPI/RRT*): {time_ratio:.3f}")
        
        if length_ratio < 1.0:
            print(f"  → MPPI path is {((1-length_ratio)*100):.1f}% shorter")
        else:
            print(f"  → RRT* path is {((length_ratio-1)*100):.1f}% shorter")
        
        if time_ratio < 1.0:
            print(f"  → MPPI is {((1-time_ratio)*100):.1f}% faster")
        else:
            print(f"  → RRT* is {((time_ratio-1)*100):.1f}% faster")
    
    print(f"{'='*60}\n")
    
    return results


def visualize_comparison(env, start, goal, results, scenario_name, save_path):
    """Visualize comparison between MPPI and RRT*
    
    Args:
        env: Environment2D
        start: Start position
        goal: Goal position
        results: Dictionary with 'mppi' and 'rrt_star' results
        scenario_name: Name for title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    vis = Visualizer(env, figsize=(10, 10))
    
    # Left: RRT* path with tree
    vis.plot_environment(axes[0])
    if results['rrt_star'] is not None:
        rrt_planner = results['rrt_star']['planner']
        rrt_planner.visualize_tree(axes[0], show_path=True, path_color='red')
        rrt_path = results['rrt_star']['path']
        axes[0].plot(rrt_path[:, 0], rrt_path[:, 1],
                    color='red', linewidth=3, label=f"RRT* (length: {results['rrt_star']['length']:.2f})",
                    linestyle='-', alpha=0.8, zorder=6)
    else:
        axes[0].scatter(start[0], start[1], color='green', s=300, marker='o',
                       edgecolor='black', linewidth=2, zorder=5, label='Start')
        axes[0].scatter(goal[0], goal[1], color='red', s=300, marker='*',
                       edgecolor='black', linewidth=2, zorder=5, label='Goal')
    
    axes[0].set_title(f'RRT* Path Planning - {scenario_name}', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].set_aspect('equal')
    
    # Right: MPPI path
    vis.plot_environment(axes[1])
    if results['mppi'] is not None:
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


def compare_christmas_market():
    """Compare MPPI and RRT* on Christmas Market scenario"""
    env, start, goal, bounds, crowd_regions = create_christmas_market_environment()
    results = compare_algorithms(env, start, goal, bounds, "Christmas Market Scenario", robot_radius=0.3, crowd_regions=crowd_regions)
    
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'comparison_christmas_market_mppi_rrtstar.png')
    visualize_comparison(env, start, goal, results, "Christmas Market", save_path)
    
    return results


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" " * 15 + "MPPI vs RRT* Comparison")
    print(" " * 10 + "Christmas Market Scenario")
    print("=" * 70 + "\n")
    
    results = compare_christmas_market()
    
    print("\n" + "=" * 70)
    print("Comparison complete!")
    print("=" * 70 + "\n")

