"""
Quick test for Christmas Market with crowd density - minimal iterations for fast testing
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scenarios.christmas_market import create_christmas_market_environment
from mppi_core import CrowdDensityCost

def test_crowd_density():
    """Quick test to verify crowd density regions work correctly"""
    print("=" * 60)
    print("Christmas Market Crowd Density - Quick Test")
    print("=" * 60)
    
    # Create environment
    env, start, goal, bounds, crowd_regions = create_christmas_market_environment(variant="v2")
    
    print(f"Environment: {len(env.obstacles)} obstacles")
    print(f"Start: ({start[0]:.1f}, {start[1]:.1f})")
    print(f"Goal: ({goal[0]:.1f}, {goal[1]:.1f})")
    print(f"\nCrowd regions: {len(crowd_regions)}")
    for region in crowd_regions:
        print(f"  - {region.name}: density {region.density_multiplier}x")
        print(f"    Bounds: x=[{region.x_min}, {region.x_max}], y=[{region.y_min}, {region.y_max}]")
    
    # Test crowd density cost function
    print("\n" + "=" * 60)
    print("Testing CrowdDensityCost function...")
    print("=" * 60)
    
    crowd_cost = CrowdDensityCost(crowd_regions=crowd_regions, weight=30.0)
    
    # Create test paths from start to goal
    # Path 1: Direct diagonal path (goes through center + high density areas)
    path1 = np.linspace(start, goal, 50).reshape(1, 50, 2)
    cost1 = crowd_cost(path1)[0]
    
    # Path 2: Detour around center via right side (avoids center, through right-medium)
    waypoint = np.array([5.0, 0.0])
    path2_part1 = np.linspace(start, waypoint, 25)
    path2_part2 = np.linspace(waypoint, goal, 25)
    path2 = np.vstack([path2_part1, path2_part2]).reshape(1, 50, 2)
    cost2 = crowd_cost(path2)[0]
    
    # Path 3: Detour via bottom right, then up along right edge
    waypoint3a = np.array([5.0, -5.0])
    waypoint3b = np.array([6.0, 0.0])
    path3_part1 = np.linspace(start, waypoint3a, 17)
    path3_part2 = np.linspace(waypoint3a, waypoint3b, 17)
    path3_part3 = np.linspace(waypoint3b, goal, 16)
    path3 = np.vstack([path3_part1, path3_part2, path3_part3]).reshape(1, 50, 2)
    cost3 = crowd_cost(path3)[0]
    
    print(f"\nPath comparison:")
    print(f"  Path 1 - Direct diagonal (through center 8x): cost = {cost1:.2f}")
    print(f"  Path 2 - Right detour (avoids center): cost = {cost2:.2f}")
    print(f"  Path 3 - Bottom-right detour: cost = {cost3:.2f}")
    print(f"\nBest path: ", end="")
    best_cost = min(cost1, cost2, cost3)
    if best_cost == cost1:
        print(f"Direct (but passes through center)")
    elif best_cost == cost2:
        print(f"Right detour - saves {100*(cost1-cost2)/cost1:.1f}% vs direct")
    else:
        print(f"Bottom-right detour - saves {100*(cost1-cost3)/cost1:.1f}% vs direct")
    
    # Visualize
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Draw crowd density regions (background first, then center on top)
    regions_sorted = sorted(crowd_regions, key=lambda r: r.density_multiplier)
    
    for region in regions_sorted:
        x_min, x_max, y_min, y_max = region.x_min, region.x_max, region.y_min, region.y_max
        width = x_max - x_min
        height = y_max - y_min
        
        if region.density_multiplier >= 7.0:
            color = 'darkred'
            alpha = 0.4
        elif region.density_multiplier >= 4.0:
            color = 'red'
            alpha = 0.25
        elif region.density_multiplier >= 2.0:
            color = 'orange'
            alpha = 0.2
        elif region.density_multiplier >= 1.5:
            color = 'yellow'
            alpha = 0.15
        else:
            color = 'lightgreen'
            alpha = 0.15
        
        rect = plt.Rectangle((x_min, y_min), width, height,
                            facecolor=color, alpha=alpha, edgecolor='black',
                            linewidth=2, linestyle='--', zorder=1)
        ax.add_patch(rect)
        
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        ax.text(cx, cy, f'{region.name}\n{region.density_multiplier}x',
                ha='center', va='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Draw obstacles
    for obs in env.obstacles:
        if hasattr(obs, 'center'):  # Circle
            circle = plt.Circle(obs.center, obs.radius, color='gray', alpha=0.7, zorder=3)
            ax.add_patch(circle)
        else:  # Rectangle
            rect = plt.Rectangle((obs.x_min, obs.y_min),
                                obs.x_max - obs.x_min,
                                obs.y_max - obs.y_min,
                                facecolor='gray', alpha=0.7, zorder=3)
            ax.add_patch(rect)
    
    # Draw paths
    ax.plot(path1[0, :, 0], path1[0, :, 1], 'b-', linewidth=3, 
            label=f'Path 1: Direct (cost={cost1:.0f})', zorder=4)
    ax.plot(path2[0, :, 0], path2[0, :, 1], 'g--', linewidth=3,
            label=f'Path 2: Right detour (cost={cost2:.0f})', zorder=4)
    ax.plot(path3[0, :, 0], path3[0, :, 1], 'm:', linewidth=3,
            label=f'Path 3: Bottom-right (cost={cost3:.0f})', zorder=4)
    
    # Mark start and goal
    ax.scatter(start[0], start[1], color='green', s=400, marker='o',
              edgecolor='black', linewidth=3, zorder=5, label='Start')
    ax.scatter(goal[0], goal[1], color='red', s=500, marker='*',
              edgecolor='black', linewidth=3, zorder=5, label='Goal')
    
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Christmas Market - Crowd Density with 3x3 Center Zone', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    
    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'christmas_market_crowd_density.png'),
                dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved: outputs/christmas_market_crowd_density.png")
    
    print("\n" + "=" * 60)
    print("Quick test completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    test_crowd_density()

