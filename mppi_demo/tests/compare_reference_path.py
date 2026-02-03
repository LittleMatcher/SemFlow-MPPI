"""
对比测试：使用RRT*参考路径 vs 不使用参考路径
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scenarios.christmas_market import create_christmas_market_environment
from mppi_core import (
    CollisionCost, SmoothnessCost, GoalCost, CompositeCost,
    GoalApproachCost, PathLengthCost, ReferencePathCost, TerminalVelocityCost,
    TurnCost, CrowdDensityCost, BoundaryConstraintCost, MPPI_BSpline, Visualizer
)
from rrt_star_path_planner import RRTStarPlanner


def run_mppi_with_reference(env, start, goal, bounds, crowd_regions, guide_path):
    """使用参考路径运行MPPI"""
    print("\n" + "=" * 70)
    print("测试 1: 使用 RRT* 参考路径")
    print("=" * 70)
    
    robot_radius = 0.01
    
    # Cost functions WITH reference path
    collision_cost = CollisionCost(env=env, robot_radius=robot_radius, epsilon=0.2, 
                                  weight=120.0, use_hard_constraint=True, hard_penalty=1e6)
    smoothness_cost = SmoothnessCost(penalize='acceleration', weight=0.5)
    goal_cost = GoalCost(goal=goal, weight=100.0)
    goal_approach_cost = GoalApproachCost(goal=goal, weight=20.0, power=2.0)
    path_length_cost = PathLengthCost(weight=2.0)
    turn_cost = TurnCost(weight=5.0, method='angle_diff', dt=0.1)
    terminal_vel_cost = TerminalVelocityCost(weight=15.0, last_fraction=0.25)
    crowd_density_cost = CrowdDensityCost(crowd_regions=crowd_regions, weight=100.0)
    boundary_cost = BoundaryConstraintCost(bounds=bounds, margin=0.5, weight=200.0, 
                                          use_hard_constraint=True, hard_penalty=1e6)
    
    # Reference path cost (关键区别)
    reference_cost = ReferencePathCost(reference_path=guide_path, weight=3.0, 
                                      progress_weight=1.0, backtrack_weight=40.0, lateral_power=2.0)
    
    cost_function = CompositeCost([
        collision_cost, smoothness_cost, reference_cost, goal_cost,
        goal_approach_cost, path_length_cost, terminal_vel_cost,
        turn_cost, crowd_density_cost, boundary_cost
    ])
    
    mppi = MPPI_BSpline(
        cost_function=cost_function, n_samples=400, n_control_points=10,
        bspline_degree=3, time_horizon=8.0, n_timesteps=80,
        temperature=0.6, noise_std=0.7, bounds=bounds, elite_ratio=0.05
    )
    
    start_time = time.time()
    result = mppi.optimize(start=start, goal=goal, n_iterations=70, verbose=False)
    optimization_time = time.time() - start_time
    
    return result, optimization_time


def run_mppi_without_reference(env, start, goal, bounds, crowd_regions):
    """不使用参考路径运行MPPI"""
    print("\n" + "=" * 70)
    print("测试 2: 不使用参考路径")
    print("=" * 70)
    
    robot_radius = 0.01
    
    # Cost functions WITHOUT reference path
    collision_cost = CollisionCost(env=env, robot_radius=robot_radius, epsilon=0.2, 
                                  weight=120.0, use_hard_constraint=True, hard_penalty=1e6)
    smoothness_cost = SmoothnessCost(penalize='acceleration', weight=0.5)
    goal_cost = GoalCost(goal=goal, weight=100.0)
    goal_approach_cost = GoalApproachCost(goal=goal, weight=20.0, power=2.0)
    path_length_cost = PathLengthCost(weight=2.0)
    turn_cost = TurnCost(weight=5.0, method='angle_diff', dt=0.1)
    terminal_vel_cost = TerminalVelocityCost(weight=15.0, last_fraction=0.25)
    crowd_density_cost = CrowdDensityCost(crowd_regions=crowd_regions, weight=100.0)
    boundary_cost = BoundaryConstraintCost(bounds=bounds, margin=0.5, weight=200.0, 
                                          use_hard_constraint=True, hard_penalty=1e6)
    
    # NO reference_cost! (关键区别)
    
    cost_function = CompositeCost([
        collision_cost, smoothness_cost, goal_cost,
        goal_approach_cost, path_length_cost, terminal_vel_cost,
        turn_cost, crowd_density_cost, boundary_cost
    ])
    
    mppi = MPPI_BSpline(
        cost_function=cost_function, n_samples=400, n_control_points=10,
        bspline_degree=3, time_horizon=8.0, n_timesteps=80,
        temperature=0.6, noise_std=0.7, bounds=bounds, elite_ratio=0.05
    )
    
    start_time = time.time()
    result = mppi.optimize(start=start, goal=goal, n_iterations=70, verbose=False)
    optimization_time = time.time() - start_time
    
    return result, optimization_time


def calculate_crowd_cost(trajectory, crowd_regions):
    """计算轨迹的人流密度成本"""
    total_crowd_cost = 0.0
    total_distance = 0.0
    
    for i in range(len(trajectory) - 1):
        p1 = trajectory[i]
        p2 = trajectory[i + 1]
        segment_length = np.linalg.norm(p2 - p1)
        midpoint = (p1 + p2) / 2
        
        # 找到最高密度
        max_density = 1.0
        for region in crowd_regions:
            if region.contains_point(midpoint):
                max_density = max(max_density, region.density_multiplier)
        
        total_crowd_cost += segment_length * max_density
        total_distance += segment_length
    
    avg_density = total_crowd_cost / total_distance if total_distance > 0 else 0
    return total_crowd_cost, avg_density, total_distance


def main():
    print("\n" + "=" * 70)
    print("MPPI 对比测试：使用 vs 不使用 RRT* 参考路径")
    print("=" * 70)
    
    # 创建环境
    env, start, goal, bounds, crowd_regions = create_christmas_market_environment(variant="v2")
    
    print(f"\n环境信息:")
    print(f"  起点: ({start[0]:.1f}, {start[1]:.1f})")
    print(f"  终点: ({goal[0]:.1f}, {goal[1]:.1f})")
    print(f"  障碍物: {len(env.obstacles)}")
    print(f"  人流区域: {len(crowd_regions)}")
    
    # 生成RRT*参考路径
    print(f"\n生成 RRT* 参考路径...")
    planner = RRTStarPlanner(env, resolution=0.1, robot_radius=0.35, step_size=0.8,
                            goal_sample_rate=0.10, max_iterations=3000, search_radius=2.0)
    guide_path, planner_info = planner.plan(start, goal)
    
    if guide_path is None:
        print("❌ RRT* 规划失败!")
        return
    
    # 简化为4个点
    if len(guide_path) > 4:
        indices = np.linspace(0, len(guide_path) - 1, 4, dtype=int)
        guide_path = guide_path[indices]
    
    print(f"✓ RRT* 路径: {planner_info['path_length']:.2f}m, 简化为 {len(guide_path)} 个关键点")
    
    # 测试1: 使用参考路径
    result_with_ref, time_with_ref = run_mppi_with_reference(
        env, start, goal, bounds, crowd_regions, guide_path
    )
    
    # 测试2: 不使用参考路径
    result_no_ref, time_no_ref = run_mppi_without_reference(
        env, start, goal, bounds, crowd_regions
    )
    
    # 分析结果
    print("\n" + "=" * 70)
    print("结果对比")
    print("=" * 70)
    
    # 计算人流成本
    crowd_cost_with, avg_dens_with, dist_with = calculate_crowd_cost(
        result_with_ref['trajectory'], crowd_regions
    )
    crowd_cost_no, avg_dens_no, dist_no = calculate_crowd_cost(
        result_no_ref['trajectory'], crowd_regions
    )
    
    print(f"\n{'指标':<25} | {'使用RRT*参考':>15} | {'不使用参考':>15} | {'差异':>15}")
    print("-" * 70)
    print(f"{'优化时间 (秒)':<25} | {time_with_ref:>15.2f} | {time_no_ref:>15.2f} | {time_no_ref-time_with_ref:>+15.2f}")
    print(f"{'最终代价':<25} | {result_with_ref['best_cost_all_time']:>15.2f} | {result_no_ref['best_cost_all_time']:>15.2f} | {result_no_ref['best_cost_all_time']-result_with_ref['best_cost_all_time']:>+15.2f}")
    print(f"{'路径长度 (m)':<25} | {dist_with:>15.2f} | {dist_no:>15.2f} | {dist_no-dist_with:>+15.2f}")
    print(f"{'人流总成本':<25} | {crowd_cost_with:>15.2f} | {crowd_cost_no:>15.2f} | {crowd_cost_no-crowd_cost_with:>+15.2f}")
    print(f"{'平均人流密度 (×)':<25} | {avg_dens_with:>15.2f} | {avg_dens_no:>15.2f} | {avg_dens_no-avg_dens_with:>+15.2f}")
    print(f"{'最优迭代':<25} | {result_with_ref['best_iteration']:>15} | {result_no_ref['best_iteration']:>15} | {result_no_ref['best_iteration']-result_with_ref['best_iteration']:>+15}")
    
    # 判断哪个更好
    print(f"\n{'性能总结':^70}")
    print("-" * 70)
    
    if result_with_ref['best_cost_all_time'] < result_no_ref['best_cost_all_time']:
        improvement = (result_no_ref['best_cost_all_time'] - result_with_ref['best_cost_all_time']) / result_no_ref['best_cost_all_time'] * 100
        print(f"✓ 使用RRT*参考路径的代价更低 (改善 {improvement:.1f}%)")
    else:
        improvement = (result_with_ref['best_cost_all_time'] - result_no_ref['best_cost_all_time']) / result_with_ref['best_cost_all_time'] * 100
        print(f"✓ 不使用参考路径的代价更低 (改善 {improvement:.1f}%)")
    
    if crowd_cost_with < crowd_cost_no:
        print(f"✓ 使用RRT*参考路径的人流成本更低")
    else:
        print(f"✓ 不使用参考路径的人流成本更低")
    
    if time_with_ref < time_no_ref:
        print(f"✓ 使用RRT*参考路径收敛更快")
    else:
        print(f"✓ 不使用参考路径收敛更快")
    
    # 可视化对比
    print(f"\n生成对比可视化...")
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    
    for idx, (ax, result, title, color) in enumerate([
        (axes[0], result_with_ref, 'With RRT* Reference', 'blue'),
        (axes[1], result_no_ref, 'Without Reference', 'red')
    ]):
        # 绘制人流密度区域
        for region in crowd_regions:
            x_min, x_max, y_min, y_max = region.x_min, region.x_max, region.y_min, region.y_max
            width, height = x_max - x_min, y_max - y_min
            
            if region.density_multiplier >= 15.0:
                fcolor, alpha = 'darkred', 0.3
            elif region.density_multiplier >= 10.0:
                fcolor, alpha = 'red', 0.25
            elif region.density_multiplier >= 5.0:
                fcolor, alpha = 'orange', 0.2
            elif region.density_multiplier >= 2.0:
                fcolor, alpha = 'yellow', 0.15
            else:
                fcolor, alpha = 'lightgreen', 0.1
            
            rect = plt.Rectangle((x_min, y_min), width, height, facecolor=fcolor, 
                                alpha=alpha, edgecolor='gray', linewidth=1, linestyle='--', zorder=1)
            ax.add_patch(rect)
        
        # 绘制障碍物
        for obs in env.obstacles:
            if hasattr(obs, 'center'):
                circle = plt.Circle(obs.center, obs.radius, color='gray', alpha=0.7, zorder=3)
                ax.add_patch(circle)
            else:
                rect = plt.Rectangle((obs.x_min, obs.y_min), obs.x_max - obs.x_min,
                                    obs.y_max - obs.y_min, facecolor='gray', alpha=0.7, zorder=3)
                ax.add_patch(rect)
        
        # 绘制参考路径（仅左图）
        if idx == 0 and guide_path is not None:
            ax.plot(guide_path[:, 0], guide_path[:, 1], 'c--', linewidth=2, 
                   alpha=0.7, label='RRT* Reference', zorder=2)
        
        # 绘制MPPI轨迹
        trajectory = result['trajectory']
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=3, 
               label='MPPI Solution', zorder=4)
        
        # 起点和终点
        ax.scatter(start[0], start[1], color='green', s=300, marker='o',
                  edgecolor='black', linewidth=2, zorder=5, label='Start')
        ax.scatter(goal[0], goal[1], color='red', s=300, marker='*',
                  edgecolor='black', linewidth=2, zorder=5, label='Goal')
        
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title(f'{title}\nCost: {result["best_cost_all_time"]:.0f}, '
                    f'Length: {dist_with if idx==0 else dist_no:.1f}m, '
                    f'Avg Density: {avg_dens_with if idx==0 else avg_dens_no:.1f}×',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='upper left')
    
    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'comparison_reference_path.png'),
                dpi=150, bbox_inches='tight')
    print(f"✓ 对比图保存: outputs/comparison_reference_path.png")
    
    print("\n" + "=" * 70)
    print("对比测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
