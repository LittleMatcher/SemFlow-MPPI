"""
对比单核和多核并行处理的性能
比较不同CPU核心数下MPPI的优化时间
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mppi_core.mppi import MPPI_BSpline
from mppi_core.cost_functions import (
    CompositeCost, CollisionCost, SmoothnessCost, 
    GoalCost, PathLengthCost, GoalApproachCost,
    ReferencePathCost, TerminalVelocityCost, TurnCost,
    CrowdDensityCost, BoundaryConstraintCost
)
from scenarios.christmas_market import create_christmas_market_environment
from rrt_star_path_planner import RRTStarPlanner
from multiprocessing import cpu_count


def run_mppi_with_n_jobs(n_jobs, n_iterations=50):
    """
    使用指定数量的CPU核心运行MPPI
    
    Args:
        n_jobs: CPU核心数，1表示单核，-1表示所有核心
        n_iterations: 迭代次数
    
    Returns:
        optimization_time: 优化耗时（秒）
        final_cost: 最终代价
    """
    # 创建圣诞市场场景
    env, start, goal, bounds, crowd_regions = create_christmas_market_environment("v1")
    
    # 生成RRT*参考路径
    rrt_planner = RRTStarPlanner(
        env=env,
        resolution=0.1,
        robot_radius=0.35,
        step_size=0.8,
        max_iterations=3000,
        search_radius=2.0
    )
    
    rrt_path, rrt_info = rrt_planner.plan(start, goal)
    if rrt_path is None:
        print("RRT* failed to find a path!")
        return None, None
    
    # 简化参考路径为4个关键点
    indices = np.linspace(0, len(rrt_path) - 1, 4, dtype=int)
    reference_path = rrt_path[indices]
    
    # 定义代价函数
    cost_components = [
        CollisionCost(env=env, robot_radius=0.35, weight=200.0),
        SmoothnessCost(weight=1.0),
        GoalCost(goal=goal, weight=50.0),
        PathLengthCost(weight=2.0),
        GoalApproachCost(goal=goal, weight=20.0),
        ReferencePathCost(reference_path=reference_path, weight=3.0),
        TerminalVelocityCost(weight=10.0),
        TurnCost(weight=1.0),
        CrowdDensityCost(crowd_regions=crowd_regions, weight=100.0),
        BoundaryConstraintCost(bounds=env.bounds, robot_radius=0.35, weight=200.0)
    ]
    cost_function = CompositeCost(cost_components)
    
    # 创建MPPI规划器
    mppi = MPPI_BSpline(
        cost_function=cost_function,
        n_samples=200,
        n_control_points=12,
        bspline_degree=3,
        time_horizon=6.0,
        n_timesteps=60,
        temperature=0.5,
        noise_std=0.3,
        bounds=env.bounds,
        elite_ratio=0.05,
        n_jobs=n_jobs  # 设置CPU核心数
    )
    
    # 设置起点和终点
    mppi.set_start_goal(start, goal)
    
    # 运行优化
    start_time = time.time()
    for i in range(n_iterations):
        info = mppi.step()
        if (i + 1) % 10 == 0:
            print(f"  Iteration {i+1}/{n_iterations}, Best cost: {info['best_cost']:.2f}")
    
    optimization_time = time.time() - start_time
    final_cost = mppi.best_cost_all_time
    
    return optimization_time, final_cost


def main():
    """主函数 - 对比不同CPU核心数的性能"""
    
    print("="*80)
    print("MPPI 多核并行性能测试")
    print("="*80)
    
    # 获取CPU核心数
    n_cpu = cpu_count()
    print(f"\n系统CPU核心数: {n_cpu}")
    
    # 测试不同的核心配置
    test_configs = [
        (1, "单核（串行）"),
        (2, "双核"),
        (4, "四核"),
        (-1, f"全部核心 ({n_cpu}核)")
    ]
    
    # 只测试可用的配置
    test_configs = [(n, desc) for n, desc in test_configs if n == -1 or n <= n_cpu]
    
    results = []
    
    for n_jobs, description in test_configs:
        print(f"\n{'='*60}")
        print(f"测试配置: {description}")
        print(f"{'='*60}")
        
        opt_time, final_cost = run_mppi_with_n_jobs(n_jobs=n_jobs, n_iterations=50)
        
        if opt_time is not None:
            results.append({
                'n_jobs': n_jobs if n_jobs > 0 else n_cpu,
                'description': description,
                'time': opt_time,
                'cost': final_cost
            })
            print(f"\n优化时间: {opt_time:.2f} 秒")
            print(f"最终代价: {final_cost:.2f}")
    
    # 打印对比结果
    print("\n" + "="*80)
    print("性能对比总结")
    print("="*80)
    
    baseline_time = results[0]['time']
    
    print(f"\n{'配置':<20} {'时间(秒)':<12} {'加速比':<12} {'最终代价':<12}")
    print("-"*60)
    
    for result in results:
        speedup = baseline_time / result['time']
        print(f"{result['description']:<20} {result['time']:<12.2f} "
              f"{speedup:<12.2f}x {result['cost']:<12.2f}")
    
    # 绘制性能对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 时间对比
    descriptions = [r['description'] for r in results]
    times = [r['time'] for r in results]
    speedups = [baseline_time / r['time'] for r in results]
    
    ax1.bar(descriptions, times, color=['red', 'orange', 'green', 'blue'][:len(times)])
    ax1.set_ylabel('优化时间 (秒)', fontsize=12)
    ax1.set_title('不同CPU核心配置的优化时间', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # 在柱状图上添加数值标签
    for i, (desc, t) in enumerate(zip(descriptions, times)):
        ax1.text(i, t, f'{t:.1f}s', ha='center', va='bottom', fontsize=10)
    
    # 加速比对比
    ax2.plot(range(len(speedups)), speedups, 'o-', linewidth=2, markersize=8, color='green')
    ax2.set_xticks(range(len(descriptions)))
    ax2.set_xticklabels(descriptions)
    ax2.set_ylabel('加速比 (相对于单核)', fontsize=12)
    ax2.set_title('并行加速比', fontsize=14, fontweight='bold')
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='基准线')
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    # 在点上添加数值标签
    for i, s in enumerate(speedups):
        ax2.text(i, s, f'{s:.2f}x', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('outputs/parallel_performance.png', dpi=150, bbox_inches='tight')
    print(f"\n性能对比图已保存至: outputs/parallel_performance.png")
    
    plt.show()


if __name__ == '__main__':
    main()
