"""
快速测试多核并行处理功能
只比较单核vs多核，迭代次数较少
"""
import numpy as np
import time
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mppi_core.mppi import MPPI_BSpline
from mppi_core.cost_functions import (
    CompositeCost, CollisionCost, SmoothnessCost, 
    GoalCost, PathLengthCost, CrowdDensityCost
)
from scenarios.christmas_market import create_christmas_market_environment
from multiprocessing import cpu_count


def run_quick_test():
    """快速测试单核vs多核性能"""
    
    print("="*70)
    print("MPPI 多核并行快速测试")
    print("="*70)
    
    # 获取CPU核心数
    n_cpu = cpu_count()
    print(f"\n系统CPU核心数: {n_cpu}\n")
    
    # 创建环境
    env, start, goal, bounds, crowd_regions = create_christmas_market_environment("v1")
    
    # 定义代价函数
    cost_components = [
        CollisionCost(env=env, robot_radius=0.35, weight=200.0),
        SmoothnessCost(weight=1.0),
        GoalCost(goal=goal, weight=50.0),
        PathLengthCost(weight=2.0),
        CrowdDensityCost(crowd_regions=crowd_regions, weight=100.0),
    ]
    cost_function = CompositeCost(cost_components)
    
    # 测试配置
    n_samples = 300  # 增加样本数以展示并行优势
    n_iterations = 30  # 增加迭代次数
    
    results = []
    
    # 测试1：单核（串行）
    print("-"*70)
    print("测试1：单核（串行）")
    print("-"*70)
    
    mppi_serial = MPPI_BSpline(
        cost_function=cost_function,
        n_samples=n_samples,
        n_control_points=10,
        time_horizon=5.0,
        n_timesteps=50,
        temperature=0.5,
        noise_std=0.3,
        bounds=bounds,
        elite_ratio=0.05,
        n_jobs=1  # 单核
    )
    
    mppi_serial.initialize(start, goal)
    
    start_time = time.time()
    for i in range(n_iterations):
        info = mppi_serial.step()
        if (i + 1) % 5 == 0:
            print(f"  迭代 {i+1}/{n_iterations}, 代价: {info['best_cost']:.2f}")
    
    serial_time = time.time() - start_time
    serial_cost = mppi_serial.best_cost_all_time
    
    print(f"\n✓ 优化时间: {serial_time:.2f} 秒")
    print(f"✓ 最终代价: {serial_cost:.2f}")
    
    results.append({
        'name': '单核',
        'n_jobs': 1,
        'time': serial_time,
        'cost': serial_cost
    })
    
    # 测试2：多核（全部核心）
    print("\n" + "-"*70)
    print(f"测试2：多核（使用全部{n_cpu}个核心）")
    print("-"*70)
    
    mppi_parallel = MPPI_BSpline(
        cost_function=cost_function,
        n_samples=n_samples,
        n_control_points=10,
        time_horizon=5.0,
        n_timesteps=50,
        temperature=0.5,
        noise_std=0.3,
        bounds=bounds,
        elite_ratio=0.05,
        n_jobs=-1  # 全部核心
    )
    
    mppi_parallel.initialize(start, goal)
    
    start_time = time.time()
    for i in range(n_iterations):
        info = mppi_parallel.step()
        if (i + 1) % 5 == 0:
            print(f"  迭代 {i+1}/{n_iterations}, 代价: {info['best_cost']:.2f}")
    
    parallel_time = time.time() - start_time
    parallel_cost = mppi_parallel.best_cost_all_time
    
    print(f"\n✓ 优化时间: {parallel_time:.2f} 秒")
    print(f"✓ 最终代价: {parallel_cost:.2f}")
    
    results.append({
        'name': f'多核({n_cpu}核)',
        'n_jobs': n_cpu,
        'time': parallel_time,
        'cost': parallel_cost
    })
    
    # 打印对比结果
    print("\n" + "="*70)
    print("性能对比")
    print("="*70)
    
    speedup = serial_time / parallel_time
    time_saved = serial_time - parallel_time
    percent_saved = (time_saved / serial_time) * 100
    
    print(f"\n{'配置':<15} {'CPU核心':<10} {'时间(秒)':<12} {'代价':<12}")
    print("-"*55)
    for r in results:
        print(f"{r['name']:<15} {r['n_jobs']:<10} {r['time']:<12.2f} {r['cost']:<12.2f}")
    
    print("\n" + "-"*70)
    print(f"加速比: {speedup:.2f}x")
    print(f"时间节省: {time_saved:.2f} 秒 ({percent_saved:.1f}%)")
    print("-"*70)
    
    if speedup > 1.5:
        print(f"\n✓ 并行处理效果显著！加速比达到 {speedup:.2f}x")
    elif speedup > 1.1:
        print(f"\n✓ 并行处理有效，加速比为 {speedup:.2f}x")
    else:
        print(f"\n⚠ 并行处理效果不明显，可能是因为样本数太少或环境太简单")
    
    print(f"\n提示：增加 n_samples 和 n_iterations 可以获得更大的加速效果")


if __name__ == '__main__':
    run_quick_test()
