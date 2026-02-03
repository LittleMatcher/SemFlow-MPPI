"""
验证多核并行是否正常工作的测试脚本
显示实际使用的CPU核心数
"""
import numpy as np
import time
import os
import sys
import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mppi_core.mppi import MPPI_BSpline
from mppi_core.cost_functions import (
    CompositeCost, CollisionCost, SmoothnessCost, 
    GoalCost, PathLengthCost
)
from mppi_core.environment_2d import Environment2D, Rectangle


def monitor_cpu_usage():
    """监控当前进程及其子进程的CPU使用情况"""
    process = psutil.Process()
    children = process.children(recursive=True)
    
    cpu_count = len([p for p in children if p.is_running()])
    total_cpu = process.cpu_percent(interval=0.1)
    
    return cpu_count, total_cpu


def test_parallel():
    """测试并行处理"""
    
    print("="*70)
    print("多核并行验证测试")
    print("="*70)
    print(f"系统CPU核心数: {psutil.cpu_count()}")
    print()
    
    # 创建简单环境
    bounds = (0, 10, 0, 10)
    env = Environment2D(bounds=bounds)
    # 手动添加障碍物到列表
    env.obstacles = [
        Rectangle(x_min=3, x_max=4, y_min=2, y_max=8),
        Rectangle(x_min=6, x_max=7, y_min=2, y_max=8)
    ]
    
    start = np.array([1.0, 5.0])
    goal = np.array([9.0, 5.0])
    
    # 定义代价函数
    cost_components = [
        CollisionCost(env=env, robot_radius=0.3, weight=100.0),
        SmoothnessCost(weight=1.0),
        GoalCost(goal=goal, weight=50.0),
        PathLengthCost(weight=1.0),
    ]
    cost_function = CompositeCost(cost_components)
    
    # 测试配置
    n_samples = 400  # 足够大以展示并行优势
    n_jobs = 16
    
    print(f"配置: n_samples={n_samples}, n_jobs={n_jobs}")
    print()
    
    # 创建MPPI规划器
    mppi = MPPI_BSpline(
        cost_function=cost_function,
        n_samples=n_samples,
        n_control_points=10,
        time_horizon=5.0,
        n_timesteps=50,
        temperature=0.5,
        noise_std=0.3,
        bounds=bounds,
        elite_ratio=0.05,
        n_jobs=n_jobs
    )
    
    mppi.initialize(start, goal)
    
    print("\n开始运行MPPI优化...")
    print("监控CPU使用情况...\n")
    
    # 运行几次迭代并监控
    for i in range(10):
        start_time = time.time()
        
        # 执行一次迭代
        info = mppi.step()
        
        iter_time = time.time() - start_time
        
        # 监控CPU使用
        process = psutil.Process()
        children = process.children(recursive=True)
        active_workers = len([p for p in children if p.is_running()])
        
        print(f"迭代 {i+1}: 耗时 {iter_time:.3f}s, "
              f"代价 {info['best_cost']:.2f}, "
              f"活跃子进程: {active_workers}")
    
    print("\n" + "="*70)
    print("测试完成")
    print("="*70)
    
    # 最终统计
    process = psutil.Process()
    children = process.children(recursive=True)
    
    if len(children) > 0:
        print(f"\n✓ 检测到 {len(children)} 个子进程")
        print(f"  预期: {n_jobs} 个工作进程")
        if len(children) >= n_jobs:
            print(f"  ✓ 并行处理正常工作！")
        else:
            print(f"  ⚠ 子进程数少于预期，可能存在问题")
    else:
        print("\n✗ 未检测到子进程！")
        print("  并行处理可能没有正确启用")
    
    # 显示每个核心的使用情况
    print("\n每个CPU核心的使用率:")
    cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
    active_cores = sum(1 for p in cpu_percent if p > 10)
    print(f"  活跃核心数 (>10%): {active_cores} / {len(cpu_percent)}")
    
    if active_cores >= n_jobs * 0.8:  # 至少80%的核心在工作
        print(f"  ✓ 检测到 {active_cores} 个核心活跃，并行效果良好")
    elif active_cores >= 4:
        print(f"  ⚠ 检测到 {active_cores} 个核心活跃，部分并行")
    else:
        print(f"  ✗ 只有 {active_cores} 个核心活跃，并行效果不佳")


if __name__ == '__main__':
    test_parallel()
