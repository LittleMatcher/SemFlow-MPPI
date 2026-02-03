"""
简单示例：使用多核并行MPPI进行路径规划
展示如何启用并行处理来加速优化
"""
import numpy as np
import matplotlib.pyplot as plt
import time

from mppi_core.mppi import MPPI_BSpline
from mppi_core.cost_functions import (
    CompositeCost, CollisionCost, SmoothnessCost, 
    GoalCost, PathLengthCost
)
from mppi_core.environment_2d import Environment2D, Rectangle
from mppi_core.visualization import plot_environment_and_trajectory


def main():
    """主函数 - 对比串行和并行MPPI"""
    
    print("="*70)
    print("多核并行MPPI示例")
    print("="*70)
    
    # 创建简单环境
    env = Environment2D(bounds=(0, 10, 0, 10))
    
    # 添加一些障碍物
    env.add_obstacle(Rectangle(x_min=3, x_max=4, y_min=2, y_max=8))
    env.add_obstacle(Rectangle(x_min=6, x_max=7, y_min=2, y_max=8))
    
    # 起点和终点
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
    n_samples = 400  # 较大样本数以展示并行优势
    n_iterations = 40
    
    print(f"\n配置:")
    print(f"  样本数: {n_samples}")
    print(f"  迭代次数: {n_iterations}")
    
    # 方法1: 串行MPPI
    print(f"\n{'-'*70}")
    print("运行串行MPPI (单核)...")
    print(f"{'-'*70}")
    
    mppi_serial = MPPI_BSpline(
        cost_function=cost_function,
        n_samples=n_samples,
        n_control_points=10,
        time_horizon=5.0,
        n_timesteps=50,
        temperature=0.5,
        noise_std=0.3,
        bounds=env.bounds,
        elite_ratio=0.05,
        n_jobs=1  # 单核
    )
    
    mppi_serial.initialize(start, goal)
    
    start_time = time.time()
    for i in range(n_iterations):
        info_serial = mppi_serial.step()
        if (i + 1) % 10 == 0:
            print(f"  迭代 {i+1}/{n_iterations}, 代价: {info_serial['best_cost']:.2f}")
    
    serial_time = time.time() - start_time
    serial_trajectory = mppi_serial.best_trajectory_all_time
    serial_cost = mppi_serial.best_cost_all_time
    
    print(f"\n✓ 串行优化完成")
    print(f"  时间: {serial_time:.2f} 秒")
    print(f"  最终代价: {serial_cost:.2f}")
    
    # 方法2: 并行MPPI
    print(f"\n{'-'*70}")
    print("运行并行MPPI (多核)...")
    print(f"{'-'*70}")
    
    mppi_parallel = MPPI_BSpline(
        cost_function=cost_function,
        n_samples=n_samples,
        n_control_points=10,
        time_horizon=5.0,
        n_timesteps=50,
        temperature=0.5,
        noise_std=0.3,
        bounds=env.bounds,
        elite_ratio=0.05,
        n_jobs=-1  # 使用所有核心
    )
    
    mppi_parallel.initialize(start, goal)
    
    start_time = time.time()
    for i in range(n_iterations):
        info_parallel = mppi_parallel.step()
        if (i + 1) % 10 == 0:
            print(f"  迭代 {i+1}/{n_iterations}, 代价: {info_parallel['best_cost']:.2f}")
    
    parallel_time = time.time() - start_time
    parallel_trajectory = mppi_parallel.best_trajectory_all_time
    parallel_cost = mppi_parallel.best_cost_all_time
    
    print(f"\n✓ 并行优化完成")
    print(f"  时间: {parallel_time:.2f} 秒")
    print(f"  最终代价: {parallel_cost:.2f}")
    
    # 性能对比
    speedup = serial_time / parallel_time
    time_saved = serial_time - parallel_time
    
    print(f"\n{'='*70}")
    print("性能对比")
    print(f"{'='*70}")
    print(f"\n  串行时间: {serial_time:.2f} 秒")
    print(f"  并行时间: {parallel_time:.2f} 秒")
    print(f"  加速比: {speedup:.2f}x")
    print(f"  时间节省: {time_saved:.2f} 秒 ({(time_saved/serial_time)*100:.1f}%)")
    
    if speedup > 1.5:
        print(f"\n  ✓ 并行处理效果显著！")
    elif speedup > 1.1:
        print(f"\n  ✓ 并行处理有效")
    else:
        print(f"\n  ⚠ 提示：增加样本数可能获得更好的加速效果")
    
    # 可视化对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 串行结果
    ax1 = axes[0]
    plot_environment_and_trajectory(
        env, serial_trajectory, start, goal,
        title=f'串行MPPI\n时间: {serial_time:.2f}s, 代价: {serial_cost:.2f}',
        ax=ax1
    )
    
    # 并行结果
    ax2 = axes[1]
    plot_environment_and_trajectory(
        env, parallel_trajectory, start, goal,
        title=f'并行MPPI (加速{speedup:.2f}x)\n时间: {parallel_time:.2f}s, 代价: {parallel_cost:.2f}',
        ax=ax2
    )
    
    plt.tight_layout()
    plt.savefig('outputs/parallel_example.png', dpi=150, bbox_inches='tight')
    print(f"\n结果已保存至: outputs/parallel_example.png")
    
    plt.show()


if __name__ == '__main__':
    main()
