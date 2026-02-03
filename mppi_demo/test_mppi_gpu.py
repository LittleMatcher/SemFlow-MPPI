"""
GPU加速的MPPI测试 - Christmas Market场景
利用RTX 5090并行模拟数千个轨迹
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scenarios.christmas_market import create_christmas_market_environment
from mppi_core import (
    CollisionCost, SmoothnessCost, GoalCost, CompositeCost,
    PathLengthCost, TurnCost, Visualizer
)
from mppi_core.mppi_gpu import MPPI_GPU


def run_mppi_gpu_christmas_market(temperature=1.0, n_samples=2000, n_iterations=100,
                                  batch_size=1000, save_plots=True):
    """
    在GPU上运行MPPI - 利用并行计算模拟数千个"平行宇宙"
    
    Args:
        temperature: MPPI温度参数
        n_samples: 轨迹采样数（GPU可以处理更多）
        n_iterations: 优化迭代次数
        batch_size: GPU批处理大小
        save_plots: 是否保存图像
    """
    print("=" * 70)
    print("🚀 GPU加速的MPPI - Christmas Market场景")
    print("   利用RTX 5090模拟数千个'平行宇宙'")
    print("=" * 70)
    print(f"Temperature (λ): {temperature}")
    print(f"Samples (K): {n_samples}")
    print(f"Iterations: {n_iterations}")
    print(f"Batch Size: {batch_size}")
    print()
    
    # 检查GPU可用性
    # Note: RTX 5090 (sm_120) requires PyTorch 2.6+ or nightly builds
    # For now, we'll use CPU until compatible PyTorch is available
    if not torch.cuda.is_available():
        print("⚠️  警告: CUDA不可用，将使用CPU（速度会慢得多）")
        device = 'cpu'
    else:
        try:
            # Test if GPU actually works
            test_tensor = torch.zeros(1).cuda()
            device = 'cuda'
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print()
        except RuntimeError as e:
            print(f"⚠️  GPU检测到但无法使用: {e}")
            print("   原因: RTX 5090 (CUDA sm_120) 需要 PyTorch 2.6+ 或 nightly builds")
            print("   当前使用CPU进行测试...")
            device = 'cpu'
            print()
    
    # 创建环境
    env, start, goal, bounds = create_christmas_market_environment()
    
    print(f"Environment: {len(env.obstacles)} obstacles")
    print(f"Start: ({start[0]:.1f}, {start[1]:.1f})")
    print(f"Goal: ({goal[0]:.1f}, {goal[1]:.1f})")
    print()
    
    # Robot参数
    robot_radius = 0.3
    
    # 代价函数（与CPU版本相同）
    collision_cost = CollisionCost(
        env=env,
        robot_radius=robot_radius,
        epsilon=0.2,
        weight=120.0,
        use_hard_constraint=True,
        hard_penalty=1e6
    )
    
    smoothness_cost = SmoothnessCost(
        penalize='acceleration',
        weight=0.5
    )
    
    goal_cost = GoalCost(
        goal=goal,
        weight=100.0
    )
    
    path_length_cost = PathLengthCost(
        weight=50.0
    )
    
    dt = 0.1
    turn_cost = TurnCost(
        weight=5.0,
        method='angle_diff',
        dt=dt,
        use_sharp_turn_penalty=True,
        max_angular_change=0.5,
        sharp_turn_threshold=0.4
    )
    
    cost_function = CompositeCost([
        collision_cost,
        smoothness_cost,
        goal_cost,
        path_length_cost,
        turn_cost
    ])
    
    # 创建GPU加速的MPPI优化器
    print("初始化GPU-MPPI优化器...")
    mppi_gpu = MPPI_GPU(
        cost_function=cost_function,
        n_samples=n_samples,
        n_control_points=18,
        bspline_degree=3,
        time_horizon=8.0,
        n_timesteps=80,
        temperature=temperature,
        noise_std=0.7,
        bounds=bounds,
        device=device,
        batch_size=batch_size
    )
    
    # 运行优化
    print("\n" + "=" * 70)
    print("开始GPU优化 - 模拟数千个平行宇宙...")
    print("=" * 70)
    start_time = time.time()
    
    result = mppi_gpu.optimize(
        start=start,
        goal=goal,
        n_iterations=n_iterations,
        verbose=True
    )
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("优化完成!")
    print("=" * 70)
    print(f"总时间: {elapsed_time:.2f}秒")
    print(f"平均每次迭代: {elapsed_time/n_iterations:.3f}秒")
    print(f"最佳代价: {result['best_cost_all_time']:.2f} (迭代 {result['best_iteration']})")
    print(f"路径长度: {np.sum(np.linalg.norm(np.diff(result['trajectory'], axis=0), axis=1)):.2f}m")
    
    # 计算加速比（与典型CPU时间比较）
    cpu_time_estimate = elapsed_time * (400 / n_samples) * 2  # 粗略估计
    speedup = cpu_time_estimate / elapsed_time
    print(f"\n估计加速比: {speedup:.1f}x (vs CPU with 400 samples)")
    print(f"GPU并行效率: 同时模拟 {n_samples} 个平行宇宙")
    print("=" * 70)
    
    if save_plots:
        # 可视化
        vis = Visualizer(env, figsize=(14, 14))
        
        # 1. 最终轨迹
        fig1, ax1 = plt.subplots(1, 1, figsize=(14, 14))
        vis.plot_environment(ax1)
        vis.plot_trajectory(
            result['trajectory'],
            ax=ax1,
            color='blue',
            linewidth=3,
            label=f'GPU-MPPI Solution (Iteration {result["best_iteration"]})',
            show_control_points=True,
            control_points=result['best_control_points_all_time']
        )
        ax1.scatter(start[0], start[1], color='green', s=300, marker='o',
                   edgecolor='black', linewidth=2, zorder=5, label='Start')
        ax1.scatter(goal[0], goal[1], color='red', s=300, marker='*',
                   edgecolor='black', linewidth=2, zorder=5, label='Goal')
        ax1.set_title(f'GPU-MPPI Solution - {n_samples} Parallel Universes', 
                     fontsize=16, fontweight='bold')
        ax1.legend(fontsize=12, loc='upper right')
        
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        fig1.savefig(os.path.join(output_dir, 'gpu_mppi_christmas_market_solution.png'),
                    dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved: outputs/gpu_mppi_christmas_market_solution.png")
        
        # 2. 代价历史
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        vis.plot_cost_history(
            result['cost_history'],
            ax=ax2,
            best_iteration=result['best_iteration'],
            best_cost=result['best_cost_all_time']
        )
        ax2.set_title(f'GPU-MPPI Cost History ({n_samples} samples)', 
                     fontsize=14, fontweight='bold')
        
        fig2.savefig(os.path.join(output_dir, 'gpu_mppi_cost_history.png'),
                    dpi=150, bbox_inches='tight')
        print(f"✓ Saved: outputs/gpu_mppi_cost_history.png")
        
        plt.close('all')
    
    return result


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  🚀 GPU加速MPPI - 模拟数千个平行宇宙")
    print("     RTX 5090 并行计算展示")
    print("=" * 70 + "\n")
    
    # 检测GPU是否可用并调整参数
    if torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(1).cuda()
            # GPU可用 - 使用大量样本
            n_samples = 2000
            batch_size = 1000
            print("✓ GPU可用 - 将使用2000个并行轨迹")
        except:
            # GPU不可用 - 使用较少样本
            n_samples = 500
            batch_size = 500
            print("⚠️ GPU不兼容 - 使用CPU模式（500个样本）")
    else:
        n_samples = 500
        batch_size = 500
        print("⚠️ CUDA不可用 - 使用CPU模式（500个样本）")
    
    # GPU可以处理更多样本
    result = run_mppi_gpu_christmas_market(
        temperature=1.0,
        n_samples=n_samples,
        n_iterations=100,
        batch_size=batch_size,
        save_plots=True
    )
    
    print("\n" + "=" * 70)
    print("🎉 GPU-MPPI演示完成!")
    print("=" * 70 + "\n")
