"""
比较CPU MPPI vs GPU MPPI性能
展示GPU加速的潜力（当PyTorch支持RTX 5090后）
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scenarios.christmas_market import create_christmas_market_environment
from mppi_core import (
    CollisionCost, SmoothnessCost, GoalCost, CompositeCost,
    PathLengthCost, TurnCost, MPPI, Visualizer
)
from mppi_core.mppi_gpu import MPPI_GPU


def run_comparison():
    """比较CPU和GPU版本的MPPI"""
    
    print("=" * 80)
    print("  🔥 MPPI性能比较: CPU vs GPU")
    print("  评估GPU并行化的效果")
    print("=" * 80)
    print()
    
    # 创建环境
    env, start, goal, bounds = create_christmas_market_environment()
    robot_radius = 0.3
    
    # 相同的代价函数
    collision_cost = CollisionCost(
        env=env,
        robot_radius=robot_radius,
        epsilon=0.2,
        weight=120.0,
        use_hard_constraint=True,
        hard_penalty=1e6
    )
    
    smoothness_cost = SmoothnessCost(penalize='acceleration', weight=0.5)
    goal_cost = GoalCost(goal=goal, weight=100.0)
    path_length_cost = PathLengthCost(weight=50.0)
    turn_cost = TurnCost(weight=5.0, method='angle_diff', dt=0.1,
                        use_sharp_turn_penalty=True, max_angular_change=0.5)
    
    cost_function = CompositeCost([
        collision_cost, smoothness_cost, goal_cost, path_length_cost, turn_cost
    ])
    
    results = []
    
    # ============ 测试1: CPU基准 (400样本) ============
    print("\n" + "="*80)
    print("📊 测试1: CPU MPPI (400 samples)")
    print("="*80)
    
    mppi_cpu = MPPI(
        cost_function=cost_function,
        n_samples=400,
        n_control_points=18,
        bspline_degree=3,
        time_horizon=8.0,
        n_timesteps=80,
        temperature=1.0,
        noise_std=0.7,
        bounds=bounds
    )
    
    start_time = time.time()
    result_cpu = mppi_cpu.optimize(start=start, goal=goal, n_iterations=50, verbose=False)
    elapsed_cpu = time.time() - start_time
    
    path_length_cpu = np.sum(np.linalg.norm(np.diff(result_cpu['trajectory'], axis=0), axis=1))
    
    print(f"✓ 完成!")
    print(f"  时间: {elapsed_cpu:.2f}秒")
    print(f"  平均每次迭代: {elapsed_cpu/50:.3f}秒")
    print(f"  最佳代价: {result_cpu['best_cost_all_time']:.2f}")
    print(f"  路径长度: {path_length_cpu:.2f}m")
    
    results.append({
        'name': 'CPU (400 samples)',
        'time': elapsed_cpu,
        'time_per_iter': elapsed_cpu/50,
        'cost': result_cpu['best_cost_all_time'],
        'path_length': path_length_cpu,
        'samples': 400,
        'trajectory': result_cpu['trajectory']
    })
    
    # ============ 测试2: GPU加速 (2000样本) ============
    print("\n" + "="*80)
    print("📊 测试2: GPU MPPI (2000 samples)")
    print("="*80)
    
    mppi_gpu = MPPI_GPU(
        cost_function=cost_function,
        n_samples=2000,
        n_control_points=18,
        bspline_degree=3,
        time_horizon=8.0,
        n_timesteps=80,
        temperature=1.0,
        noise_std=0.7,
        bounds=bounds,
        device='cuda',  # 会自动回退到CPU如果GPU不可用
        batch_size=1000
    )
    
    start_time = time.time()
    result_gpu = mppi_gpu.optimize(start=start, goal=goal, n_iterations=50, verbose=False)
    elapsed_gpu = time.time() - start_time
    
    path_length_gpu = np.sum(np.linalg.norm(np.diff(result_gpu['trajectory'], axis=0), axis=1))
    
    print(f"✓ 完成!")
    print(f"  时间: {elapsed_gpu:.2f}秒")
    print(f"  平均每次迭代: {elapsed_gpu/50:.3f}秒")
    print(f"  最佳代价: {result_gpu['best_cost_all_time']:.2f}")
    print(f"  路径长度: {path_length_gpu:.2f}m")
    
    results.append({
        'name': 'GPU (2000 samples)',
        'time': elapsed_gpu,
        'time_per_iter': elapsed_gpu/50,
        'cost': result_gpu['best_cost_all_time'],
        'path_length': path_length_gpu,
        'samples': 2000,
        'trajectory': result_gpu['trajectory']
    })
    
    # ============ 性能对比 ============
    print("\n" + "="*80)
    print("📈 性能对比总结")
    print("="*80)
    
    print("\n┌─────────────────────┬────────────┬────────────┬─────────┐")
    print("│ 方法                │ 时间(秒)   │ 迭代(秒)   │ 样本数  │")
    print("├─────────────────────┼────────────┼────────────┼─────────┤")
    for r in results:
        print(f"│ {r['name']:19s} │ {r['time']:10.2f} │ {r['time_per_iter']:10.3f} │ {r['samples']:7d} │")
    print("└─────────────────────┴────────────┴────────────┴─────────┘")
    
    print("\n┌─────────────────────┬────────────┬──────────────┐")
    print("│ 方法                │ 代价       │ 路径长度(m)  │")
    print("├─────────────────────┼────────────┼──────────────┤")
    for r in results:
        print(f"│ {r['name']:19s} │ {r['cost']:10.2f} │ {r['path_length']:12.2f} │")
    print("└─────────────────────┴────────────┴──────────────┘")
    
    # 计算加速比
    speedup = elapsed_cpu / elapsed_gpu
    sample_ratio = 2000 / 400
    
    print("\n" + "="*80)
    print("🚀 关键指标")
    print("="*80)
    print(f"⏱️  加速比: {speedup:.2f}x")
    print(f"📊 样本增加: {sample_ratio:.1f}x (400 → 2000)")
    print(f"📉 路径改善: {((path_length_cpu - path_length_gpu) / path_length_cpu * 100):.1f}%")
    print(f"💰 代价改善: {((results[0]['cost'] - results[1]['cost']) / results[0]['cost'] * 100):.1f}%")
    
    if speedup < 1.0:
        print("\n⚠️  注意: GPU当前未启用 (RTX 5090需要PyTorch 2.6+)")
        print("   预期GPU加速比: 5-10x (当PyTorch支持sm_120后)")
    else:
        print("\n✅ GPU加速成功! 享受并行计算的威力!")
    
    print("="*80)
    
    # ============ 可视化 ============
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 轨迹对比
    vis = Visualizer(env, figsize=(16, 8))
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    for idx, (ax, r) in enumerate(zip(axes, results)):
        vis.plot_environment(ax)
        vis.plot_trajectory(
            r['trajectory'],
            ax=ax,
            color='blue' if idx == 0 else 'red',
            linewidth=3,
            label=f"{r['name']}\nTime: {r['time']:.2f}s\nPath: {r['path_length']:.2f}m",
            show_control_points=False
        )
        ax.scatter(start[0], start[1], color='green', s=300, marker='o',
                  edgecolor='black', linewidth=2, zorder=5)
        ax.scatter(goal[0], goal[1], color='red', s=300, marker='*',
                  edgecolor='black', linewidth=2, zorder=5)
        ax.set_title(r['name'], fontsize=18, fontweight='bold')
        ax.legend(fontsize=12, loc='upper right')
    
    fig.suptitle('CPU vs GPU MPPI 性能对比', fontsize=20, fontweight='bold', y=0.98)
    fig.savefig(os.path.join(output_dir, 'cpu_vs_gpu_comparison.png'),
                dpi=150, bbox_inches='tight')
    print(f"\n✓ 可视化保存: outputs/cpu_vs_gpu_comparison.png")
    
    # 2. 性能图表
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    
    # 时间对比
    ax1 = axes2[0, 0]
    names = [r['name'] for r in results]
    times = [r['time'] for r in results]
    colors = ['steelblue', 'crimson']
    bars1 = ax1.bar(names, times, color=colors, alpha=0.7)
    ax1.set_ylabel('总时间 (秒)', fontsize=12)
    ax1.set_title('执行时间对比', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for bar, time_val in zip(bars1, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.2f}s', ha='center', va='bottom', fontsize=11)
    
    # 每次迭代时间
    ax2 = axes2[0, 1]
    iter_times = [r['time_per_iter'] for r in results]
    bars2 = ax2.bar(names, iter_times, color=colors, alpha=0.7)
    ax2.set_ylabel('时间/迭代 (秒)', fontsize=12)
    ax2.set_title('每次迭代时间', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for bar, time_val in zip(bars2, iter_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.3f}s', ha='center', va='bottom', fontsize=11)
    
    # 路径长度对比
    ax3 = axes2[1, 0]
    path_lengths = [r['path_length'] for r in results]
    bars3 = ax3.bar(names, path_lengths, color=colors, alpha=0.7)
    ax3.set_ylabel('路径长度 (m)', fontsize=12)
    ax3.set_title('路径长度对比', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    for bar, length in zip(bars3, path_lengths):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{length:.2f}m', ha='center', va='bottom', fontsize=11)
    
    # 样本数量
    ax4 = axes2[1, 1]
    samples = [r['samples'] for r in results]
    bars4 = ax4.bar(names, samples, color=colors, alpha=0.7)
    ax4.set_ylabel('样本数量', fontsize=12)
    ax4.set_title('并行样本数量', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    for bar, sample_count in zip(bars4, samples):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{sample_count}', ha='center', va='bottom', fontsize=11)
    
    fig2.suptitle('性能指标对比', fontsize=16, fontweight='bold', y=0.99)
    fig2.tight_layout(rect=[0, 0, 1, 0.97])
    fig2.savefig(os.path.join(output_dir, 'performance_metrics.png'),
                 dpi=150, bbox_inches='tight')
    print(f"✓ 性能图表保存: outputs/performance_metrics.png")
    
    plt.close('all')
    
    return results


if __name__ == "__main__":
    results = run_comparison()
    
    print("\n" + "="*80)
    print("🎯 测试完成!")
    print("="*80)
    print("\n💡 提示:")
    print("   - 当前GPU可能使用CPU回退 (RTX 5090需要PyTorch 2.6+)")
    print("   - 一旦PyTorch支持，GPU将提供5-10x加速")
    print("   - 更多样本 (2000 vs 400) 通常产生更优路径")
    print("\n查看输出文件:")
    print("   - outputs/cpu_vs_gpu_comparison.png")
    print("   - outputs/performance_metrics.png")
    print("="*80 + "\n")
