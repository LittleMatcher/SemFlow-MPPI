"""
最简单场景测试：无障碍空间中的最优路径

验证目标：
1. MPPI 能否在没有障碍物的情况下找到接近直线的最优路径
2. 路径长度是否接近理论最短距离（欧氏距离）
3. 算法收敛性和稳定性
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mppi_core.environment_2d import Environment2D
from mppi_core.cost_functions import CollisionCost, SmoothnessCost, GoalCost, CompositeCost, PathLengthCost
from mppi_core.mppi import MPPI_BSpline
from mppi_core.visualization import Visualizer


def create_free_space_scenario():
    """创建完全无障碍的简单场景"""
    # Environment bounds
    bounds = (-5, 5, -5, 5)
    env = Environment2D(bounds)
    
    # 没有障碍物！
    
    # Start and goal positions
    start = np.array([-3.0, -3.0])
    goal = np.array([3.0, 3.0])
    
    return env, start, goal, bounds


def compute_theoretical_optimal(start, goal):
    """计算理论最优路径（直线距离）"""
    optimal_length = np.linalg.norm(goal - start)
    
    # 生成理论最优路径（直线）
    n_points = 100
    t = np.linspace(0, 1, n_points).reshape(-1, 1)
    optimal_trajectory = (1 - t) * start + t * goal
    
    return optimal_length, optimal_trajectory


def compute_path_length(trajectory):
    """计算轨迹的实际长度"""
    diffs = np.diff(trajectory, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    return np.sum(segment_lengths)


def run_free_space_test(temperature=1.0, n_samples=100, n_iterations=80,
                        show_plots=True, save_plots=True):
    """
    运行无障碍空间测试
    
    Args:
        temperature: MPPI 温度参数
        n_samples: 轨迹采样数
        n_iterations: 迭代次数
        show_plots: 是否显示图表
        save_plots: 是否保存图表
    """
    print("=" * 70)
    print("MPPI 无障碍空间测试 - 验证最优路径查找能力")
    print("=" * 70)
    print(f"温度 (λ): {temperature}")
    print(f"采样数 (K): {n_samples}")
    print(f"迭代次数: {n_iterations}")
    print()
    
    # 创建环境
    env, start, goal, bounds = create_free_space_scenario()
    
    # 计算理论最优
    optimal_length, optimal_trajectory = compute_theoretical_optimal(start, goal)
    print(f"起始点: ({start[0]:.1f}, {start[1]:.1f})")
    print(f"目标点: ({goal[0]:.1f}, {goal[1]:.1f})")
    print(f"理论最优路径长度: {optimal_length:.4f}")
    print()
    
    # Robot parameters
    robot_radius = 0.2
    
    # 创建代价函数
    # 无障碍空间，碰撞代价权重可以较低
    collision_cost = CollisionCost(
        env=env,
        robot_radius=robot_radius,
        epsilon=0.1,
        weight=50.0,  # 较低权重（因为没有障碍物）
        use_hard_constraint=True,
        hard_penalty=1e6
    )
    
    smoothness_cost = SmoothnessCost(
        penalize='acceleration',
        weight=0.5
    )
    
    goal_cost = GoalCost(
        goal=goal,
        weight=50.0
    )
    
    # 路径长度代价应该占主导（因为没有障碍物）
    path_length_cost = PathLengthCost(
        weight=20.0  # 较高权重，鼓励最短路径
    )
    
    cost_function = CompositeCost([
        collision_cost,
        smoothness_cost,
        goal_cost,
        path_length_cost
    ])
    
    # 创建 MPPI 优化器
    mppi = MPPI_BSpline(
        cost_function=cost_function,
        n_samples=n_samples,
        n_control_points=10,
        bspline_degree=3,
        time_horizon=5.0,
        n_timesteps=50,
        temperature=temperature,
        noise_std=0.3,
        bounds=bounds
    )
    
    # 运行优化
    print("运行 MPPI 优化...")
    result = mppi.optimize(
        start=start,
        goal=goal,
        n_iterations=n_iterations,
        verbose=True,
        return_best_all_time=True
    )
    
    print()
    print("=" * 70)
    print("优化结果分析")
    print("=" * 70)
    
    # 计算实际路径长度
    actual_length = compute_path_length(result['trajectory'])
    length_ratio = actual_length / optimal_length
    length_error = (actual_length - optimal_length) / optimal_length * 100
    
    print(f"理论最优路径长度: {optimal_length:.4f}")
    print(f"MPPI 找到的路径长度: {actual_length:.4f}")
    print(f"长度比率: {length_ratio:.4f} (1.0 表示完美)")
    print(f"长度误差: {length_error:.2f}%")
    print()
    print(f"最终迭代代价: {result['final_cost']:.2f}")
    print(f"所有迭代中的最佳代价: {result['best_cost_all_time']:.2f} (迭代 {result['best_iteration']})")
    print()
    
    # 评估结果
    if length_ratio < 1.01:
        print("✓✓✓ 优秀！路径长度非常接近理论最优 (< 1% 误差)")
    elif length_ratio < 1.05:
        print("✓✓ 良好！路径长度接近理论最优 (< 5% 误差)")
    elif length_ratio < 1.10:
        print("✓ 可接受！路径长度较为接近理论最优 (< 10% 误差)")
    else:
        print("⚠ 警告：路径长度与理论最优差距较大 (> 10% 误差)")
    
    # 检查是否接近直线
    # 计算路径点到直线的平均距离
    line_vector = goal - start
    line_length = np.linalg.norm(line_vector)
    line_unit = line_vector / line_length
    
    distances_to_line = []
    for point in result['trajectory']:
        vec_to_point = point - start
        projection_length = np.dot(vec_to_point, line_unit)
        projection_point = start + projection_length * line_unit
        distance = np.linalg.norm(point - projection_point)
        distances_to_line.append(distance)
    
    avg_deviation = np.mean(distances_to_line)
    max_deviation = np.max(distances_to_line)
    
    print(f"路径到直线的平均偏差: {avg_deviation:.4f}")
    print(f"路径到直线的最大偏差: {max_deviation:.4f}")
    print()
    
    if avg_deviation < 0.05:
        print("✓✓✓ 优秀！路径非常接近直线 (< 0.05 平均偏差)")
    elif avg_deviation < 0.1:
        print("✓✓ 良好！路径接近直线 (< 0.1 平均偏差)")
    elif avg_deviation < 0.2:
        print("✓ 可接受！路径较为接近直线 (< 0.2 平均偏差)")
    else:
        print("⚠ 警告：路径偏离直线较远 (> 0.2 平均偏差)")
    print()
    
    # 可视化
    vis = Visualizer(env, figsize=(14, 10))
    
    # 1. 对比图：理论最优 vs MPPI 结果
    fig1, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 理论最优路径
    vis.plot_environment(axes[0])
    axes[0].plot(optimal_trajectory[:, 0], optimal_trajectory[:, 1],
                color='green', linewidth=3, linestyle='--',
                label=f'Optimal Path (Length: {optimal_length:.4f})', alpha=0.8)
    axes[0].scatter(start[0], start[1], color='green', s=200, marker='o',
                   edgecolor='black', linewidth=2, zorder=5, label='Start')
    axes[0].scatter(goal[0], goal[1], color='red', s=200, marker='*',
                   edgecolor='black', linewidth=2, zorder=5, label='Goal')
    axes[0].set_title('Optimal Path (Straight Line)', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # MPPI 找到的路径
    vis.plot_environment(axes[1])
    vis.plot_trajectory(
        result['trajectory'],
        ax=axes[1],
        color='blue',
        linewidth=3,
        label=f'MPPI Path (Length: {actual_length:.4f})',
        show_control_points=True,
        control_points=result['best_control_points_all_time']
    )
    axes[1].plot(optimal_trajectory[:, 0], optimal_trajectory[:, 1],
                color='green', linewidth=2, linestyle='--',
                label=f'Optimal (Length: {optimal_length:.4f})', alpha=0.6)
    axes[1].set_title(f'MPPI Optimization Result (Error: {length_error:.2f}%)', 
                     fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        fig1.savefig(os.path.join(output_dir, 'free_space_comparison.png'),
                    dpi=150, bbox_inches='tight')
        print(f"Saved: outputs/free_space_comparison.png")
    
    # 2. 代价历史和收敛分析
    fig2, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # 代价历史
    vis.plot_cost_history(
        result['cost_history'],
        ax=axes[0],
        best_iteration=result['best_iteration'],
        best_cost=result['best_cost_all_time']
    )
    
    # 路径长度收敛
    path_lengths = []
    for info in result['info_history']:
        traj = info['best_trajectory']
        length = compute_path_length(traj)
        path_lengths.append(length)
    
    axes[1].plot(path_lengths, linewidth=2, color='blue', label='Path Length per Iteration')
    axes[1].axhline(y=optimal_length, color='green', linestyle='--', 
                   linewidth=2, label=f'Optimal ({optimal_length:.4f})')
    axes[1].scatter([result['best_iteration']], [compute_path_length(result['best_trajectory_all_time'])],
                   color='red', s=200, marker='*', zorder=5,
                   label=f'Global Best (Iteration {result["best_iteration"]})',
                   edgecolor='black', linewidth=2)
    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('Path Length', fontsize=12)
    axes[1].set_title('Path Length Convergence Analysis', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        fig2.savefig(os.path.join(output_dir, 'free_space_convergence.png'),
                    dpi=150, bbox_inches='tight')
        print(f"Saved: outputs/free_space_convergence.png")
    
    # 3. 偏差分析
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
    
    # 计算每个点的偏差
    point_indices = np.arange(len(distances_to_line))
    
    ax3.plot(point_indices, distances_to_line, linewidth=2, color='blue',
            label='Distance to Line')
    ax3.axhline(y=avg_deviation, color='red', linestyle='--', linewidth=2,
               label=f'Average Deviation: {avg_deviation:.4f}')
    ax3.fill_between(point_indices, 0, distances_to_line, alpha=0.3, color='blue')
    ax3.set_xlabel('Trajectory Point Index', fontsize=12)
    ax3.set_ylabel('Distance to Line', fontsize=12)
    ax3.set_title('Deviation from Optimal Line Analysis', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        fig3.savefig(os.path.join(output_dir, 'free_space_deviation.png'),
                    dpi=150, bbox_inches='tight')
        print(f"Saved: outputs/free_space_deviation.png")
    
    if show_plots:
        plt.show()
    
    print("=" * 70)
    print("测试完成！")
    print("=" * 70)
    
    return {
        'result': result,
        'optimal_length': optimal_length,
        'actual_length': actual_length,
        'length_ratio': length_ratio,
        'length_error': length_error,
        'avg_deviation': avg_deviation,
        'max_deviation': max_deviation
    }


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MPPI 无障碍空间测试")
    print("验证算法是否能找到最优路径（直线）")
    print("=" * 70 + "\n")
    
    # 运行测试
    test_result = run_free_space_test(
        temperature=1.0,
        n_samples=100,
        n_iterations=80,
        show_plots=True,
        save_plots=True
    )
    
    print("\n总结：")
    print(f"  路径长度误差: {test_result['length_error']:.2f}%")
    print(f"  平均偏差: {test_result['avg_deviation']:.4f}")
    print(f"  最大偏差: {test_result['max_deviation']:.4f}")
    print()
    print("如果误差 < 5% 且平均偏差 < 0.1，说明 MPPI 能很好地找到最优路径！")

