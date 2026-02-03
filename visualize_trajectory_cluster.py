#!/usr/bin/env python3
"""
可视化 L2 轨迹聚类数据

从 npz 文件中加载轨迹数据并生成可视化图表
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_npz_data(npz_path):
    """加载 npz 文件数据"""
    data = np.load(npz_path)
    
    print("=" * 60)
    print("数据信息")
    print("=" * 60)
    print(f"轨迹形状: {data['trajectories'].shape}")
    print(f"速度形状: {data['velocities'].shape}")
    print(f"加速度形状: {data['accelerations'].shape}")
    print(f"完整状态形状: {data['full_states'].shape}")
    print(f"样本数: {data['num_samples']}")
    print(f"时间范围: {data['time_horizon']}")
    print(f"起始状态: {data['start_state']}")
    print(f"目标状态: {data['goal_state']}")
    print("=" * 60)
    
    return {
        'trajectories': data['trajectories'],
        'velocities': data['velocities'],
        'accelerations': data['accelerations'],
        'full_states': data['full_states'],
        'num_samples': int(data['num_samples']),
        'time_horizon': float(data['time_horizon']),
        'start_state': data['start_state'],
        'goal_state': data['goal_state'],
    }


def plot_trajectories_2d(data, save_path=None):
    """绘制 2D 轨迹图"""
    trajectories = data['trajectories']
    start_state = data['start_state']
    goal_state = data['goal_state']
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 绘制所有轨迹
    num_samples = trajectories.shape[0]
    for i in range(num_samples):
        traj = trajectories[i]  # [T, 2]
        ax.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.3, linewidth=1)
    
    # 绘制起始点
    start_pos = start_state[:2]
    ax.plot(start_pos[0], start_pos[1], 'go', markersize=15, 
            label=f'Start ({start_pos[0]:.2f}, {start_pos[1]:.2f})', zorder=5)
    
    # 绘制目标点
    goal_pos = goal_state[:2]
    ax.plot(goal_pos[0], goal_pos[1], 'r*', markersize=20, 
            label=f'Goal ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})', zorder=5)
    
    # 绘制起始速度方向
    if len(start_state) >= 4:
        start_vel = start_state[2:4]
        if np.linalg.norm(start_vel) > 0.01:
            ax.arrow(start_pos[0], start_pos[1], 
                    start_vel[0] * 0.1, start_vel[1] * 0.1,
                    head_width=0.02, head_length=0.02, fc='green', ec='green',
                    label='Start Velocity')
    
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title(f'L2 Trajectory Cluster ({num_samples} trajectories)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_trajectory_statistics(data, save_path=None):
    """绘制轨迹统计信息"""
    trajectories = data['trajectories']
    velocities = data['velocities']
    accelerations = data['accelerations']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 路径长度分布
    ax = axes[0, 0]
    path_lengths = []
    for i in range(trajectories.shape[0]):
        traj = trajectories[i]
        length = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
        path_lengths.append(length)
    
    ax.hist(path_lengths, bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Path Length', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Path Length Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 2. 速度大小分布
    ax = axes[0, 1]
    vel_magnitudes = np.linalg.norm(velocities, axis=-1).flatten()
    ax.hist(vel_magnitudes, bins=30, edgecolor='black', alpha=0.7, color='orange')
    ax.set_xlabel('Velocity Magnitude', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Velocity Magnitude Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 3. 加速度大小分布
    ax = axes[1, 0]
    acc_magnitudes = np.linalg.norm(accelerations, axis=-1).flatten()
    ax.hist(acc_magnitudes, bins=30, edgecolor='black', alpha=0.7, color='red')
    ax.set_xlabel('Acceleration Magnitude', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Acceleration Magnitude Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 4. 目标到达误差
    ax = axes[1, 1]
    goal_state = data['goal_state']
    goal_pos = goal_state[:2]
    goal_errors = []
    for i in range(trajectories.shape[0]):
        final_pos = trajectories[i, -1, :]
        error = np.linalg.norm(final_pos - goal_pos)
        goal_errors.append(error)
    
    ax.hist(goal_errors, bins=20, edgecolor='black', alpha=0.7, color='purple')
    ax.set_xlabel('Goal Reaching Error', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Goal Reaching Error Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Trajectory Statistics', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_velocity_and_acceleration_fields(data, save_path=None):
    """绘制速度和加速度场"""
    trajectories = data['trajectories']
    velocities = data['velocities']
    accelerations = data['accelerations']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 选择一条代表性轨迹
    traj_idx = trajectories.shape[0] // 2
    traj = trajectories[traj_idx]
    vel = velocities[traj_idx]
    acc = accelerations[traj_idx]
    
    # 速度场
    ax = axes[0]
    ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, label='Trajectory', zorder=3)
    
    # 绘制速度向量
    step = max(1, len(traj) // 20)  # 每 20 个点绘制一个向量
    for i in range(0, len(traj), step):
        ax.arrow(traj[i, 0], traj[i, 1],
                vel[i, 0] * 0.05, vel[i, 1] * 0.05,
                head_width=0.02, head_length=0.02, fc='green', ec='green',
                alpha=0.6, zorder=2)
    
    ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=10, label='Start', zorder=4)
    ax.plot(traj[-1, 0], traj[-1, 1], 'r*', markersize=15, label='End', zorder=4)
    ax.set_xlabel('X Position', fontsize=11)
    ax.set_ylabel('Y Position', fontsize=11)
    ax.set_title('Velocity Field', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # 加速度场
    ax = axes[1]
    ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, label='Trajectory', zorder=3)
    
    # 绘制加速度向量
    for i in range(0, len(traj), step):
        ax.arrow(traj[i, 0], traj[i, 1],
                acc[i, 0] * 0.1, acc[i, 1] * 0.1,
                head_width=0.02, head_length=0.02, fc='red', ec='red',
                alpha=0.6, zorder=2)
    
    ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=10, label='Start', zorder=4)
    ax.plot(traj[-1, 0], traj[-1, 1], 'r*', markersize=15, label='End', zorder=4)
    ax.set_xlabel('X Position', fontsize=11)
    ax.set_ylabel('Y Position', fontsize=11)
    ax.set_title('Acceleration Field', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.suptitle(f'Velocity and Acceleration Fields (Trajectory #{traj_idx})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_time_evolution(data, save_path=None):
    """绘制轨迹随时间的变化"""
    trajectories = data['trajectories']
    velocities = data['velocities']
    accelerations = data['accelerations']
    time_horizon = data['time_horizon']
    
    T = trajectories.shape[1]
    time_steps = np.linspace(0, time_horizon, T)
    
    # 选择几条代表性轨迹
    num_show = min(5, trajectories.shape[0])
    indices = np.linspace(0, trajectories.shape[0] - 1, num_show, dtype=int)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. X 位置随时间变化
    ax = axes[0, 0]
    for idx in indices:
        ax.plot(time_steps, trajectories[idx, :, 0], alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('X Position', fontsize=11)
    ax.set_title('X Position vs Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 2. Y 位置随时间变化
    ax = axes[0, 1]
    for idx in indices:
        ax.plot(time_steps, trajectories[idx, :, 1], alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Y Position', fontsize=11)
    ax.set_title('Y Position vs Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 3. 速度大小随时间变化
    ax = axes[1, 0]
    for idx in indices:
        vel_mag = np.linalg.norm(velocities[idx], axis=-1)
        ax.plot(time_steps, vel_mag, alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Velocity Magnitude', fontsize=11)
    ax.set_title('Velocity Magnitude vs Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 4. 加速度大小随时间变化
    ax = axes[1, 1]
    for idx in indices:
        acc_mag = np.linalg.norm(accelerations[idx], axis=-1)
        ax.plot(time_steps, acc_mag, alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Acceleration Magnitude', fontsize=11)
    ax.set_title('Acceleration Magnitude vs Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Trajectory Time Evolution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='可视化 L2 轨迹聚类数据')
    parser.add_argument('--input', type=str, 
                       default='validation_results/l2_trajectory_cluster.npz',
                       help='输入 npz 文件路径')
    parser.add_argument('--output_dir', type=str, default='validation_results',
                       help='输出目录')
    parser.add_argument('--all', action='store_true',
                       help='生成所有可视化图表')
    parser.add_argument('--trajectories', action='store_true',
                       help='生成轨迹 2D 图')
    parser.add_argument('--statistics', action='store_true',
                       help='生成统计信息图')
    parser.add_argument('--fields', action='store_true',
                       help='生成速度和加速度场图')
    parser.add_argument('--evolution', action='store_true',
                       help='生成时间演化图')
    
    args = parser.parse_args()
    
    # 如果没有指定任何选项，默认生成所有图表
    if not any([args.all, args.trajectories, args.statistics, args.fields, args.evolution]):
        args.all = True
    
    # 加载数据
    print(f"正在加载数据: {args.input}")
    data = load_npz_data(args.input)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成可视化
    if args.all or args.trajectories:
        print("\n生成轨迹 2D 图...")
        plot_trajectories_2d(data, save_path=output_dir / 'l2_trajectories_2d.png')
    
    if args.all or args.statistics:
        print("\n生成统计信息图...")
        plot_trajectory_statistics(data, save_path=output_dir / 'l2_trajectory_statistics.png')
    
    if args.all or args.fields:
        print("\n生成速度和加速度场图...")
        plot_velocity_and_acceleration_fields(data, 
                                            save_path=output_dir / 'l2_velocity_acceleration_fields.png')
    
    if args.all or args.evolution:
        print("\n生成时间演化图...")
        plot_time_evolution(data, save_path=output_dir / 'l2_time_evolution.png')
    
    print("\n" + "=" * 60)
    print("可视化完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()

