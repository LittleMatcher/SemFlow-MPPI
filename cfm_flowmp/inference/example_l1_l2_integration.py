"""
L1-L2 集成示例

展示如何使用 L1 反应控制层接收 L2 (CFM-FlowMP) 的输出。

工作流程：
1. L2 层（TrajectoryGenerator）生成 K 个锚点轨迹
2. L1 层（L1ReactiveController）接收这些锚点，进行 MPPI 优化
3. L1 返回最优控制序列，可用于闭环回流
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import numpy as np
from typing import Dict, Optional

from cfm_flowmp.inference.generator import TrajectoryGenerator, GeneratorConfig
from cfm_flowmp.inference.l1_reactive_control import L1ReactiveController, L1Config


def example_l1_l2_integration():
    """
    示例：L1-L2 集成
    
    这个示例展示了：
    1. L2 生成多个锚点轨迹
    2. L1 接收 L2 输出并进行 MPPI 优化
    3. 闭环回流：L1 的最优控制用于下一帧的 L2 初始化
    """
    
    # ============ 假设：已有训练好的模型 ============
    # model = ...  # 你的 FlowMP 模型
    # 这里我们假设模型已经加载
    
    # ============ 步骤 1: 配置 L2 生成器 ============
    generator_config = GeneratorConfig(
        state_dim=2,  # 2D 平面
        seq_len=64,   # 轨迹长度
        num_samples=5,  # 生成 5 个锚点轨迹（K=5）
        use_bspline_smoothing=True,
    )
    
    # generator = TrajectoryGenerator(model, generator_config)
    
    # ============ 步骤 2: 配置 L1 控制器 ============
    l1_config = L1Config(
        n_samples_per_mode=100,  # 每个模式的采样数
        n_timesteps=64,  # 与 L2 输出一致
        tube_radius=0.5,  # 管道半径
        tube_covariance=0.1,  # 管道协方差
        w_semantic=1.0,  # 语义场权重
        w_tube=10.0,  # 管道约束权重
        w_energy=0.1,  # 能量项权重
        temperature=1.0,  # 逆温度 λ
        use_warm_start=True,  # 启用热启动
    )
    
    # 可选：定义语义场函数（障碍物场、SDF 等）
    def semantic_cost_fn(positions: torch.Tensor) -> torch.Tensor:
        """
        语义场代价函数
        
        Args:
            positions: [B, T, D] 位置序列
            
        Returns:
            costs: [B] 语义场代价
        """
        # 示例：简单的障碍物场
        # 这里可以替换为实际的 SDF 或其他语义场
        return torch.zeros(positions.shape[0], device=positions.device)
    
    l1_controller = L1ReactiveController(
        config=l1_config,
        semantic_fn=semantic_cost_fn,
    )
    
    # ============ 步骤 3: 在线规划循环 ============
    # 假设的起始和目标位置
    start_pos = torch.tensor([[0.0, 0.0]])  # [B=1, D=2]
    goal_pos = torch.tensor([[5.0, 5.0]])   # [B=1, D=2]
    start_vel = torch.tensor([[0.0, 0.0]])  # [B=1, D=2] (可选)
    
    n_frames = 10  # 规划 10 帧
    
    for frame in range(n_frames):
        print(f"\n============ 帧 {frame} ============")
        
        # ============ L2 层生成 ============
        # 获取热启动状态（如果有上一帧）
        warm_start_state = l1_controller.get_warm_start_state()
        
        # L2 生成 K 个锚点轨迹
        # l2_output = generator.generate(
        #     start_pos=start_pos,
        #     goal_pos=goal_pos,
        #     start_vel=start_vel,
        #     num_samples=5,  # 生成 5 个锚点
        # )
        
        # 模拟 L2 输出（用于演示）
        K, T, D = 5, 64, 2
        l2_output = {
            'positions': torch.randn(K, T, D),  # [K, T, D]
            'velocities': torch.randn(K, T, D),  # [K, T, D]
            'accelerations': torch.randn(K, T, D),  # [K, T, D]
        }
        
        print(f"L2 输出: {l2_output['positions'].shape} 个锚点轨迹")
        
        # ============ L1 层优化 ============
        # 初始化 L1 控制器（从 L2 输出）
        l1_controller.initialize_from_l2_output(l2_output)
        
        # 执行 MPPI 优化
        l1_result = l1_controller.optimize(
            n_iterations=10,
            verbose=True,
        )
        
        print(f"L1 最优控制形状: {l1_result['optimal_control'].shape}")
        print(f"L1 最佳代价: {l1_result['best_cost']:.4f}")
        print(f"L1 最佳模式: {l1_result['best_mode']}")
        
        # ============ 获取最优控制（用于闭环回流） ============
        # 这个 u*_k 将在下一帧 k+1 成为 L2 的 z_init
        optimal_control = l1_controller.get_next_control(
            l2_output,
            n_iterations=10,
        )
        
        print(f"最优控制序列: {optimal_control.shape}")
        
        # ============ 更新状态（模拟执行） ============
        # 在实际应用中，这里会执行第一个控制动作
        # 然后更新 start_pos, start_vel 等状态
        # start_pos = start_pos + optimal_control[0] * dt  # 简化示例
    
    print("\n============ 规划完成 ============")


def example_l2_output_format():
    """
    示例：L2 输出格式说明
    
    展示 generator.generate() 的输出格式，以及 L1 如何接收它。
    """
    
    # L2 输出格式（来自 TrajectoryGenerator.generate()）
    # 当 num_samples > 1 时：
    K = 5  # 锚点数量
    T = 64  # 时间步数
    D = 2   # 状态维度
    
    l2_output = {
        'positions': torch.randn(K, T, D),      # [K, T, D] 或 [B*N, T, D]
        'velocities': torch.randn(K, T, D),     # [K, T, D]
        'accelerations': torch.randn(K, T, D),  # [K, T, D]
    }
    
    print("L2 输出格式:")
    print(f"  positions: {l2_output['positions'].shape}")
    print(f"  velocities: {l2_output['velocities'].shape}")
    print(f"  accelerations: {l2_output['accelerations'].shape}")
    
    # L1 接收 L2 输出
    l1_config = L1Config()
    l1_controller = L1ReactiveController(config=l1_config)
    
    # 初始化 L1（从 L2 输出）
    l1_controller.initialize_from_l2_output(l2_output)
    
    print(f"\nL1 初始化完成，创建了 {len(l1_controller.optimizers)} 个并行 MPPI 优化器")
    
    # 执行优化
    result = l1_controller.optimize(n_iterations=5)
    
    print(f"\nL1 优化结果:")
    print(f"  最优控制: {result['optimal_control'].shape}")
    print(f"  最佳代价: {result['best_cost']:.4f}")
    print(f"  最佳模式: {result['best_mode']}")


if __name__ == "__main__":
    print("============ L1-L2 集成示例 ============")
    
    # 运行示例
    example_l2_output_format()
    print("\n" + "="*50 + "\n")
    # example_l1_l2_integration()  # 需要实际的模型

