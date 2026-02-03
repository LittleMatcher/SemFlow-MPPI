"""
L1 反应控制层 (The Reactive Legs)

在 L2 提供的先验附近进行局部优化，确保硬约束。

数学目标：
在 L2 提供的先验附近进行局部优化，确保硬约束满足。

核心功能：
1. 拓扑并行采样 (Topology-Concurrent Sampling)
   - L2 输出 K 个锚点 {ū^1, ..., ū^K}
   - L1 实例化 K 个并行 MPPI 优化器
   - 对于第 m 个模式，控制序列采样分布为：u_i^m ~ N(ū^m, Σ_tube)

2. 能量-安全双重代价 (Dual Objective)
   - 总代价函数：J(u) = w1 * J_sem(x) + w2 * J_tube(x, ū^m) + w3 * u^T R u
   - 管道约束：J_tube = 0 if ||x - x_anchor|| < r_tube else ∞

3. 最优控制更新 (Update Law)
   - MPPI 经典公式：u*_k = Σ_{i,m} exp(-J(u_i^m) / λ) * u_i^m / Σ_{i,m} exp(-J(u_i^m) / λ)
   - 闭环回流：u*_k 将在下一帧 k+1 成为 L2 的 z_init

接口说明：
- 接收 L2 输出（来自 TrajectoryGenerator.generate()）
- L2 输出格式：{'positions': [K, T, D], 'velocities': [K, T, D], ...}
- 返回最优控制序列 u*_k [T, D]
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass

# Note: L1 layer doesn't directly use models, but may need them for type hints
# from ..models import FlowMPTransformer  # Commented out if not available


@dataclass
class L1Config:
    """L1 反应控制层配置"""
    
    # 采样参数
    n_samples_per_mode: int = 100  # 每个模式的采样数
    n_control_points: int = 10  # 控制点数量
    time_horizon: float = 5.0  # 时间范围
    n_timesteps: int = 50  # 时间步数
    
    # 管道约束参数
    tube_radius: float = 0.5  # 管道半径 r_tube
    tube_covariance: float = 0.1  # 管道协方差 Σ_tube 的标准差
    
    # 代价函数权重
    w_semantic: float = 1.0  # w1: L3 语义场权重
    w_tube: float = 10.0  # w2: 管道约束权重
    w_energy: float = 0.1  # w3: 能量项权重
    
    # MPPI 参数
    temperature: float = 1.0  # 逆温度 λ
    energy_matrix_scale: float = 1.0  # 能量矩阵 R 的缩放
    
    # 热启动参数（On-Policy 特性）
    use_warm_start: bool = True  # 是否使用热启动
    warm_start_noise_scale: float = 0.1  # 热启动时的噪声缩放
    shift_padding_mode: str = "zero"  # 移位填充模式: "zero" 或 "extrapolate"
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SemanticFieldCost:
    """
    L3 语义场代价函数
    
    这是一个接口，实际实现应该由用户提供语义场函数。
    语义场可以是：
    - 障碍物场（SDF）
    - 语义分割场
    - 其他环境感知场
    """
    
    def __init__(self, semantic_fn: Optional[Callable] = None):
        """
        Args:
            semantic_fn: 语义场函数，接受位置 [B, T, D] 返回代价 [B]
        """
        self.semantic_fn = semantic_fn
    
    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        """
        计算语义场代价
        
        Args:
            positions: 位置序列 [B, T, D]
            
        Returns:
            costs: 语义场代价 [B]
        """
        if self.semantic_fn is not None:
            return self.semantic_fn(positions)
        else:
            # 默认：零代价（如果没有提供语义场函数）
            return torch.zeros(positions.shape[0], device=positions.device)


class TubeConstraintCost:
    """
    管道约束代价函数
    
    确保轨迹保持在锚点附近的"管道"内：
    J_tube = 0 if ||x - x_anchor|| < r_tube else ∞
    """
    
    def __init__(self, tube_radius: float = 0.5):
        """
        Args:
            tube_radius: 管道半径 r_tube
        """
        self.tube_radius = tube_radius
    
    def __call__(
        self, 
        positions: torch.Tensor, 
        anchor_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        计算管道约束代价
        
        Args:
            positions: 当前位置序列 [B, T, D]
            anchor_positions: 锚点位置序列 [M, T, D] 或 [T, D]
            
        Returns:
            costs: 管道约束代价 [B]
        """
        B, T, D = positions.shape
        
        # 如果 anchor_positions 是 [T, D]，扩展到 [1, T, D]
        if anchor_positions.dim() == 2:
            anchor_positions = anchor_positions.unsqueeze(0)
        M = anchor_positions.shape[0]
        
        # 计算每个样本到每个锚点的距离
        # positions: [B, T, D], anchor_positions: [M, T, D]
        # 需要广播到 [B, M, T, D]
        positions_expanded = positions.unsqueeze(1)  # [B, 1, T, D]
        anchor_expanded = anchor_positions.unsqueeze(0)  # [1, M, T, D]
        
        # 计算每个时间步的距离
        distances = torch.norm(
            positions_expanded - anchor_expanded, 
            dim=-1
        )  # [B, M, T]
        
        # 找到每个样本到最近锚点的最小距离（在所有时间步）
        min_distances = distances.min(dim=1)[0]  # [B, T]
        max_distance_per_sample = min_distances.max(dim=1)[0]  # [B]
        
        # 硬约束：如果任何时间步超出管道，代价为无穷
        # 使用大的惩罚值代替真正的无穷
        costs = torch.where(
            max_distance_per_sample < self.tube_radius,
            torch.zeros_like(max_distance_per_sample),
            torch.full_like(max_distance_per_sample, 1e6)
        )
        
        return costs


class EnergyCost:
    """
    能量项代价：u^T R u
    
    惩罚控制输入的能量，鼓励平滑控制。
    """
    
    def __init__(self, energy_matrix_scale: float = 1.0):
        """
        Args:
            energy_matrix_scale: 能量矩阵 R 的缩放因子
        """
        self.energy_matrix_scale = energy_matrix_scale
    
    def __call__(self, control_sequence: torch.Tensor) -> torch.Tensor:
        """
        计算能量代价
        
        Args:
            control_sequence: 控制序列 [B, T, D] 或 [B, n_control_points, D]
            
        Returns:
            costs: 能量代价 [B]
        """
        # 如果控制序列是控制点，先转换为轨迹
        # 这里假设 control_sequence 已经是轨迹形式 [B, T, D]
        B, T, D = control_sequence.shape
        
        # 计算控制输入的能量（使用 L2 范数的平方）
        # 对每个时间步的控制输入计算能量
        control_energy = torch.sum(control_sequence ** 2, dim=-1)  # [B, T]
        total_energy = torch.sum(control_energy, dim=1)  # [B]
        
        return self.energy_matrix_scale * total_energy


class DualObjectiveCost:
    """
    能量-安全双重代价函数 (Dual Objective)
    
    数学公式：
    J(u) = w1 * J_sem(x) + w2 * J_tube(x, ū^m) + w3 * u^T R u
    
    其中：
    - J_sem(x): L3 语义场代价（障碍物、语义分割等）
    - J_tube(x, ū^m): 管道约束代价，确保轨迹保持在锚点附近的"管道"内
    - u^T R u: 能量项，惩罚控制输入的能量，鼓励平滑控制
    - w1, w2, w3: 权重参数
    
    管道约束 J_tube:
    J_tube = 0  if ||x - x_anchor|| < r_tube
    J_tube = ∞  otherwise
    
    这确保了优化只在锚点附近的局部区域内进行。
    """
    
    def __init__(
        self,
        semantic_cost: SemanticFieldCost,
        tube_cost: TubeConstraintCost,
        energy_cost: EnergyCost,
        w_semantic: float = 1.0,
        w_tube: float = 10.0,
        w_energy: float = 0.1,
    ):
        """
        Args:
            semantic_cost: 语义场代价函数
            tube_cost: 管道约束代价函数
            energy_cost: 能量代价函数
            w_semantic: 语义场权重
            w_tube: 管道约束权重
            w_energy: 能量项权重
        """
        self.semantic_cost = semantic_cost
        self.tube_cost = tube_cost
        self.energy_cost = energy_cost
        self.w_semantic = w_semantic
        self.w_tube = w_tube
        self.w_energy = w_energy
    
    def __call__(
        self,
        positions: torch.Tensor,
        control_sequence: torch.Tensor,
        anchor_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算总代价
        
        Args:
            positions: 位置序列 [B, T, D]
            control_sequence: 控制序列 [B, T, D] 或 [B, n_control_points, D]
            anchor_positions: 锚点位置序列 [M, T, D] 或 [T, D]
            
        Returns:
            costs: 总代价 [B]
        """
        # 语义场代价
        J_sem = self.semantic_cost(positions)
        
        # 管道约束代价
        J_tube = self.tube_cost(positions, anchor_positions)
        
        # 能量代价
        J_energy = self.energy_cost(control_sequence)
        
        # 总代价：J(u) = w1 * J_sem(x) + w2 * J_tube(x, ū^m) + w3 * u^T R u
        total_cost = (
            self.w_semantic * J_sem +      # L3 语义场代价
            self.w_tube * J_tube +          # 管道约束代价
            self.w_energy * J_energy        # 能量项
        )
        
        return total_cost


class ParallelMPPIOptimizer:
    """
    单个模式的 MPPI 优化器
    
    在锚点附近进行局部优化。
    """
    
    def __init__(
        self,
        anchor_control: torch.Tensor,  # [T, D] 或 [n_control_points, D]
        cost_function: DualObjectiveCost,
        config: L1Config,
    ):
        """
        Args:
            anchor_control: 锚点控制序列
            cost_function: 双重代价函数
            config: L1 配置
        """
        self.anchor_control = anchor_control
        self.cost_function = cost_function
        self.config = config
        self.device = torch.device(config.device)
        
        # 当前控制序列（初始化为锚点）
        self.current_control = anchor_control.clone().to(self.device)
        
        # 管道协方差矩阵
        self.tube_covariance = (
            config.tube_covariance ** 2 * 
            torch.eye(anchor_control.shape[-1], device=self.device)
        )
    
    def sample_control_sequences(self) -> torch.Tensor:
        """
        在锚点附近采样控制序列（拓扑并行采样）
        
        数学原理：
        对于第 m 个模式，控制序列采样分布为：
        u_i^m ~ N(ū^m, Σ_tube)
        
        其中：
        - ū^m: 第 m 个锚点控制序列 [T, D]
        - Σ_tube: 管道协方差矩阵，比标准 MPPI 小，限制在"管道"内探索
        - i: 样本索引 (i = 1, ..., n_samples_per_mode)
        
        实现细节：
        - 使用对角协方差矩阵简化计算：Σ_tube = σ_tube^2 * I
        - σ_tube 由 config.tube_covariance 指定
        
        Returns:
            sampled_controls: 采样的控制序列 [n_samples, T, D]
        """
        n_samples = self.config.n_samples_per_mode
        T, D = self.anchor_control.shape
        
        # 将锚点扩展到批次
        anchor_expanded = self.anchor_control.unsqueeze(0).repeat(
            n_samples, 1, 1
        )  # [n_samples, T, D]
        
        # 采样标准高斯噪声
        noise = torch.randn(n_samples, T, D, device=self.device)
        
        # 应用管道协方差（对角协方差：Σ_tube = σ_tube^2 * I）
        # 这限制了采样在锚点附近的"管道"内
        noise = noise * self.config.tube_covariance
        
        # 采样控制序列：u_i^m = ū^m + ε_i^m, 其中 ε_i^m ~ N(0, Σ_tube)
        sampled_controls = anchor_expanded + noise
        
        return sampled_controls
    
    def control_to_trajectory(
        self, 
        control_sequence: torch.Tensor
    ) -> torch.Tensor:
        """
        将控制序列转换为轨迹位置
        
        这里简化处理：假设控制序列直接对应位置序列。
        实际应用中可能需要通过动力学模型积分。
        
        对于更复杂的系统，可以使用：
        - 双积分器模型：x_{t+1} = x_t + v_t * dt, v_{t+1} = v_t + u_t * dt
        - 或其他动力学模型
        
        Args:
            control_sequence: 控制序列 [B, T, D] 或 [B, n_control_points, D]
            
        Returns:
            positions: 位置序列 [B, T, D]
        """
        # 如果控制序列是控制点，需要插值到轨迹
        if control_sequence.shape[1] != self.config.n_timesteps:
            # 控制点 -> 轨迹（使用线性插值）
            B, n_cp, D = control_sequence.shape
            T = self.config.n_timesteps
            
            # 创建插值索引
            t_indices = torch.linspace(0, n_cp - 1, T, device=control_sequence.device)
            t_floor = t_indices.floor().long()
            t_ceil = (t_floor + 1).clamp(max=n_cp - 1)
            alpha = (t_indices - t_floor).unsqueeze(-1)
            
            # 线性插值
            positions = (
                control_sequence[:, t_floor, :] * (1 - alpha) +
                control_sequence[:, t_ceil, :] * alpha
            )
        else:
            # 直接使用控制序列作为位置（简化假设）
            positions = control_sequence
        
        return positions
    
    def evaluate_costs(
        self,
        sampled_controls: torch.Tensor,
        anchor_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        评估采样控制序列的代价
        
        Args:
            sampled_controls: 采样的控制序列 [n_samples, T, D]
            anchor_positions: 锚点位置序列 [T, D]
            
        Returns:
            costs: 代价 [n_samples]
        """
        # 将控制序列转换为轨迹
        positions = self.control_to_trajectory(sampled_controls)
        
        # 计算代价
        costs = self.cost_function(
            positions=positions,
            control_sequence=sampled_controls,
            anchor_positions=anchor_positions,
        )
        
        return costs
    
    def compute_weights(self, costs: torch.Tensor) -> torch.Tensor:
        """
        计算重要性权重
        
        w_i = exp(-cost_i / λ) / Σ exp(-cost_j / λ)
        
        Args:
            costs: 代价 [n_samples]
            
        Returns:
            weights: 权重 [n_samples]
        """
        # 归一化代价（数值稳定性）
        costs_normalized = costs - costs.min()
        
        # 计算权重
        weights = torch.exp(-costs_normalized / (self.config.temperature + 1e-8))
        
        # 归一化
        weights = weights / (weights.sum() + 1e-8)
        
        return weights
    
    def update_control(
        self,
        sampled_controls: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        更新控制序列（加权平均）
        
        Args:
            sampled_controls: 采样的控制序列 [n_samples, T, D]
            weights: 权重 [n_samples]
            
        Returns:
            updated_control: 更新的控制序列 [T, D]
        """
        # 加权平均
        weights_expanded = weights.view(-1, 1, 1)  # [n_samples, 1, 1]
        updated_control = (weights_expanded * sampled_controls).sum(dim=0)
        
        return updated_control
    
    def step(self, anchor_positions: torch.Tensor) -> Dict:
        """
        执行一次 MPPI 迭代
        
        Args:
            anchor_positions: 锚点位置序列 [T, D]
            
        Returns:
            info: 迭代信息字典
        """
        # 采样控制序列
        sampled_controls = self.sample_control_sequences()
        
        # 评估代价
        costs = self.evaluate_costs(sampled_controls, anchor_positions)
        
        # 计算权重
        weights = self.compute_weights(costs)
        
        # 更新控制序列
        self.current_control = self.update_control(
            sampled_controls, weights
        )
        
        # 找到最佳样本
        best_idx = costs.argmin()
        best_cost = costs[best_idx].item()
        
        info = {
            'best_cost': best_cost,
            'mean_cost': costs.mean().item(),
            'best_control': sampled_controls[best_idx].clone(),
            'current_control': self.current_control.clone(),
        }
        
        return info


class L1ReactiveController:
    """
    L1 反应控制层
    
    接收 L2 输出的 K 个锚点，实例化 K 个并行 MPPI 优化器，
    进行局部优化并返回最优控制。
    
    支持 On-Policy 特性：
    - 在线热启动（Warm-Start）：使用上一帧的最优控制作为先验
    - 策略延续：确保相邻时间步的策略连续性
    """
    
    def __init__(
        self,
        config: L1Config,
        semantic_fn: Optional[Callable] = None,
    ):
        """
        Args:
            config: L1 配置
            semantic_fn: 语义场函数（可选）
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # 创建代价函数组件
        semantic_cost = SemanticFieldCost(semantic_fn)
        tube_cost = TubeConstraintCost(config.tube_radius)
        energy_cost = EnergyCost(config.energy_matrix_scale)
        
        self.cost_function = DualObjectiveCost(
            semantic_cost=semantic_cost,
            tube_cost=tube_cost,
            energy_cost=energy_cost,
            w_semantic=config.w_semantic,
            w_tube=config.w_tube,
            w_energy=config.w_energy,
        )
        
        # 并行 MPPI 优化器列表（延迟初始化）
        self.optimizers: List[ParallelMPPIOptimizer] = []
        
        # 热启动状态管理（On-Policy 特性）
        self.previous_optimal_control: Optional[torch.Tensor] = None  # 上一帧的最优控制
        self.previous_optimal_state: Optional[torch.Tensor] = None  # 上一帧的最优状态（用于 CFM）
    
    def initialize_from_l2_output(
        self,
        l2_output: Dict[str, torch.Tensor],
    ):
        """
        从 L2 输出初始化 K 个并行 MPPI 优化器
        
        L2 输出格式（来自 generator.generate()）：
        - 'positions': [B*N, T, D] 或 [B, T, D] - K 个锚点轨迹
          当 num_samples > 1 时，格式为 [B*num_samples, T, D]，即 K = B*num_samples
          当 num_samples = 1 时，格式为 [B, T, D]，即 K = B
        - 'velocities': [K, T, D] (可选)
        - 'accelerations': [K, T, D] (可选)
        
        数学原理：
        L2 输出 K 个锚点 {ū^1, ..., ū^K}，L1 为每个锚点实例化一个 MPPI 优化器。
        对于第 m 个模式，采样分布为：u_i^m ~ N(ū^m, Σ_tube)
        
        Args:
            l2_output: L2 层输出字典，支持两种格式：
                - 来自 TrajectoryGenerator.generate(): {'positions': ...}
                - 来自 L2SafetyCFM.generate_trajectory_anchors(): {'trajectories': ...}
        """
        # 提取锚点位置 - 支持两种键名
        if 'trajectories' in l2_output:
            # 来自 generate_trajectory_anchors()
            anchor_positions = l2_output['trajectories']  # [B*N, T, D]
        elif 'positions' in l2_output:
            # 来自 TrajectoryGenerator.generate()
            anchor_positions = l2_output['positions']  # [K, T, D] 或 [B, K, T, D] 或 [B*N, T, D]
        else:
            raise KeyError(
                f"l2_output 必须包含 'trajectories' 或 'positions' 键。"
                f"当前键: {list(l2_output.keys())}"
            )
        
        # 处理不同的输入格式
        if anchor_positions.dim() == 4:
            # [B, K, T, D] -> 取第一个批次，得到 [K, T, D]
            anchor_positions = anchor_positions[0]
        elif anchor_positions.dim() == 3:
            # [B*N, T, D] 或 [B, T, D] -> 直接使用，第一维就是 K
            # 这是 generator.generate() 的标准输出格式
            pass
        else:
            raise ValueError(
                f"不支持的 anchor_positions 维度: {anchor_positions.shape}. "
                f"期望 [K, T, D] 或 [B, K, T, D] 或 [B*N, T, D]"
            )
        
        # 确保在正确的设备上
        anchor_positions = anchor_positions.to(self.device)
        
        K, T, D = anchor_positions.shape
        
        # 更新配置中的时间步数（如果与 L2 输出不一致）
        if T != self.config.n_timesteps:
            # 如果时间步数不匹配，可以选择插值或更新配置
            # 这里我们更新配置以匹配 L2 输出
            self.config.n_timesteps = T
        
        # 清空现有优化器
        self.optimizers = []
        
        # 为每个锚点创建 MPPI 优化器（拓扑并行采样）
        # 每个优化器在对应的锚点附近进行局部优化
        for k in range(K):
            anchor_control = anchor_positions[k]  # [T, D] - 第 k 个锚点
            
            optimizer = ParallelMPPIOptimizer(
                anchor_control=anchor_control,
                cost_function=self.cost_function,
                config=self.config,
            )
            
            self.optimizers.append(optimizer)
    
    def optimize(
        self,
        n_iterations: int = 10,
        verbose: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        执行并行 MPPI 优化
        
        Args:
            n_iterations: 每个优化器的迭代次数
            verbose: 是否打印进度
            
        Returns:
            result: 包含最优控制的字典
        """
        if len(self.optimizers) == 0:
            raise ValueError("优化器未初始化。请先调用 initialize_from_l2_output()")
        
        K = len(self.optimizers)
        
        # 存储所有模式的采样和代价
        all_sampled_controls = []
        all_costs = []
        all_anchor_positions = []
        
        # 对每个模式进行优化
        for m, optimizer in enumerate(self.optimizers):
            # 获取锚点位置
            anchor_positions = optimizer.anchor_control  # [T, D]
            
            # 执行多次迭代
            for iteration in range(n_iterations):
                info = optimizer.step(anchor_positions)
                
                if verbose and iteration % 5 == 0:
                    print(
                        f"模式 {m}, 迭代 {iteration}: "
                        f"最佳代价 = {info['best_cost']:.4f}"
                    )
            
            # 收集最终采样和代价（用于全局最优控制计算）
            sampled_controls = optimizer.sample_control_sequences()
            costs = optimizer.evaluate_costs(sampled_controls, anchor_positions)
            
            all_sampled_controls.append(sampled_controls)
            all_costs.append(costs)
            all_anchor_positions.append(anchor_positions)
        
        # ============ 全局最优控制更新 (Update Law) ============
        # 数学公式（MPPI 经典公式）：
        # u*_k = Σ_{i,m} exp(-J(u_i^m) / λ) * u_i^m / Σ_{i,m} exp(-J(u_i^m) / λ)
        # 其中：
        #   - i: 样本索引 (i = 1, ..., n_samples_per_mode)
        #   - m: 模式索引 (m = 1, ..., K)
        #   - u_i^m: 第 m 个模式下的第 i 个采样控制序列
        #   - J(u_i^m): 第 m 个模式下的第 i 个样本的代价
        #   - λ: 逆温度参数 (temperature)
        
        # 合并所有模式的采样和代价
        all_controls = torch.cat(all_sampled_controls, dim=0)  # [K*n_samples, T, D]
        all_costs_tensor = torch.cat(all_costs, dim=0)  # [K*n_samples]
        
        # 计算全局权重（重要性采样权重）
        # 数值稳定性：先归一化代价（减去最小值）
        costs_normalized = all_costs_tensor - all_costs_tensor.min()
        
        # 计算权重：w_i^m = exp(-J(u_i^m) / λ)
        global_weights = torch.exp(
            -costs_normalized / (self.config.temperature + 1e-8)
        )
        
        # 归一化权重：w_i^m = w_i^m / Σ_{j,n} w_j^n
        global_weights = global_weights / (global_weights.sum() + 1e-8)
        
        # 计算全局最优控制（加权平均）
        # u*_k = Σ_{i,m} w_i^m * u_i^m
        weights_expanded = global_weights.view(-1, 1, 1)  # [K*n_samples, 1, 1]
        optimal_control = (weights_expanded * all_controls).sum(dim=0)  # [T, D]
        
        # 找到最佳模式
        best_mode_idx = all_costs_tensor.argmin() // self.config.n_samples_per_mode
        best_control = all_controls[all_costs_tensor.argmin()]
        
        result = {
            'optimal_control': optimal_control,  # [T, D] - 全局加权最优
            'best_control': best_control,  # [T, D] - 最佳单个样本
            'best_mode': best_mode_idx,
            'best_cost': all_costs_tensor.min().item(),
            'mean_cost': all_costs_tensor.mean().item(),
            'all_controls': all_controls,  # [K*n_samples, T, D]
            'all_costs': all_costs_tensor,  # [K*n_samples]
        }
        
        return result
    
    def shift_control_sequence(
        self,
        control_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """
        移位操作：将控制序列向前移动一步
        
        在 t 时刻规划出的最优轨迹 τ*_t，在 t+1 时刻应该成为强有力的先验。
        移位操作：丢弃第一个动作，末尾补零或补预测。
        
        Args:
            control_sequence: 控制序列 [T, D] 或 [B, T, D]
            
        Returns:
            shifted_control: 移位后的控制序列 [T, D] 或 [B, T, D]
        """
        if control_sequence.dim() == 2:
            # [T, D]
            T, D = control_sequence.shape
            shifted = torch.zeros_like(control_sequence)
            
            # 向前移动：u_{t+1} = u_t (丢弃第一个，其余前移)
            shifted[:-1] = control_sequence[1:]
            
            # 末尾填充
            if self.config.shift_padding_mode == "zero":
                # 补零
                shifted[-1] = 0.0
            elif self.config.shift_padding_mode == "extrapolate":
                # 外推：使用最后两个点的差值
                if T >= 2:
                    shifted[-1] = control_sequence[-1] + (
                        control_sequence[-1] - control_sequence[-2]
                    )
                else:
                    shifted[-1] = control_sequence[-1]
            else:
                shifted[-1] = control_sequence[-1]  # 保持最后一个值
            
            return shifted
        else:
            # [B, T, D]
            B, T, D = control_sequence.shape
            shifted = torch.zeros_like(control_sequence)
            shifted[:, :-1] = control_sequence[:, 1:]
            
            if self.config.shift_padding_mode == "zero":
                shifted[:, -1] = 0.0
            elif self.config.shift_padding_mode == "extrapolate":
                if T >= 2:
                    shifted[:, -1] = control_sequence[:, -1] + (
                        control_sequence[:, -1] - control_sequence[:, -2]
                    )
                else:
                    shifted[:, -1] = control_sequence[:, -1]
            else:
                shifted[:, -1] = control_sequence[:, -1]
            
            return shifted
    
    def prepare_warm_start_state(
        self,
        shifted_control: torch.Tensor,
    ) -> torch.Tensor:
        """
        准备热启动状态（用于 CFM 反向注入）
        
        将移位后的控制序列加噪后作为 CFM 的初始状态 z_T
        （Reverse Process 的起点），而不是从纯高斯噪声 N(0, I) 开始。
        
        Args:
            shifted_control: 移位后的控制序列 [T, D]
            
        Returns:
            warm_start_state: 热启动状态 [T, D*3] (pos + vel + acc)
        """
        T, D = shifted_control.shape
        
        # 将控制序列转换为完整状态（pos + vel + acc）
        # 简化：假设控制序列是位置，计算速度和加速度
        positions = shifted_control  # [T, D]
        
        # 计算速度（有限差分）
        dt = self.config.time_horizon / self.config.n_timesteps
        velocities = torch.diff(positions, dim=0) / dt
        # 填充最后一个速度
        velocities = torch.cat([velocities, velocities[-1:]], dim=0)
        
        # 计算加速度
        accelerations = torch.diff(velocities, dim=0) / dt
        accelerations = torch.cat([accelerations, accelerations[-1:]], dim=0)
        
        # 拼接完整状态
        full_state = torch.cat([positions, velocities, accelerations], dim=-1)  # [T, D*3]
        
        # 添加噪声（用于 CFM 反向注入）
        # 噪声缩放：较小的噪声保持先验信息，同时允许探索
        noise = torch.randn_like(full_state) * self.config.warm_start_noise_scale
        
        warm_start_state = full_state + noise
        
        return warm_start_state
    
    def get_next_control(
        self,
        l2_output: Dict[str, torch.Tensor],
        n_iterations: int = 10,
    ) -> torch.Tensor:
        """
        获取下一帧的最优控制（用于闭环回流）
        
        这个 u*_k 将在下一帧 k+1 成为 L2 的 z_init。
        
        Args:
            l2_output: L2 层输出
            n_iterations: 优化迭代次数
            
        Returns:
            next_control: 下一帧的最优控制 [T, D]
        """
        # 初始化优化器
        self.initialize_from_l2_output(l2_output)
        
        # 执行优化
        result = self.optimize(n_iterations=n_iterations)
        
        # 获取最优控制
        optimal_control = result['optimal_control']  # [T, D]
        
        # 保存当前最优控制（用于下一帧的热启动）
        if self.config.use_warm_start:
            self.previous_optimal_control = optimal_control.clone()
            
            # 准备热启动状态（用于 CFM 反向注入）
            shifted_control = self.shift_control_sequence(optimal_control)
            warm_start_state = self.prepare_warm_start_state(shifted_control)
            self.previous_optimal_state = warm_start_state
        
        # 返回最优控制（用于下一帧的初始化）
        return optimal_control
    
    def get_warm_start_state(
        self,
    ) -> Optional[torch.Tensor]:
        """
        获取热启动状态（用于 CFM 反向注入）
        
        返回上一帧移位并加噪后的状态，作为 CFM 的初始状态 z_T。
        如果没有上一帧的状态，返回 None（使用纯噪声）。
        
        Returns:
            warm_start_state: 热启动状态 [T, D*3] 或 None
        """
        if not self.config.use_warm_start:
            return None
        
        if self.previous_optimal_state is None:
            return None
        
        return self.previous_optimal_state.clone()

