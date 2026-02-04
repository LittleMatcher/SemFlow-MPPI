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

# MPPI demo backend (@mppi_demo)

from mppi_demo.mppi_core import (
    CostFunction as MPPICostFunction,
    MPPI_BSpline,
    CompositeCost,
    SmoothnessCost,
    PathLengthCost,
    GoalCost,
)

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


class AnchorTubeCost(MPPICostFunction if MPPICostFunction is not None else object):
    """
    使用 L2 输出的多模态锚点构建"软管道约束"代价：
    
    对于每条 MPC 轨迹 τ，计算其到任一锚点轨迹的最近距离：
        d_min(t) = min_k ||x_t - x_t^k||
    管道损失:
        J_tube = Σ_t max(0, d_min(t) - r)^2
    """
    
    def __init__(self, anchors: np.ndarray, tube_radius: float = 0.5, weight: float = 1.0):
        """
        Args:
            anchors: [K, T, D] L2 锚点轨迹 (numpy)
        """
        assert anchors.ndim == 3, f"anchors 期望 [K, T, D]，得到 {anchors.shape}"
        self.anchors = anchors.astype(np.float32)
        self.tube_radius = float(tube_radius)
        self.weight = float(weight)
    
    def __call__(
        self,
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        accelerations: Optional[np.ndarray] = None,
        jerks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # positions: [B, T, D], anchors: [K, T, D]
        B, T, D = positions.shape
        K = self.anchors.shape[0]
        
        # [B, 1, T, D] - [1, K, T, D] -> [B, K, T]
        diff = positions[:, None, :, :] - self.anchors[None, :, :, :]
        dist = np.linalg.norm(diff, axis=-1)  # [B, K, T]
        
        # 每个时间步对所有锚点取最小距离 -> [B, T]
        min_dist = dist.min(axis=1)
        
        # 软管道惩罚
        excess = np.maximum(0.0, min_dist - self.tube_radius)
        costs = self.weight * np.sum(excess ** 2, axis=1)  # [B]
        return costs.astype(np.float32)


class L1ReactiveController:
    """
    L1 反应控制层（重构版，基于 @mppi_demo）
    
    - 接收 L2 输出的 K 条锚点轨迹（`trajectories` 或 `positions`，形状兼容 L2）
    - 将这些锚点编码为 mppi_demo 的代价函数（语义代价 + 管道约束 + 平滑/长度等）
    - 使用 `MPPI_BSpline` 在 B-Spline 控制点上进行优化，输出一条最优轨迹
    
    外部接口保持不变：
    - `initialize_from_l2_output(l2_output: Dict[str, Tensor])`
    - `optimize(n_iterations: int, verbose: bool) -> Dict[str, Tensor]`
    - `get_next_control(l2_output, n_iterations) -> Tensor[T, D]`
    - `get_warm_start_state() -> Optional[Tensor[T, 3D]]`
    """
    
    def __init__(
        self,
        config: L1Config,
        semantic_fn: Optional[Callable] = None,
    ):
        """
        Args:
            config: L1 配置
            semantic_fn: 语义场函数（可选），签名为
                         `semantic_fn(positions: torch.Tensor[B, T, D]) -> torch.Tensor[B]`
        """
        self.config = config
        self.device = torch.device(config.device)
        
        if MPPI_BSpline is None:
            raise ImportError(
                "无法导入 `mppi_demo.mppi_core`。请确保 @mppi_demo 子模块可用，"
                "或将其添加到 Python 路径。"
            )
        
        # 保存语义代价函数（通过适配器接入 MPPI-BSpline）
        self.semantic_fn = semantic_fn
        
        # L2 锚点（torch）以及 numpy 形式
        self.anchor_positions: Optional[torch.Tensor] = None  # [K, T, D]
        self._anchors_np: Optional[np.ndarray] = None         # [K, T, D]
        
        # 当前规划问题的起点与终点（来自 L2 锚点）
        self._start_np: Optional[np.ndarray] = None  # (D,)
        self._goal_np: Optional[np.ndarray] = None   # (D,)
        
        # mppi_demo 后端
        self._mppi: Optional[MPPI_BSpline] = None
        self._mppi_cost = None
        
        # 热启动状态管理（给 CFM L2 使用）
        self.previous_optimal_control: Optional[torch.Tensor] = None  # [T, D]
        self.previous_optimal_state: Optional[torch.Tensor] = None    # [T, 3D]
    
    # ------------------------------------------------------------------
    # 内部：构建 mppi_demo 代价函数 & 初始化 MPPI 优化器
    # ------------------------------------------------------------------
    def _build_mppi_cost(self):
        """基于当前配置与 L2 锚点构建 mppi_demo 的 CompositeCost。"""
        costs = []
        
        # 1) 管道约束：鼓励靠近 L2 多模态锚点
        if self._anchors_np is not None and self._anchors_np.shape[0] > 0:
            costs.append(
                AnchorTubeCost(
                    anchors=self._anchors_np,
                    tube_radius=self.config.tube_radius,
                    weight=self.config.w_tube,
                )
            )
        
        # 2) 平滑 / 能量项（使用加速度平滑近似能量代价）
        if SmoothnessCost is not None and self.config.w_energy > 0:
            costs.append(
                SmoothnessCost(
                    penalize='acceleration',
                    weight=float(self.config.w_energy),
                )
            )
        
        # 3) 路径长度（偏好更短路径）
        if PathLengthCost is not None:
            costs.append(PathLengthCost(weight=10.0))
        
        # 4) 终点到目标的距离（使用 L2 均值目标）
        if GoalCost is not None and self._goal_np is not None:
            costs.append(GoalCost(goal=self._goal_np, weight=50.0 * float(self.config.w_semantic)))
        
        if len(costs) == 0:
            # 退化情况：始终返回 0 代价
            class ZeroCost(MPPICostFunction):  # type: ignore[misc]
                def __call__(self, positions: np.ndarray, **kwargs) -> np.ndarray:
                    return np.zeros(positions.shape[0], dtype=np.float32)
            self._mppi_cost = ZeroCost()
        elif len(costs) == 1:
            self._mppi_cost = costs[0]
        else:
            self._mppi_cost = CompositeCost(costs)
    
    def _ensure_mppi_created(self):
        """延迟创建 MPPI_BSpline 实例，以当前配置和代价为准。"""
        if self._mppi is not None:
            return
        if self._start_np is None or self._goal_np is None:
            raise RuntimeError("在创建 MPPI 之前需要先通过 L2 输出设置 start/goal。")
        if self._mppi_cost is None:
            self._build_mppi_cost()
        
        self._mppi = MPPI_BSpline(
            cost_function=self._mppi_cost,
            n_samples=int(self.config.n_samples_per_mode),
            n_control_points=int(self.config.n_control_points),
            bspline_degree=3,
            time_horizon=float(self.config.time_horizon),
            n_timesteps=int(self.config.n_timesteps),
            temperature=float(self.config.temperature),
            noise_std=float(self.config.tube_covariance),
            bounds=(0.0, 1.0, 0.0, 1.0),  # L2 轨迹在 [0,1]×[0,1] 上
            elite_ratio=0.2,
            n_jobs=1,
        )
    
    # ------------------------------------------------------------------
    # 公共 API：与旧版 L1ReactiveController 保持函数签名一致
    # ------------------------------------------------------------------
    def initialize_from_l2_output(
        self,
        l2_output: Dict[str, torch.Tensor],
    ):
        """
        从 L2 输出初始化 L1 层（接口保持与旧实现一致）。
        
        支持的 L2 输出格式：
        - `{'trajectories': [B*N, T, D], ...}`  来自 `L2SafetyCFM.generate_trajectory_anchors`
        - `{'positions': [K, T, D] 或 [B, K, T, D], ...}` 来自 `TrajectoryGenerator.generate`
        """
        # 提取锚点位置 - 支持两种键名
        if 'trajectories' in l2_output:
            anchor_positions = l2_output['trajectories']  # [B*N, T, D]
        elif 'positions' in l2_output:
            anchor_positions = l2_output['positions']
        else:
            raise KeyError(
                f"l2_output 必须包含 'trajectories' 或 'positions' 键，当前键: {list(l2_output.keys())}"
            )
        
        # 统一为 [K, T, D]
        if anchor_positions.dim() == 4:
            # [B, K, T, D] -> 取第一个 batch: [K, T, D]
            anchor_positions = anchor_positions[0]
        elif anchor_positions.dim() == 3:
            # [K, T, D] 或 [B*N, T, D]
            pass
        else:
            raise ValueError(
                f"不支持的 anchor_positions 维度: {anchor_positions.shape}。"
                f"期望 [K, T, D] 或 [B, K, T, D] 或 [B*N, T, D]"
            )
        
        anchor_positions = anchor_positions.to(self.device)
        K, T, D = anchor_positions.shape
        self.anchor_positions = anchor_positions
        self._anchors_np = anchor_positions.detach().cpu().numpy().astype(np.float32)
        
        # 若时间步与配置不一致，更新 n_timesteps 以匹配 L2
        if T != self.config.n_timesteps:
            self.config.n_timesteps = T
        
        # 根据锚点估计全局 start / goal（使用所有锚点的均值）
        start_mean = anchor_positions[:, 0, :].mean(dim=0)   # [D]
        goal_mean = anchor_positions[:, -1, :].mean(dim=0)   # [D]
        self._start_np = start_mean.detach().cpu().numpy().astype(np.float32)
        self._goal_np = goal_mean.detach().cpu().numpy().astype(np.float32)
        
        # 重建 MPPI 代价 & 优化器（因为锚点发生变化）
        self._mppi = None
        self._mppi_cost = None
        self._build_mppi_cost()
        self._ensure_mppi_created()
        
        # 使用第一条锚点轨迹拟合 B-spline 控制点，作为 MPPI 的初始控制
        first_anchor = anchor_positions[0].detach().cpu().numpy().astype(np.float32)  # [T, D]
        if self._mppi is not None:
            cp = self._mppi.bspline.fit_trajectory(first_anchor)
            self._mppi.control_points = cp
    
    def optimize(
        self,
        n_iterations: int = 10,
        verbose: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        使用 @mppi_demo 的 `MPPI_BSpline` 执行优化。
        
        Returns (与旧版尽量保持一致的键):
            - 'optimal_control': [T, D] torch.Tensor
            - 'best_control'   : [T, D] torch.Tensor
            - 'best_mode'      : int（此处退化为 0）
            - 'best_cost'      : float
            - 'mean_cost'      : float（所有迭代 mean_cost 的平均）
            - 'all_controls'   : [0, T, D] 空张量（占位，避免下游代码出错）
            - 'all_costs'      : [N] numpy->torch 张量（每次迭代的 best_cost 历史）
        """
        if self._mppi is None:
            self._ensure_mppi_created()
        
        # 如果启用了 L1 自身的热启动，并且有上一帧的最优轨迹，
        # 则用它来初始化 MPPI 的控制点（拟合 B-spline）
        if self.config.use_warm_start and self.previous_optimal_control is not None:
            prev_traj = self.previous_optimal_control.detach().cpu().numpy().astype(np.float32)
            cp = self._mppi.bspline.fit_trajectory(prev_traj)
            self._mppi.control_points = cp
        
        result_np = self._mppi.optimize(
            start=self._start_np,
            goal=self._goal_np,
            n_iterations=int(n_iterations),
            verbose=verbose,
            return_best_all_time=True,
        )
        
        # 最优轨迹（positions）-> torch
        optimal_traj_np = result_np['trajectory'].astype(np.float32)  # [T, 2]
        optimal_control = torch.from_numpy(optimal_traj_np).to(self.device)
        
        # 统计信息
        best_cost = float(result_np['best_cost_all_time'])
        info_history = result_np.get('info_history', [])
        if len(info_history) > 0 and 'mean_cost' in info_history[0]:
            mean_cost = float(np.mean([info['mean_cost'] for info in info_history]))
        else:
            mean_cost = best_cost
        
        cost_history = np.array(result_np.get('cost_history', []), dtype=np.float32)
        all_costs_t = torch.from_numpy(cost_history).to(self.device)
        
        result = {
            'optimal_control': optimal_control,          # [T, D]
            'best_control': optimal_control.clone(),     # 与 optimal 相同
            'best_mode': 0,                              # 单一 MPPI 实例，模式退化为 0
            'best_cost': best_cost,
            'mean_cost': mean_cost,
            'all_controls': torch.zeros(                 # 占位张量，防止下游 KeyError
                0,
                self.config.n_timesteps,
                self.anchor_positions.shape[-1] if self.anchor_positions is not None else 2,
                device=self.device,
            ),
            'all_costs': all_costs_t,
        }
        
        return result
    
    # ------------------------------------------------------------------
    # 热启动：保持与旧 API 一致，用于给 L2 / TrajectoryGenerator 提供 warm-start 状态
    # ------------------------------------------------------------------
    def shift_trajectory(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """
        移位操作：将轨迹向前移动一步 (Predict Step)
        
        实现 Warm Start 的"Predict"阶段：
        在 t 时刻规划出的最优轨迹 τ*_t，在 t+1 时刻应该成为强有力的先验。
        移位操作：丢弃第一个动作，末尾补零或补预测。
        
        数学表达:
            u_pred = Shift(u*_{t-1})
        
        Args:
            trajectory: 轨迹 [B, T, D] 或 [T, D]
                D 可以是位置维度 (2D) 或完整状态维度 (pos+vel+acc, 6D)
            
        Returns:
            shifted_traj: 移位后的轨迹 [B, T, D] 或 [T, D]
        """
        if trajectory.dim() == 2:
            # [T, D]
            T, D = trajectory.shape
            shifted = torch.zeros_like(trajectory)
            
            # 向前移动：u_{t+1} = u_t (丢弃第一个，其余前移)
            shifted[:-1] = trajectory[1:]
            
            # 末尾填充
            if self.config.shift_padding_mode == "zero":
                # 补零
                shifted[-1] = 0.0
            elif self.config.shift_padding_mode == "extrapolate":
                # 外推：使用最后两个点的差值（恒速度假设）
                if T >= 2:
                    shifted[-1] = trajectory[-1] + (
                        trajectory[-1] - trajectory[-2]
                    )
                else:
                    shifted[-1] = trajectory[-1]
            else:
                shifted[-1] = trajectory[-1]  # 保持最后一个值
            
            return shifted
        else:
            # [B, T, D]
            B, T, D = trajectory.shape
            shifted = torch.zeros_like(trajectory)
            shifted[:, :-1] = trajectory[:, 1:]
            
            if self.config.shift_padding_mode == "zero":
                shifted[:, -1] = 0.0
            elif self.config.shift_padding_mode == "extrapolate":
                if T >= 2:
                    shifted[:, -1] = trajectory[:, -1] + (
                        trajectory[:, -1] - trajectory[:, -2]
                    )
                else:
                    shifted[:, -1] = trajectory[:, -1]
            else:
                shifted[:, -1] = trajectory[:, -1]
            
            return shifted
    
    def shift_control_sequence(
        self,
        control_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """
        移位操作：将控制序列向前移动一步
        
        别名方法，调用 shift_trajectory 以保持向后兼容性。
        
        Args:
            control_sequence: 控制序列 [T, D] 或 [B, T, D]
            
        Returns:
            shifted_control: 移位后的控制序列 [T, D] 或 [B, T, D]
        """
        return self.shift_trajectory(control_sequence)
    
    def prepare_warm_start_latent(
        self,
        prev_opt_traj: torch.Tensor,
        tau_warm: float = 0.8,
    ) -> torch.Tensor:
        """
        准备热启动潜在状态（Revert Step）
        
        实现 Warm Start 的"Revert"阶段，使用 Optimal Transport (OT) 插值：
        1. 移位前一帧的最优轨迹
        2. 采样高斯噪声 ε ~ N(0, I)
        3. 使用 OT 插值构建中间时刻 τ_warm 的潜在状态
        
        数学表达:
            u_shift = Shift(u*_{t-1})
            ε ~ N(0, I)
            z_τ = τ_warm · u_shift + (1 - τ_warm) · ε
        
        这遵循 SDEdit 原理：而不是从 t=0 (纯噪声) 开始，我们从 t=τ_warm
        开始，这保留了前一轨迹的结构，同时允许模型纠正错误。
        
        Args:
            prev_opt_traj: 前一帧的最优轨迹 [T, D] 或 [B, T, D]
                对于完整状态，D 应该是 state_dim * 3 (pos + vel + acc)
                对于仅位置，D 应该是 state_dim (将自动扩展)
            tau_warm: OT 插值参数，范围 [0, 1]
                0.0 = 纯噪声（标准 CFM）
                1.0 = 完全确定性（无探索）
                0.8 = 推荐值（80% 先验，20% 噪声）
            
        Returns:
            z_tau: 时间 τ_warm 的热启动潜在状态 [T, D*3] 或 [B, T, D*3]
        """
        # 步骤 1: 移位轨迹（Predict）
        shifted_traj = self.shift_trajectory(prev_opt_traj)
        
        # 检查输入维度并扩展为完整状态（如果需要）
        if shifted_traj.dim() == 2:
            # [T, D]
            T, D = shifted_traj.shape
            batch_size = None
            
            # 如果输入仅为位置（D = state_dim），扩展为完整状态
            if D == 2:  # 仅位置 (x, y)
                positions = shifted_traj  # [T, 2]
                
                # 计算速度（有限差分）
                dt = self.config.time_horizon / self.config.n_timesteps
                velocities = torch.diff(positions, dim=0) / dt
                velocities = torch.cat([velocities, velocities[-1:]], dim=0)  # [T, 2]
                
                # 计算加速度
                accelerations = torch.diff(velocities, dim=0) / dt
                accelerations = torch.cat([accelerations, accelerations[-1:]], dim=0)  # [T, 2]
                
                # 拼接完整状态 [T, 6]
                u_shift = torch.cat([positions, velocities, accelerations], dim=-1)
            else:
                # 已经是完整状态
                u_shift = shifted_traj
        else:
            # [B, T, D]
            B, T, D = shifted_traj.shape
            batch_size = B
            
            if D == 2:  # 仅位置
                positions = shifted_traj  # [B, T, 2]
                
                # 计算速度
                dt = self.config.time_horizon / self.config.n_timesteps
                velocities = torch.diff(positions, dim=1) / dt
                velocities = torch.cat([velocities, velocities[:, -1:, :]], dim=1)
                
                # 计算加速度
                accelerations = torch.diff(velocities, dim=1) / dt
                accelerations = torch.cat([accelerations, accelerations[:, -1:, :]], dim=1)
                
                # 拼接完整状态 [B, T, 6]
                u_shift = torch.cat([positions, velocities, accelerations], dim=-1)
            else:
                u_shift = shifted_traj
        
        # 步骤 2: 采样噪声 ε ~ N(0, I)
        epsilon = torch.randn_like(u_shift)
        
        # 步骤 3: OT 插值 z_τ = τ · u_shift + (1 - τ) · ε
        # 这是 Optimal Transport 路径公式，与 CFM 训练一致
        z_tau = tau_warm * u_shift + (1.0 - tau_warm) * epsilon
        
        return z_tau
    
    def prepare_warm_start_state(
        self,
        shifted_control: torch.Tensor,
    ) -> torch.Tensor:
        """
        准备热启动状态（用于 CFM 反向注入）
        
        旧方法，保持向后兼容性。
        推荐使用 prepare_warm_start_latent() 以获得更好的 OT 插值控制。
        
        Args:
            shifted_control: 移位后的控制序列 [T, D]
            
        Returns:
            warm_start_state: 热启动状态 [T, D*3] (pos + vel + acc)
        """
        # 使用默认 tau_warm = 0.8 调用新方法
        # 注意：此方法添加简单的高斯噪声，而 prepare_warm_start_latent
        # 使用正确的 OT 插值
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
        获取下一帧的最优控制（用于闭环回流），接口保持与旧版一致。
        
        这个 u*_k 将在下一帧 k+1 成为 L2 的 z_init。
        """
        # 使用新的 L2 输出更新锚点 / 起终点
        self.initialize_from_l2_output(l2_output)
        
        # 执行 MPPI 优化
        result = self.optimize(n_iterations=n_iterations)
        optimal_control = result['optimal_control']  # [T, D]
        
        # 保存当前最优控制（用于下一帧的热启动）
        if self.config.use_warm_start:
            self.previous_optimal_control = optimal_control.clone()
            
            shifted_control = self.shift_control_sequence(optimal_control)
            warm_start_state = self.prepare_warm_start_state(shifted_control)
            self.previous_optimal_state = warm_start_state
        
        return optimal_control
    
    def get_warm_start_state(
        self,
    ) -> Optional[torch.Tensor]:
        """
        获取热启动状态（用于 CFM 反向注入）。
        
        返回上一帧移位并加噪后的状态，作为 L2 / TrajectoryGenerator 的初始状态 z_T。
        如果没有上一帧的状态，返回 None（使用纯噪声）。
        """
        if not self.config.use_warm_start:
            return None
        
        if self.previous_optimal_state is None:
            return None
        
        return self.previous_optimal_state.clone()
