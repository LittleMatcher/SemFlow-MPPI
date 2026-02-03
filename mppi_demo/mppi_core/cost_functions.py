"""
MPPI 的代价函数
参考 Motion Planning Diffusion 原理 (arXiv:2308.01557)
"""
import numpy as np
from .environment_2d import Environment2D
from typing import Optional


class CostFunction:
    """代价函数基类"""
    
    def __call__(self, positions: np.ndarray, 
                 velocities: Optional[np.ndarray] = None,
                 accelerations: Optional[np.ndarray] = None,
                 jerks: Optional[np.ndarray] = None) -> np.ndarray:
        """计算代价
        Args:
            positions: 形状 (batch, T, 2)
            velocities: 形状 (batch, T, 2)，可选
            accelerations: 形状 (batch, T, 2)，可选
            jerks: 形状 (batch, T, 2)，可选
        Returns:
            costs: 形状 (batch,)
        """
        raise NotImplementedError


class CollisionCost(CostFunction):
    """
    基于 SDF 的碰撞代价：使用硬约束和 barrier 函数确保障碍物绝对不可触碰
    
    设计原则：
    1. 硬约束：如果轨迹有任何点穿透障碍物，给予极大的惩罚（1e6）
    2. Barrier 函数：使用 1/(SDF - r - ε) 形式的惩罚，确保接近障碍物时代价趋向无穷
    3. 二次惩罚：在安全距离内使用二次惩罚引导优化
    
    这确保了障碍物是硬约束，绝对不可触碰。
    """
    
    def __init__(self, env: Environment2D, 
                 robot_radius: float = 0.2,
                 epsilon: float = 0.1,
                 weight: float = 100.0,
                 use_hard_constraint: bool = True,
                 hard_penalty: float = 1e6):
        """
        Args:
            env: 带障碍物的 2D 环境
            robot_radius: 机器人安全半径
            epsilon: 安全边距
            weight: 代价权重
            use_hard_constraint: 是否使用硬约束（碰撞时给予极大惩罚）
            hard_penalty: 硬约束的惩罚值（如果轨迹碰撞）
        """
        self.env = env
        self.robot_radius = robot_radius
        self.epsilon = epsilon
        self.weight = weight
        self.use_hard_constraint = use_hard_constraint
        self.hard_penalty = hard_penalty
        
    def __call__(self, positions: np.ndarray, **kwargs) -> np.ndarray:
        """
        Args:
            positions: 形状 (batch, T, 2)
        Returns:
            costs: 形状 (batch,)
        """
        batch_size, T, _ = positions.shape
        
        # 为 SDF 计算重塑形状
        positions_flat = positions.reshape(-1, 2)
        
        # 计算 SDF
        sdf = self.env.compute_sdf(positions_flat)
        sdf = sdf.reshape(batch_size, T)
        
        # 安全距离阈值
        safety_threshold = self.robot_radius + self.epsilon
        
        # 检查是否有碰撞（硬约束检查）
        if self.use_hard_constraint:
            # 检查是否有任何点穿透障碍物（SDF < robot_radius）
            collisions = sdf < self.robot_radius
            has_collision = np.any(collisions, axis=1)  # (batch,)
            
            # 如果有碰撞，给予极大惩罚（硬约束）
            hard_penalties = np.where(has_collision, self.hard_penalty, 0.0)
        else:
            hard_penalties = np.zeros(batch_size)
        
        # 硬约束：障碍物绝对不可触碰
        # 使用 barrier 函数确保接近障碍物时代价趋向无穷
        
        # 计算到安全边界的距离
        safety_margin = sdf - safety_threshold  # 正值=安全，负值=不安全
        
        # 区分三个区域：
        # 1. 完全安全区域 (safety_margin > 0)：代价为0或很小
        # 2. 接近边界区域 (0 >= safety_margin > -epsilon)：使用barrier函数
        # 3. 碰撞区域 (safety_margin <= -epsilon)：已经在硬约束中处理
        
        # 对于完全安全的区域，代价很小（用于引导优化）
        safe_penalty = np.zeros_like(safety_margin)
        
        # 对于接近边界或不安全的区域，使用barrier函数
        # barrier函数：1 / (distance + eps)，当distance -> 0时 -> 无穷
        unsafe_mask = safety_margin <= 0
        if np.any(unsafe_mask):
            # 计算到安全边界的距离（绝对值）
            barrier_distance = -safety_margin[unsafe_mask] + 1e-8  # 避免除零
            # 使用 barrier 函数：惩罚与 1/distance 成正比
            barrier_penalty = 10.0 / barrier_distance  # 系数10控制barrier的陡峭程度
            safe_penalty[unsafe_mask] = barrier_penalty
        
        # 对时间求和，然后应用权重
        soft_costs = self.weight * np.sum(safe_penalty, axis=1)
        
        # 总代价 = 硬约束惩罚 + 软代价（barrier + 二次惩罚）
        total_costs = hard_penalties + soft_costs
        
        return total_costs
    
    def compute_per_timestep(self, positions: np.ndarray) -> np.ndarray:
        """计算每个时间步的碰撞代价用于可视化
        Args:
            positions: 形状 (batch, T, 2)
        Returns:
            costs: 形状 (batch, T)
        """
        batch_size, T, _ = positions.shape
        positions_flat = positions.reshape(-1, 2)
        
        sdf = self.env.compute_sdf(positions_flat)
        sdf = sdf.reshape(batch_size, T)
        
        safety_threshold = self.robot_radius + self.epsilon
        safety_margin = sdf - safety_threshold
        
        # 检查碰撞（硬约束）
        if self.use_hard_constraint:
            collisions = sdf < self.robot_radius
            hard_penalties = np.where(collisions, self.hard_penalty / T, 0.0)  # 平均分配到各时间步
        else:
            hard_penalties = np.zeros_like(sdf)
        
        # Barrier 函数：只对接近边界或不安全的区域应用
        safe_penalty = np.zeros_like(safety_margin)
        unsafe_mask = safety_margin <= 0
        if np.any(unsafe_mask):
            barrier_distance = -safety_margin[unsafe_mask] + 1e-8
            barrier_penalty = 10.0 / barrier_distance
            safe_penalty[unsafe_mask] = barrier_penalty
        
        return hard_penalties + self.weight * safe_penalty


class SmoothnessCost(CostFunction):
    """
    惩罚大加速度或加加速度以实现光滑轨迹
    """
    
    def __init__(self, penalize: str = 'acceleration',
                 weight: float = 1.0):
        """
        Args:
            penalize: 'acceleration' 或 'jerk'
            weight: 代价权重
        """
        assert penalize in ['acceleration', 'jerk']
        self.penalize = penalize
        self.weight = weight
        
    def __call__(self, positions: np.ndarray,
                 velocities: Optional[np.ndarray] = None,
                 accelerations: Optional[np.ndarray] = None,
                 jerks: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Args:
            positions: 形状 (batch, T, 2)
            accelerations: 形状 (batch, T, 2)
            jerks: 形状 (batch, T, 2)
        Returns:
            costs: 形状 (batch,)
        """
        if self.penalize == 'acceleration':
            if accelerations is None:
                raise ValueError("加速度平滑需要 accelerations 参数")
            # 加速度的 L2 范数
            accel_mag = np.linalg.norm(accelerations, axis=-1)  # (batch, T)
            costs = self.weight * np.sum(accel_mag**2, axis=1)
            
        elif self.penalize == 'jerk':
            if jerks is None:
                raise ValueError("加加速度平滑需要 jerks 参数")
            # 加加速度的 L2 范数
            jerk_mag = np.linalg.norm(jerks, axis=-1)  # (batch, T)
            costs = self.weight * np.sum(jerk_mag**2, axis=1)
            
        return costs


class GoalCost(CostFunction):
    """
    惩罚最后时间步到目标的距离
    """
    
    def __init__(self, goal: np.ndarray, weight: float = 50.0):
        """
        Args:
            goal: 目标位置，形状 (2,)
            weight: 代价权重
        """
        self.goal = goal
        self.weight = weight
        
    def __call__(self, positions: np.ndarray, **kwargs) -> np.ndarray:
        """
        Args:
            positions: 形状 (batch, T, 2)
        Returns:
            costs: 形状 (batch,)
        """
        # 最后时间步到目标的距离
        final_pos = positions[:, -1, :]  # (batch, 2)
        dist = np.linalg.norm(final_pos - self.goal, axis=-1)
        
        costs = self.weight * dist**2
        
        return costs


class GoalApproachCost(CostFunction):
    """Penalize moving away from the goal during the trajectory.

    This discourages "orbiting" / looping near the goal: once a trajectory is
    approaching the goal, increasing goal-distance becomes expensive.
    """

    def __init__(self, goal: np.ndarray, weight: float = 50.0, power: float = 2.0):
        self.goal = np.asarray(goal, dtype=np.float64)
        self.weight = weight
        self.power = max(1.0, float(power))

    def __call__(self, positions: np.ndarray, **kwargs) -> np.ndarray:
        # Distance-to-goal over time
        d = np.linalg.norm(positions - self.goal[None, None, :], axis=-1)  # (batch, T)
        # Positive increases in distance (moving away)
        inc = np.diff(d, axis=1)
        away = np.maximum(0.0, inc)
        return self.weight * np.sum(away ** self.power, axis=1)


class TerminalVelocityCost(CostFunction):
    """Penalize speed near the end of the horizon to avoid end spirals."""

    def __init__(self, weight: float = 10.0, last_fraction: float = 0.25):
        self.weight = weight
        self.last_fraction = float(np.clip(last_fraction, 0.0, 1.0))

    def __call__(self,
                 positions: np.ndarray,
                 velocities: Optional[np.ndarray] = None,
                 **kwargs) -> np.ndarray:
        if velocities is None:
            raise ValueError("TerminalVelocityCost requires velocities")

        T = velocities.shape[1]
        if T == 0:
            return np.zeros(positions.shape[0])

        start_idx = int(np.floor((1.0 - self.last_fraction) * T))
        start_idx = max(0, min(T - 1, start_idx))
        v_mag = np.linalg.norm(velocities[:, start_idx:, :], axis=-1)  # (batch, t_tail)
        return self.weight * np.sum(v_mag ** 2, axis=1)


class TerrainCost(CostFunction):
    """
    根据地形类型对路径进行惩罚
    不同地形（水泥、草地、沼泽）有不同的通行成本
    成本 = 路径长度 × 地形系数
    """
    
    def __init__(self, get_terrain_cost_fn, weight: float = 50.0):
        """
        Args:
            get_terrain_cost_fn: 函数，接受位置(x,y)返回地形成本系数
            weight: 代价权重
        """
        self.get_terrain_cost_fn = get_terrain_cost_fn
        self.weight = weight
    
    def __call__(self,
                 positions: np.ndarray,
                 **kwargs) -> np.ndarray:
        """
        计算每条轨迹的总地形成本
        
        Args:
            positions: 形状为 (batch_size, time_steps, 2) 的位置数组
            
        Returns:
            形状为 (batch_size,) 的成本数组
        """
        batch_size, T, _ = positions.shape
        costs = np.zeros(batch_size)
        
        for b in range(batch_size):
            total_cost = 0.0
            for t in range(T - 1):
                # 计算段长度
                segment = positions[b, t+1] - positions[b, t]
                segment_length = np.linalg.norm(segment)
                
                # 使用中点的地形成本
                midpoint = (positions[b, t] + positions[b, t+1]) / 2.0
                terrain_cost = self.get_terrain_cost_fn(midpoint)
                
                # 累积成本 = 长度 × 地形系数
                total_cost += segment_length * terrain_cost
            
            costs[b] = total_cost
        
        return self.weight * costs


class PathLengthCost(CostFunction):
    """
    惩罚路径长度，鼓励更短的路径
    这是路径优化中的关键组件，确保找到最短路径
    """
    
    def __init__(self, weight: float = 10.0):
        """
        Args:
            weight: 代价权重
        """
        self.weight = weight
        
    def __call__(self, positions: np.ndarray, **kwargs) -> np.ndarray:
        """
        Args:
            positions: 形状 (batch, T, 2)
        Returns:
            costs: 形状 (batch,)
        """
        # 计算路径长度：相邻点之间的距离之和
        diffs = np.diff(positions, axis=1)  # (batch, T-1, 2)
        segment_lengths = np.linalg.norm(diffs, axis=-1)  # (batch, T-1)
        path_length = np.sum(segment_lengths, axis=1)  # (batch,)
        
        costs = self.weight * path_length
        
        return costs


class ReferencePathCost(CostFunction):
    """Penalty for deviating from a guidance polyline (e.g., A* path)."""

    def __init__(self,
                 reference_path: np.ndarray,
                 weight: float = 20.0,
                 progress_weight: float = 5.0,
                 backtrack_weight: float = 50.0,
                 lateral_power: float = 2.0):
        """Args:
            reference_path: Waypoints describing the guide path, shape (N, 2)
            weight: Lateral deviation weight
            progress_weight: Penalty for lagging behind the guide-path progress
            lateral_power: Exponent applied to lateral distances (>=1)
        """
        if reference_path is None or len(reference_path) < 2:
            raise ValueError("Reference path must have at least two waypoints")

        self.reference_path = np.asarray(reference_path, dtype=np.float64)
        diffs = np.diff(self.reference_path, axis=0)
        lengths = np.linalg.norm(diffs, axis=1)

        # Remove zero-length segments to avoid division by zero
        valid_mask = lengths > 1e-6
        self.segment_starts = self.reference_path[:-1][valid_mask]
        self.segment_vecs = diffs[valid_mask]
        self.segment_lengths = lengths[valid_mask]

        if len(self.segment_lengths) == 0:
            raise ValueError("Reference path segments have zero length")

        self.segment_len_sq = self.segment_lengths ** 2 + 1e-9
        self.segment_offsets = np.concatenate((
            [0.0],
            np.cumsum(self.segment_lengths)[:-1]
        ))
        self.total_length = np.sum(self.segment_lengths)

        self.weight = weight
        self.progress_weight = progress_weight
        self.backtrack_weight = backtrack_weight
        self.lateral_power = max(1.0, lateral_power)

    def __call__(self, positions: np.ndarray, **kwargs) -> np.ndarray:
        batch_size, T, _ = positions.shape
        points = positions.reshape(-1, 2)

        distances, progress = self._distance_and_progress(points)
        distances = distances.reshape(batch_size, T)
        progress = progress.reshape(batch_size, T)

        # Lateral deviation penalty (encourage staying close to guide path)
        lateral_cost = np.sum(distances ** self.lateral_power, axis=1)
        total_cost = self.weight * lateral_cost

        if self.progress_weight > 0:
            # Encourage steady progress along the reference path by penalizing lag
            ideal_progress = np.linspace(0.0, self.total_length, T, dtype=np.float64)
            lag = np.maximum(0.0, ideal_progress - progress)
            progress_cost = np.sum(lag ** 2, axis=1)
            total_cost += self.progress_weight * progress_cost

        if self.backtrack_weight > 0:
            # Penalize any backward motion along the guide path (loops/backtracking)
            dp = np.diff(progress, axis=1)  # (batch, T-1)
            back = np.maximum(0.0, -dp)
            total_cost += self.backtrack_weight * np.sum(back ** 2, axis=1)

        return total_cost

    def _distance_and_progress(self, points: np.ndarray):
        """Compute lateral distance and arc-length progress for each point."""
        # Broadcast points against segments
        diff = points[:, None, :] - self.segment_starts[None, :, :]
        proj = np.sum(diff * self.segment_vecs[None, :, :], axis=-1) / self.segment_len_sq[None, :]
        t = np.clip(proj, 0.0, 1.0)
        closest = self.segment_starts[None, :, :] + t[..., None] * self.segment_vecs[None, :, :]
        delta = points[:, None, :] - closest
        distances = np.linalg.norm(delta, axis=-1)

        min_idx = np.argmin(distances, axis=1)
        min_dist = distances[np.arange(points.shape[0]), min_idx]
        t_min = t[np.arange(points.shape[0]), min_idx]
        progress = self.segment_offsets[min_idx] + t_min * self.segment_lengths[min_idx]

        return min_dist, progress


class TurnCost(CostFunction):
    """
    惩罚路径中的拐弯（方向变化），特别是急转弯
    
    通过计算相邻方向向量之间的角度差来量化拐弯程度。
    对超过物理限制的急转弯给予更大的惩罚，确保转弯符合物理规律。
    
    物理约束：
    - 考虑最大角速度限制，每个时间步的最大角度变化应该有限
    - 对超过阈值的大角度变化使用更强的惩罚（如平方或指数）
    """
    
    def __init__(self, weight: float = 5.0, 
                 method: str = 'angle_diff',
                 max_angular_change: float = None,
                 dt: float = 0.1,
                 use_sharp_turn_penalty: bool = True,
                 sharp_turn_threshold: float = None):
        """
        Args:
            weight: 代价权重
            method: 计算方法
                - 'angle_diff': 计算相邻方向向量之间的角度差（推荐）
                - 'curvature': 基于曲率的方法
            max_angular_change: 每个时间步的最大角度变化（弧度），如果None则使用max_omega*dt
            dt: 时间步长，用于计算物理限制
            use_sharp_turn_penalty: 是否对急转弯使用更强的惩罚
            sharp_turn_threshold: 急转弯阈值（弧度），超过此值的角度变化会被额外惩罚
        """
        self.weight = weight
        self.method = method
        self.dt = dt
        self.use_sharp_turn_penalty = use_sharp_turn_penalty
        
        # 设置最大角度变化（物理限制）
        # 假设最大角速度约为 π rad/s，时间步长为 0.1s
        if max_angular_change is None:
            # 默认：max_omega = π, dt = 0.1 -> max_change ≈ 0.314 rad (约18度)
            self.max_angular_change = np.pi * dt
        else:
            self.max_angular_change = max_angular_change
        
        # 设置急转弯阈值（超过此值认为是急转弯）
        if sharp_turn_threshold is None:
            # 默认：阈值设为最大角度变化的80%，超过此值认为是急转弯
            self.sharp_turn_threshold = self.max_angular_change * 0.8
        else:
            self.sharp_turn_threshold = sharp_turn_threshold
        
        assert method in ['angle_diff', 'curvature']
        
    def __call__(self, positions: np.ndarray, **kwargs) -> np.ndarray:
        """
        计算拐弯代价
        Args:
            positions: 形状 (batch, T, 2)
        Returns:
            costs: 形状 (batch,)
        """
        batch_size, T, _ = positions.shape
        
        if T < 3:  # 至少需要3个点才能计算角度差
            return np.zeros(batch_size)
        
        if self.method == 'angle_diff':
            costs = self._compute_angle_difference_cost(positions)
        elif self.method == 'curvature':
            costs = self._compute_curvature_cost(positions)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self.weight * costs
    
    def _compute_angle_difference_cost(self, positions: np.ndarray) -> np.ndarray:
        """计算相邻方向向量之间的角度差，对急转弯给予额外惩罚
        Args:
            positions: 形状 (batch, T, 2)
        Returns:
            costs: 形状 (batch,)
        """
        # 计算相邻点之间的方向向量
        # 从点 i 到点 i+1 的向量
        vecs = np.diff(positions, axis=1)  # (batch, T-1, 2)
        
        # 计算方向向量的角度（使用atan2）
        angles = np.arctan2(vecs[:, :, 1], vecs[:, :, 0])  # (batch, T-1)
        
        # 计算相邻角度之间的差
        angle_diffs = np.diff(angles, axis=1)  # (batch, T-2)
        
        # 将角度差标准化到 [-pi, pi] 范围
        angle_diffs = np.arctan2(np.sin(angle_diffs), np.cos(angle_diffs))
        
        # 计算角度差的绝对值
        angle_diffs_abs = np.abs(angle_diffs)
        
        if self.use_sharp_turn_penalty:
            # 对急转弯使用更强的惩罚
            # 1. 基础代价：角度差的平方
            base_cost = angle_diffs_abs ** 2
            
            # 2. 急转弯额外惩罚：如果角度差超过阈值，使用指数惩罚
            sharp_turn_mask = angle_diffs_abs > self.sharp_turn_threshold
            
            # 对急转弯使用更强的惩罚：使用 (angle_diff / threshold)^3 或更高次幂
            # 这样急转弯的代价会快速增长
            sharp_turn_penalty = np.where(
                sharp_turn_mask,
                (angle_diffs_abs / self.sharp_turn_threshold) ** 4,  # 4次方惩罚急转弯
                0.0
            )
            
            # 3. 超过物理限制的额外惩罚：如果角度差超过最大物理限制，使用极大惩罚
            violation_mask = angle_diffs_abs > self.max_angular_change
            physical_violation_penalty = np.where(
                violation_mask,
                10.0 * (angle_diffs_abs / self.max_angular_change) ** 2,  # 违反物理限制的极大惩罚
                0.0
            )
            
            # 总代价 = 基础代价 + 急转弯惩罚 + 物理限制违反惩罚
            total_cost_per_timestep = base_cost + sharp_turn_penalty + physical_violation_penalty
        else:
            # 简单方法：只使用角度差的平方
            total_cost_per_timestep = angle_diffs_abs ** 2
        
        # 对所有时间步求和
        total_cost = np.sum(total_cost_per_timestep, axis=1)  # (batch,)
        
        return total_cost
    
    def _compute_curvature_cost(self, positions: np.ndarray) -> np.ndarray:
        """基于曲率计算拐弯代价
        Args:
            positions: 形状 (batch, T, 2)
        Returns:
            costs: 形状 (batch,)
        """
        # 计算一阶和二阶导数（速度和加速度）
        velocities = np.diff(positions, axis=1)  # (batch, T-1, 2)
        accelerations = np.diff(velocities, axis=1)  # (batch, T-2, 2)
        
        # 速度大小
        vel_mag = np.linalg.norm(velocities[:, :-1, :], axis=-1)  # (batch, T-2)
        
        # 曲率公式: |v x a| / |v|^3
        # 对于2D，叉积的大小 = |vx*ay - vy*ax|
        vel_x = velocities[:, :-1, 0]
        vel_y = velocities[:, :-1, 1]
        acc_x = accelerations[:, :, 0]
        acc_y = accelerations[:, :, 1]
        
        cross_product = np.abs(vel_x * acc_y - vel_y * acc_x)
        
        # 避免除零
        vel_mag_cubed = vel_mag ** 3
        vel_mag_cubed = np.where(vel_mag_cubed < 1e-8, 1e-8, vel_mag_cubed)
        
        # 曲率
        curvature = cross_product / vel_mag_cubed  # (batch, T-2)
        
        # 对曲率的平方求和
        total_cost = np.sum(curvature ** 2, axis=1)  # (batch,)
        
        return total_cost


class CompositeCost(CostFunction):
    """
    组合多个代价函数
    """
    
    def __init__(self, cost_functions: list):
        """
        Args:
            cost_functions: CostFunction 对象列表
        """
        self.cost_functions = cost_functions
        
    def __call__(self, positions: np.ndarray,
                 velocities: Optional[np.ndarray] = None,
                 accelerations: Optional[np.ndarray] = None,
                 jerks: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Args:
            positions: 形状 (batch, T, 2)
            velocities: 形状 (batch, T, 2)
            accelerations: 形状 (batch, T, 2)
            jerks: 形状 (batch, T, 2)
        Returns:
            costs: 形状 (batch,)
        """
        total_cost = np.zeros(positions.shape[0])
        
        for cost_fn in self.cost_functions:
            total_cost += cost_fn(
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
                jerks=jerks
            )
        
        return total_cost
    
    def compute_breakdown(self, positions: np.ndarray,
                         velocities: Optional[np.ndarray] = None,
                         accelerations: Optional[np.ndarray] = None,
                         jerks: Optional[np.ndarray] = None) -> dict:
        """计算代价分解用于分析
        Returns:
            cost_dict: 字典，键为代价名称，值为代价值
        """
        costs = {}
        
        for i, cost_fn in enumerate(self.cost_functions):
            name = cost_fn.__class__.__name__
            costs[f"{name}_{i}"] = cost_fn(
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
                jerks=jerks
            )
        
        costs['total'] = sum(costs.values())
        
        return costs


class CrowdDensityCost(CostFunction):
    """
    人流密度代价：根据路径经过区域的人流密度计算代价
    
    设计思路：
    1. 路径尽量避开高人流密度区域
    2. 计算路径每段在对应区域的密度加权长度
    3. 总代价 = Σ(段长度 × 区域密度倍数)
    """
    
    def __init__(self, crowd_regions: list, weight: float = 10.0):
        """
        Args:
            crowd_regions: CrowdRegion对象列表
            weight: 人流密度代价权重
        """
        self.crowd_regions = crowd_regions
        self.weight = weight
    
    def get_density_at_point(self, point: np.ndarray) -> float:
        """获取某点的人流密度倍数（如有重叠区域，取最高密度）"""
        max_density = 1.0  # 默认密度
        for region in self.crowd_regions:
            if region.contains_point(point):
                max_density = max(max_density, region.density_multiplier)
        return max_density
    
    def __call__(self, positions: np.ndarray, 
                 velocities: Optional[np.ndarray] = None,
                 accelerations: Optional[np.ndarray] = None,
                 jerks: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算人流密度代价
        
        Args:
            positions: (batch, T, 2) 位置序列
            
        Returns:
            costs: (batch,) 每条轨迹的密度代价
        """
        batch_size, T, _ = positions.shape
        costs = np.zeros(batch_size)
        
        for b in range(batch_size):
            total_density_cost = 0.0
            
            # 遍历轨迹的每一段
            for t in range(T - 1):
                p1 = positions[b, t]
                p2 = positions[b, t + 1]
                
                # 计算段长度
                segment_length = np.linalg.norm(p2 - p1)
                
                # 使用段中点的密度
                midpoint = (p1 + p2) / 2
                density = self.get_density_at_point(midpoint)
                
                # 累加加权代价
                total_density_cost += segment_length * density
            
            costs[b] = self.weight * total_density_cost
        
        return costs


class BoundaryConstraintCost(CostFunction):
    """
    边界约束代价：严格限制轨迹不能超出指定边界
    
    设计原则：
    1. 硬约束：任何超出边界的点都给予极大惩罚
    2. Barrier函数：接近边界时代价增大
    3. 确保机器人绝对不会移动到边界外
    """
    
    def __init__(self, bounds: tuple, 
                 margin: float = 0.5,
                 weight: float = 100.0,
                 use_hard_constraint: bool = True,
                 hard_penalty: float = 1e6):
        """
        Args:
            bounds: (x_min, x_max, y_min, y_max) 边界范围
            margin: 安全边距（距离边界多远开始惩罚）
            weight: 代价权重
            use_hard_constraint: 是否使用硬约束（超出边界直接极大惩罚）
            hard_penalty: 硬约束惩罚值
        """
        self.x_min, self.x_max, self.y_min, self.y_max = bounds
        self.margin = margin
        self.weight = weight
        self.use_hard_constraint = use_hard_constraint
        self.hard_penalty = hard_penalty
    
    def __call__(self, positions: np.ndarray, 
                 velocities: Optional[np.ndarray] = None,
                 accelerations: Optional[np.ndarray] = None,
                 jerks: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算边界约束代价
        
        Args:
            positions: (batch, T, 2) 位置序列
            
        Returns:
            costs: (batch,) 每条轨迹的边界代价
        """
        batch_size, T, _ = positions.shape
        costs = np.zeros(batch_size)
        
        for b in range(batch_size):
            boundary_cost = 0.0
            violated = False
            
            for t in range(T):
                x, y = positions[b, t, 0], positions[b, t, 1]
                
                # 检查是否超出边界
                if (x < self.x_min or x > self.x_max or 
                    y < self.y_min or y > self.y_max):
                    if self.use_hard_constraint:
                        # 硬约束：直接给予极大惩罚
                        costs[b] = self.hard_penalty
                        violated = True
                        break
                    else:
                        # 软约束：计算超出程度
                        out_x = max(0, self.x_min - x) + max(0, x - self.x_max)
                        out_y = max(0, self.y_min - y) + max(0, y - self.y_max)
                        boundary_cost += (out_x + out_y) ** 2
                
                # 使用barrier函数：接近边界时增加代价
                if not violated:
                    # 计算到各边界的距离
                    dist_to_left = x - self.x_min
                    dist_to_right = self.x_max - x
                    dist_to_bottom = y - self.y_min
                    dist_to_top = self.y_max - y
                    
                    min_dist = min(dist_to_left, dist_to_right, 
                                  dist_to_bottom, dist_to_top)
                    
                    # 在安全边距内使用barrier函数
                    if min_dist < self.margin:
                        # barrier: 1 / (dist - epsilon)
                        epsilon = 0.01
                        barrier = 1.0 / (min_dist + epsilon)
                        boundary_cost += barrier
            
            if not violated:
                costs[b] = self.weight * boundary_cost
        
        return costs
