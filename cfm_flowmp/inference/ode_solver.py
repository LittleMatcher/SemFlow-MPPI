"""
流匹配推理的 ODE 求解器

实现用于求解 ODE 的数值积分方法:
    dx/dt = v_θ(x_t, t, c)

从 t=0 到 t=1 生成轨迹。

可用求解器:
- EulerSolver: 简单的一阶方法
- RK4Solver: 经典四阶龙格-库塔方法（推荐）
- AdaptiveRK45Solver: 自适应步长 RK45 方法
"""

import torch
import torch.nn as nn
from typing import Callable, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class SolverConfig:
    """ODE 求解器配置"""
    
    # 积分步数（用于均匀步进）
    num_steps: int = 20
    
    # 时间范围
    t_start: float = 0.0
    t_end: float = 1.0
    
    # 用于自适应求解器
    atol: float = 1e-5
    rtol: float = 1e-5
    max_steps: int = 1000
    
    # 是否返回中间状态
    return_trajectory: bool = False
    
    # 自定义时间调度（如果提供则覆盖 num_steps）
    # 遵循"统一生成-细化规划"方法
    time_schedule: Optional[List[float]] = None
    
    # 预定义调度
    use_8step_schedule: bool = False  # 使用激进的 8 步调度


# 基于"统一生成-细化规划"的预定义时间调度
SCHEDULE_8STEP = [0.0, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
SCHEDULE_UNIFORM_10 = [i/10 for i in range(11)]
SCHEDULE_UNIFORM_20 = [i/20 for i in range(21)]


class EulerSolver:
    """
    欧拉方法（一阶）ODE 求解器
    
    简单但精度较低。适用于调试和快速推理。
    支持自定义时间调度以实现非均匀步进。
    
    更新规则:
        x_{n+1} = x_n + dt * v(x_n, t_n)
    """
    
    def __init__(self, config: SolverConfig = None):
        self.config = config or SolverConfig()
    
    def _get_time_schedule(self, num_steps: int = None) -> List[float]:
        """Get the time schedule for ODE integration."""
        if self.config.time_schedule is not None:
            return self.config.time_schedule
        elif self.config.use_8step_schedule:
            return SCHEDULE_8STEP
        else:
            n = num_steps or self.config.num_steps
            return [self.config.t_start + i * (self.config.t_end - self.config.t_start) / n 
                    for i in range(n + 1)]
    
    def solve(
        self,
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_0: torch.Tensor,
        num_steps: int = None,
        time_schedule: List[float] = None,
    ) -> torch.Tensor:
        """
        使用欧拉方法求解 ODE
        
        参数:
            velocity_fn: 函数 v(x, t)，返回状态 x 和时间 t 处的速度
            x_0: 初始状态 [B, T, D]
            num_steps: 积分步数（覆盖配置）
            time_schedule: 自定义时间调度（覆盖配置）
            
        返回:
            t=1 处的最终状态 x_1
        """
        # 获取时间调度
        if time_schedule is not None:
            schedule = time_schedule
        else:
            schedule = self._get_time_schedule(num_steps)
        
        x = x_0.clone()
        B = x_0.shape[0]
        
        trajectory = [x.clone()] if self.config.return_trajectory else None
        
        # 通过时间调度进行积分
        for i in range(len(schedule) - 1):
            t_curr = schedule[i]
            t_next = schedule[i + 1]
            dt = t_next - t_curr
            
            t = torch.full((B,), t_curr, device=x_0.device, dtype=x_0.dtype)
            
            # 计算速度并更新
            v = velocity_fn(x, t)
            x = x + dt * v
            
            if trajectory is not None:
                trajectory.append(x.clone())
        
        if self.config.return_trajectory:
            return torch.stack(trajectory, dim=1)  # [B, num_steps+1, T, D]
        
        return x


class RK4Solver:
    """
    经典四阶龙格-库塔 ODE 求解器
    
    在精度和计算成本之间提供良好平衡。
    这是 FlowMP 推理的推荐求解器。
    
    支持均匀步进和自定义时间调度，如"统一生成-细化规划"中所述
    （例如，8 步调度）。
    
    更新规则 (RK4):
        k1 = v(x_n, t_n)
        k2 = v(x_n + dt/2 * k1, t_n + dt/2)
        k3 = v(x_n + dt/2 * k2, t_n + dt/2)
        k4 = v(x_n + dt * k3, t_n + dt)
        x_{n+1} = x_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    """
    
    def __init__(self, config: SolverConfig = None):
        self.config = config or SolverConfig()
    
    def _get_time_schedule(self, num_steps: int = None) -> List[float]:
        """获取 ODE 积分的时间调度"""
        # 优先级: 自定义调度 > 8 步标志 > 均匀步进
        if self.config.time_schedule is not None:
            return self.config.time_schedule
        elif self.config.use_8step_schedule:
            return SCHEDULE_8STEP
        else:
            n = num_steps or self.config.num_steps
            return [self.config.t_start + i * (self.config.t_end - self.config.t_start) / n 
                    for i in range(n + 1)]
    
    def solve(
        self,
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_0: torch.Tensor,
        num_steps: int = None,
        time_schedule: List[float] = None,
    ) -> torch.Tensor:
        """
        使用 RK4 方法求解 ODE
        
        参数:
            velocity_fn: 函数 v(x, t)，返回状态 x 和时间 t 处的速度
            x_0: 初始状态 [B, T, D] 或 [B, D]
            num_steps: 积分步数（覆盖配置，如果提供 time_schedule 则忽略）
            time_schedule: 自定义时间调度（覆盖配置）
            
        返回:
            t=1 处的最终状态 x_1，或如果 return_trajectory=True 则返回轨迹
        """
        # 获取时间调度
        if time_schedule is not None:
            schedule = time_schedule
        else:
            schedule = self._get_time_schedule(num_steps)
        
        x = x_0.clone()
        B = x_0.shape[0]
        
        trajectory = [x.clone()] if self.config.return_trajectory else None
        
        # 通过时间调度进行积分
        for i in range(len(schedule) - 1):
            t_curr = schedule[i]
            t_next = schedule[i + 1]
            dt = t_next - t_curr
            
            # 当前时间作为张量
            t = torch.full((B,), t_curr, device=x_0.device, dtype=x_0.dtype)
            
            # RK4 阶段
            k1 = velocity_fn(x, t)
            k2 = velocity_fn(x + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = velocity_fn(x + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = velocity_fn(x + dt * k3, t + dt)
            
            # 更新状态
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            
            if trajectory is not None:
                trajectory.append(x.clone())
        
        if self.config.return_trajectory:
            return torch.stack(trajectory, dim=1)  # [B, num_steps+1, ...]
        
        return x
    
    def solve_with_intermediates(
        self,
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_0: torch.Tensor,
        num_steps: int = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        求解 ODE 并返回中间状态和速度
        
        适用于调试和可视化
        
        返回:
            (最终状态, 状态列表, 速度列表) 的元组
        """
        num_steps = num_steps or self.config.num_steps
        dt = (self.config.t_end - self.config.t_start) / num_steps
        
        x = x_0.clone()
        B = x_0.shape[0]
        t = torch.full((B,), self.config.t_start, device=x_0.device, dtype=x_0.dtype)
        
        states = [x.clone()]
        velocities = []
        times = [t.clone()]
        
        for step in range(num_steps):
            k1 = velocity_fn(x, t)
            k2 = velocity_fn(x + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = velocity_fn(x + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = velocity_fn(x + dt * k3, t + dt)
            
            # 存储此步骤的平均速度
            v_avg = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            velocities.append(v_avg.clone())
            
            x = x + dt * v_avg
            t = t + dt
            
            states.append(x.clone())
            times.append(t.clone())
        
        return x, states, velocities, times


class AdaptiveRK45Solver:
    """
    自适应步长 RK45（Dormand-Prince）求解器
    
    自动调整步长以平衡精度和速度。
    比固定步长 RK4 更准确，但可能更慢。
    
    使用嵌入式 RK4/RK5 对进行误差估计。
    """
    
    def __init__(self, config: SolverConfig = None):
        self.config = config or SolverConfig()
        
        # Dormand-Prince 系数
        self.a = [
            [],
            [1/5],
            [3/40, 9/40],
            [44/45, -56/15, 32/9],
            [19372/6561, -25360/2187, 64448/6561, -212/729],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84],
        ]
        
        # 五阶权重
        self.b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
        
        # 四阶权重
        self.b4 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
        
        # 时间分数
        self.c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    
    def solve(
        self,
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_0: torch.Tensor,
    ) -> torch.Tensor:
        """
        使用自适应 RK45 求解 ODE
        
        参数:
            velocity_fn: 函数 v(x, t)，返回速度
            x_0: 初始状态
            
        返回:
            t=1 处的最终状态
        """
        x = x_0.clone()
        B = x_0.shape[0]
        t = torch.full((B,), self.config.t_start, device=x_0.device, dtype=x_0.dtype)
        
        # 初始步长
        dt = (self.config.t_end - self.config.t_start) / self.config.num_steps
        
        trajectory = [x.clone()] if self.config.return_trajectory else None
        
        step_count = 0
        while t[0] < self.config.t_end and step_count < self.config.max_steps:
            # 不要超出范围
            dt = min(dt, self.config.t_end - t[0].item())
            
            # 计算 RK 阶段
            k = []
            k.append(velocity_fn(x, t))
            
            for i in range(1, 7):
                x_stage = x.clone()
                for j in range(i):
                    x_stage = x_stage + dt * self.a[i][j] * k[j]
                t_stage = t + self.c[i] * dt
                k.append(velocity_fn(x_stage, t_stage))
            
            # 计算四阶和五阶解
            x5 = x.clone()
            x4 = x.clone()
            for i in range(7):
                x5 = x5 + dt * self.b5[i] * k[i]
                x4 = x4 + dt * self.b4[i] * k[i]
            
            # 误差估计
            error = (x5 - x4).abs().max()
            
            # 容差检查
            tol = self.config.atol + self.config.rtol * x.abs().max()
            
            if error < tol:
                # 接受步骤
                x = x5
                t = t + dt
                step_count += 1
                
                if trajectory is not None:
                    trajectory.append(x.clone())
            
            # 调整步长
            if error > 0:
                dt = 0.9 * dt * (tol / error) ** 0.2
            else:
                dt = 2 * dt
            
            dt = max(dt, 1e-6)  # 最小步长
        
        if self.config.return_trajectory:
            return torch.stack(trajectory, dim=1)
        
        return x


class MidpointSolver:
    """
    中点方法（二阶）ODE 求解器
    
    比欧拉方法更好，比 RK4 更简单。
    
    更新规则:
        k1 = v(x_n, t_n)
        x_{n+1} = x_n + dt * v(x_n + dt/2 * k1, t_n + dt/2)
    """
    
    def __init__(self, config: SolverConfig = None):
        self.config = config or SolverConfig()
    
    def solve(
        self,
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_0: torch.Tensor,
        num_steps: int = None,
    ) -> torch.Tensor:
        """使用中点方法求解 ODE"""
        num_steps = num_steps or self.config.num_steps
        dt = (self.config.t_end - self.config.t_start) / num_steps
        
        x = x_0.clone()
        B = x_0.shape[0]
        t = torch.full((B,), self.config.t_start, device=x_0.device, dtype=x_0.dtype)
        
        trajectory = [x.clone()] if self.config.return_trajectory else None
        
        for step in range(num_steps):
            k1 = velocity_fn(x, t)
            k2 = velocity_fn(x + 0.5 * dt * k1, t + 0.5 * dt)
            
            x = x + dt * k2
            t = t + dt
            
            if trajectory is not None:
                trajectory.append(x.clone())
        
        if self.config.return_trajectory:
            return torch.stack(trajectory, dim=1)
        
        return x


def create_solver(
    solver_type: str = "rk4",
    config: SolverConfig = None,
):
    """
    创建 ODE 求解器的工厂函数
    
    参数:
        solver_type: 求解器类型 ("euler", "midpoint", "rk4", "rk45")
        config: 求解器配置
        
    返回:
        ODE 求解器实例
    """
    solvers = {
        "euler": EulerSolver,
        "midpoint": MidpointSolver,
        "rk4": RK4Solver,
        "rk45": AdaptiveRK45Solver,
    }
    
    if solver_type not in solvers:
        raise ValueError(f"Unknown solver type: {solver_type}. Choose from {list(solvers.keys())}")
    
    return solvers[solver_type](config)
