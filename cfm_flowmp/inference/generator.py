"""
FlowMP 轨迹生成器

完整的轨迹生成流程：
1. 采样初始噪声
2. 从 t=0 到 t=1 求解 ODE
3. 使用 B 样条平滑进行后处理以确保物理一致性
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from .ode_solver import RK4Solver, SolverConfig, create_solver


@dataclass
class GeneratorConfig:
    """轨迹生成器配置"""
    
    # 求解器设置
    solver_type: str = "rk4"
    num_steps: int = 20  # 用于均匀步进
    
    # 时间调度（按照"统一生成-细化规划"方法）
    # 非均匀调度：早期大步长，接近 t=1 时小步长
    use_8step_schedule: bool = True  # 默认使用激进的 8 步调度
    custom_time_schedule: list = None  # 使用自定义调度覆盖
    
    # 自定义时间调度（如果提供则覆盖 num_steps）
    # 示例：8 步调度为 [0.0, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
    # 此非均匀调度在早期使用较大步长，在接近 t=1 时使用较小步长
    # 以便在细化阶段获得更好的细粒度控制
    time_schedule: Optional[List[float]] = None
    
    # 状态维度
    state_dim: int = 2
    seq_len: int = 64
    
    # 平滑（B 样条拟合以确保物理一致性）
    use_bspline_smoothing: bool = True
    bspline_degree: int = 3
    bspline_num_control_points: int = 20
    
    # 采样
    num_samples: int = 1  # 每个条件生成的轨迹数量
    
    # ============ 热启动（同策略）设置 ============
    # 实现"短期记忆"策略延续，类似于同策略强化学习
    # 时间 t 的最优轨迹成为时间 t+1 的强先验
    enable_warm_start: bool = False  # 启用时间热启动机制
    warm_start_noise_scale: float = 0.1  # 热启动初始状态的噪声尺度
    warm_start_shift_mode: str = "zero_pad"  # 'zero_pad', 'repeat_last', 'predict'
    warm_start_memory_length: int = 1  # 要记住的先前轨迹数量


# 来自"统一生成-细化规划"的 8 步调度
# 前载式：早期大步长（探索），后期小步长（细化）
DEFAULT_8STEP_SCHEDULE = [0.0, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]


class BSplineSmoother:
    """
    B 样条平滑器，用于轨迹后处理
    
    将 B 样条拟合到生成的轨迹，以确保：
    - 平滑性（连续导数）
    - 物理一致性
    - 减少来自 ODE 积分误差的噪声
    """
    
    def __init__(
        self,
        degree: int = 3,
        num_control_points: int = 20,
    ):
        """
        参数:
            degree: B 样条次数（3 = 三次）
            num_control_points: 用于拟合的控制点数量
        """
        self.degree = degree
        self.num_control_points = num_control_points
    
    def smooth(
        self,
        trajectory: torch.Tensor,
        return_derivatives: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        使用 B 样条拟合平滑轨迹
        
        参数:
            trajectory: 位置轨迹 [B, T, D]
            return_derivatives: 是否计算速度和加速度
            
        返回:
            包含平滑后的 'positions', 'velocities', 'accelerations' 的字典
        """
        B, T, D = trajectory.shape
        device = trajectory.device
        dtype = trajectory.dtype
        
        # Convert to numpy for scipy operations
        traj_np = trajectory.detach().cpu().numpy()
        
        try:
            from scipy.interpolate import splprep, splev
            
            smoothed_positions = []
            smoothed_velocities = []
            smoothed_accelerations = []
            
            # 轨迹点的参数值
            t_eval = np.linspace(0, 1, T)
            
            for b in range(B):
                traj_b = traj_np[b]  # [T, D]
                
                # 转置以供 splprep 使用：期望 [D, T]
                traj_b_t = traj_b.T
                
                # 拟合 B 样条
                # s=0 表示插值，s>0 表示平滑
                smoothing_factor = max(0, T - np.sqrt(2 * T))  # 自适应平滑
                
                try:
                    tck, u = splprep(
                        traj_b_t,
                        k=self.degree,
                        s=smoothing_factor,
                    )
                    
                    # 在原始参数值处评估
                    pos = np.array(splev(t_eval, tck)).T  # [T, D]
                    
                    if return_derivatives:
                        # 一阶导数（速度）
                        vel = np.array(splev(t_eval, tck, der=1)).T
                        vel = vel / (T - 1)  # 按时间步长缩放
                        
                        # 二阶导数（加速度）
                        acc = np.array(splev(t_eval, tck, der=2)).T
                        acc = acc / ((T - 1) ** 2)
                        
                        smoothed_velocities.append(vel)
                        smoothed_accelerations.append(acc)
                    
                    smoothed_positions.append(pos)
                    
                except Exception as e:
                    # 如果样条拟合失败，使用原始轨迹
                    smoothed_positions.append(traj_b)
                    if return_derivatives:
                        # 计算数值导数
                        vel = np.gradient(traj_b, axis=0)
                        acc = np.gradient(vel, axis=0)
                        smoothed_velocities.append(vel)
                        smoothed_accelerations.append(acc)
            
            # Convert back to torch
            result = {
                'positions': torch.tensor(
                    np.stack(smoothed_positions, axis=0),
                    device=device, dtype=dtype
                )
            }
            
            if return_derivatives:
                result['velocities'] = torch.tensor(
                    np.stack(smoothed_velocities, axis=0),
                    device=device, dtype=dtype
                )
                result['accelerations'] = torch.tensor(
                    np.stack(smoothed_accelerations, axis=0),
                    device=device, dtype=dtype
                )
            
            return result
            
        except ImportError:
            # 如果 scipy 不可用，回退到简单移动平均
            return self._smooth_moving_average(trajectory, return_derivatives)
    
    def _smooth_moving_average(
        self,
        trajectory: torch.Tensor,
        return_derivatives: bool = True,
        window_size: int = 5,
    ) -> Dict[str, torch.Tensor]:
        """
        简单移动平均平滑回退方法
        """
        B, T, D = trajectory.shape
        
        # 填充并应用移动平均
        pad = window_size // 2
        padded = torch.nn.functional.pad(
            trajectory.permute(0, 2, 1),  # [B, D, T]
            (pad, pad),
            mode='replicate'
        )
        
        # 通过 conv1d 进行移动平均
        kernel = torch.ones(1, 1, window_size, device=trajectory.device) / window_size
        
        smoothed = []
        for d in range(D):
            smoothed_d = torch.nn.functional.conv1d(
                padded[:, d:d+1, :],
                kernel,
                padding=0
            )
            smoothed.append(smoothed_d)
        
        positions = torch.cat(smoothed, dim=1).permute(0, 2, 1)  # [B, T, D]
        
        result = {'positions': positions}
        
        if return_derivatives:
            # 数值导数
            velocities = torch.gradient(positions, dim=1)[0]
            accelerations = torch.gradient(velocities, dim=1)[0]
            result['velocities'] = velocities
            result['accelerations'] = accelerations
        
        return result


class TrajectoryGenerator:
    """
    完整轨迹生成流程
    
    通过以下方式生成轨迹：
    1. 从 N(0, I) 采样初始噪声
    2. 从 t=0 到 t=1 求解 ODE dx/dt = v_θ(x, t, c)
    3. 可选地使用 B 样条平滑以确保物理一致性
    
    **热启动（同策略）特性：**
    启用后，实现类似于同策略强化学习的时间连续性：
    - 在时间 t，MPPI 输出最优控制序列 u*_t
    - 在时间 t+1，u*_t 向前移位以创建先验 ũ_t+1
    - CFM 从 ũ_t+1 的加噪版本开始，而不是纯高斯噪声
    - 这创建了"策略延续"，其中决策建立在先前步骤的基础上
    
    用法:
        generator = TrajectoryGenerator(model, config)
        trajectories = generator.generate(
            start_pos, goal_pos, start_vel
        )
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: GeneratorConfig = None,
    ):
        """
        初始化轨迹生成器
        
        参数:
            model: 训练好的 FlowMP transformer 模型
            config: 生成器配置
        """
        self.model = model
        self.config = config or GeneratorConfig()
        
        # 确定时间调度
        if self.config.custom_time_schedule is not None:
            time_schedule = self.config.custom_time_schedule
        elif self.config.use_8step_schedule:
            time_schedule = DEFAULT_8STEP_SCHEDULE
        else:
            time_schedule = None  # 使用均匀步进
        
        # 创建带时间调度的 ODE 求解器
        solver_config = SolverConfig(
            num_steps=self.config.num_steps,
            time_schedule=time_schedule,  # 使用计算出的 time_schedule
            return_trajectory=False,
            use_8step_schedule=self.config.use_8step_schedule,
        )
        self.solver = create_solver(self.config.solver_type, solver_config)
        self.time_schedule = time_schedule
        
        # 创建 B 样条平滑器以确保物理一致性
        # 按照规范："通过 B 样条进行输出平滑以消除数值漂移"
        if self.config.use_bspline_smoothing:
            self.smoother = BSplineSmoother(
                degree=self.config.bspline_degree,
                num_control_points=self.config.bspline_num_control_points,
            )
        else:
            self.smoother = None
        
        # ============ 热启动内存 ============
        # 存储最近的最优轨迹用于时间热启动
        # 这实现了类似于同策略强化学习的"短期记忆"
        self.warm_start_cache: Optional[Dict[str, torch.Tensor]] = None
        self.warm_start_timestep: int = 0
    
    def _shift_trajectory_forward(
        self, 
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """
        将轨迹在时间上向前移动一步（时间移位操作）
        
        实现来自同策略强化学习热启动的"移位操作"：
        - 丢弃第一个控制/状态（已执行）
        - 将剩余序列向前移位
        - 根据 shift_mode 填充末尾
        
        参数:
            trajectory: 控制/状态序列 [B, T, D]
            
        返回:
            移位后的轨迹 [B, T, D]
        """
        B, T, D = trajectory.shape
        device = trajectory.device
        dtype = trajectory.dtype
        
        # 移位：移除第一个时间步，追加新的最后一个时间步
        shifted = trajectory[:, 1:, :]  # [B, T-1, D]
        
        # 根据 shift_mode 填充末尾
        if self.config.warm_start_shift_mode == "zero_pad":
            # 追加零（减速/停止）
            padding = torch.zeros(B, 1, D, device=device, dtype=dtype)
        elif self.config.warm_start_shift_mode == "repeat_last":
            # 重复最后一个状态（恒定速度/控制）
            padding = trajectory[:, -1:, :]
        elif self.config.warm_start_shift_mode == "predict":
            # 从最后两个步骤进行线性外推
            if T >= 2:
                last_two = trajectory[:, -2:, :]
                delta = last_two[:, 1:] - last_two[:, 0:1]
                padding = last_two[:, 1:] + delta
            else:
                padding = trajectory[:, -1:, :]
        else:
            raise ValueError(f"未知的 shift_mode: {self.config.warm_start_shift_mode}")
        
        shifted_traj = torch.cat([shifted, padding], dim=1)  # [B, T, D]
        return shifted_traj
    
    def _create_warm_start_prior(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        从缓存的轨迹创建热启动初始状态
        
        实现来自同策略强化学习的"加噪先验"：
        - 从 t-1 获取缓存的最优轨迹
        - 在时间上向前移位
        - 添加受控噪声以保持探索
        
        参数:
            batch_size: 批次大小
            device: 设备
            dtype: 数据类型
            
        返回:
            热启动初始状态 x_0 [B, T, D*3]
        """
        if self.warm_start_cache is None:
            # 无缓存，返回纯高斯噪声
            return torch.randn(
                batch_size, 
                self.config.seq_len, 
                self.config.state_dim * 3,
                device=device, 
                dtype=dtype
            )
        
        # 将缓存的轨迹向前移位
        cached_state = self.warm_start_cache['raw_output']  # [B_cache, T, D*3]
        
        # 处理批次大小不匹配（如需要则重复）
        B_cache = cached_state.shape[0]
        if B_cache < batch_size:
            repeat_factor = (batch_size + B_cache - 1) // B_cache
            cached_state = cached_state.repeat(repeat_factor, 1, 1)[:batch_size]
        elif B_cache > batch_size:
            cached_state = cached_state[:batch_size]
        
        # 在时间上向前移位
        shifted_prior = self._shift_trajectory_forward(cached_state)
        
        # 添加探索噪声（缩放的高斯噪声）
        noise = torch.randn_like(shifted_prior) * self.config.warm_start_noise_scale
        warm_start_x0 = shifted_prior + noise
        
        return warm_start_x0
    
    def update_warm_start_cache(
        self,
        optimal_trajectory: Dict[str, torch.Tensor],
    ):
        """
        使用最新的最优轨迹更新热启动缓存
        
        这应该在 MPPI 优化产生 u*_t 之后调用。
        在完整实现中，这应该由 L1 MPPI 层调用。
        
        参数:
            optimal_trajectory: 包含最优轨迹的字典
                - 必须包含 'raw_output' 键，值为完整状态 [B, T, D*3]
        """
        self.warm_start_cache = {
            'raw_output': optimal_trajectory['raw_output'].detach().clone(),
            'timestep': self.warm_start_timestep,
        }
        self.warm_start_timestep += 1
    
    def reset_warm_start(self):
        """
        重置热启动缓存
        
        在开始新回合或轨迹连续性中断时调用此方法。
        """
        self.warm_start_cache = None
        self.warm_start_timestep = 0
    
    def _create_velocity_fn(
        self,
        start_pos: torch.Tensor,
        goal_pos: torch.Tensor,
        start_vel: Optional[torch.Tensor] = None,
        goal_vel: Optional[torch.Tensor] = None,
        env_encoding: Optional[torch.Tensor] = None,
    ):
        """
        为 ODE 求解器创建速度函数
        
        速度函数包装模型并处理条件化。
        
        参数:
            start_pos: 起始位置 [B, D]
            goal_pos: 目标位置 [B, D]
            start_vel: 起始速度 [B, D] (可选)
            goal_vel: 目标速度 [B, D] (可选，用于 L2 层)
            env_encoding: 环境编码 [B, env_dim] (可选，用于 L2 层)
        """
        def velocity_fn(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """
            计算状态 x_t 和时间 t 处的速度
            
            参数:
                x_t: 当前状态 [B, T, 6] (位置, 速度, 加速度)
                t: 当前时间 [B]
                
            返回:
                速度场 [B, T, 6]
            """
            with torch.no_grad():
                # 调用模型（所有参数都是可选的，模型 forward 方法会处理）
                # FlowMPTransformer 和 FlowMPUNet1D 都支持这些可选参数
                output = self.model(
                    x_t=x_t,
                    t=t,
                    start_pos=start_pos,
                    goal_pos=goal_pos,
                    start_vel=start_vel,
                    goal_vel=goal_vel,
                    env_encoding=env_encoding,
                )
            return output
        
        return velocity_fn
    
    @torch.no_grad()
    def generate(
        self,
        start_pos: torch.Tensor,
        goal_pos: torch.Tensor,
        start_vel: Optional[torch.Tensor] = None,
        goal_vel: Optional[torch.Tensor] = None,
        env_encoding: Optional[torch.Tensor] = None,
        num_samples: int = None,
        return_raw: bool = False,
        warm_start_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        为给定条件生成轨迹
        
        参数:
            start_pos: 起始位置 [B, D]
            goal_pos: 目标位置 [B, D]
            start_vel: 起始速度 [B, D] (可选)
            goal_vel: 目标速度 [B, D] (可选，用于 L2 层)
            env_encoding: 环境编码 [B, env_dim] (可选，用于 L2 层)
            num_samples: 每个条件的样本数
                        当 > 1 时，为每个条件生成多个轨迹
                        输出批次大小变为 B * num_samples
            return_raw: 是否在返回的字典中包含原始（未平滑）轨迹输出
                       如果为 False，仅返回平滑后的轨迹
                        注意：原始输出始终在内部计算（用于热启动），
                        但仅在 return_raw=True 时包含在返回字典中
            
        返回:
            包含以下内容的字典:
                - 'positions': 生成的位置轨迹 [B*num_samples, T, D]
                  当 num_samples=1 时: [B, T, D]
                  当 num_samples>1 时: [B*num_samples, T, D] (L1 的 K 个锚点轨迹)
                - 'velocities': 生成的速度轨迹 [B*num_samples, T, D]
                - 'accelerations': 生成的加速度轨迹 [B*num_samples, T, D]
                - 'raw_output': 原始模型输出 [B*num_samples, T, D*3] (仅当 return_raw=True)
                - 'raw_positions': 平滑前的位置 [B*num_samples, T, D] (仅当 return_raw=True)
                - 'raw_velocities': 平滑前的速度 [B*num_samples, T, D] (仅当 return_raw=True)
                - 'raw_accelerations': 平滑前的加速度 [B*num_samples, T, D] (仅当 return_raw=True)
                
        注意:
            此方法设计用于与 L1 MPPI 层配合：
            - 当 num_samples > 1 时，输出包含 K = B*num_samples 个锚点轨迹
            - L1 层可以直接使用这些作为锚点: initialize_from_l2_output(l2_output)
            - 支持 L2 层的额外参数 (goal_vel, env_encoding) 以消除代码重复
        """
        self.model.eval()
        
        B_original = start_pos.shape[0]  # 原始批次大小（在 num_samples 扩展之前）
        D = self.config.state_dim
        T = self.config.seq_len
        device = start_pos.device
        dtype = start_pos.dtype
        
        num_samples = num_samples or self.config.num_samples
        
        # 处理每个条件的多个样本
        if num_samples > 1:
            start_pos = start_pos.repeat(num_samples, 1)
            goal_pos = goal_pos.repeat(num_samples, 1)
            if start_vel is not None:
                start_vel = start_vel.repeat(num_samples, 1)
            if goal_vel is not None:
                goal_vel = goal_vel.repeat(num_samples, 1)
            if env_encoding is not None:
                env_encoding = env_encoding.repeat(num_samples, 1)
            B = B_original * num_samples  # 扩展后的批次大小
        else:
            B = B_original  # 无需扩展
        
        # ============ 热启动初始状态 ============
        # 采样或初始化初始状态 x_0
        # 优先级:
        #   1) 显式 warm_start_state (来自 L1 / 外部控制器)
        #   2) 内部热启动缓存 (GeneratorConfig.enable_warm_start)
        #   3) 标准 CFM: 纯高斯噪声 N(0, I)
        if warm_start_state is not None:
            # 接受的形状:
            #   [T, D*3] -> 广播到 [B, T, D*3]
            #   [1, T, D*3] -> 广播到 [B, T, D*3]
            #   [B_original, T, D*3] -> 如果 num_samples>1 则重复到 [B_original*num_samples, T, D*3]
            #   [B, T, D*3] -> 直接使用（已匹配扩展后的批次大小）
            ws = warm_start_state
            if ws.dim() == 2:
                ws = ws.unsqueeze(0)
            if ws.shape[-1] != D * 3 or ws.shape[-2] != T:
                raise ValueError(
                    f"warm_start_state 必须具有形状 [*, {T}, {D*3}], "
                    f"得到 {tuple(ws.shape)}"
                )
            ws = ws.to(device=device, dtype=dtype)
            
            # 处理不同的输入形状
            ws_batch_size = ws.shape[0]
            
            # 情况 1: 单个轨迹 [1, T, D*3] -> 广播到完整批次
            if ws_batch_size == 1 and B > 1:
                ws = ws.repeat(B, 1, 1)
            # 情况 2: 每个条件的热启动 [B_original, T, D*3] -> 为 num_samples 扩展
            elif ws_batch_size == B_original and num_samples > 1:
                ws = ws.repeat_interleave(num_samples, dim=0)
            # 情况 3: 已匹配扩展后的批次大小 [B, T, D*3] -> 直接使用
            elif ws_batch_size == B:
                pass  # 形状已正确
            # 情况 4: 无效形状
            else:
                raise ValueError(
                    f"不兼容的 warm_start_state 批次大小 {ws_batch_size}。 "
                    f"期望以下之一: 1, {B_original} (每个条件), 或 {B} (完整批次)。 "
                    f"对于 B_original={B_original}, num_samples={num_samples}, B={B} 得到形状 {tuple(ws.shape)}"
                )
            x_0 = ws
        elif self.config.enable_warm_start:
            # 使用内部热启动缓存: 移位先验 + 噪声
            x_0 = self._create_warm_start_prior(B, device, dtype)
        else:
            # 标准 CFM: 从 N(0, I) 采样
            # 状态有 6 个通道: 位置(2) + 速度(2) + 加速度(2)
            x_0 = torch.randn(B, T, D * 3, device=device, dtype=dtype)
        
        # 创建速度函数（支持 L2 层的额外参数）
        velocity_fn = self._create_velocity_fn(
            start_pos, goal_pos, start_vel, goal_vel, env_encoding
        )
        
        # 求解 ODE
        x_1 = self.solver.solve(velocity_fn, x_0)
        
        # 提取组件
        positions_raw = x_1[..., :D]
        velocities_raw = x_1[..., D:D*2]
        accelerations_raw = x_1[..., D*2:D*3]
        
        result = {}
        
        # 如果启用则应用平滑
        if self.smoother is not None:
            smoothed = self.smoother.smooth(positions_raw, return_derivatives=True)
            result['positions'] = smoothed['positions']
            result['velocities'] = smoothed['velocities']
            result['accelerations'] = smoothed['accelerations']
        else:
            result['positions'] = positions_raw
            result['velocities'] = velocities_raw
            result['accelerations'] = accelerations_raw
        
        # 根据 return_raw 参数有条件地存储原始输出
        # 原始输出始终计算（平滑需要），但仅在 return_raw=True 时包含在返回字典中
        # 这允许用户控制输出大小
        # 注意: 对于热启动功能，用户应使用来自先前生成的 raw_output 调用 update_warm_start_cache()
        # (当 return_raw=True 时)
        if return_raw:
            result['raw_positions'] = positions_raw
            result['raw_velocities'] = velocities_raw
            result['raw_accelerations'] = accelerations_raw
            result['raw_output'] = x_1
        
        return result
    
    @torch.no_grad()
    def generate_with_guidance(
        self,
        start_pos: torch.Tensor,
        goal_pos: torch.Tensor,
        start_vel: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        obstacle_fn: Optional[callable] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        使用分类器自由引导生成轨迹
        
        允许将生成引导到期望的属性，如避障
        
        参数:
            start_pos: 起始位置 [B, D]
            goal_pos: 目标位置 [B, D]
            start_vel: 起始速度 [B, D]
            guidance_scale: 引导尺度 (1.0 = 无引导)
            obstacle_fn: 返回避障梯度的函数
            
        返回:
            包含以下内容的字典:
                - 'positions': 生成的位置轨迹 [B, T, D]
                - 'velocities': 生成的速度轨迹 [B, T, D]
                - 'accelerations': 生成的加速度轨迹 [B, T, D]
                - 'raw_output': 原始模型输出 [B, T, D*3] (与 generate() 保持一致)
        """
        # 注意: 完整的 CFG 需要训练时使用条件丢弃的模型
        # 这是带有可选避障引导的简化版本
        
        self.model.eval()
        
        B = start_pos.shape[0]
        D = self.config.state_dim
        T = self.config.seq_len
        device = start_pos.device
        dtype = start_pos.dtype
        
        x_0 = torch.randn(B, T, D * 3, device=device, dtype=dtype)
        
        def guided_velocity_fn(x_t, t):
            # 条件速度
            v_cond = self.model(
                x_t=x_t,
                t=t,
                start_pos=start_pos,
                goal_pos=goal_pos,
                start_vel=start_vel,
            )
            
            # 如果提供则添加避障梯度
            if obstacle_fn is not None and guidance_scale != 1.0:
                x_t_clone = x_t.clone().requires_grad_(True)
                obstacle_cost = obstacle_fn(x_t_clone[..., :D])
                
                if obstacle_cost.requires_grad:
                    grad = torch.autograd.grad(
                        obstacle_cost.sum(),
                        x_t_clone,
                        create_graph=False
                    )[0]
                    
                    # 应用引导
                    v_cond = v_cond - guidance_scale * grad
            
            return v_cond
        
        x_1 = self.solver.solve(guided_velocity_fn, x_0)
        
        # 提取并可选地平滑
        positions_raw = x_1[..., :D]
        
        result = {}
        
        if self.smoother is not None:
            smoothed = self.smoother.smooth(positions_raw, return_derivatives=True)
            result['positions'] = smoothed['positions']
            result['velocities'] = smoothed['velocities']
            result['accelerations'] = smoothed['accelerations']
        else:
            result['positions'] = positions_raw
            result['velocities'] = x_1[..., D:D*2]
            result['accelerations'] = x_1[..., D*2:D*3]
        
        # 包含原始输出以与 generate() 方法保持一致
        result['raw_output'] = x_1
        result['raw_positions'] = positions_raw
        result['raw_velocities'] = x_1[..., D:D*2]
        result['raw_accelerations'] = x_1[..., D*2:D*3]
        
        return result
    
    @torch.no_grad()
    def generate_batch(
        self,
        conditions: List[Dict[str, torch.Tensor]],
        batch_size: int = 32,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        批量生成多个条件的轨迹
        
        参数:
            conditions: 条件字典列表
            batch_size: 推理的最大批次大小
            
        返回:
            轨迹字典列表
        """
        results = []
        
        for i in range(0, len(conditions), batch_size):
            batch_conds = conditions[i:i+batch_size]
            
            # 堆叠条件
            start_pos = torch.stack([c['start_pos'] for c in batch_conds])
            goal_pos = torch.stack([c['goal_pos'] for c in batch_conds])
            
            start_vel = None
            if 'start_vel' in batch_conds[0]:
                start_vel = torch.stack([c['start_vel'] for c in batch_conds])
            
            # 生成
            batch_result = self.generate(start_pos, goal_pos, start_vel)
            
            # 拆分结果
            for j in range(len(batch_conds)):
                result = {
                    'positions': batch_result['positions'][j],
                    'velocities': batch_result['velocities'][j],
                    'accelerations': batch_result['accelerations'][j],
                }
                results.append(result)
        
        return results


def create_8step_schedule() -> List[float]:
    """
    创建实现策略中指定的 8 步非均匀时间调度
    
    此调度在早期使用较大步长（粗生成），在接近 t=1 时使用较小步长（细粒度细化）
    以在最终阶段保留更多细节
    
    返回:
        时间值列表: [0.0, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
    """
    return [0.0, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]


def compute_trajectory_metrics(
    generated: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor] = None,
) -> Dict[str, float]:
    """
    计算生成轨迹的指标
    
    参数:
        generated: 生成的轨迹字典
        target: 真实轨迹字典 (可选)
        
    返回:
        指标值字典
    """
    metrics = {}
    
    positions = generated['positions']
    velocities = generated['velocities']
    accelerations = generated['accelerations']
    
    # 平滑度指标（越小越好）
    # 急动度: 加速度的变化率
    if positions.dim() == 3:
        jerk = torch.diff(accelerations, dim=1)
        metrics['avg_jerk'] = jerk.norm(dim=-1).mean().item()
        
        # 曲率变化
        vel_norm = velocities.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        curvature = (velocities[..., 0] * accelerations[..., 1] - 
                    velocities[..., 1] * accelerations[..., 0]) / (vel_norm.squeeze(-1) ** 3)
        metrics['curvature_var'] = curvature.var(dim=1).mean().item()
    
    # 如果提供目标，计算误差
    if target is not None:
        # 位置误差
        pos_error = (generated['positions'] - target['positions']).norm(dim=-1)
        metrics['pos_mse'] = pos_error.pow(2).mean().item()
        metrics['pos_mae'] = pos_error.mean().item()
        
        # 目标到达误差（最终位置）
        goal_error = (generated['positions'][:, -1] - target['positions'][:, -1]).norm(dim=-1)
        metrics['goal_error'] = goal_error.mean().item()
        
        # 速度误差
        if 'velocities' in target:
            vel_error = (generated['velocities'] - target['velocities']).norm(dim=-1)
            metrics['vel_mse'] = vel_error.pow(2).mean().item()
    
    return metrics
