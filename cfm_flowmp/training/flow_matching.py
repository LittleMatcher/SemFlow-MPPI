"""
流匹配训练逻辑

实现用于轨迹生成的核心流匹配算法：
1. 插值路径构建（FlowMP 的公式 6, 8, 10）
2. 目标场计算
3. 流匹配损失

参考: FlowMP 论文中条件概率路径的公式。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FlowMatchingConfig:
    """流匹配训练配置"""
    
    # 状态维度
    state_dim: int = 2  # 位置维度 (x, y)
    
    # 损失权重
    lambda_vel: float = 1.0      # 速度场损失的权重
    lambda_acc: float = 1.0      # 加速度场损失的权重
    lambda_jerk: float = 1.0     # 急动场损失的权重
    
    # 插值参数
    sigma_min: float = 1e-4      # t=1 处的最小噪声尺度
    
    # 可选: 每个状态分量的不同 sigma
    sigma_pos: float = 1e-4      # 位置的噪声尺度
    sigma_vel: float = 1e-4      # 速度的噪声尺度
    sigma_acc: float = 1e-4      # 加速度的噪声尺度
    
    # 时间采样
    t_min: float = 0.0           # 最小流时间
    t_max: float = 1.0           # 最大流时间
    

class FlowInterpolator:
    """
    构建流匹配的插值路径
    
    实现条件流匹配中使用的条件概率路径 p_t(x|x_1)。
    在 t=0 时，分布为 N(0, I)，
    在 t=1 时，分布集中在目标 x_1 处。
    
    插值遵循:
        x_t = t * x_1 + (1 - t) * epsilon
        
    其中 epsilon ~ N(0, I)，x_1 是目标轨迹。
    
    对于 FlowMP，我们有三个耦合的插值:
        q_t = interpolate(q_0, q_1, t)     # 位置
        q_dot_t = interpolate(q_dot_0, q_dot_1, t)    # 速度
        q_ddot_t = interpolate(q_ddot_0, q_ddot_1, t)  # 加速度
    """
    
    def __init__(self, config: FlowMatchingConfig = None):
        """
        参数:
            config: 流匹配配置
        """
        self.config = config or FlowMatchingConfig()
    
    def sample_time(
        self, 
        batch_size: int, 
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        从 [t_min, t_max] 均匀采样流时间 t
        
        参数:
            batch_size: 样本数
            device: 目标设备
            dtype: 目标数据类型
            
        返回:
            形状为 [batch_size] 的时间值
        """
        t = torch.rand(batch_size, device=device, dtype=dtype)
        t = self.config.t_min + t * (self.config.t_max - self.config.t_min)
        return t
    
    def sample_noise(
        self,
        target_shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        采样高斯噪声 epsilon ~ N(0, I)
        
        参数:
            target_shape: 噪声张量的形状
            device: 目标设备
            dtype: 目标数据类型
            
        返回:
            给定形状的噪声张量
        """
        return torch.randn(target_shape, device=device, dtype=dtype)
    
    def interpolate_simple(
        self,
        x_0: torch.Tensor,  # Noise
        x_1: torch.Tensor,  # Target
        t: torch.Tensor,    # Time
    ) -> torch.Tensor:
        """
        简单线性插值: x_t = t * x_1 + (1 - t) * x_0
        
        这是修正流/流匹配中使用的标准 OT（最优传输）路径
        
        参数:
            x_0: 初始状态（噪声），形状 [B, ...]
            x_1: 目标状态，形状 [B, ...]
            t: 时间值，形状 [B] 或 [B, 1, ...]
            
        返回:
            插值状态 x_t
        """
        # 扩展 t 以进行广播
        while t.dim() < x_0.dim():
            t = t.unsqueeze(-1)
        
        return t * x_1 + (1 - t) * x_0
    
    def compute_target_velocity(
        self,
        x_0: torch.Tensor,  # Noise (or x_t)
        x_1: torch.Tensor,  # Target
        t: torch.Tensor,    # Time
        from_interpolated: bool = True,
    ) -> torch.Tensor:
        """
        计算目标速度场
        
        对于 OT 路径 x_t = t * x_1 + (1-t) * x_0:
            dx_t/dt = x_1 - x_0
        
        或者等价地，用 x_t 表示:
            v_target = (x_1 - x_t) / (1 - t)
        
        参数:
            x_0: 初始状态（噪声）或插值状态
            x_1: 目标状态
            t: 时间值
            from_interpolated: 如果为 True，x_0 实际上是 x_t（插值后的）
            
        返回:
            目标速度场
        """
        # 扩展 t 以进行广播
        while t.dim() < x_1.dim():
            t = t.unsqueeze(-1)
        
        if from_interpolated:
            # x_0 实际上是 x_t
            x_t = x_0
            # v = (x_1 - x_t) / (1 - t)
            # 添加小的 epsilon 以避免在 t=1 时除以零
            eps = 1e-6
            v_target = (x_1 - x_t) / (1 - t + eps)
        else:
            # 简单形式: v = x_1 - x_0
            v_target = x_1 - x_0
        
        return v_target
    
    def interpolate_trajectory(
        self,
        q_1: torch.Tensor,       # Target position [B, T, 2]
        q_dot_1: torch.Tensor,   # Target velocity [B, T, 2]
        q_ddot_1: torch.Tensor,  # Target acceleration [B, T, 2]
        t: torch.Tensor,         # Flow time [B]
        epsilon_q: torch.Tensor = None,      # Position noise
        epsilon_q_dot: torch.Tensor = None,  # Velocity noise
        epsilon_q_ddot: torch.Tensor = None, # Acceleration noise
    ) -> Dict[str, torch.Tensor]:
        """
        在流时间 t 处构建插值轨迹状态
        
        同时实现位置、速度和加速度的 FlowMP 插值（公式 6, 8, 10）
        
        插值遵循最优传输路径:
            x_t = t * x_1 + (1 - t) * epsilon
        
        目标速度场为:
            u_target = (x_1 - x_t) / (1 - t)
        
        注意: 对于接近 1 的 t，我们使用数值稳定化
        
        参数:
            q_1: 目标位置轨迹 [B, T, state_dim]
            q_dot_1: 目标速度轨迹 [B, T, state_dim]
            q_ddot_1: 目标加速度轨迹 [B, T, state_dim]
            t: 流时间值 [B]
            epsilon_*: 可选的预采样噪声张量
            
        返回:
            包含以下内容的字典:
                - 'q_t': 插值位置 [B, T, state_dim]
                - 'q_dot_t': 插值速度 [B, T, state_dim]
                - 'q_ddot_t': 插值加速度 [B, T, state_dim]
                - 'u_target': 位置速度场目标 [B, T, state_dim]
                - 'v_target': 速度加速度场目标 [B, T, state_dim]
                - 'w_target': 加速度急动场目标 [B, T, state_dim]
                - 'x_t': 拼接状态 [B, T, state_dim * 3]
                - 'target': 拼接目标场 [B, T, state_dim * 3]
        """
        B, T, D = q_1.shape
        device = q_1.device
        dtype = q_1.dtype
        
        # 如果未提供则采样噪声
        if epsilon_q is None:
            epsilon_q = self.sample_noise((B, T, D), device, dtype)
        if epsilon_q_dot is None:
            epsilon_q_dot = self.sample_noise((B, T, D), device, dtype)
        if epsilon_q_ddot is None:
            epsilon_q_ddot = self.sample_noise((B, T, D), device, dtype)
        
        # 扩展 t 以进行广播: [B] -> [B, 1, 1]
        t_expanded = t[:, None, None]
        
        # ============ 插值状态（公式 6, 8, 10）============
        # q_t = t * q_1 + (1 - t) * epsilon_q
        q_t = t_expanded * q_1 + (1 - t_expanded) * epsilon_q
        
        # q_dot_t = t * q_dot_1 + (1 - t) * epsilon_q_dot
        q_dot_t = t_expanded * q_dot_1 + (1 - t_expanded) * epsilon_q_dot
        
        # q_ddot_t = t * q_ddot_1 + (1 - t) * epsilon_q_ddot
        q_ddot_t = t_expanded * q_ddot_1 + (1 - t_expanded) * epsilon_q_ddot
        
        # ============ 计算目标场 ============
        # 根据 FlowMP 算法 1:
        # u_target = (q_1 - q_t) / (1 - t)
        # v_target = (q_dot_1 - q_dot_t) / (1 - t)
        # w_target = (q_ddot_1 - q_ddot_t) / (1 - t)
        #
        # 注意: 由于 q_t = t * q_1 + (1-t) * epsilon,
        #       (q_1 - q_t) / (1-t) = q_1 - epsilon
        # 我们使用显式 (x_1 - x_t) / (1-t) 形式以保持一致性
        
        # 小的 epsilon 以避免在 t 接近 1 时除以零
        eps = 1e-6
        one_minus_t = (1 - t_expanded).clamp(min=eps)
        
        u_target = (q_1 - q_t) / one_minus_t
        v_target = (q_dot_1 - q_dot_t) / one_minus_t
        w_target = (q_ddot_1 - q_ddot_t) / one_minus_t
        
        # ============ 为网络输入/输出拼接 ============
        # 输入状态: [位置, 速度, 加速度] -> [B, T, 6]
        x_t = torch.cat([q_t, q_dot_t, q_ddot_t], dim=-1)
        
        # 目标场: [u, v, w] -> [B, T, 6]
        target = torch.cat([u_target, v_target, w_target], dim=-1)
        
        return {
            'q_t': q_t,
            'q_dot_t': q_dot_t,
            'q_ddot_t': q_ddot_t,
            'u_target': u_target,
            'v_target': v_target,
            'w_target': w_target,
            'x_t': x_t,
            'target': target,
            'epsilon_q': epsilon_q,
            'epsilon_q_dot': epsilon_q_dot,
            'epsilon_q_ddot': epsilon_q_ddot,
            't': t,
        }


class FlowMatchingLoss(nn.Module):
    """
    FlowMP 的流匹配损失
    
    计算预测和目标向量场之间的加权 MSE 损失:
    
    L = ||u_pred - u_target||^2 
        + λ_acc * ||v_pred - v_target||^2 
        + λ_jerk * ||w_pred - w_target||^2
    
    其中:
        - u: 速度场（用于位置）
        - v: 加速度场（用于速度）
        - w: 急动场（用于加速度）
    """
    
    def __init__(self, config: FlowMatchingConfig = None):
        """
        参数:
            config: 带有损失权重的流匹配配置
        """
        super().__init__()
        self.config = config or FlowMatchingConfig()
        self.interpolator = FlowInterpolator(config)
    
    def forward(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        reduction: str = 'mean',
    ) -> Dict[str, torch.Tensor]:
        """
        计算流匹配损失
        
        参数:
            model_output: 预测的向量场 [B, T, 6]
            target: 目标向量场 [B, T, 6]
            reduction: 损失归约方法 ('mean', 'sum', 'none')
            
        返回:
            包含以下内容的字典:
                - 'loss': 总加权损失
                - 'loss_vel': 速度场损失 (u)
                - 'loss_acc': 加速度场损失 (v)
                - 'loss_jerk': 急动场损失 (w)
        """
        D = self.config.state_dim
        
        # 提取场分量
        u_pred = model_output[..., :D]
        v_pred = model_output[..., D:D*2]
        w_pred = model_output[..., D*2:D*3]
        
        u_target = target[..., :D]
        v_target = target[..., D:D*2]
        w_target = target[..., D*2:D*3]
        
        # 计算每个场的 MSE
        loss_vel = F.mse_loss(u_pred, u_target, reduction=reduction)
        loss_acc = F.mse_loss(v_pred, v_target, reduction=reduction)
        loss_jerk = F.mse_loss(w_pred, w_target, reduction=reduction)
        
        # 加权和
        total_loss = (
            self.config.lambda_vel * loss_vel +
            self.config.lambda_acc * loss_acc +
            self.config.lambda_jerk * loss_jerk
        )
        
        return {
            'loss': total_loss,
            'loss_vel': loss_vel,
            'loss_acc': loss_acc,
            'loss_jerk': loss_jerk,
        }
    
    def compute_training_loss(
        self,
        model: nn.Module,
        q_1: torch.Tensor,
        q_dot_1: torch.Tensor,
        q_ddot_1: torch.Tensor,
        start_pos: torch.Tensor,
        goal_pos: torch.Tensor,
        start_vel: torch.Tensor = None,
        t: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        完整的训练损失计算，包括插值
        
        此方法处理完整的训练流程:
        1. 采样流时间 t
        2. 采样噪声
        3. 构建插值状态
        4. 计算目标场
        5. 运行模型前向传播
        6. 计算损失
        
        参数:
            model: FlowMP transformer 模型
            q_1: 目标位置轨迹 [B, T, D]
            q_dot_1: 目标速度轨迹 [B, T, D]
            q_ddot_1: 目标加速度轨迹 [B, T, D]
            start_pos: 起始位置 [B, D]
            goal_pos: 目标位置 [B, D]
            start_vel: 起始速度 [B, D] (可选)
            t: 预采样时间 (可选，如果为 None 则采样)
            
        返回:
            损失字典
        """
        B = q_1.shape[0]
        device = q_1.device
        dtype = q_1.dtype
        
        # 如果未提供则采样流时间
        if t is None:
            t = self.interpolator.sample_time(B, device, dtype)
        
        # 构建插值轨迹和目标
        interp_result = self.interpolator.interpolate_trajectory(
            q_1=q_1,
            q_dot_1=q_dot_1,
            q_ddot_1=q_ddot_1,
            t=t,
        )
        
        x_t = interp_result['x_t']
        target = interp_result['target']
        
        # 模型前向传播
        model_output = model(
            x_t=x_t,
            t=t,
            start_pos=start_pos,
            goal_pos=goal_pos,
            start_vel=start_vel,
        )
        
        # 计算损失
        loss_dict = self.forward(model_output, target)
        
        # 添加插值信息用于调试
        loss_dict['t'] = t.mean()
        
        return loss_dict


class VelocityConsistencyLoss(nn.Module):
    """
    用于物理一致性的可选辅助损失
    
    鼓励速度场与位置场的时间导数保持一致
    """
    
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
    
    def forward(
        self,
        q_pred: torch.Tensor,
        q_dot_pred: torch.Tensor,
        dt: float = 0.1,
    ) -> torch.Tensor:
        """
        计算速度一致性损失
        
        参数:
            q_pred: 预测位置 [B, T, D]
            q_dot_pred: 预测速度 [B, T, D]
            dt: 轨迹点之间的时间步长
            
        返回:
            一致性损失标量
        """
        # 有限差分速度: dq/dt ≈ (q[t+1] - q[t]) / dt
        q_diff = (q_pred[:, 1:, :] - q_pred[:, :-1, :]) / dt
        
        # 与预测速度比较（排除最后一步）
        q_dot_mid = (q_dot_pred[:, 1:, :] + q_dot_pred[:, :-1, :]) / 2
        
        loss = F.mse_loss(q_diff, q_dot_mid)
        return self.weight * loss
