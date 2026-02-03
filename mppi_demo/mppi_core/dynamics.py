"""
机器人动力学模型
- 差分驱动
- 简单积分器（用于比较）
"""
import numpy as np
from typing import Tuple


class DifferentialDriveRobot:
    """
    差分驱动机器人，状态为 [x, y, theta, v, omega]
    - (x, y): 位置
    - theta: 航向角
    - v: 线速度
    - omega: 角速度
    """
    
    def __init__(self, dt: float = 0.1, 
                 max_v: float = 2.0, max_omega: float = np.pi,
                 max_accel: float = 1.0, max_alpha: float = np.pi/2):
        """
        Args:
            dt: 时间步长
            max_v: 最大线速度
            max_omega: 最大角速度
            max_accel: 最大线加速度
            max_alpha: 最大角加速度
        """
        self.dt = dt
        self.max_v = max_v
        self.max_omega = max_omega
        self.max_accel = max_accel
        self.max_alpha = max_alpha
        
        self.state_dim = 5  # [x, y, theta, v, omega]
        self.control_dim = 2  # [accel, alpha]
        
    def step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """前向动力学：计算下一状态
        Args:
            state: 形状 (..., 5) [x, y, theta, v, omega]
            control: 形状 (..., 2) [accel, alpha]
        Returns:
            next_state: 形状 (..., 5)
        """
        x, y, theta, v, omega = np.split(state, 5, axis=-1)
        accel, alpha = np.split(control, 2, axis=-1)
        
        # 限制控制输入
        accel = np.clip(accel, -self.max_accel, self.max_accel)
        alpha = np.clip(alpha, -self.max_alpha, self.max_alpha)
        
        # 更新速度
        v_next = v + accel * self.dt
        omega_next = omega + alpha * self.dt
        
        # 限制速度
        v_next = np.clip(v_next, -self.max_v, self.max_v)
        omega_next = np.clip(omega_next, -self.max_omega, self.max_omega)
        
        # 使用当前速度更新位置和方向
        x_next = x + v * np.cos(theta) * self.dt
        y_next = y + v * np.sin(theta) * self.dt
        theta_next = theta + omega * self.dt
        
        # 将 theta 限制到 [-pi, pi]
        theta_next = np.arctan2(np.sin(theta_next), np.cos(theta_next))
        
        next_state = np.concatenate([
            x_next, y_next, theta_next, v_next, omega_next
        ], axis=-1)
        
        return next_state
    
    def rollout(self, initial_state: np.ndarray, 
                controls: np.ndarray) -> np.ndarray:
        """给定控制序列进行轨迹展开
        Args:
            initial_state: 形状 (..., 5) 或 (5,)
            controls: 形状 (..., T, 2) 或 (T, 2)
        Returns:
            states: 形状 (..., T+1, 5)
        """
        # 处理批量和单个轨迹
        if initial_state.ndim == 1:
            initial_state = initial_state[np.newaxis, :]
            controls = controls[np.newaxis, :]
            squeeze = True
        else:
            squeeze = False
        
        batch_size = initial_state.shape[0]
        T = controls.shape[1]
        
        states = np.zeros((batch_size, T + 1, self.state_dim))
        states[:, 0, :] = initial_state
        
        for t in range(T):
            states[:, t + 1, :] = self.step(states[:, t, :], controls[:, t, :])
        
        return states[0] if squeeze else states
    
    def get_position(self, state: np.ndarray) -> np.ndarray:
        """从状态中提取位置
        Args:
            state: 形状 (..., 5)
        Returns:
            position: 形状 (..., 2) [x, y]
        """
        return state[..., :2]
    
    def create_state(self, x: float, y: float, theta: float = 0.0,
                    v: float = 0.0, omega: float = 0.0) -> np.ndarray:
        """创建状态向量的辅助函数"""
        return np.array([x, y, theta, v, omega])


class SimpleIntegratorRobot:
    """
    简单双积分器机器人，状态为 [x, y, vx, vy]
    - (x, y): 位置  
    - (vx, vy): 速度
    
    用于比较和调试
    """
    
    def __init__(self, dt: float = 0.1,
                 max_vel: float = 2.0, max_accel: float = 1.0):
        """
        Args:
            dt: 时间步长
            max_vel: 最大速度大小
            max_accel: 最大加速度大小
        """
        self.dt = dt
        self.max_vel = max_vel
        self.max_accel = max_accel
        
        self.state_dim = 4  # [x, y, vx, vy]
        self.control_dim = 2  # [ax, ay]
        
    def step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """前向动力学
        Args:
            state: 形状 (..., 4) [x, y, vx, vy]
            control: 形状 (..., 2) [ax, ay]
        Returns:
            next_state: 形状 (..., 4)
        """
        x, y, vx, vy = np.split(state, 4, axis=-1)
        ax, ay = np.split(control, 2, axis=-1)
        
        # 限制控制输入
        accel_mag = np.sqrt(ax**2 + ay**2)
        scale = np.where(accel_mag > self.max_accel, 
                        self.max_accel / (accel_mag + 1e-8), 1.0)
        ax = ax * scale
        ay = ay * scale
        
        # 更新速度
        vx_next = vx + ax * self.dt
        vy_next = vy + ay * self.dt
        
        # 限制速度
        vel_mag = np.sqrt(vx_next**2 + vy_next**2)
        scale = np.where(vel_mag > self.max_vel,
                        self.max_vel / (vel_mag + 1e-8), 1.0)
        vx_next = vx_next * scale
        vy_next = vy_next * scale
        
        # 更新位置
        x_next = x + vx * self.dt
        y_next = y + vy * self.dt
        
        next_state = np.concatenate([
            x_next, y_next, vx_next, vy_next
        ], axis=-1)
        
        return next_state
    
    def rollout(self, initial_state: np.ndarray,
                controls: np.ndarray) -> np.ndarray:
        """给定控制序列进行轨迹展开
        Args:
            initial_state: 形状 (..., 4) 或 (4,)
            controls: 形状 (..., T, 2) 或 (T, 2)
        Returns:
            states: 形状 (..., T+1, 4)
        """
        if initial_state.ndim == 1:
            initial_state = initial_state[np.newaxis, :]
            controls = controls[np.newaxis, :]
            squeeze = True
        else:
            squeeze = False
        
        batch_size = initial_state.shape[0]
        T = controls.shape[1]
        
        states = np.zeros((batch_size, T + 1, self.state_dim))
        states[:, 0, :] = initial_state
        
        for t in range(T):
            states[:, t + 1, :] = self.step(states[:, t, :], controls[:, t, :])
        
        return states[0] if squeeze else states
    
    def get_position(self, state: np.ndarray) -> np.ndarray:
        """从状态中提取位置"""
        return state[..., :2]
    
    def create_state(self, x: float, y: float,
                    vx: float = 0.0, vy: float = 0.0) -> np.ndarray:
        """创建状态向量的辅助函数"""
        return np.array([x, y, vx, vy])
