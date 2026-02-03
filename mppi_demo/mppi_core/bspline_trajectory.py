"""
B-Spline 轨迹参数化
参考 Motion Planning Diffusion 方法 (arXiv:2308.01557)
- 光滑轨迹 (C2 连续)
- 比原始控制序列参数更少
"""
import numpy as np
from scipy import interpolate
from typing import Optional


class BSplineTrajectory:
    """B-Spline 轨迹表示"""
    
    def __init__(self, degree: int = 3, n_control_points: int = 10,
                 time_horizon: float = 5.0, dim: int = 2):
        """
        Args:
            degree: B-spline 度数 (3 表示三次，保证 C2 连续性)
            n_control_points: 控制点数量
            time_horizon: 总时间长度
            dim: 轨迹维度 (2 表示 2D 位置)
        """
        self.degree = degree
        self.n_control_points = n_control_points
        self.time_horizon = time_horizon
        self.dim = dim
        
        # 创建均匀节点向量
        # 对于开放均匀 B-spline: 首尾节点重复 (degree+1) 次
        n_knots = n_control_points + degree + 1
        self.knots = self._create_knot_vector(n_knots, degree)
        
    def _create_knot_vector(self, n_knots: int, degree: int) -> np.ndarray:
        """创建开放均匀节点向量
        对于开放均匀 B-spline: n_knots = n_control_points + degree + 1
        """
        n_internal = n_knots - 2 * (degree + 1)
        if n_internal < 0:
            # 对于少量控制点，使用夹紧节点向量
            knots = np.concatenate([
                np.zeros(degree + 1),
                np.ones(degree + 1)
            ])
        elif n_internal == 0:
            # 无内部节点
            knots = np.concatenate([
                np.zeros(degree + 1),
                np.ones(degree + 1)
            ])
        else:
            # 内部节点（排除 0 和 1，因为它们已在夹紧端点中）
            knots = np.concatenate([
                np.zeros(degree + 1),
                np.linspace(0, 1, n_internal + 2)[1:-1],  # 排除端点
                np.ones(degree + 1)
            ])
        return knots
    
    def evaluate(self, control_points: np.ndarray, 
                 n_samples: int = 100) -> np.ndarray:
        """计算 B-spline 轨迹
        Args:
            control_points: 形状 (n_control_points, dim)
            n_samples: 沿轨迹采样的点数
        Returns:
            trajectory: 形状 (n_samples, dim)
        """
        t_eval = np.linspace(0, 1, n_samples)
        
        # 使用 scipy 的 BSpline
        traj = []
        for d in range(self.dim):
            spline = interpolate.BSpline(
                self.knots, control_points[:, d], self.degree
            )
            traj.append(spline(t_eval))
        
        return np.stack(traj, axis=-1)
    
    def evaluate_derivatives(self, control_points: np.ndarray,
                            n_samples: int = 100,
                            max_derivative: int = 2) -> list:
        """计算 B-spline 及其导数
        Args:
            control_points: 形状 (n_control_points, dim)
            n_samples: 采样点数
            max_derivative: 最大导数阶数 (0=位置, 1=速度, 2=加速度)
        Returns:
            derivatives: 数组列表 [位置, 速度, 加速度, ...]
                        每个形状 (n_samples, dim)
        """
        t_eval = np.linspace(0, 1, n_samples)
        
        derivatives = [[] for _ in range(max_derivative + 1)]
        
        for d in range(self.dim):
            spline = interpolate.BSpline(
                self.knots, control_points[:, d], self.degree
            )
            
            for order in range(max_derivative + 1):
                if order == 0:
                    derivatives[order].append(spline(t_eval))
                else:
                    # 按时间缩放的导数
                    derivatives[order].append(
                        spline.derivative(order)(t_eval) / (self.time_horizon ** order)
                    )
        
        # 沿最后一个维度堆叠
        return [np.stack(d, axis=-1) for d in derivatives]
    
    def fit_trajectory(self, waypoints: np.ndarray) -> np.ndarray:
        """将 B-spline 控制点拟合到路径点
        Args:
            waypoints: 形状 (n_waypoints, dim)
        Returns:
            control_points: 形状 (n_control_points, dim)
        """
        n_waypoints = len(waypoints)
        t_waypoints = np.linspace(0, 1, n_waypoints)
        
        control_points = []
        for d in range(self.dim):
            # 将样条拟合到路径点
            tck, _ = interpolate.splprep(
                [waypoints[:, d]], 
                s=0,  # 精确插值
                k=min(self.degree, n_waypoints - 1),
                u=t_waypoints
            )
            
            # 重采样以获得所需数量的控制点
            u_control = np.linspace(0, 1, self.n_control_points)
            control_pts = interpolate.splev(u_control, tck)[0]
            control_points.append(control_pts)
        
        return np.stack(control_points, axis=-1)
    
    def sample_random_control_points(self, start: np.ndarray, 
                                    goal: np.ndarray,
                                    bounds: tuple,
                                    n_samples: int = 1) -> np.ndarray:
        """采样随机 B-spline 控制点
        Args:
            start: 起始位置，形状 (dim,)
            goal: 目标位置，形状 (dim,)
            bounds: (min_x, max_x, min_y, max_y)
            n_samples: 要采样的轨迹数
        Returns:
            control_points: 形状 (n_samples, n_control_points, dim)
        """
        min_x, max_x, min_y, max_y = bounds
        
        control_points = np.zeros((n_samples, self.n_control_points, self.dim))
        
        # 第一个和最后一个控制点固定在起始和目标
        control_points[:, 0, :] = start
        control_points[:, -1, :] = goal
        
        # 中间控制点随机采样
        control_points[:, 1:-1, 0] = np.random.uniform(
            min_x, max_x, (n_samples, self.n_control_points - 2)
        )
        control_points[:, 1:-1, 1] = np.random.uniform(
            min_y, max_y, (n_samples, self.n_control_points - 2)
        )
        
        return control_points
    
    def add_noise(self, control_points: np.ndarray, 
                  noise_std: float) -> np.ndarray:
        """向控制点添加高斯噪声（除了起始/目标）
        Args:
            control_points: 形状 (n_control_points, dim) 或 
                          (n_samples, n_control_points, dim)
            noise_std: 噪声标准差
        Returns:
            noisy_control_points: 与输入相同的形状
        """
        noise = np.random.randn(*control_points.shape) * noise_std
        
        # 不扰动起始和目标
        if control_points.ndim == 2:
            noise[0, :] = 0
            noise[-1, :] = 0
        else:
            noise[:, 0, :] = 0
            noise[:, -1, :] = 0
        
        return control_points + noise
