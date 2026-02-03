"""
GPU加速的MPPI实现
利用PyTorch在RTX 5090上并行模拟数千个轨迹
"""
import torch
import numpy as np
from typing import Optional, Tuple, Dict
from .cost_functions import CostFunction


class BSplineGPU:
    """GPU加速的B-Spline轨迹生成"""
    
    def __init__(self, degree: int = 3, n_control_points: int = 10,
                 time_horizon: float = 5.0, dim: int = 2, device: str = 'cuda'):
        self.degree = degree
        self.n_control_points = n_control_points
        self.time_horizon = time_horizon
        self.dim = dim
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 创建节点向量
        n_knots = n_control_points + degree + 1
        self.knots = self._create_knot_vector(n_knots, degree)
        
    def _create_knot_vector(self, n_knots: int, degree: int) -> torch.Tensor:
        """创建开放均匀节点向量"""
        n_internal = n_knots - 2 * (degree + 1)
        if n_internal <= 0:
            knots = torch.cat([
                torch.zeros(degree + 1),
                torch.ones(degree + 1)
            ])
        else:
            knots = torch.cat([
                torch.zeros(degree + 1),
                torch.linspace(0, 1, n_internal + 2)[1:-1],
                torch.ones(degree + 1)
            ])
        return knots.to(self.device)
    
    def _basis_functions(self, t: torch.Tensor, k: int) -> torch.Tensor:
        """计算B-spline基函数（Cox-de Boor递归）
        Args:
            t: 参数值 (n_samples,)
            k: 基函数索引
        Returns:
            basis: 基函数值 (n_samples,)
        """
        knots = self.knots
        degree = self.degree
        
        # 递归基函数计算
        if degree == 0:
            return ((t >= knots[k]) & (t < knots[k+1])).float()
        
        # 递归情况
        denom1 = knots[k + degree] - knots[k]
        denom2 = knots[k + degree + 1] - knots[k + 1]
        
        term1 = torch.zeros_like(t)
        term2 = torch.zeros_like(t)
        
        if denom1 > 1e-10:
            term1 = (t - knots[k]) / denom1 * self._basis_functions_recursive(
                t, k, degree - 1)
        if denom2 > 1e-10:
            term2 = (knots[k + degree + 1] - t) / denom2 * self._basis_functions_recursive(
                t, k + 1, degree - 1)
        
        return term1 + term2
    
    def evaluate_batch(self, control_points: torch.Tensor, 
                      n_samples: int = 100) -> torch.Tensor:
        """批量评估B-spline轨迹（GPU并行）
        Args:
            control_points: (batch, n_control_points, dim)
        Returns:
            trajectories: (batch, n_samples, dim)
        """
        batch_size = control_points.shape[0]
        t_eval = torch.linspace(0, 1, n_samples, device=self.device)
        
        # 使用线性插值作为快速近似（对于大批量）
        # 更精确的实现可以使用完整的B-spline基函数
        trajectories = []
        for b in range(batch_size):
            # 简化：使用线性插值（快速但不完全是B-spline）
            # 对于高性能，应实现完整的B-spline评估
            cp = control_points[b]  # (n_control_points, dim)
            t_cp = torch.linspace(0, 1, self.n_control_points, device=self.device)
            
            # 逐维度插值
            traj_dims = []
            for d in range(self.dim):
                # 使用torch.nn.functional.interpolate或手动插值
                traj_d = torch.nn.functional.interpolate(
                    cp[:, d].unsqueeze(0).unsqueeze(0),
                    size=n_samples,
                    mode='linear',
                    align_corners=True
                ).squeeze()
                traj_dims.append(traj_d)
            
            trajectories.append(torch.stack(traj_dims, dim=-1))
        
        return torch.stack(trajectories, dim=0)
    
    def add_noise(self, control_points: torch.Tensor, 
                 noise_std: float) -> torch.Tensor:
        """向控制点添加噪声（保持起始和目标固定）
        Args:
            control_points: (batch, n_control_points, dim)
            noise_std: 噪声标准差
        Returns:
            noisy_control_points: (batch, n_control_points, dim)
        """
        noise = torch.randn_like(control_points) * noise_std
        # 保持第一个和最后一个控制点不变
        noise[:, 0, :] = 0
        noise[:, -1, :] = 0
        return control_points + noise


class MPPI_GPU:
    """GPU加速的MPPI优化器"""
    
    def __init__(self,
                 cost_function: CostFunction,
                 n_samples: int = 1000,
                 n_control_points: int = 10,
                 bspline_degree: int = 3,
                 time_horizon: float = 5.0,
                 n_timesteps: int = 50,
                 temperature: float = 1.0,
                 noise_std: float = 0.5,
                 bounds: Tuple[float, float, float, float] = (-10, 10, -10, 10),
                 device: str = 'cuda',
                 batch_size: int = 500):
        """
        Args:
            cost_function: 代价函数（需要支持批处理）
            n_samples: 总采样数
            batch_size: GPU批处理大小（避免内存溢出）
            device: 'cuda' 或 'cpu'
        """
        self.cost_function = cost_function
        self.n_samples = n_samples
        self.n_control_points = n_control_points
        self.time_horizon = time_horizon
        self.n_timesteps = n_timesteps
        self.temperature = temperature
        self.noise_std = noise_std
        self.bounds = bounds
        self.batch_size = min(batch_size, n_samples)
        
        # GPU设置 - 检查真实可用性
        self.device = torch.device('cpu')  # Default to CPU
        actual_device = 'cpu'
        
        if device == 'cuda' and torch.cuda.is_available():
            try:
                # 测试GPU是否真的可以工作 - 不要用import warnings.warn这种
                # 直接尝试一个简单的GPU操作
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    test = torch.zeros(1, device='cuda')
                    _ = test + 1
                    del test
                    torch.cuda.synchronize()
                
                # 如果到这里没有崩溃，GPU是好的
                self.device = torch.device('cuda')
                actual_device = 'cuda'
                print(f"🚀 MPPI-GPU initialized on: cuda")
                print(f"   GPU: {torch.cuda.get_device_name(0)}")
                print(f"   CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            except (RuntimeError, AssertionError) as e:
                print(f"⚠️  GPU不可用 (RTX 5090需要PyTorch 2.6+)")
                print(f"   回退到CPU模式 (但仍然会使用向量化加速)")
                self.device = torch.device('cpu')
                actual_device = 'cpu'
        else:
            print(f"🚀 MPPI-GPU initialized on: cpu")
            print(f"   (使用CPU加速计算)")
        print()
        
        # B-Spline生成器 - 使用实际可用的设备
        self.bspline = BSplineGPU(
            degree=bspline_degree,
            n_control_points=n_control_points,
            time_horizon=time_horizon,
            dim=2,
            device=actual_device
        )
        
        # 当前控制点
        self.control_points = None
        self.iteration = 0
        self.cost_history = []
        
        # 全局最佳
        self.best_cost_all_time = float('inf')
        self.best_trajectory_all_time = None
        self.best_control_points_all_time = None
        self.best_iteration = -1
        
    def initialize(self, start: np.ndarray, goal: np.ndarray):
        """初始化控制点"""
        alphas = np.linspace(0, 1, self.n_control_points).reshape(-1, 1)
        control_points_np = (1 - alphas) * start + alphas * goal
        
        self.control_points = torch.from_numpy(control_points_np).float().to(self.device)
        self.start = torch.from_numpy(start).float().to(self.device)
        self.goal = torch.from_numpy(goal).float().to(self.device)
        
    def sample_trajectories(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """GPU并行采样轨迹
        Returns:
            sampled_control_points: (n_samples, n_control_points, 2)
            trajectories: (n_samples, n_timesteps, 2)
        """
        # 复制当前控制点
        control_points_batch = self.control_points.unsqueeze(0).repeat(
            self.n_samples, 1, 1
        )  # (n_samples, n_control_points, 2)
        
        # 添加噪声
        sampled_control_points = self.bspline.add_noise(
            control_points_batch, self.noise_std
        )
        
        # 批量评估轨迹（分批以避免内存溢出）
        trajectories = []
        for i in range(0, self.n_samples, self.batch_size):
            batch_end = min(i + self.batch_size, self.n_samples)
            batch_cp = sampled_control_points[i:batch_end]
            batch_traj = self.bspline.evaluate_batch(batch_cp, self.n_timesteps)
            trajectories.append(batch_traj)
        
        trajectories = torch.cat(trajectories, dim=0)
        
        return sampled_control_points, trajectories
    
    def evaluate_trajectories_gpu(self, trajectories: torch.Tensor) -> torch.Tensor:
        """GPU批量评估轨迹代价
        Args:
            trajectories: (n_samples, n_timesteps, 2)
        Returns:
            costs: (n_samples,)
        """
        # 计算导数（有限差分）
        dt = self.time_horizon / self.n_timesteps
        velocities = torch.diff(trajectories, dim=1) / dt
        velocities = torch.cat([velocities, velocities[:, -1:, :]], dim=1)
        
        accelerations = torch.diff(velocities, dim=1) / dt
        accelerations = torch.cat([
            accelerations, 
            accelerations[:, -1:, :], 
            accelerations[:, -1:, :]
        ], dim=1)
        
        jerks = torch.diff(accelerations, dim=1) / dt
        jerks = torch.cat([
            jerks,
            jerks[:, -1:, :],
            jerks[:, -1:, :],
            jerks[:, -1:, :]
        ], dim=1)
        
        # 转换回numpy以使用现有代价函数
        # TODO: 实现完全GPU的代价函数
        positions_np = trajectories.cpu().numpy()
        velocities_np = velocities.cpu().numpy()
        accelerations_np = accelerations.cpu().numpy()
        jerks_np = jerks.cpu().numpy()
        
        costs_np = self.cost_function(
            positions=positions_np,
            velocities=velocities_np,
            accelerations=accelerations_np,
            jerks=jerks_np
        )
        
        return torch.from_numpy(costs_np).float().to(self.device)
    
    def compute_weights(self, costs: torch.Tensor) -> torch.Tensor:
        """计算重要性权重"""
        costs_normalized = costs - costs.min()
        weights = torch.exp(-costs_normalized / (self.temperature + 1e-8))
        weights = weights / (weights.sum() + 1e-8)
        return weights
    
    def update(self, sampled_control_points: torch.Tensor, 
              weights: torch.Tensor):
        """加权更新控制点"""
        weights_expanded = weights.view(-1, 1, 1)
        new_control_points = (weights_expanded * sampled_control_points).sum(dim=0)
        
        # 保持起始和目标固定
        new_control_points[0] = self.start
        new_control_points[-1] = self.goal
        
        self.control_points = new_control_points
    
    def step(self) -> Dict:
        """执行一次MPPI迭代"""
        # 采样轨迹
        sampled_control_points, trajectories = self.sample_trajectories()
        
        # 评估代价
        costs = self.evaluate_trajectories_gpu(trajectories)
        
        # 计算权重
        weights = self.compute_weights(costs)
        
        # 更新控制点
        self.update(sampled_control_points, weights)
        
        # 跟踪最佳
        best_idx = costs.argmin().item()
        best_cost = costs[best_idx].item()
        self.cost_history.append(best_cost)
        
        best_trajectory = trajectories[best_idx].cpu().numpy()
        
        # 更新全局最佳
        if best_cost < self.best_cost_all_time:
            self.best_cost_all_time = best_cost
            self.best_trajectory_all_time = best_trajectory.copy()
            self.best_control_points_all_time = sampled_control_points[best_idx].cpu().numpy()
            self.best_iteration = self.iteration
        
        self.iteration += 1
        
        info = {
            'iteration': self.iteration,
            'best_cost': best_cost,
            'mean_cost': costs.mean().item(),
            'best_trajectory': best_trajectory,
            'best_cost_all_time': self.best_cost_all_time,
            'best_iteration': self.best_iteration
        }
        
        return info
    
    def optimize(self, start: np.ndarray, goal: np.ndarray,
                n_iterations: int = 50,
                verbose: bool = True) -> Dict:
        """运行GPU加速的MPPI优化"""
        self.initialize(start, goal)
        self.cost_history = []
        self.iteration = 0
        
        self.best_cost_all_time = float('inf')
        self.best_trajectory_all_time = None
        self.best_control_points_all_time = None
        self.best_iteration = -1
        
        info_history = []
        
        for i in range(n_iterations):
            info = self.step()
            info_history.append(info)
            
            if verbose and (i % 10 == 0 or i == n_iterations - 1):
                print(f"迭代 {i}: 最佳代价 = {info['best_cost']:.2f}, "
                      f"平均代价 = {info['mean_cost']:.2f}, "
                      f"全局最佳 = {self.best_cost_all_time:.2f} (迭代 {self.best_iteration})")
        
        # 获取最终轨迹
        final_trajectory = self.bspline.evaluate_batch(
            self.control_points.unsqueeze(0), 
            self.n_timesteps
        )[0].cpu().numpy()
        
        result = {
            'trajectory': self.best_trajectory_all_time,
            'control_points': self.best_control_points_all_time,
            'cost_history': np.array(self.cost_history),
            'info_history': info_history,
            'best_trajectory_all_time': self.best_trajectory_all_time,
            'best_control_points_all_time': self.best_control_points_all_time,
            'best_cost_all_time': self.best_cost_all_time,
            'best_iteration': self.best_iteration,
            'final_cost': self.cost_history[-1] if len(self.cost_history) > 0 else np.inf,
            'final_trajectory': final_trajectory,
            'final_control_points': self.control_points.cpu().numpy()
        }
        
        return result
