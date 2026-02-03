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
    
    # ============ CBF 安全约束设置 ============
    use_cbf_guidance: bool = False
    cbf_weight: float = 1.0
    cbf_margin: float = 0.1
    cbf_alpha: float = 1.0
    
    # ============ 多模态锚点生成设置 ============
    # 步骤4: 多模态锚点筛选 (Anchors Selection)
    enable_multimodal_anchors: bool = False  # 启用多模态锚点生成
    multimodal_batch_size: int = 64         # 并行生成的轨迹数量
    num_anchor_clusters: int = 3            # 同伦类数量 (如: 左、中、右)
    clustering_method: str = "kmeans"       # "kmeans", "gmm", "simple"
    
    # 聚类特征配置
    clustering_features: str = "midpoint"   # "midpoint", "endpoint", "curvature", "full_traj"
    clustering_weight_position: float = 1.0
    clustering_weight_velocity: float = 0.5
    clustering_weight_curvature: float = 0.3


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
            use_cbf_guidance=self.config.use_cbf_guidance,
            cbf_weight=self.config.cbf_weight,
            cbf_margin=self.config.cbf_margin,
            cbf_alpha=self.config.cbf_alpha,
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
        
        # 创建多模态锚点选择器
        if self.config.enable_multimodal_anchors:
            self.anchor_selector = MultimodalAnchorSelector(self.config)
        else:
            self.anchor_selector = None
        
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


class MultimodalAnchorSelector:
    """
    多模态锚点选择器
    
    实现步骤4: 多模态锚点筛选 (Anchors Selection)
    1. 并行生成 N 条轨迹（不同随机噪声）
    2. 使用 K-Means 或 GMM 进行聚类，识别同伦类
    3. 选择每个聚类的代表轨迹作为离散锚点
    """
    
    def __init__(self, config: GeneratorConfig):
        self.config = config
    
    def extract_clustering_features(
        self, 
        trajectories: torch.Tensor,
        velocities: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        从轨迹中提取聚类特征
        
        参数:
            trajectories: 轨迹 [N, T, D]
            velocities: 速度 [N, T, D] (可选)
            
        返回:
            特征向量 [N, feature_dim]
        """
        N, T, D = trajectories.shape
        device = trajectories.device
        
        if self.config.clustering_features == "midpoint":
            # 使用轨迹中点作为特征
            mid_idx = T // 2
            features = trajectories[:, mid_idx, :]  # [N, D]
            
        elif self.config.clustering_features == "endpoint":
            # 使用起点和终点作为特征
            start_points = trajectories[:, 0, :]   # [N, D]
            end_points = trajectories[:, -1, :]    # [N, D]
            features = torch.cat([start_points, end_points], dim=-1)  # [N, 2*D]
            
        elif self.config.clustering_features == "curvature":
            # 使用曲率特征
            # 计算轨迹的二阶导数作为曲率的近似
            if T >= 3:
                second_derivative = trajectories[:, 2:, :] - 2 * trajectories[:, 1:-1, :] + trajectories[:, :-2, :]
                curvature = torch.norm(second_derivative, dim=-1)  # [N, T-2]
                features = torch.cat([
                    trajectories[:, T//2, :],           # 中点位置
                    curvature.mean(dim=-1, keepdim=True),  # 平均曲率
                    curvature.max(dim=-1, keepdim=True)[0],  # 最大曲率
                ], dim=-1)  # [N, D+2]
            else:
                features = trajectories[:, T//2, :]  # 回退到中点
                
        elif self.config.clustering_features == "full_traj":
            # 使用完整轨迹（降维处理）
            # 选择几个关键时间点
            key_indices = torch.linspace(0, T-1, min(8, T), dtype=torch.long, device=device)
            key_points = trajectories[:, key_indices, :].reshape(N, -1)  # [N, 8*D]
            
            # 加上速度特征（如果提供）
            if velocities is not None:
                key_velocities = velocities[:, key_indices, :].reshape(N, -1)
                features = torch.cat([
                    self.config.clustering_weight_position * key_points,
                    self.config.clustering_weight_velocity * key_velocities,
                ], dim=-1)
            else:
                features = key_points
                
        else:
            # 默认：中点特征
            mid_idx = T // 2
            features = trajectories[:, mid_idx, :]
        
        return features
    
    def cluster_trajectories(
        self,
        trajectories: torch.Tensor,
        velocities: Optional[torch.Tensor] = None,
        accelerations: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        对轨迹进行聚类分析
        
        参数:
            trajectories: 轨迹 [N, T, D]
            velocities: 速度 [N, T, D] (可选)
            accelerations: 加速度 [N, T, D] (可选)
            
        返回:
            包含聚类结果的字典
        """
        N = trajectories.shape[0]
        K = min(self.config.num_anchor_clusters, N)  # 确保聚类数不超过样本数
        
        # 提取特征
        features = self.extract_clustering_features(trajectories, velocities)  # [N, F]
        
        if self.config.clustering_method == "simple":
            # 简单方法：基于终点位置进行聚类
            end_positions = trajectories[:, -1, :]  # [N, D]
            
            if D == 2:
                # 2D情况：按角度分组（左、中、右）
                angles = torch.atan2(end_positions[:, 1], end_positions[:, 0])
                angle_bins = torch.linspace(-np.pi, np.pi, K+1, device=angles.device)
                cluster_labels = torch.bucketize(angles, angle_bins) - 1
                cluster_labels = torch.clamp(cluster_labels, 0, K-1)
            else:
                # 多维情况：按第一维坐标分组
                coord_bins = torch.linspace(
                    end_positions[:, 0].min(), end_positions[:, 0].max(), K+1, 
                    device=end_positions.device
                )
                cluster_labels = torch.bucketize(end_positions[:, 0], coord_bins) - 1
                cluster_labels = torch.clamp(cluster_labels, 0, K-1)
        
        else:
            # K-Means 或 GMM 聚类
            try:
                if self.config.clustering_method == "kmeans":
                    cluster_labels = self._kmeans_clustering(features, K)
                elif self.config.clustering_method == "gmm":
                    cluster_labels = self._gmm_clustering(features, K)
                else:
                    raise ValueError(f"未知聚类方法: {self.config.clustering_method}")
                    
            except Exception as e:
                # 聚类失败时的回退策略
                print(f"聚类失败，使用简单分组: {e}")
                cluster_labels = torch.arange(N, device=trajectories.device) % K
        
        return {
            'cluster_labels': cluster_labels,  # [N,]
            'num_clusters': K,
            'features': features,
        }
    
    def _kmeans_clustering(self, features: torch.Tensor, K: int) -> torch.Tensor:
        """K-Means 聚类实现"""
        N, F = features.shape
        device = features.device
        
        # 初始化聚类中心
        centers = features[torch.randperm(N, device=device)[:K]]  # [K, F]
        
        # K-Means 迭代
        for _ in range(20):  # 最大迭代次数
            # 分配样本到最近的中心
            distances = torch.cdist(features, centers)  # [N, K]
            labels = distances.argmin(dim=-1)  # [N,]
            
            # 更新聚类中心
            new_centers = torch.zeros_like(centers)
            for k in range(K):
                mask = (labels == k)
                if mask.sum() > 0:
                    new_centers[k] = features[mask].mean(dim=0)
                else:
                    new_centers[k] = centers[k]  # 保持原中心
            
            # 检查收敛
            if torch.allclose(centers, new_centers, atol=1e-4):
                break
            centers = new_centers
        
        return labels
    
    def _gmm_clustering(self, features: torch.Tensor, K: int) -> torch.Tensor:
        """GMM 聚类实现（简化版）"""
        # 这里可以实现 EM 算法，或者回退到 K-Means
        return self._kmeans_clustering(features, K)
    
    def select_anchor_trajectories(
        self,
        trajectories: torch.Tensor,
        velocities: Optional[torch.Tensor] = None,
        accelerations: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        选择代表性锚点轨迹
        
        参数:
            trajectories: 所有生成的轨迹 [N, T, D]
            velocities: 速度 [N, T, D] (可选)
            accelerations: 加速度 [N, T, D] (可选)
            
        返回:
            包含锚点轨迹的字典
        """
        # 聚类分析
        clustering_result = self.cluster_trajectories(trajectories, velocities, accelerations)
        cluster_labels = clustering_result['cluster_labels']
        K = clustering_result['num_clusters']
        
        # 为每个聚类选择代表轨迹
        anchor_indices = []
        anchor_trajectories = []
        anchor_velocities = []
        anchor_accelerations = []
        
        for k in range(K):
            # 找到属于当前聚类的轨迹
            cluster_mask = (cluster_labels == k)
            cluster_indices = torch.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                # 空聚类：跳过或使用随机轨迹
                continue
            
            # 选择聚类中心最近的轨迹作为代表
            cluster_trajs = trajectories[cluster_mask]  # [n_k, T, D]
            cluster_center = cluster_trajs.mean(dim=0)   # [T, D]
            
            # 计算每个轨迹到聚类中心的距离
            distances = torch.norm(cluster_trajs - cluster_center.unsqueeze(0), dim=(1, 2))
            closest_idx = distances.argmin()
            representative_idx = cluster_indices[closest_idx]
            
            # 收集代表轨迹
            anchor_indices.append(representative_idx.item())
            anchor_trajectories.append(trajectories[representative_idx])
            
            if velocities is not None:
                anchor_velocities.append(velocities[representative_idx])
            if accelerations is not None:
                anchor_accelerations.append(accelerations[representative_idx])
        
        # 构建结果
        result = {
            'anchor_indices': torch.tensor(anchor_indices, device=trajectories.device),
            'anchor_trajectories': torch.stack(anchor_trajectories, dim=0) if anchor_trajectories else torch.empty(0, *trajectories.shape[1:], device=trajectories.device),
            'num_anchors': len(anchor_trajectories),
            'clustering_result': clustering_result,
        }
        
        if anchor_velocities:
            result['anchor_velocities'] = torch.stack(anchor_velocities, dim=0)
        if anchor_accelerations:
            result['anchor_accelerations'] = torch.stack(anchor_accelerations, dim=0)
        
        return result
    
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
        obstacle_positions: Optional[torch.Tensor] = None,
        cost_map: Optional[torch.Tensor] = None,
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
            warm_start_state: 热启动状态 (可选)
            obstacle_positions: 障碍物位置 [N_obs, D] (可选，用于 CBF)
            cost_map: 语义代价图 [B, H, W] (可选，用于 CBF)
            
        返回:
            包含轨迹和可选的多模态锚点的字典
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
        
        # 创建 CBF 引导函数（如果启用）
        cbf_guidance_fn = None
        if self.config.use_cbf_guidance and (obstacle_positions is not None or cost_map is not None):
            from ..training.flow_matching import FlowMatchingConfig, compute_cbf_guidance
            
            # 创建 CBF 配置
            cbf_config = FlowMatchingConfig(
                state_dim=self.config.state_dim,
                use_cbf_constraint=True,
                cbf_weight=self.config.cbf_weight,
                cbf_margin=self.config.cbf_margin,
                cbf_alpha=self.config.cbf_alpha,
            )
            
            def cbf_guidance_fn(positions, velocities):
                return compute_cbf_guidance(
                    positions, velocities, cbf_config,
                    obstacle_positions, cost_map
                )
        
        # 求解 ODE（带可选的 CBF 引导）
        if hasattr(self.solver, 'solve') and 'cbf_guidance_fn' in self.solver.solve.__code__.co_varnames:
            x_1 = self.solver.solve(velocity_fn, x_0, cbf_guidance_fn=cbf_guidance_fn)
        else:
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
        
        # ============ Multimodal Anchor Selection ============
        # Step 4: Multimodal Anchor Selection
        if self.config.enable_multimodal_anchors and self.anchor_selector is not None and num_samples > 1:
            # Only perform clustering when generating multiple samples
            anchor_result = self.anchor_selector.select_anchor_trajectories(
                trajectories=result['positions'],
                velocities=result.get('velocities'),
                accelerations=result.get('accelerations'),
            )
            
            # Replace original output with clustered anchor trajectories
            # This ensures L1 MPPI receives discrete anchors representing different homotopy classes
            result.update({
                'positions': anchor_result['anchor_trajectories'],
                'velocities': anchor_result.get('anchor_velocities', result.get('velocities')),
                'accelerations': anchor_result.get('anchor_accelerations', result.get('accelerations')),
                'num_anchors': anchor_result['num_anchors'],
                'anchor_indices': anchor_result['anchor_indices'],
                'clustering_info': anchor_result['clustering_result'],
                
                # Keep original multi-sample outputs for debugging
                'all_positions': result['positions'] if 'all_positions' not in result else result['positions'],
                'all_velocities': result['velocities'] if 'all_velocities' not in result else result['velocities'],
                'all_accelerations': result['accelerations'] if 'all_accelerations' not in result else result['accelerations'],
            })
            
            # Update batch size to match number of anchors
            B = anchor_result['num_anchors']
        
        # Update warm start cache (if enabled)
        if self.config.enable_warm_start and x_1 is not None:
            self.update_warm_start_cache({'raw_output': x_1})
        
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
        Use classifier-free guidance to generate trajectories.
        
        Allow guiding generation towards desired properties, such as obstacle avoidance.
        
        Args:
            start_pos: Starting position [B, D]
            goal_pos: Goal position [B, D]
            start_vel: Starting velocity [B, D]
            guidance_scale: Guidance scale (1.0 = no guidance)
            obstacle_fn: Function returning obstacle avoidance gradients
            
        Returns:
            Dictionary containing:
                - 'positions': Generated position trajectories [B, T, D]
                - 'velocities': Generated velocity trajectories [B, T, D]
                - 'accelerations': Generated acceleration trajectories [B, T, D]
                - 'raw_output': Raw model output [B, T, D*3] (consistent with generate())
        """
        # Note: Full CFG requires model trained with condition dropout
        # This is a simplified version with optional obstacle avoidance guidance
        
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
