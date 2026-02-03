"""
MPPI 轨迹的可视化工具
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle as MPLRectangle
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter
from .environment_2d import Environment2D, Circle as CircleObs, Rectangle as RectangleObs
from typing import Optional, List


class Visualizer:
    """可视化 2D 环境和轨迹"""
    
    def __init__(self, env: Environment2D, figsize=(10, 10)):
        """
        Args:
            env: 2D 环境
            figsize: 图形大小
        """
        self.env = env
        self.figsize = figsize
        
    def plot_environment(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """绘制障碍物
        Args:
            ax: matplotlib 轴（如果为 None 则创建新的）
        Returns:
            ax: matplotlib 轴
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        # 绘制障碍物
        for obs in self.env.obstacles:
            if isinstance(obs, CircleObs):
                circle = Circle(obs.center, obs.radius, 
                              color='gray', alpha=0.7, zorder=1)
                ax.add_patch(circle)
            elif isinstance(obs, RectangleObs):
                width = obs.x_max - obs.x_min
                height = obs.y_max - obs.y_min
                rect = MPLRectangle((obs.x_min, obs.y_min), width, height,
                                   color='gray', alpha=0.7, zorder=1)
                ax.add_patch(rect)
        
        # 设置限制
        ax.set_xlim(self.env.x_min, self.env.x_max)
        ax.set_ylim(self.env.y_min, self.env.y_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_trajectory(self, trajectory: np.ndarray, 
                       ax: Optional[plt.Axes] = None,
                       color='blue', linewidth=2, alpha=1.0,
                       label: Optional[str] = None,
                       show_control_points: bool = False,
                       control_points: Optional[np.ndarray] = None) -> plt.Axes:
        """绘制单条轨迹
        Args:
            trajectory: 形状 (T, 2)
            ax: matplotlib 轴
            color: 线条颜色
            linewidth: 线宽
            alpha: 透明度
            label: 图例标签
            show_control_points: 是否显示 B-spline 控制点
            control_points: 形状 (n_control_points, 2)
        Returns:
            ax: matplotlib 轴
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            self.plot_environment(ax)
        
        # 绘制轨迹
        ax.plot(trajectory[:, 0], trajectory[:, 1], 
               color=color, linewidth=linewidth, alpha=alpha, 
               label=label, zorder=3)
        
        # 绘制起始和结束点
        ax.scatter(trajectory[0, 0], trajectory[0, 1], 
                  color='green', s=200, marker='o', 
                  edgecolor='black', linewidth=2, zorder=5, label='Start')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
                  color='red', s=200, marker='*',
                  edgecolor='black', linewidth=2, zorder=5, label='Goal')
        
        # 如果提供，绘制控制点
        if show_control_points and control_points is not None:
            ax.scatter(control_points[:, 0], control_points[:, 1],
                      color='orange', s=50, marker='x', zorder=4,
                      label='Control Points')
            ax.plot(control_points[:, 0], control_points[:, 1],
                   color='orange', linewidth=1, alpha=0.5, 
                   linestyle='--', zorder=2)
        
        return ax
    
    def plot_multiple_trajectories(self, trajectories: np.ndarray,
                                   ax: Optional[plt.Axes] = None,
                                   colors: Optional[np.ndarray] = None,
                                   alpha: float = 0.3,
                                   linewidth: float = 1) -> plt.Axes:
        """绘制多条轨迹（例如，MPPI 采样）
        Args:
            trajectories: 形状 (n_trajs, T, 2)
            ax: matplotlib 轴
            colors: 每条轨迹的颜色，形状 (n_trajs, 3) 或 (n_trajs, 4)
            alpha: 透明度
            linewidth: 线宽
        Returns:
            ax: matplotlib 轴
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            self.plot_environment(ax)
        
        n_trajs = len(trajectories)
        
        if colors is None:
            # 默认：蓝色，强度变化
            colors = plt.cm.Blues(np.linspace(0.3, 0.8, n_trajs))
        
        for i, traj in enumerate(trajectories):
            color = colors[i] if colors.ndim == 2 else colors
            ax.plot(traj[:, 0], traj[:, 1],
                   color=color, alpha=alpha, linewidth=linewidth, zorder=2)
        
        return ax
    
    def plot_weighted_trajectories(self, trajectories: np.ndarray,
                                   weights: np.ndarray,
                                   ax: Optional[plt.Axes] = None,
                                   cmap: str = 'YlOrRd') -> plt.Axes:
        """按权重着色绘制轨迹
        Args:
            trajectories: 形状 (n_trajs, T, 2)
            weights: 形状 (n_trajs,)，归一化权重
            ax: matplotlib 轴
            cmap: 颜色映射名称
        Returns:
            ax: matplotlib 轴
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            self.plot_environment(ax)
        
        # 为着色归一化权重
        weights_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
        
        cmap_obj = plt.cm.get_cmap(cmap)
        colors = cmap_obj(weights_norm)
        
        # 绘制轨迹
        for i, traj in enumerate(trajectories):
            alpha = 0.2 + 0.6 * weights_norm[i]  # 权重越高 = 越可见
            linewidth = 0.5 + 2.0 * weights_norm[i]
            ax.plot(traj[:, 0], traj[:, 1],
                   color=colors[i], alpha=alpha, linewidth=linewidth, zorder=2)
        
        return ax
    
    def visualize_sdf(self, resolution: int = 200,
                     ax: Optional[plt.Axes] = None,
                     robot_radius: float = 0.2) -> plt.Axes:
        """可视化符号距离场
        Args:
            resolution: 网格分辨率
            ax: matplotlib 轴
            robot_radius: 用于可视化的机器人半径
        Returns:
            ax: matplotlib 轴
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        # 创建网格
        x = np.linspace(self.env.x_min, self.env.x_max, resolution)
        y = np.linspace(self.env.y_min, self.env.y_max, resolution)
        X, Y = np.meshgrid(x, y)
        points = np.stack([X.flatten(), Y.flatten()], axis=-1)
        
        # 计算 SDF
        sdf = self.env.compute_sdf(points).reshape(resolution, resolution)
        
        # 绘制
        im = ax.contourf(X, Y, sdf, levels=20, cmap='RdYlGn', alpha=0.6)
        
        # 绘制零水平集（障碍物边界）
        ax.contour(X, Y, sdf, levels=[0], colors='black', linewidths=2)
        
        # 绘制安全边界（robot_radius）
        ax.contour(X, Y, sdf, levels=[robot_radius], 
                  colors='red', linewidths=2, linestyles='--')
        
        plt.colorbar(im, ax=ax, label='Signed Distance')
        ax.set_aspect('equal')
        ax.set_title('Signed Distance Field')
        
        return ax
    
    def create_animation(self, info_history: List[dict],
                        save_path: Optional[str] = None,
                        show_samples: bool = True,
                        fps: int = 10) -> FuncAnimation:
        """创建 MPPI 优化过程的动画
        Args:
            info_history: 来自 MPPI.optimize() 的信息字典列表
            save_path: 保存动画的路径（None = 不保存）
            show_samples: 是否显示采样的轨迹
            fps: 每秒帧数
        Returns:
            animation: FuncAnimation 对象
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        # 绘制环境
        self.plot_environment(ax)
        
        # 获取起始和目标
        traj0 = info_history[0]['best_trajectory']
        ax.scatter(traj0[0, 0], traj0[0, 1], color='green', s=200, 
                  marker='o', edgecolor='black', linewidth=2, zorder=5)
        ax.scatter(traj0[-1, 0], traj0[-1, 1], color='red', s=200,
                  marker='*', edgecolor='black', linewidth=2, zorder=5)
        
        # 要更新的元素
        best_line, = ax.plot([], [], color='blue', linewidth=3, 
                            label='Best Trajectory', zorder=4)
        sample_lines = []
        
        if show_samples:
            for _ in range(info_history[0]['all_trajectories'].shape[0]):
                line, = ax.plot([], [], color='gray', alpha=0.1, 
                              linewidth=0.5, zorder=2)
                sample_lines.append(line)
        
        title = ax.set_title('Iteration 0')
        ax.legend()
        
        def update(frame):
            info = info_history[frame]
            
            # 更新最佳轨迹
            best_traj = info['best_trajectory']
            best_line.set_data(best_traj[:, 0], best_traj[:, 1])
            
            # 更新采样轨迹
            if show_samples:
                all_trajs = info['all_trajectories']
                for i, line in enumerate(sample_lines):
                    if i < len(all_trajs):
                        line.set_data(all_trajs[i, :, 0], all_trajs[i, :, 1])
            
            # 更新标题
            title.set_text(f"Iteration {frame}, Cost: {info['best_cost']:.2f}")
            
            return [best_line, title] + sample_lines
        
        anim = FuncAnimation(fig, update, frames=len(info_history),
                           interval=1000/fps, blit=True)
        
        if save_path is not None:
            print(f"Saving animation to {save_path}...")
            writer = PillowWriter(fps=fps)
            anim.save(save_path, writer=writer)
            print("Animation saved!")
        
        return anim
    
    def plot_cost_history(self, cost_history: np.ndarray,
                         ax: Optional[plt.Axes] = None,
                         best_iteration: Optional[int] = None,
                         best_cost: Optional[float] = None) -> plt.Axes:
        """绘制迭代过程中的代价
        Args:
            cost_history: 代价数组
            ax: matplotlib 轴
            best_iteration: 所有迭代中的最佳迭代（可选，用于标记）
            best_cost: 所有迭代中的最佳代价（可选，用于标记）
        Returns:
            ax: matplotlib 轴
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        
        ax.plot(cost_history, linewidth=2, label='Best Cost per Iteration', alpha=0.7)
        
        # 标记所有迭代中的最佳迭代
        if best_iteration is not None and best_iteration < len(cost_history):
            if best_cost is None:
                best_cost = cost_history[best_iteration]
            ax.scatter([best_iteration], [best_cost], 
                      color='red', s=200, marker='*', 
                      zorder=5, label=f'Global Best (Iteration {best_iteration})',
                      edgecolor='black', linewidth=2)
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Best Cost', fontsize=12)
        ax.set_title('MPPI Optimization Progress', fontsize=14)
        ax.grid(True, alpha=0.3)
        if best_iteration is not None:
            ax.legend(fontsize=10)
        
        return ax
