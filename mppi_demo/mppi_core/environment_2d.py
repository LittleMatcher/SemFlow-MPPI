"""
基于 SDF 的 2D 环境障碍物表示
参考 Motion Planning Diffusion (arXiv:2308.01557)
"""
import numpy as np
from typing import List, Tuple, Union
from dataclasses import dataclass


@dataclass
class Circle:
    """圆形障碍物"""
    center: np.ndarray  # [x, y]
    radius: float
    
    def sdf(self, points: np.ndarray) -> np.ndarray:
        """符号距离函数
        Args:
            points: 形状 (N, 2) 或 (2,)
        Returns:
            distances: 形状 (N,) 或标量。负值表示内部，正值表示外部
        """
        if points.ndim == 1:
            points = points.reshape(1, -1)
        dist_to_center = np.linalg.norm(points - self.center, axis=-1)
        return dist_to_center - self.radius


@dataclass
class Rectangle:
    """轴对齐矩形障碍物"""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    
    def sdf(self, points: np.ndarray) -> np.ndarray:
        """轴对齐矩形的符号距离函数
        Args:
            points: 形状 (N, 2) 或 (2,)
        Returns:
            distances: 形状 (N,) 或标量
        """
        if points.ndim == 1:
            points = points.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False
            
        # 到各边的距离
        dx = np.maximum(self.x_min - points[:, 0], points[:, 0] - self.x_max)
        dy = np.maximum(self.y_min - points[:, 1], points[:, 1] - self.y_max)
        
        # 矩形外部
        outside_mask = (dx > 0) | (dy > 0)
        dist_outside = np.sqrt(np.maximum(dx, 0)**2 + np.maximum(dy, 0)**2)
        
        # 矩形内部
        dist_inside = np.maximum(dx, dy)
        
        dist = np.where(outside_mask, dist_outside, dist_inside)
        
        return dist[0] if squeeze else dist


class Environment2D:
    """带障碍物和 SDF 计算的 2D 环境"""
    
    def __init__(self, bounds: Tuple[float, float, float, float]):
        """
        Args:
            bounds: (x_min, x_max, y_min, y_max)
        """
        self.x_min, self.x_max, self.y_min, self.y_max = bounds
        self.obstacles: List[Union[Circle, Rectangle]] = []
        
    def add_circle_obstacle(self, center: np.ndarray, radius: float):
        """添加圆形障碍物"""
        self.obstacles.append(Circle(center=center, radius=radius))
        
    def add_rectangle_obstacle(self, x_min: float, x_max: float, 
                               y_min: float, y_max: float):
        """添加矩形障碍物"""
        self.obstacles.append(Rectangle(x_min, x_max, y_min, y_max))
        
    def compute_sdf(self, points: np.ndarray) -> np.ndarray:
        """计算到所有障碍物的最小符号距离
        Args:
            points: 形状 (N, 2) 或 (2,)
        Returns:
            sdf: 形状 (N,) 或标量。负值表示在障碍物内部
        """
        if len(self.obstacles) == 0:
            if points.ndim == 1:
                return np.inf
            return np.full(len(points), np.inf)
        
        sdfs = []
        for obs in self.obstacles:
            sdfs.append(obs.sdf(points))
        
        # 到任意障碍物的最小距离
        sdf = np.min(np.stack(sdfs, axis=0), axis=0)
        return sdf
    
    def is_collision(self, points: np.ndarray, robot_radius: float = 0.0) -> np.ndarray:
        """检查点是否与障碍物碰撞
        Args:
            points: 形状 (N, 2) 或 (2,)
            robot_radius: 机器人安全半径
        Returns:
            collision: 形状 (N,) 或 bool
        """
        sdf = self.compute_sdf(points)
        return sdf < robot_radius
    
    def create_u_trap(self, center: Tuple[float, float], 
                      width: float = 3.0, height: float = 4.0, 
                      thickness: float = 0.3):
        """创建 U 形陷阱障碍物"""
        cx, cy = center
        # 左侧墙
        self.add_rectangle_obstacle(
            cx - width/2 - thickness, cx - width/2,
            cy - height/2, cy + height/2
        )
        # 右侧墙
        self.add_rectangle_obstacle(
            cx + width/2, cx + width/2 + thickness,
            cy - height/2, cy + height/2
        )
        # 底部墙
        self.add_rectangle_obstacle(
            cx - width/2 - thickness, cx + width/2 + thickness,
            cy - height/2 - thickness, cy - height/2
        )
        
    def create_narrow_passage(self, start: Tuple[float, float],
                             end: Tuple[float, float],
                             passage_width: float = 0.6,
                             wall_thickness: float = 2.0):
        """创建两堵墙之间的狭窄通道"""
        x1, y1 = start
        x2, y2 = end
        
        # 上墙
        self.add_rectangle_obstacle(
            x1, x2,
            y1 + passage_width/2, y1 + passage_width/2 + wall_thickness
        )
        # 下墙  
        self.add_rectangle_obstacle(
            x1, x2,
            y1 - passage_width/2 - wall_thickness, y1 - passage_width/2
        )
