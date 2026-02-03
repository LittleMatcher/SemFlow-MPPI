"""
Terrain Cost Scenario - 地形代价场景

包含三个不同地形区域：
1. 水泥路面（Concrete）：代价最低，最优通行
2. 草地（Grass）：代价中等，可通行但不推荐
3. 沼泽地（Swamp）：代价很高，应尽量避免

同时包含类似圣诞市场的障碍物布局
"""
import numpy as np
import sys
import os
from typing import List, Tuple

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

from mppi_core.environment_2d import Environment2D, Circle, Rectangle


class TerrainRegion:
    """地形区域定义"""
    def __init__(self, vertices: np.ndarray, cost_multiplier: float, name: str):
        """
        Args:
            vertices: 区域顶点坐标 (N, 2)
            cost_multiplier: 地形代价倍数（1.0=正常，>1更贵）
            name: 区域名称
        """
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.cost_multiplier = float(cost_multiplier)
        self.name = name
    
    def contains_point(self, point: np.ndarray) -> bool:
        """判断点是否在区域内（射线法）"""
        x, y = point[0], point[1]
        n = len(self.vertices)
        inside = False
        
        p1x, p1y = self.vertices[0]
        for i in range(1, n + 1):
            p2x, p2y = self.vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside


def create_terrain_cost_environment():
    """
    创建带地形代价的环境
    
    新设计布局：
    - 左侧：草地区域（无障碍，高代价10×）- 简单但昂贵
    - 右侧：水泥路面（有障碍，低代价1×）- 复杂但便宜
    
    Returns:
        env: 环境对象
        terrain_regions: 地形区域列表
        start: 起点
        goal: 终点
        bounds: 边界
    """
    bounds = (-10, 10, -10, 10)
    env = Environment2D(bounds=bounds)
    
    # === 定义两个地形区域 ===
    terrain_regions = []
    
    # 1. 左侧草地区域（整个左半边，无障碍，成本很高）
    grass_left_vertices = np.array([
        [-10, -10],
        [0, -10],
        [0, 10],
        [-10, 10]
    ])
    terrain_regions.append(TerrainRegion(
        grass_left_vertices,
        cost_multiplier=10.0,  # 10倍代价
        name="Grass (Left)"
    ))
    
    # 2. 右侧水泥路面区域（整个右半边，有障碍，成本低）
    concrete_right_vertices = np.array([
        [0, -10],
        [10, -10],
        [10, 10],
        [0, 10]
    ])
    terrain_regions.append(TerrainRegion(
        concrete_right_vertices,
        cost_multiplier=1.0,  # 基准代价
        name="Concrete (Right)"
    ))
    
    # === 添加障碍物（仅在右侧水泥地）===
    # 下部障碍
    env.add_rectangle_obstacle(1.0, 4.0, -9.5, -6.5)
    env.add_rectangle_obstacle(6.0, 9.0, -9.5, -6.5)
    
    # 中下部障碍
    env.add_rectangle_obstacle(1.0, 4.0, -6.0, -3.0)
    env.add_rectangle_obstacle(6.0, 9.0, -6.0, -3.0)
    
    # 中部障碍
    env.add_rectangle_obstacle(1.0, 4.0, -2.5, 0.5)
    env.add_rectangle_obstacle(6.0, 9.0, -2.5, 0.5)
    
    # 中央障碍
    env.add_rectangle_obstacle(1.0, 4.0, 1.0, 4.0)
    env.add_rectangle_obstacle(6.0, 9.0, 1.0, 4.0)
    
    # 中上部障碍
    env.add_rectangle_obstacle(1.0, 4.0, 4.5, 7.5)
    env.add_rectangle_obstacle(6.0, 9.0, 4.5, 7.5)
    
    # 上部障碍
    env.add_rectangle_obstacle(1.5, 4.5, 8.0, 9.5)
    env.add_rectangle_obstacle(5.5, 8.5, 8.0, 9.5)
    
    # 圆形障碍物（仅在右侧）
    env.add_circle_obstacle(np.array([2.5, -5.0]), 0.6)
    env.add_circle_obstacle(np.array([7.5, -5.0]), 0.6)
    env.add_circle_obstacle(np.array([5.0, -1.0]), 0.5)
    env.add_circle_obstacle(np.array([2.5, 2.5]), 0.6)
    env.add_circle_obstacle(np.array([7.5, 2.5]), 0.6)
    env.add_circle_obstacle(np.array([5.0, 6.0]), 0.5)
    
    # 起点和终点
    start = np.array([-8.0, -8.0])
    goal = np.array([8.0, 8.0])
    
    return env, terrain_regions, start, goal, bounds

def get_terrain_cost_at_point(point: np.ndarray, terrain_regions: List[TerrainRegion]) -> float:
    """
    获取某点的地形代价倍数
    
    Args:
        point: 位置 (2,)
        terrain_regions: 地形区域列表
    Returns:
        cost_multiplier: 地形代价倍数
    """
    for region in terrain_regions:
        if region.contains_point(point):
            return region.cost_multiplier
    return 1.0  # 默认代价


def get_terrain_cost_batch(points: np.ndarray, terrain_regions: List[TerrainRegion]) -> np.ndarray:
    """
    批量获取地形代价
    
    Args:
        points: 位置数组 (N, 2)
        terrain_regions: 地形区域列表
    Returns:
        costs: 代价倍数数组 (N,)
    """
    costs = np.ones(len(points))
    for i, point in enumerate(points):
        costs[i] = get_terrain_cost_at_point(point, terrain_regions)
    return costs


if __name__ == "__main__":
    # 简单测试
    env, terrain_regions, start, goal, bounds = create_terrain_cost_environment()
    print(f"Environment created with {len(env.obstacles)} obstacles")
    print(f"Terrain regions: {[r.name for r in terrain_regions]}")
    print(f"Start: {start}, Goal: {goal}")
    
    # 测试地形代价
    test_points = [
        np.array([-5.0, 0.0]),  # 沼泽
        np.array([0.0, 0.0]),   # 水泥
        np.array([5.0, 0.0])    # 草地
    ]
    for p in test_points:
        cost = get_terrain_cost_at_point(p, terrain_regions)
        print(f"Point {p}: terrain cost = {cost}")
