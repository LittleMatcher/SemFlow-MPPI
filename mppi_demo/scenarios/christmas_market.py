"""
Christmas Market Scenario

A complex scenario with many obstacles (stalls, booths) arranged
in a crowded market layout, with regional crowd density considerations.
"""
import numpy as np
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

from mppi_core.environment_2d import Environment2D, Circle, Rectangle


class CrowdRegion:
    """人流量区域定义"""
    def __init__(self, bounds, density_multiplier: float, name: str):
        """
        Args:
            bounds: (x_min, x_max, y_min, y_max) 区域边界
            density_multiplier: 人流密度倍数 (1.0为基准)
            name: 区域名称
        """
        self.x_min, self.x_max, self.y_min, self.y_max = bounds
        self.density_multiplier = float(density_multiplier)
        self.name = name
    
    def contains_point(self, point: np.ndarray) -> bool:
        """判断点是否在区域内"""
        x, y = point[0], point[1]
        return (self.x_min <= x <= self.x_max and 
                self.y_min <= y <= self.y_max)
    
    def get_density(self, point: np.ndarray) -> float:
        """获取指定点的人流密度"""
        if self.contains_point(point):
            return self.density_multiplier
        return 1.0  # 默认密度


def create_christmas_market_environment(variant: str = "v1"):
    """
    Create a complex "Christmas Market" environment with many obstacles
    and crowd density regions
    
    Design (Balanced version with clear but challenging passages):
    - Multiple stalls/booths with strategic spacing
    - Clear main corridors with some obstacles
    - Balanced between challenge and navigability
    - Goal is to navigate from one end to the other
    - Four crowd density regions: Left (high), Right (medium), Bottom (low), Top (very high)
    """
    bounds = (-8, 8, -8, 8)  # x_min, x_max, y_min, y_max
    env = Environment2D(bounds=bounds)

    variant = (variant or "v1").lower()

    # Variant notes:
    # - v1: baseline balanced layout
    # - v2: slightly shifted/perturbed layout to test robustness (still navigable)
    if variant not in {"v1", "v2"}:
        raise ValueError(f"Unknown Christmas Market variant: {variant}")

    def add_rect(x1, x2, y1, y2):
        env.add_rectangle_obstacle(x1, x2, y1, y2)

    def add_circle(center, radius):
        env.add_circle_obstacle(np.array(center, dtype=float), float(radius))

    # Small deterministic perturbations for v2
    dx_c = 0.25 if variant == "v2" else 0.0
    dy_c = -0.15 if variant == "v2" else 0.0
    dx_r = -0.20 if variant == "v2" else 0.0
    dy_r = 0.10 if variant == "v2" else 0.0
    
    # === Row 1: Entry area (wider spacing) ===
    add_rect(-6.2, -4.8, -7.0, -5.5)
    # add_rect(-3.2, -1.8, -7.0, -5.5)
    add_rect(-0.2, 1.2, -7.0, -5.5)
    add_rect(3.8, 5.2, -7.0, -5.5)
    
    # === Row 2: Second row (reduced height for wider passages) ===
    # add_rect(-7.0, -5.3, -4.0, -2.3)
    add_rect(-2.8, -1.2, -4.0, -2.3)
    add_rect(1.5, 3.2, -4.0, -2.3)
    add_rect(5.3, 7.0, -4.0, -2.3)
    
    # === Row 3: Central area (key navigation zone) ===
    # Left side
    add_rect(-7.0, -5.7, -0.9, 0.9)
    add_rect(-4.2, -2.2, -1.2, 0.4)
    # Center corridor - clear passage with side obstacles
    # In v2, shift the central corridor slightly to change the optimal route
    add_rect(-1.9 + dx_c, -0.9 + dx_c, -0.9 + dy_c, 0.9 + dy_c)
    add_rect(1.1 + dx_c, 2.1 + dx_c, -0.9 + dy_c, 0.9 + dy_c)
    # Right side
    # add_rect(3.5, 4.7, -1.2, 0.4)
    add_rect(5.7, 7.0, -0.9, 0.9)
    
    # === Row 4: Upper area (strategic gaps) ===
    add_rect(-7.0, -5.3, 2.2, 3.8)
    add_rect(-3.2, -1.5, 2.2, 3.8)
    add_rect(1.8, 3.5, 2.2, 3.8)
    add_rect(5.3, 7.0, 2.2, 3.8)
    
    # === Row 5: Exit area (more open) ===
    add_rect(-6.3, -4.8, 5.2, 6.8)
    add_rect(-2.8 + dx_r, -1.3 + dx_r, 5.2 + dy_r, 6.8 + dy_r)
    add_rect(1.5 + dx_r, 3.0 + dx_r, 5.2 + dy_r, 6.8 + dy_r)
    add_rect(5.0, 6.5, 5.2, 6.8)
    
    # === Circular obstacles (strategic placement) ===
    # Lower area guidance
    add_circle([-5.5, -3.0], 0.32)
    add_circle([2.3, -3.0], 0.32)
    
    # Central area - minimal but strategic
    add_circle([-6.2, 0.0], 0.28)
    add_circle([6.2, 0.0], 0.28)
    add_circle([-0.3 + dx_c, 0.0 + dy_c], 0.22)  # Central marker (shifted in v2)
    
    # Upper area
    add_circle([-5.0, 3.0], 0.3)
    # add_circle([4.5, 3.0], 0.3)
    
    # === Small obstacles for path guidance ===
    add_rect(-1.5, -1.0, -5.3, -4.9)
    add_rect(0.8 + dx_c, 1.3 + dx_c, 1.0 + dy_c, 1.5 + dy_c)

    # Extra small obstacle only in v2 to alter the straight-up corridor slightly
    if variant == "v2":
        add_rect(-0.55, -0.10, 3.95, 4.35)
    
    # Start and goal positions
    start = np.array([-7.5, -7.5])  # Bottom-left entrance
    goal = np.array([2, 7.5])     # Top-right corner exit
    
    # === 人流量区域定义 ===
    crowd_regions = []
    
    # 中心区域 (3x3): 极高人流量 - 核心拥挤区域
    crowd_regions.append(CrowdRegion(
        bounds=(-1.5, 1.5, -1.5, 1.5),
        density_multiplier=18.0,
        name="Center-VeryHigh"
    ))
    
    # 左侧区域 (x < 0, y >= 0): 高人流量 - 热门区域
    crowd_regions.append(CrowdRegion(
        bounds=(-8, 0, 0, 8),
        density_multiplier=10.0,
        name="Left-High"
    ))
    
    # 右侧区域 (x >= 0, y >= 0): 中等人流量 - 普通区域
    crowd_regions.append(CrowdRegion(
        bounds=(0, 8, 0, 8),
        density_multiplier=2.0,
        name="Right-Medium"
    ))
    
    # 下方区域 (y < 0, x < 0): 低人流量 - 入口区域
    crowd_regions.append(CrowdRegion(
        bounds=(-8, 0, -8, 0),
        density_multiplier=1.0,
        name="Bottom-Low"
    ))

    crowd_regions.append(CrowdRegion(
        bounds=(-5, -3, -5, -3),
        density_multiplier=5.0,
        name="Bottom-Low"
    ))
    
    # 下方右侧 (y < 0, x >= 0): 中低人流量
    crowd_regions.append(CrowdRegion(
        bounds=(0, 8, -8, 0),
        density_multiplier=1.5,
        name="Bottom-Right-MedLow"
    ))
    
    return env, start, goal, bounds, crowd_regions
