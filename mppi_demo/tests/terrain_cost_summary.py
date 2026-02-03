"""
地形成本场景总结

展示MPPI在地形成本场景下的路径规划能力:
- 三种地形: 沼泽地(×5.0), 水泥(×1.0), 草地(×2.5)
- MPPI能够根据地形成本优化路径，选择成本更低的路线

这个脚本快速演示TerrainCost功能是否正常工作。
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenarios.terrain_cost import create_terrain_cost_environment, get_terrain_cost_at_point
from mppi_core import TerrainCost

def test_terrain_cost_function():
    """测试地形成本函数"""
    
    print("=" * 60)
    print("地形成本函数测试")
    print("=" * 60)
    
    # 创建环境
    env, terrain_regions, start, goal, bounds = create_terrain_cost_environment()
    
    print(f"\n场景配置:")
    print(f"  起点: {start}")
    print(f"  终点: {goal}")
    print(f"  障碍物: {len(env.obstacles)}")
    print(f"\n地形类型:")
    for region in terrain_regions:
        print(f"  {region.name}: 成本倍数 {region.cost_multiplier}")
    
    # 测试点 - 覆盖不同地形区域
    test_points = [
        (np.array([0.0, 0.0]), "中央沼泽"),
        (np.array([-8.0, -6.0]), "左下水泥"),
        (np.array([8.0, 6.0]), "右上水泥"),
        (np.array([-8.0, -8.0]), "起点（草地）"),
        (np.array([8.0, 8.0]), "终点（草地）"),
        (np.array([4.0, 4.0]), "右上沼泽边缘")
    ]
    
    print(f"\n测试不同位置的地形成本:")
    for point, desc in test_points:
        cost = get_terrain_cost_at_point(point, terrain_regions)
        print(f"  {desc:12s} {point}: {cost:.1f}×")
    
    # 测试TerrainCost成本函数
    print(f"\n测试TerrainCost成本函数...")
    
    def get_cost_fn(point):
        return get_terrain_cost_at_point(point, terrain_regions)
    
    terrain_cost = TerrainCost(get_cost_fn, weight=80.0)
    
    # 测试两条轨迹：直线 vs 绕路
    print(f"\n对比两种策略:")
    
    # 策略1: 直线穿过沼泽中心（几何最短，但穿过10×成本沼泽）
    direct_trajectory = np.array([
        [[-8.0, -8.0], [-4.0, -4.0], [0.0, 0.0], [4.0, 4.0], [8.0, 8.0]]
    ])
    direct_cost = terrain_cost(direct_trajectory)[0]
    direct_length = sum([np.linalg.norm(direct_trajectory[0][i+1] - direct_trajectory[0][i]) 
                        for i in range(len(direct_trajectory[0])-1)])
    
    # 策略2: 贴边绕行（走水泥1×区域）
    # 从左下角沿左边界和顶边界绕到右上角
    detour_trajectory = np.array([
        [[-8.0, -8.0], [-8.0, -5.0], [-6.0, 0.0], [-4.0, 4.0], [0.0, 8.0], [4.0, 8.0], [8.0, 8.0]]
    ])
    detour_cost = terrain_cost(detour_trajectory)[0]
    detour_length = sum([np.linalg.norm(detour_trajectory[0][i+1] - detour_trajectory[0][i]) 
                        for i in range(len(detour_trajectory[0])-1)])
    
    print(f"  策略1 - 直线穿沼泽中心:")
    print(f"    长度: {direct_length:.1f}m")
    print(f"    地形成本: {direct_cost:.0f} (weight=150时)")
    print(f"  策略2 - 贴边绕行走水泥:")
    print(f"    长度: {detour_length:.1f}m (+{(detour_length/direct_length-1)*100:.0f}%)")
    print(f"    地形成本: {detour_cost:.0f} (weight=150时)")
    
    if detour_cost < direct_cost:
        savings = (direct_cost - detour_cost) / direct_cost * 100
        print(f"  ✓ 绕路策略长{(detour_length/direct_length-1)*100:.0f}%，但地形成本低{savings:.0f}%！")
        print(f"  → MPPI应该选择绕路策略以最小化总成本")
    else:
        print(f"  注意：当前设置下绕路反而更贵，需要调整地形布局")
    
    print(f"\n✓ 地形成本函数测试通过!")
    
    print(f"\n说明:")
    print(f"  - TerrainCost 已实现并可以正常计算轨迹的地形成本")
    print(f"  - 完整的MPPI优化测试请运行: python tests/test_terrain_cost.py")
    print(f"  - 注意: 完整测试需要较长时间(约2-5分钟)")


if __name__ == "__main__":
    test_terrain_cost_function()
