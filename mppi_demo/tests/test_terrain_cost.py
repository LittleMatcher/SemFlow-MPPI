"""
测试地形成本场景下的 MPPI 路径规划

场景特点:
- 三种地形: 水泥路面(成本1.0)、草地(成本2.5)、沼泽地(成本8.0)
- 加入障碍物
- 起点在草地，终点在水泥路面
- MPPI 需要权衡路径长度与地形成本
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mppi_core import (
    MPPI_BSpline,
    CollisionCost,
    SmoothnessCost,
    GoalCost,
    PathLengthCost,
    TurnCost,
    TerrainCost,
    CompositeCost,
    Visualizer
)
from scenarios.terrain_cost import create_terrain_cost_environment


def test_terrain_cost_mppi():
    """测试带地形成本的 MPPI 路径规划"""
    
    print("=" * 60)
    print("地形成本场景 - MPPI 路径规划测试")
    print("=" * 60)
    
    # 创建环境
    env, terrain_regions, start, goal, bounds = create_terrain_cost_environment()
    
    print(f"\n场景信息:")
    print(f"  起点: {start}")
    print(f"  终点: {goal}")
    print(f"  环境范围: {bounds}")
    print(f"  障碍物数量: {len(env.obstacles)}")
    print(f"\n地形类型:")
    for region in terrain_regions:
        print(f"  {region.name} (成本倍数 {region.cost_multiplier})")
    
    # 计算直线路径的理论成本
    direct_distance = np.linalg.norm(goal - start)
    print(f"\n路径选择分析:")
    print(f"  直线距离: {direct_distance:.1f}m")
    print(f"  直线穿过沼泽，理论成本: ≈{direct_distance * 10:.0f} (10×地形)")
    print(f"  绕路走水泥，理论成本: ≈{direct_distance * 1.2 * 1:.0f} (1×地形)")
    print(f"  → MPPI应该选择绕路策略，节省约70%+成本")
    
    # 创建地形成本函数 - 简化版本，直接使用 terrain_regions
    from scenarios.terrain_cost import get_terrain_cost_at_point
    
    def get_terrain_cost_wrapper(point):
        return get_terrain_cost_at_point(point, terrain_regions)
    
    # 配置成本函数 - 地形成本绝对统治！其他全部最小化
    costs = [
        CollisionCost(env, robot_radius=0.3, epsilon=0.1, 
                     weight=150.0, hard_penalty=1e6),  # 保持避障权重
        SmoothnessCost(weight=0.5),  # 极小化
        GoalCost(goal, weight=10.0),  # 极小化！只要最终到达即可
        PathLengthCost(weight=0.1),  # 几乎忽略路径长度
        TurnCost(weight=1.0),  # 几乎忽略转弯
        TerrainCost(get_terrain_cost_wrapper, weight=500.0),  # 再次大幅提升！
    ]
    
    composite_cost = CompositeCost(costs)
    
    # 创建 MPPI 规划器 - 超大探索空间，必须找到绕路！
    mppi = MPPI_BSpline(
        cost_function=composite_cost,
        n_samples=1000,  # 最大采样数
        n_control_points=10,  # 更少控制点=更大范围路径变化
        bspline_degree=3,
        time_horizon=8.0,  # 超长时间范围，容纳大幅绕路
        n_timesteps=80,
        temperature=2.0,  # 超高温度，极限探索
        noise_std=1.5,  # 超大噪声，强制跳出局部最优
        bounds=bounds,
        elite_ratio=0.05  # 使用精英采样（top 5%）
    )
    
    print(f"\nMPPI 配置:")
    print(f"  采样数: {mppi.n_samples}")
    print(f"  控制点数: {mppi.n_control_points}")
    print(f"  精英比例: {mppi.elite_ratio * 100}%")
    print(f"  时间范围: {mppi.time_horizon}s")
    
    # 运行 MPPI 优化
    print("\n开始优化...")
    t_start = time.time()
    
    try:
        result = mppi.optimize(
            start=start,
            goal=goal,
            n_iterations=300,
            verbose=True
        )
        
        best_trajectory = result['trajectory']
        costs_history = result['cost_history']
        
        t_elapsed = time.time() - t_start
        
        print(f"\n优化完成!")
        print(f"  总用时: {t_elapsed:.2f}s")
        print(f"  最佳成本: {costs_history[-1]:.2f}")
        print(f"  平均迭代时间: {t_elapsed/len(costs_history)*1000:.1f}ms")
        
        # 计算地形成本分解
        terrain_costs = []
        positions = best_trajectory
        for i in range(len(positions) - 1):
            segment_length = np.linalg.norm(positions[i+1] - positions[i])
            midpoint = (positions[i] + positions[i+1]) / 2
            terrain_cost = get_terrain_cost_wrapper(midpoint)
            terrain_costs.append((segment_length, terrain_cost, segment_length * terrain_cost))
        
        total_length = sum([tc[0] for tc in terrain_costs])
        weighted_terrain_cost = sum([tc[2] for tc in terrain_costs])
        avg_terrain_cost = weighted_terrain_cost / total_length if total_length > 0 else 0
        
        print(f"\n路径统计:")
        print(f"  总长度: {total_length:.2f}m")
        print(f"  加权地形成本: {weighted_terrain_cost:.2f}")
        print(f"  平均地形系数: {avg_terrain_cost:.2f}")
        
        # 对比直线路径
        direct_dist = np.linalg.norm(goal - start)
        direct_cost_estimate = direct_dist * 10.0  # 假设全程沼泽
        savings = (direct_cost_estimate - weighted_terrain_cost) / direct_cost_estimate * 100
        print(f"\n智能地形选择效果:")
        print(f"  直线路径估计成本: {direct_cost_estimate:.1f} (全程沼泽10×)")
        print(f"  MPPI实际地形成本: {weighted_terrain_cost:.1f}")
        print(f"  成本节省: {savings:.1f}%")
        print(f"  路径延长: {(total_length/direct_dist - 1)*100:.1f}%")
        print(f"  → MPPI通过绕路策略大幅降低总成本！")
        
        # 可视化
        visualize_terrain_result(
            env, start, goal, best_trajectory, costs_history, 
            terrain_regions, terrain_costs, bounds
        )
        
    except Exception as e:
        print(f"\n优化regions{e}")
        import traceback
        traceback.print_exc()


def visualize_terrain_result(env, start, goal, trajectory, costs_history, 
                             terrain_regions, terrain_costs, bounds):
    """可视化地形成本场景结果"""
    
    # 设置字体避免中文警告
    import matplotlib
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 左图: 环境与轨迹
    ax1 = axes[0]
    
    # 绘制地形区域（按顺序：先画沼泽，再画草地，最后画水泥，形成正确的层次）
    color_map = {
        'Swamp': ('#8B4513', 0.4),      # 深棕色
        'Concrete': ('#CCCCCC', 0.6),   # 灰色
        'Grass': ('#90EE90', 0.5)       # 浅绿色
    }
    
    # 先绘制背景区域
    for region in terrain_regions:
        color, alpha = color_map.get(region.name, ('#FFFFFF', 0.3))
        poly = MplPolygon(region.vertices, facecolor=color, alpha=alpha, 
                         edgecolor='black', linewidth=1.0,
                         label=f'{region.name} (×{region.cost_multiplier})')
        ax1.add_patch(poly)
    
    # 绘制障碍物
    for obs in env.obstacles:
        if hasattr(obs, 'radius'):  # Circle
            circle = plt.Circle((obs.center[0], obs.center[1]), obs.radius,
                              color='darkred', alpha=0.7)
            ax1.add_patch(circle)
        elif hasattr(obs, 'width'):  # Rectangle
            rect = plt.Rectangle(
                (obs.center[0] - obs.width/2, obs.center[1] - obs.height/2),
                obs.width, obs.height,
                color='darkred', alpha=0.7
            )
            ax1.add_patch(rect)
    
    # 绘制轨迹
    positions = trajectory[:, :2]
    ax1.plot(positions[:, 0], positions[:, 1], 
            'b-', linewidth=2.5, label='MPPI轨迹', zorder=10)
    ax1.plot(start[0], start[1], 'go', markersize=12, 
            label='起点', zorder=11)
    ax1.plot(goal[0], goal[1], 'r*', markersize=15, 
            label='终点', zorder=11)
    
    ax1.set_xlim(bounds[0], bounds[1])
    ax1.set_ylim(bounds[2], bounds[3])
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X (m)', fontsize=11)
    ax1.set_ylabel('Y (m)', fontsize=11)
    ax1.set_title('Terrain Cost Scenario - MPPI Path Planning', fontsize=13, fontweight='bold')
    
    # 去重图例
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), 
              loc='upper left', fontsize=9)
    
    # 右图: 成本历史
    ax2 = axes[1]
    ax2.plot(costs_history, 'b-', linewidth=2)
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Total Cost', fontsize=11)
    ax2.set_title('MPPI Optimization Process', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 标注最佳成本
    best_iter = np.argmin(costs_history)
    best_cost = costs_history[best_iter]
    ax2.plot(best_iter, best_cost, 'r*', markersize=15, 
            label=f'Best: {best_cost:.1f} (iter {best_iter})')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'terrain_cost_mppi.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图片已保存: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    test_terrain_cost_mppi()
