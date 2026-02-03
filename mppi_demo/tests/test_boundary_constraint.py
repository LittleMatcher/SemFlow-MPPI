"""
测试边界约束功能
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mppi_core import BoundaryConstraintCost

def test_boundary_visualization():
    """可视化边界约束效果"""
    print("=" * 60)
    print("边界约束代价函数 - 可视化测试")
    print("=" * 60)
    
    bounds = (-8, 8, -8, 8)
    boundary_cost = BoundaryConstraintCost(
        bounds=bounds,
        margin=0.5,
        weight=200.0,
        use_hard_constraint=True
    )
    
    # 创建网格来可视化代价场
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    
    # 计算每个点的代价
    costs = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([[[X[i, j], Y[i, j]]]])
            costs[i, j] = boundary_cost(point)[0]
    
    # 限制显示范围以便看清楚
    costs_display = np.log10(costs + 1)  # 对数尺度
    
    # 绘图
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # 绘制代价场
    im = ax.contourf(X, Y, costs_display, levels=20, cmap='YlOrRd')
    ax.contour(X, Y, costs_display, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    
    # 绘制边界
    ax.plot([bounds[0], bounds[1], bounds[1], bounds[0], bounds[0]],
            [bounds[2], bounds[2], bounds[3], bounds[3], bounds[2]],
            'b-', linewidth=3, label='Hard Boundary')
    
    # 绘制安全边距
    margin = 0.5
    ax.plot([bounds[0]+margin, bounds[1]-margin, bounds[1]-margin, bounds[0]+margin, bounds[0]+margin],
            [bounds[2]+margin, bounds[2]+margin, bounds[3]-margin, bounds[3]-margin, bounds[2]+margin],
            'g--', linewidth=2, label='Safety Margin')
    
    # 绘制测试路径
    path_inside = np.array([[-5, -5], [0, 0], [5, 5]])
    ax.plot(path_inside[:, 0], path_inside[:, 1], 'g-', linewidth=3, 
            marker='o', markersize=8, label='Safe Path')
    
    path_near = np.array([[-7.8, -7.8], [0, 0], [7.8, 7.8]])
    ax.plot(path_near[:, 0], path_near[:, 1], 'y-', linewidth=3,
            marker='s', markersize=8, label='Near Boundary Path')
    
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Boundary Constraint Cost Field (Log Scale)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log10(cost + 1)', fontsize=11)
    
    # 保存
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'boundary_constraint.png'),
                dpi=150, bbox_inches='tight')
    
    print("\n代价统计:")
    print(f"  边界内区域: cost ≈ 0")
    print(f"  接近边界(margin内): cost 快速增长 (barrier函数)")
    print(f"  超出边界: cost = 1e6 (硬约束)")
    print(f"\n✓ 可视化已保存: outputs/boundary_constraint.png")
    print("=" * 60)

if __name__ == "__main__":
    test_boundary_visualization()
