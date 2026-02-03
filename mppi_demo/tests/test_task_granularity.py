"""
测试任务粒度和并行效率
"""
import numpy as np
import time
from mppi_core.bspline_trajectory import BSplineTrajectory

def test_task_granularity():
    """测试单个任务的执行时间"""
    
    print("测试B-Spline轨迹生成的任务粒度")
    print("="*60)
    
    # 创建B-Spline参数
    degree = 3
    n_control_points = 10
    time_horizon = 5.0
    n_timesteps = 50
    
    bspline = BSplineTrajectory(
        degree=degree,
        n_control_points=n_control_points,
        time_horizon=time_horizon,
        dim=2
    )
    
    # 随机控制点
    control_points = np.random.randn(n_control_points, 2)
    
    # 测试单次执行时间
    n_tests = 100
    times = []
    
    for _ in range(n_tests):
        start = time.time()
        traj = bspline.evaluate(control_points, n_samples=n_timesteps)
        times.append(time.time() - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"\n单个轨迹生成:")
    print(f"  平均时间: {avg_time*1000:.3f} ms")
    print(f"  标准差: {std_time*1000:.3f} ms")
    print(f"  最小: {min(times)*1000:.3f} ms")
    print(f"  最大: {max(times)*1000:.3f} ms")
    
    # 估算400个样本的理论时间
    n_samples = 400
    serial_time = avg_time * n_samples
    parallel_time_16 = serial_time / 16  # 假设完美加速
    overhead = 0.001  # 假设1ms的进程通信开销
    
    print(f"\n对于{n_samples}个样本:")
    print(f"  串行执行时间: {serial_time*1000:.1f} ms")
    print(f"  理想16核并行: {parallel_time_16*1000:.1f} ms")
    print(f"  进程通信开销 (估计): {overhead*1000:.1f} ms/任务")
    print(f"  实际并行预期: {(parallel_time_16 + overhead*n_samples)*1000:.1f} ms")
    
    if avg_time < 0.001:  # 小于1ms
        print(f"\n⚠ 警告：任务粒度太小！")
        print(f"  单个任务只需 {avg_time*1000:.3f} ms")
        print(f"  进程通信开销可能超过计算时间")
        print(f"  建议：增加批处理大小或使用更复杂的计算")
    elif avg_time < 0.01:  # 小于10ms
        print(f"\n⚠ 任务粒度偏小")
        print(f"  并行效率可能受影响")
    else:
        print(f"\n✓ 任务粒度合适")

if __name__ == '__main__':
    test_task_granularity()
