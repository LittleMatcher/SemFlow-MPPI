"""
CFM FlowMP 2D 环境轨迹数据生成脚本

参照 flow_mp-main 的 generate_env_trajs.py 与 generator.py：
- 使用 B-spline 生成平滑轨迹 (pos, vel, acc)
- 生成 2D 障碍物并转为 cost_map [1, H, W]
- 保存为 .npz 供 cfm_flowmp.data.FlowMPEnvDataset 加载

用法:
    python -m cfm_flowmp.scripts.generate_data.generate_env_trajs_cfm --output_dir traj_data/cfm_env --num_trajs 500
"""

import argparse
import numpy as np
from pathlib import Path
from scipy.interpolate import make_interp_spline
from typing import Tuple, List, Optional


def generate_bspline_trajectories_2d(
    x: np.ndarray,
    y: np.ndarray,
    k: int,
    noise_scale: float,
    num_eval: int = 256,
    vel_acc: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    生成 2D B-spline 轨迹（参照 flow_mp.generator.generate_bspline_trajectories）。
    
    Args:
        x: 控制点 x 坐标 [N]
        y: 控制点 y 坐标 [N]
        k: 轨迹条数
        noise_scale: 控制点噪声
        num_eval: 每条轨迹采样点数
        vel_acc: 是否包含 vel/acc
        seed: 随机种子
        
    Returns:
        trajectories: [k, num_eval, 2] 或 [k, num_eval, 6] (pos, vel, acc)
    """
    if seed is not None:
        np.random.seed(seed)
    
    trajectories = []
    n = len(x)
    
    for _ in range(k):
        noisy_x = x.copy()
        noisy_y = y.copy()
        noisy_x[1:-1] += np.random.normal(scale=noise_scale, size=n - 2)
        noisy_y[1:-1] += np.random.normal(scale=noise_scale, size=n - 2)
        
        spl_x = make_interp_spline(
            np.arange(n), noisy_x, k=3,
            bc_type=([(1, 0)], [(1, 0)])
        )
        spl_y = make_interp_spline(
            np.arange(n), noisy_y, k=3,
            bc_type=([(1, 0)], [(1, 0)])
        )
        
        t = np.linspace(0, n - 1, num_eval)
        pos_x, pos_y = spl_x(t), spl_y(t)
        
        if not vel_acc:
            trajectories.append(np.stack([pos_x, pos_y], axis=-1))
        else:
            vel_x = spl_x.derivative(1)(t)
            vel_y = spl_y.derivative(1)(t)
            acc_x = spl_x.derivative(2)(t)
            acc_y = spl_y.derivative(2)(t)
            trajectories.append(np.stack([pos_x, pos_y, vel_x, vel_y, acc_x, acc_y], axis=-1))
    
    return np.array(trajectories, dtype=np.float32)


def obstacles_to_cost_map(
    map_size: int,
    bounds: Tuple[float, float, float, float],
    obstacles: List[dict],
) -> np.ndarray:
    """
    将障碍物列表转为 cost_map [1, H, W]，范围 [0,1] 归一化到 [0, map_size-1]。
    
    obstacles 每项可为:
        - {"type": "circle", "center": [cx, cy], "radius": r}  归一化 [0,1]
        - {"type": "rect", "min": [x0,y0], "max": [x1,y1]}    归一化 [0,1]
        - {"type": "gaussian", "center": [cx, cy], "sigma": s}
    """
    x_min, x_max, y_min, y_max = bounds
    cost_map = np.zeros((1, map_size, map_size), dtype=np.float32)
    
    def to_grid(px: float, py: float) -> Tuple[int, int]:
        ix = int((px - x_min) / (x_max - x_min) * (map_size - 1))
        iy = int((py - y_min) / (y_max - y_min) * (map_size - 1))
        return np.clip(ix, 0, map_size - 1), np.clip(iy, 0, map_size - 1)
    
    for obs in obstacles:
        if obs["type"] == "circle":
            cx, cy = obs["center"]
            r = obs["radius"]
            gx, gy = to_grid(cx, cy)
            r_pix = max(1, int(r * map_size))
            yg, xg = np.ogrid[-r_pix : r_pix + 1, -r_pix : r_pix + 1]
            mask = (xg * xg + yg * yg <= r_pix * r_pix).astype(np.float32)
            y_start = max(0, gy - r_pix)
            y_end = min(map_size, gy + r_pix + 1)
            x_start = max(0, gx - r_pix)
            x_end = min(map_size, gx + r_pix + 1)
            my0, my1 = r_pix - (gy - y_start), r_pix + (y_end - gy)
            mx0, mx1 = r_pix - (gx - x_start), r_pix + (x_end - gx)
            my1 = min(my1, 2 * r_pix + 1)
            mx1 = min(mx1, 2 * r_pix + 1)
            cost_map[0, y_start:y_end, x_start:x_end] = np.maximum(
                cost_map[0, y_start:y_end, x_start:x_end],
                mask[my0:my1, mx0:mx1]
            )
        elif obs["type"] == "gaussian":
            cx, cy = obs["center"]
            sigma = obs["sigma"]
            gx, gy = to_grid(cx, cy)
            sigma_pix = sigma * map_size
            r_pix = int(3 * sigma_pix)
            yg, xg = np.ogrid[-r_pix : r_pix + 1, -r_pix : r_pix + 1]
            g = np.exp(-(xg**2 + yg**2) / (2 * sigma_pix**2)).astype(np.float32)
            y_start = max(0, gy - r_pix)
            y_end = min(map_size, gy + r_pix + 1)
            x_start = max(0, gx - r_pix)
            x_end = min(map_size, gx + r_pix + 1)
            gy0 = max(0, -gy + r_pix)
            gy1 = gy0 + (y_end - y_start)
            gx0 = max(0, -gx + r_pix)
            gx1 = gx0 + (x_end - x_start)
            cost_map[0, y_start:y_end, x_start:x_end] = np.maximum(
                cost_map[0, y_start:y_end, x_start:x_end],
                g[gy0:gy1, gx0:gx1]
            )
        elif obs["type"] == "rect":
            x0, y0 = obs["min"]
            x1, y1 = obs["max"]
            i0, j0 = to_grid(x0, y0)
            i1, j1 = to_grid(x1, y1)
            cost_map[0, j0:j1+1, i0:i1+1] = 1.0
    
    return cost_map


def generate_random_obstacles(
    bounds: Tuple[float, float, float, float],
    num_obstacles: int,
    obstacle_type: str = "gaussian",
    sigma_range: Tuple[float, float] = (0.05, 0.15),
    seed: Optional[int] = None,
) -> List[dict]:
    """生成随机障碍物（归一化坐标）。"""
    if seed is not None:
        np.random.seed(seed)
    x_min, x_max, y_min, y_max = bounds
    obstacles = []
    for _ in range(num_obstacles):
        cx = np.random.uniform(x_min + 0.1, x_max - 0.1)
        cy = np.random.uniform(y_min + 0.1, y_max - 0.1)
        if obstacle_type == "gaussian":
            sigma = np.random.uniform(*sigma_range)
            obstacles.append({"type": "gaussian", "center": [cx, cy], "sigma": sigma})
        else:
            r = np.random.uniform(0.03, 0.08)
            obstacles.append({"type": "circle", "center": [cx, cy], "radius": r})
    return obstacles


def resample_trajectory(traj: np.ndarray, num_points: int) -> np.ndarray:
    """将轨迹 [T, D] 重采样为 num_points 个点。"""
    T, D = traj.shape
    if T == num_points:
        return traj
    indices = np.linspace(0, T - 1, num_points, dtype=np.int32)
    return traj[indices].astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Generate 2D env trajectories + cost maps for CFM FlowMP")
    parser.add_argument("--output_dir", type=str, default="traj_data/cfm_env",
                        help="Output directory for .npz and metadata")
    parser.add_argument("--num_trajs", type=int, default=500,
                        help="Number of trajectories to generate")
    parser.add_argument("--seq_len", type=int, default=64,
                        help="Trajectory sequence length (resampled)")
    parser.add_argument("--map_size", type=int, default=64,
                        help="Cost map size (map_size x map_size)")
    parser.add_argument("--bounds", type=float, nargs=4, default=[0.0, 1.0, 0.0, 1.0],
                        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
                        help="Environment bounds (normalized)")
    parser.add_argument("--num_via", type=int, default=13,
                        help="Number of via points for B-spline")
    parser.add_argument("--noise_scale", type=float, default=0.01,
                        help="B-spline control point noise")
    parser.add_argument("--num_obstacles", type=int, default=4,
                        help="Number of random obstacles per scene")
    parser.add_argument("--obstacle_type", type=str, default="gaussian",
                        choices=["gaussian", "circle"],
                        help="Obstacle shape")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    bounds = tuple(args.bounds)
    x_min, x_max, y_min, y_max = bounds
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 为每条轨迹生成：随机起点/终点 -> B-spline -> 重采样；同一场景共享障碍物与 cost_map
    # 简化：每个“场景”固定若干条轨迹，共享一个 cost_map；或每条轨迹独立随机障碍（更简单）
    all_positions = []   # [N, seq_len, 2]
    all_velocities = []
    all_accelerations = []
    all_cost_maps = []   # [N, 1, H, W]
    all_start_states = []  # [N, 6]
    all_goal_states = []  # [N, 4]
    all_style_weights = []  # [N, 2] 与 MockL3Dataset 一致
    
    rng = np.random.default_rng(args.seed)
    
    for i in range(args.num_trajs):
        # 随机起点、终点（归一化）
        start = rng.uniform([x_min + 0.1, y_min + 0.1], [x_min + 0.3, y_min + 0.3])
        goal = rng.uniform([x_max - 0.3, y_max - 0.3], [x_max - 0.1, y_max - 0.1])
        
        # 控制点：起点 -> 中间扰动 -> 终点
        num_via = args.num_via
        t_via = np.linspace(0, 1, num_via)
        x_via = start[0] + (goal[0] - start[0]) * t_via
        y_via = start[1] + (goal[1] - start[1]) * t_via
        x_via[1:-1] += rng.normal(0, 0.05, num_via - 2)
        y_via[1:-1] += rng.normal(0, 0.05, num_via - 2)
        
        trajs = generate_bspline_trajectories_2d(
            x_via, y_via, k=1, noise_scale=args.noise_scale,
            num_eval=256, vel_acc=True, seed=args.seed + i
        )
        traj = trajs[0]  # [256, 6]
        pos = traj[:, :2]
        vel = traj[:, 2:4]
        acc = traj[:, 4:6]
        
        pos = resample_trajectory(pos, args.seq_len)
        vel = resample_trajectory(vel, args.seq_len)
        acc = resample_trajectory(acc, args.seq_len)
        
        obstacles = generate_random_obstacles(
            bounds, args.num_obstacles, args.obstacle_type,
            seed=args.seed + i + 1000
        )
        cost_map = obstacles_to_cost_map(args.map_size, bounds, obstacles)
        cost_map = cost_map[np.newaxis, ...]  # [1, 1, H, W]
        
        start_state = np.concatenate([pos[0], vel[0], acc[0]], axis=-1).astype(np.float32)
        goal_state = np.concatenate([pos[-1], vel[-1]], axis=-1).astype(np.float32)
        style = rng.dirichlet([1.0, 1.0]).astype(np.float32)
        
        all_positions.append(pos)
        all_velocities.append(vel)
        all_accelerations.append(acc)
        all_cost_maps.append(cost_map)
        all_start_states.append(start_state)
        all_goal_states.append(goal_state)
        all_style_weights.append(style)
    
    positions = np.stack(all_positions, axis=0)
    velocities = np.stack(all_velocities, axis=0)
    accelerations = np.stack(all_accelerations, axis=0)
    cost_maps = np.concatenate(all_cost_maps, axis=0)
    start_states = np.stack(all_start_states, axis=0)
    goal_states = np.stack(all_goal_states, axis=0)
    style_weights = np.stack(all_style_weights, axis=0)
    
    npz_path = out_dir / "data.npz"
    np.savez_compressed(
        npz_path,
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        cost_maps=cost_maps,
        start_states=start_states,
        goal_states=goal_states,
        style_weights=style_weights,
        seq_len=args.seq_len,
        map_size=args.map_size,
    )
    print(f"Saved {args.num_trajs} samples to {npz_path}")
    print(f"  positions: {positions.shape}, cost_maps: {cost_maps.shape}")


if __name__ == "__main__":
    main()
