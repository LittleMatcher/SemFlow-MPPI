"""
CFM FlowMP 2D 环境轨迹数据生成脚本

- 使用路径规划算法（A* 或 MPPI）在 cost_map 上规划轨迹，再经 B-spline 平滑得到 (pos, vel, acc)
- MPPI：在连续空间用双积分器动力学 + 障碍/目标/控制代价迭代优化，得到平滑避障轨迹
- 起点与终点仅在自由空间中采样，不允许落在障碍物上
- 运动学参数（速度、加速度）由轨迹 B-spline 求导得到，与轨迹一致
- 保存为 .npz 供 cfm_flowmp.data.FlowMPEnvDataset 加载

用法:
    python -m cfm_flowmp.scripts.generate_data.generate_env_trajs_cfm --output_dir traj_data/cfm_env --num_trajs 500
    python -m cfm_flowmp.scripts.generate_data.generate_env_trajs_cfm --planner mppi --num_trajs 500 --num_workers 8
    python -m cfm_flowmp.scripts.generate_data.generate_env_trajs_cfm --planner astar --num_trajs 1000 --num_workers 4
"""

import argparse
import heapq
import multiprocessing as mp
import numpy as np
from pathlib import Path
from scipy.interpolate import make_interp_spline
from typing import Tuple, List, Optional, Dict, Any


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


def cost_at_point(
    px: float,
    py: float,
    cost_map: np.ndarray,
    bounds: Tuple[float, float, float, float],
    map_size: int,
) -> float:
    """查询归一化坐标 (px, py) 在 cost_map 上的代价值（双线性插值）。"""
    x_min, x_max, y_min, y_max = bounds
    ix = (px - x_min) / (x_max - x_min) * (map_size - 1)
    iy = (py - y_min) / (y_max - y_min) * (map_size - 1)
    ix = np.clip(ix, 0, map_size - 1)
    iy = np.clip(iy, 0, map_size - 1)
    # 双线性插值
    i0, j0 = int(np.floor(ix)), int(np.floor(iy))
    i1, j1 = min(i0 + 1, map_size - 1), min(j0 + 1, map_size - 1)
    fx, fy = ix - i0, iy - j0
    c00 = cost_map[0, j0, i0]
    c10 = cost_map[0, j0, i1]
    c01 = cost_map[0, j1, i0]
    c11 = cost_map[0, j1, i1]
    return float(
        (1 - fx) * (1 - fy) * c00 + fx * (1 - fy) * c10
        + (1 - fx) * fy * c01 + fx * fy * c11
    )


def is_point_in_free_space(
    px: float,
    py: float,
    cost_map: np.ndarray,
    bounds: Tuple[float, float, float, float],
    map_size: int,
    threshold: float = 0.1,
) -> bool:
    """判断归一化坐标 (px, py) 是否在自由空间（不在障碍物上）。"""
    return cost_at_point(px, py, cost_map, bounds, map_size) < threshold


def sample_start_goal_in_free_space(
    cost_map: np.ndarray,
    bounds: Tuple[float, float, float, float],
    map_size: int,
    rng: np.random.Generator,
    margin: float = 0.05,
    start_region: Optional[Tuple[float, float, float, float]] = None,
    goal_region: Optional[Tuple[float, float, float, float]] = None,
    max_attempts: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在自由空间中采样起点和终点（不允许落在障碍物上）。
    start_region / goal_region: (x_min, x_max, y_min, y_max) 归一化，None 则用左下/右上区域。
    """
    x_min, x_max, y_min, y_max = bounds
    if start_region is None:
        start_region = (x_min + margin, x_min + 0.25, y_min + margin, y_max * 0.5)
    if goal_region is None:
        goal_region = (x_max - 0.25, x_max - margin, y_max * 0.5, y_max - margin)
    sx_min, sx_max, sy_min, sy_max = start_region
    gx_min, gx_max, gy_min, gy_max = goal_region
    for _ in range(max_attempts):
        start = rng.uniform([sx_min, sy_min], [sx_max, sy_max])
        goal = rng.uniform([gx_min, gy_min], [gx_max, gy_max])
        if is_point_in_free_space(start[0], start[1], cost_map, bounds, map_size) and is_point_in_free_space(
            goal[0], goal[1], cost_map, bounds, map_size
        ):
            return start.astype(np.float32), goal.astype(np.float32)
    raise RuntimeError(
        "sample_start_goal_in_free_space: failed to sample start/goal in free space after max_attempts"
    )


def plan_path_astar(
    cost_map: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    bounds: Tuple[float, float, float, float],
    map_size: int,
    occupancy_threshold: float = 0.25,
) -> List[Tuple[float, float]]:
    """
    在栅格地图上用 A* 规划从 start 到 goal 的路径（避开障碍物）。
    返回归一化坐标下的路径点列表 [(x1,y1), (x2,y2), ...]。
    """
    x_min, x_max, y_min, y_max = bounds

    def to_grid(px: float, py: float) -> Tuple[int, int]:
        ix = int((px - x_min) / (x_max - x_min) * (map_size - 1))
        iy = int((py - y_min) / (y_max - y_min) * (map_size - 1))
        return np.clip(ix, 0, map_size - 1), np.clip(iy, 0, map_size - 1)

    def to_world(ix: int, iy: int) -> Tuple[float, float]:
        px = x_min + (ix / (map_size - 1)) * (x_max - x_min)
        py = y_min + (iy / (map_size - 1)) * (y_max - y_min)
        return px, py

    occupancy = (cost_map[0] >= occupancy_threshold).astype(np.uint8)
    s_ix, s_iy = to_grid(start[0], start[1])
    g_ix, g_iy = to_grid(goal[0], goal[1])
    if occupancy[s_iy, s_ix] or occupancy[g_iy, g_ix]:
        return [tuple(start.tolist()), tuple(goal.tolist())]

    # A* on grid (8-neighbor)
    counter = [0]
    open_set = [(0.0, 0, s_iy, s_ix)]
    came_from: dict = {}
    g_score: dict = {}
    g_score[(s_iy, s_ix)] = 0.0
    while open_set:
        _, _, cy, cx = heapq.heappop(open_set)
        if (cy, cx) == (g_iy, g_ix):
            path_grid = []
            while (cy, cx) in came_from:
                path_grid.append((cx, cy))
                cy, cx = came_from[(cy, cx)]
            path_grid.append((s_ix, s_iy))
            path_grid.reverse()
            return [to_world(ix, iy) for ix, iy in path_grid]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                ny, nx = cy + dy, cx + dx
                if nx < 0 or nx >= map_size or ny < 0 or ny >= map_size:
                    continue
                if occupancy[ny, nx]:
                    continue
                step = 1.414 if (dx != 0 and dy != 0) else 1.0
                tentative = g_score.get((cy, cx), float("inf")) + step
                if tentative < g_score.get((ny, nx), float("inf")):
                    came_from[(ny, nx)] = (cy, cx)
                    g_score[(ny, nx)] = tentative
                    h = np.hypot(nx - g_ix, ny - g_iy)
                    counter[0] += 1
                    heapq.heappush(open_set, (tentative + h, counter[0], ny, nx))
    return [tuple(start.tolist()), tuple(goal.tolist())]


def plan_trajectory_mppi(
    cost_map: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    bounds: Tuple[float, float, float, float],
    map_size: int,
    horizon: int = 64,
    num_samples: int = 512,
    num_iterations: int = 20,
    dt: float = 0.02,
    control_std: float = 2.0,
    temperature: float = 0.1,
    obstacle_scale: float = 10.0,
    goal_scale: float = 5.0,
    control_scale: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[float, float]]:
    """
    使用 MPPI (Model Predictive Path Integral) 在连续空间优化轨迹，
    得到平滑、避障的路径。动力学为二维双积分器 (x,y,vx,vy) + 控制 (ax,ay)。
    返回归一化坐标下的路径点列表，供 path_to_trajectory_with_kinematics 使用。
    """
    if rng is None:
        rng = np.random.default_rng()
    x_min, x_max, y_min, y_max = bounds

    # 状态 [x, y, vx, vy]，控制 [ax, ay]；零初速下 s = 0.5*a*T^2 => a = 2*(goal-start)/T^2
    total_time = max((horizon - 1) * dt, 1e-6)
    nominal_acc = 2.0 * (goal - start) / (total_time ** 2)
    nominal_u = np.tile(nominal_acc, (horizon, 1)).astype(np.float64)
    positions = np.linspace(start, goal, horizon + 1).astype(np.float64)

    for _ in range(num_iterations):
        # 采样控制扰动
        u_samples = nominal_u[np.newaxis, :, :] + control_std * rng.standard_normal(
            (num_samples, horizon, 2)
        )
        # 滚转轨迹并计算代价
        costs = np.zeros(num_samples, dtype=np.float64)
        all_positions = np.zeros((num_samples, horizon + 1, 2), dtype=np.float64)

        for k in range(num_samples):
            u = u_samples[k]
            pos = np.zeros((horizon + 1, 2))
            vel = np.zeros((horizon + 1, 2))
            pos[0] = start
            vel[0] = 0.0
            cost = 0.0
            for t in range(horizon):
                vel[t + 1] = vel[t] + u[t] * dt
                pos[t + 1] = pos[t] + vel[t] * dt
                # 边界裁剪
                pos[t + 1, 0] = np.clip(pos[t + 1, 0], x_min, x_max)
                pos[t + 1, 1] = np.clip(pos[t + 1, 1], y_min, y_max)
                # 障碍物代价
                c_obs = cost_at_point(
                    pos[t + 1, 0], pos[t + 1, 1], cost_map, bounds, map_size
                )
                cost += obstacle_scale * c_obs
                cost += control_scale * (u[t] ** 2).sum()
            all_positions[k] = pos
            # 终端目标代价
            cost += goal_scale * np.linalg.norm(pos[-1] - goal)
            costs[k] = cost

        # MPPI 权重：指数加权，数值稳定
        costs = np.maximum(costs, 1e-8)
        cost_min = costs.min()
        weights = np.exp(-(costs - cost_min) / temperature)
        weights = weights / (weights.sum() + 1e-10)
        # 用加权平均更新 nominal 控制
        nominal_u = np.einsum("k,kij->ij", weights, u_samples)
        # 用更新后的 nominal 滚转得到新的位置序列（用于下一轮初始或输出）
        pos_nom = np.zeros((horizon + 1, 2))
        vel_nom = np.zeros((horizon + 1, 2))
        pos_nom[0] = start
        vel_nom[0] = 0.0
        for t in range(horizon):
            vel_nom[t + 1] = vel_nom[t] + nominal_u[t] * dt
            pos_nom[t + 1] = pos_nom[t] + vel_nom[t] * dt
            pos_nom[t + 1, 0] = np.clip(pos_nom[t + 1, 0], x_min, x_max)
            pos_nom[t + 1, 1] = np.clip(pos_nom[t + 1, 1], y_min, y_max)
        positions = pos_nom

    path = [tuple(pos) for pos in positions]
    return path


def path_to_trajectory_with_kinematics(
    path: List[Tuple[float, float]],
    num_eval: int = 256,
) -> np.ndarray:
    """
    将路径点拟合成 B-spline，再在 num_eval 个点上求值并计算速度、加速度，
    使运动学参数与轨迹一致。返回 [num_eval, 6] (pos, vel, acc)。
    """
    if len(path) < 2:
        path = path + [path[-1]] if path else [(0.0, 0.0), (1.0, 1.0)]
    path = np.array(path, dtype=np.float64)
    n = len(path)
    # 若点过少，线性插值到至少 4 点以便 B-spline k=3
    if n < 4:
        t_old = np.linspace(0, 1, n)
        t_new = np.linspace(0, 1, 4)
        path = np.column_stack([
            np.interp(t_new, t_old, path[:, 0]),
            np.interp(t_new, t_old, path[:, 1]),
        ])
        n = 4
    # 弦长参数化
    seg_len = np.linalg.norm(np.diff(path, axis=0), axis=1)
    seg_len = np.maximum(seg_len, 1e-8)
    t_param = np.zeros(n)
    t_param[1:] = np.cumsum(seg_len)
    t_param = t_param / t_param[-1]
    k = min(3, n - 1)
    spl_x = make_interp_spline(t_param, path[:, 0], k=k, bc_type=([(1, 0)], [(1, 0)]))
    spl_y = make_interp_spline(t_param, path[:, 1], k=k, bc_type=([(1, 0)], [(1, 0)]))
    t_eval = np.linspace(0, 1, num_eval)
    pos_x = spl_x(t_eval)
    pos_y = spl_y(t_eval)
    vel_x = spl_x.derivative(1)(t_eval)
    vel_y = spl_y.derivative(1)(t_eval)
    acc_x = spl_x.derivative(2)(t_eval)
    acc_y = spl_y.derivative(2)(t_eval)
    return np.stack([pos_x, pos_y, vel_x, vel_y, acc_x, acc_y], axis=-1).astype(np.float32)


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


def _generate_one_trajectory(traj_index: int, args_dict: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
    """
    单条轨迹生成（供多进程调用）。使用 args_dict 中参数与 traj_index 对应种子。
    成功返回包含 pos, vel, acc, cost_map_4d, start_state, goal_state, style_weights 的 dict，失败返回 None。
    """
    bounds = tuple(args_dict["bounds"])
    map_size = args_dict["map_size"]
    num_obstacles = args_dict["num_obstacles"]
    obstacle_type = args_dict["obstacle_type"]
    planner = args_dict["planner"]
    occupancy_threshold = args_dict["occupancy_threshold"]
    seed = args_dict["seed"]
    seq_len = args_dict["seq_len"]
    base_seed = seed + traj_index + 1000
    rng = np.random.default_rng(seed + traj_index)

    obstacles = generate_random_obstacles(
        bounds, num_obstacles, obstacle_type,
        seed=base_seed,
    )
    cost_map = obstacles_to_cost_map(map_size, bounds, obstacles)
    cost_map_4d = cost_map[np.newaxis, ...]

    try:
        start, goal = sample_start_goal_in_free_space(
            cost_map, bounds, map_size, rng,
            margin=0.05,
            max_attempts=200,
        )
    except RuntimeError:
        return None

    if planner == "astar":
        path = plan_path_astar(
            cost_map, start, goal, bounds, map_size,
            occupancy_threshold=occupancy_threshold,
        )
    else:
        path = plan_trajectory_mppi(
            cost_map, start, goal, bounds, map_size,
            horizon=args_dict["mppi_horizon"],
            num_samples=args_dict["mppi_samples"],
            num_iterations=args_dict["mppi_iterations"],
            dt=args_dict["mppi_dt"],
            control_std=args_dict["mppi_control_std"],
            temperature=args_dict["mppi_temperature"],
            rng=rng,
        )

    traj = path_to_trajectory_with_kinematics(path, num_eval=256)
    pos = resample_trajectory(traj[:, :2], seq_len)
    vel = resample_trajectory(traj[:, 2:4], seq_len)
    acc = resample_trajectory(traj[:, 4:6], seq_len)

    start_state = np.concatenate([pos[0], vel[0], acc[0]], axis=-1).astype(np.float32)
    goal_state = np.concatenate([pos[-1], vel[-1]], axis=-1).astype(np.float32)
    style = rng.dirichlet([1.0, 1.0]).astype(np.float32)

    return {
        "pos": pos,
        "vel": vel,
        "acc": acc,
        "cost_map_4d": cost_map_4d,
        "start_state": start_state,
        "goal_state": goal_state,
        "style_weights": style,
    }


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
    parser.add_argument("--planner", type=str, default="mppi",
                        choices=["astar", "mppi"],
                        help="Path planner: astar (grid A*), mppi (smooth MPPI in continuous space)")
    parser.add_argument("--occupancy_threshold", type=float, default=0.25,
                        help="Cost map threshold for occupancy in planner")
    parser.add_argument("--free_space_threshold", type=float, default=0.1,
                        help="Cost threshold below which start/goal are allowed")
    # MPPI 参数（仅当 --planner mppi 时生效）
    parser.add_argument("--mppi_horizon", type=int, default=64,
                        help="MPPI horizon (number of control steps)")
    parser.add_argument("--mppi_samples", type=int, default=512,
                        help="MPPI number of sampled trajectories per iteration")
    parser.add_argument("--mppi_iterations", type=int, default=20,
                        help="MPPI number of iterations")
    parser.add_argument("--mppi_dt", type=float, default=0.02,
                        help="MPPI time step")
    parser.add_argument("--mppi_control_std", type=float, default=2.0,
                        help="MPPI control perturbation std")
    parser.add_argument("--mppi_temperature", type=float, default=0.1,
                        help="MPPI temperature (lower = sharper weighting)")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel workers (1 = sequential)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    bounds = tuple(args.bounds)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args_dict = {
        "bounds": args.bounds,
        "map_size": args.map_size,
        "num_obstacles": args.num_obstacles,
        "obstacle_type": args.obstacle_type,
        "planner": args.planner,
        "occupancy_threshold": args.occupancy_threshold,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "mppi_horizon": args.mppi_horizon,
        "mppi_samples": args.mppi_samples,
        "mppi_iterations": args.mppi_iterations,
        "mppi_dt": args.mppi_dt,
        "mppi_control_std": args.mppi_control_std,
        "mppi_temperature": args.mppi_temperature,
    }

    if args.num_workers <= 1:
        results = []
        for i in range(args.num_trajs):
            results.append(_generate_one_trajectory(i, args_dict))
    else:
        with mp.Pool(processes=args.num_workers) as pool:
            results = pool.starmap(_generate_one_trajectory, [(i, args_dict) for i in range(args.num_trajs)])

    all_positions = []
    all_velocities = []
    all_accelerations = []
    all_cost_maps = []
    all_start_states = []
    all_goal_states = []
    all_style_weights = []
    for r in results:
        if r is None:
            continue
        all_positions.append(r["pos"])
        all_velocities.append(r["vel"])
        all_accelerations.append(r["acc"])
        all_cost_maps.append(r["cost_map_4d"])
        all_start_states.append(r["start_state"])
        all_goal_states.append(r["goal_state"])
        all_style_weights.append(r["style_weights"])
    num_skipped = args.num_trajs - len(all_positions)

    if not all_positions:
        raise RuntimeError("No trajectories generated (all start/goal samples may have been on obstacles). Try more obstacles or larger free regions.")
    
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
    n = len(all_positions)
    print(f"Saved {n} samples to {npz_path}")
    if num_skipped > 0:
        print(f"  (skipped {num_skipped} due to no free start/goal)")
    print(f"  positions: {positions.shape}, cost_maps: {cost_maps.shape}")


if __name__ == "__main__":
    main()
