#!/usr/bin/env python3
"""
Baseline benchmark for validating whether L2 helps L1.

Baselines:
A. L1-only (classic MPPI)
B. L2-only (direct proposal execution)
C. L2 + L1 (proposal-guided MPPI)
D. Random proposals + L1 (same proposal count as C)

Usage:
  python mppi_demo/tests/benchmark_l2_l1_baselines.py \
      --l2-checkpoint checkpoints_l2/best_model.pt \
      --scenario all --trials 5 --num-proposals 32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cfm_flowmp.models import create_l2_safety_cfm
from mppi_demo.astar_path_planner import AStarPlanner
from mppi_demo.mppi_core import (
    BSplineTrajectory,
    CollisionCost,
    CompositeCost,
    GoalCost,
    MPPI_BSpline,
    PathLengthCost,
    SmoothnessCost,
)
from mppi_demo.scenarios import (
    create_christmas_market_environment,
    create_narrow_passage_scenario,
    create_u_trap_scenario,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark A/B/C/D baselines for L2->L1 usefulness")
    parser.add_argument("--l2-checkpoint", type=str, required=True, help="Path to trained L2 checkpoint")
    parser.add_argument("--scenario", type=str, default="all", choices=["u_trap", "narrow_passage", "christmas_market", "all"])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--num-proposals", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--map-size", type=int, default=64)
    parser.add_argument("--mppi-samples", type=int, default=300)
    parser.add_argument("--mppi-iters", type=int, default=80)
    parser.add_argument("--goal-tol", type=float, default=0.5)
    parser.add_argument("--robot-radius", type=float, default=0.2)
    parser.add_argument("--tube-radius", type=float, default=0.7)
    parser.add_argument("--tube-weight", type=float, default=15.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs/l2_l1_baseline_results.json")
    return parser.parse_args()


def world_to_norm(points: np.ndarray, bounds: Tuple[float, float, float, float]) -> np.ndarray:
    x_min, x_max, y_min, y_max = bounds
    out = points.copy()
    out[..., 0] = (out[..., 0] - x_min) / max(1e-8, (x_max - x_min))
    out[..., 1] = (out[..., 1] - y_min) / max(1e-8, (y_max - y_min))
    return np.clip(out, 0.0, 1.0)


def norm_to_world(points: np.ndarray, bounds: Tuple[float, float, float, float]) -> np.ndarray:
    x_min, x_max, y_min, y_max = bounds
    out = points.copy()
    out[..., 0] = out[..., 0] * (x_max - x_min) + x_min
    out[..., 1] = out[..., 1] * (y_max - y_min) + y_min
    return out


def env_to_cost_map(env, bounds: Tuple[float, float, float, float], map_size: int) -> torch.Tensor:
    x_min, x_max, y_min, y_max = bounds
    xs = np.linspace(x_min, x_max, map_size)
    ys = np.linspace(y_min, y_max, map_size)
    xx, yy = np.meshgrid(xs, ys)
    points = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)

    sdf = env.compute_sdf(points).reshape(map_size, map_size)
    # High cost in obstacle, decays with distance.
    cost = np.exp(-np.maximum(sdf, 0.0) / 0.35)
    cost[sdf < 0.0] = 1.0
    cost = np.clip(cost, 0.0, 1.0).astype(np.float32)

    return torch.from_numpy(cost).unsqueeze(0).unsqueeze(0)


def build_base_cost(env, goal: np.ndarray, robot_radius: float) -> CompositeCost:
    return CompositeCost([
        CollisionCost(env=env, robot_radius=robot_radius, epsilon=0.1, weight=140.0, use_hard_constraint=True, hard_penalty=1e6),
        SmoothnessCost(penalize="acceleration", weight=0.7),
        PathLengthCost(weight=8.0),
        GoalCost(goal=goal, weight=80.0),
    ])


def run_mppi(
    cost_fn: CompositeCost,
    start: np.ndarray,
    goal: np.ndarray,
    bounds: Tuple[float, float, float, float],
    n_samples: int,
    n_iters: int,
    seq_len: int,
    seed_traj: np.ndarray | None = None,
) -> Dict[str, Any]:
    mppi = MPPI_BSpline(
        cost_function=cost_fn,
        n_samples=n_samples,
        n_control_points=12,
        bspline_degree=3,
        time_horizon=6.0,
        n_timesteps=seq_len,
        temperature=0.8,
        noise_std=0.35,
        bounds=bounds,
        elite_ratio=0.1,
        n_jobs=1,
    )

    mppi.initialize(start, goal)
    mppi.cost_history = []
    mppi.iteration = 0
    mppi.best_cost_all_time = np.inf
    mppi.best_trajectory_all_time = None
    mppi.best_control_points_all_time = None
    mppi.best_iteration = -1

    if seed_traj is not None:
        cp = mppi.bspline.fit_trajectory(seed_traj)
        mppi.control_points = cp

    info_history = []
    for _ in range(n_iters):
        info_history.append(mppi.step())

    if mppi.best_trajectory_all_time is None:
        trajectory = mppi.bspline.evaluate(mppi.control_points, n_samples=seq_len)
        best_cost = np.inf
    else:
        trajectory = mppi.best_trajectory_all_time
        best_cost = float(mppi.best_cost_all_time)

    return {
        "trajectory": trajectory,
        "best_cost": best_cost,
        "info_history": info_history,
    }


def polyline_resample(path: np.ndarray, n: int) -> np.ndarray:
    if len(path) <= 1:
        return np.repeat(path[:1], n, axis=0)

    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-8:
        return np.repeat(path[:1], n, axis=0)

    t = np.linspace(0.0, s[-1], n)
    out = np.zeros((n, path.shape[-1]), dtype=np.float64)
    for d in range(path.shape[-1]):
        out[:, d] = np.interp(t, s, path[:, d])
    return out


def generate_random_proposals(
    env,
    start: np.ndarray,
    goal: np.ndarray,
    bounds: Tuple[float, float, float, float],
    k: int,
    seq_len: int,
) -> np.ndarray:
    rng = np.random.default_rng()
    proposals: List[np.ndarray] = []

    n_spline = max(1, k // 2)
    n_goal_biased = max(1, (k - n_spline) // 2)
    n_astar = max(0, k - n_spline - n_goal_biased)

    bspline = BSplineTrajectory(degree=3, n_control_points=10, time_horizon=6.0, dim=2)
    cps = bspline.sample_random_control_points(start, goal, bounds, n_samples=n_spline)
    for i in range(n_spline):
        proposals.append(bspline.evaluate(cps[i], n_samples=seq_len))

    line = np.linspace(start, goal, seq_len)
    direction = goal - start
    norm = np.linalg.norm(direction) + 1e-8
    tangent = direction / norm
    normal = np.array([-tangent[1], tangent[0]])
    for _ in range(n_goal_biased):
        amp = rng.uniform(-1.2, 1.2)
        curve = np.sin(np.linspace(0.0, np.pi, seq_len))[:, None] * amp * normal[None, :]
        traj = line + curve
        traj[:, 0] = np.clip(traj[:, 0], bounds[0], bounds[1])
        traj[:, 1] = np.clip(traj[:, 1], bounds[2], bounds[3])
        proposals.append(traj)

    if n_astar > 0:
        planner = AStarPlanner(env=env, resolution=0.2, robot_radius=0.2)
        path, _ = planner.plan(start, goal)
        if path is None or len(path) < 2:
            path = np.stack([start, goal], axis=0)

        for _ in range(n_astar):
            cut = int(rng.uniform(0.4, 1.0) * (len(path) - 1))
            cut = max(1, min(cut, len(path) - 1))
            local = path[: cut + 1]
            stitched = np.concatenate([local, goal[None, :]], axis=0)
            proposals.append(polyline_resample(stitched, seq_len))

    if len(proposals) < k:
        while len(proposals) < k:
            proposals.append(line.copy())

    return np.stack(proposals[:k], axis=0)


def load_l2_model(checkpoint_path: str, device: str):
    model = create_l2_safety_cfm(
        model_type="transformer",
        state_dim=2,
        max_seq_len=64,
        hidden_dim=256,
        num_layers=8,
        num_heads=8,
        cost_map_channels=1,
        cost_map_latent_dim=256,
        cost_map_encoder_type="single_scale",
        use_style_conditioning=True,
        style_dim=3,
        use_8step_schedule=True,
        use_bspline_smoothing=True,
        bspline_degree=3,
        bspline_num_control_points=20,
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate_l2_proposals(
    model,
    env,
    start: np.ndarray,
    goal: np.ndarray,
    bounds: Tuple[float, float, float, float],
    num_proposals: int,
    map_size: int,
    device: str,
) -> Dict[str, Any]:
    cost_map = env_to_cost_map(env, bounds, map_size).to(device)
    start_n = world_to_norm(start[None, :], bounds)[0]
    goal_n = world_to_norm(goal[None, :], bounds)[0]

    x_curr = torch.tensor([[start_n[0], start_n[1], 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    x_goal = torch.tensor([[goal_n[0], goal_n[1], 0.0, 0.0]], dtype=torch.float32, device=device)
    w_style = torch.tensor([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=torch.float32, device=device)

    out = model.generate_trajectory_anchors(
        cost_map=cost_map,
        x_curr=x_curr,
        x_goal=x_goal,
        w_style=w_style,
        num_samples=num_proposals,
    )

    traj_norm = out["trajectories"].detach().cpu().numpy()
    traj_world = norm_to_world(traj_norm, bounds)

    mode_weights = out.get("mode_weights", None)
    proposal_scores = out.get("proposal_scores", None)
    semantic_tags = out.get("semantic_tags", [])

    mode_weights_np = mode_weights.detach().cpu().numpy() if isinstance(mode_weights, torch.Tensor) else None
    proposal_scores_np = proposal_scores.detach().cpu().numpy() if isinstance(proposal_scores, torch.Tensor) else None

    return {
        "trajectories": traj_world,
        "mode_weights": mode_weights_np,
        "proposal_scores": proposal_scores_np,
        "semantic_tags": semantic_tags,
    }


def trajectory_metrics(
    traj: np.ndarray,
    env,
    goal: np.ndarray,
    goal_tol: float,
    robot_radius: float,
    elapsed_sec: float,
    objective_cost: float,
) -> Dict[str, float]:
    if traj.ndim != 2:
        raise ValueError("trajectory must be [T,2]")

    d_goal = float(np.linalg.norm(traj[-1] - goal))
    seg = np.diff(traj, axis=0)
    path_length = float(np.linalg.norm(seg, axis=1).sum()) if len(seg) > 0 else 0.0

    vel = np.diff(traj, axis=0)
    acc = np.diff(vel, axis=0)
    jerk = np.diff(acc, axis=0)

    smoothness = float(np.linalg.norm(acc, axis=1).mean()) if len(acc) > 0 else 0.0
    jitter = float(np.linalg.norm(jerk, axis=1).mean()) if len(jerk) > 0 else 0.0

    sdf = env.compute_sdf(traj)
    min_clearance = float(np.min(sdf))
    collision = bool(np.any(sdf < robot_radius))
    success = (not collision) and (d_goal <= goal_tol)

    return {
        "success": float(success),
        "collision": float(collision),
        "goal_error": d_goal,
        "path_length": path_length,
        "smoothness": smoothness,
        "jitter": jitter,
        "min_clearance": min_clearance,
        "runtime_sec": float(elapsed_sec),
        "objective_cost": float(objective_cost),
    }


def scenario_factory(name: str):
    if name == "u_trap":
        env, start, goal, bounds = create_u_trap_scenario()
        return env, start, goal, bounds
    if name == "narrow_passage":
        env, start, goal, bounds = create_narrow_passage_scenario(passage_width=0.8)
        return env, start, goal, bounds
    if name == "christmas_market":
        env, start, goal, bounds, _ = create_christmas_market_environment(variant="v2")
        return env, start, goal, bounds
    raise ValueError(f"Unknown scenario {name}")


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = [
        "success",
        "collision",
        "goal_error",
        "path_length",
        "smoothness",
        "jitter",
        "min_clearance",
        "runtime_sec",
        "objective_cost",
    ]
    out: Dict[str, Any] = {"n": len(rows)}
    for m in metrics:
        vals = np.array([r[m] for r in rows], dtype=np.float64)
        out[f"{m}_mean"] = float(vals.mean())
        out[f"{m}_std"] = float(vals.std())
    return out


def print_table(summary: Dict[str, Dict[str, Any]]) -> None:
    headers = ["baseline", "succ", "coll", "goal_err", "len", "jitter", "time"]
    print("\n" + "=" * 92)
    print(f"{headers[0]:<24} {headers[1]:>8} {headers[2]:>8} {headers[3]:>12} {headers[4]:>10} {headers[5]:>10} {headers[6]:>10}")
    print("-" * 92)
    for k, v in summary.items():
        print(
            f"{k:<24} "
            f"{v['success_mean']:>8.3f} "
            f"{v['collision_mean']:>8.3f} "
            f"{v['goal_error_mean']:>12.3f} "
            f"{v['path_length_mean']:>10.3f} "
            f"{v['jitter_mean']:>10.3f} "
            f"{v['runtime_sec_mean']:>10.3f}"
        )
    print("=" * 92 + "\n")


def run() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    l2_model = load_l2_model(args.l2_checkpoint, args.device)

    scenario_names = [args.scenario] if args.scenario != "all" else ["u_trap", "narrow_passage", "christmas_market"]

    all_results: Dict[str, Any] = {"config": vars(args), "scenarios": {}}

    for sname in scenario_names:
        print(f"Running scenario: {sname}")
        env, start, goal, bounds = scenario_factory(sname)

        rows_by_baseline: Dict[str, List[Dict[str, Any]]] = {
            "A_l1_only": [],
            "B_l2_only": [],
            "C_l2_plus_l1": [],
            "D_random_proposal_plus_l1": [],
        }

        for trial in range(args.trials):
            np.random.seed(args.seed + trial)
            torch.manual_seed(args.seed + trial)

            base_cost = build_base_cost(env, goal, args.robot_radius)

            # A. L1-only
            t0 = time.time()
            a = run_mppi(
                cost_fn=base_cost,
                start=start,
                goal=goal,
                bounds=bounds,
                n_samples=args.mppi_samples,
                n_iters=args.mppi_iters,
                seq_len=args.seq_len,
                seed_traj=None,
            )
            ta = time.time() - t0
            rows_by_baseline["A_l1_only"].append(
                trajectory_metrics(a["trajectory"], env, goal, args.goal_tol, args.robot_radius, ta, a["best_cost"])
            )

            # Shared L2 proposals for B/C
            l2_out = generate_l2_proposals(
                model=l2_model,
                env=env,
                start=start,
                goal=goal,
                bounds=bounds,
                num_proposals=args.num_proposals,
                map_size=args.map_size,
                device=args.device,
            )
            l2_trajs = l2_out["trajectories"]
            l2_weights = l2_out["mode_weights"]

            # B. L2-only
            t0 = time.time()
            if l2_weights is not None and len(l2_weights) == len(l2_trajs):
                b_idx = int(np.argmax(l2_weights))
            else:
                b_idx = 0
            b_traj = l2_trajs[b_idx]
            tb = time.time() - t0
            b_cost = float(base_cost(positions=b_traj[None, :, :])[0])
            rows_by_baseline["B_l2_only"].append(
                trajectory_metrics(b_traj, env, goal, args.goal_tol, args.robot_radius, tb, b_cost)
            )

            # C. L2 + L1
            from cfm_flowmp.inference.l1_reactive_control import AnchorTubeCost

            tube_cost_l2 = AnchorTubeCost(anchors=l2_trajs.astype(np.float32), tube_radius=args.tube_radius, weight=args.tube_weight)
            cost_c = CompositeCost([base_cost, tube_cost_l2])
            seed_idx_c = int(np.argmax(l2_weights)) if l2_weights is not None and len(l2_weights) == len(l2_trajs) else 0
            t0 = time.time()
            c = run_mppi(
                cost_fn=cost_c,
                start=start,
                goal=goal,
                bounds=bounds,
                n_samples=args.mppi_samples,
                n_iters=args.mppi_iters,
                seq_len=args.seq_len,
                seed_traj=l2_trajs[seed_idx_c],
            )
            tc = time.time() - t0
            rows_by_baseline["C_l2_plus_l1"].append(
                trajectory_metrics(c["trajectory"], env, goal, args.goal_tol, args.robot_radius, tc, c["best_cost"])
            )

            # D. Random proposals + L1
            random_trajs = generate_random_proposals(
                env=env,
                start=start,
                goal=goal,
                bounds=bounds,
                k=max(2, len(l2_trajs)),
                seq_len=args.seq_len,
            )
            tube_cost_rand = AnchorTubeCost(anchors=random_trajs.astype(np.float32), tube_radius=args.tube_radius, weight=args.tube_weight)
            cost_d = CompositeCost([base_cost, tube_cost_rand])

            rand_seed_costs = base_cost(positions=random_trajs)
            seed_idx_d = int(np.argmin(rand_seed_costs))
            t0 = time.time()
            d = run_mppi(
                cost_fn=cost_d,
                start=start,
                goal=goal,
                bounds=bounds,
                n_samples=args.mppi_samples,
                n_iters=args.mppi_iters,
                seq_len=args.seq_len,
                seed_traj=random_trajs[seed_idx_d],
            )
            td = time.time() - t0
            rows_by_baseline["D_random_proposal_plus_l1"].append(
                trajectory_metrics(d["trajectory"], env, goal, args.goal_tol, args.robot_radius, td, d["best_cost"])
            )

            print(f"  trial {trial + 1}/{args.trials} done")

        summary = {k: aggregate(v) for k, v in rows_by_baseline.items()}
        print_table(summary)

        all_results["scenarios"][sname] = {
            "raw": rows_by_baseline,
            "summary": summary,
        }

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved benchmark results to: {out_path}")


if __name__ == "__main__":
    run()
