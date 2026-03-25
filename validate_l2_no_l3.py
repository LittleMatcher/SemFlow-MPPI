"""
Validate L2 effectiveness without L3.

This script uses only current dataset fields (cost_map/start_state/goal_state/style_weights)
and evaluates whether L2-generated proposals are better than random proposals.

Evaluation setup:
- L2-only: generate K proposals from L2, select best by unified score
- Random-only: generate K random proposals, select best by the same score

Metrics (lower is better unless stated otherwise):
- goal_reach_rate (higher better)
- collision_rate
- ade (average displacement error to expert)
- fde (final displacement error to expert)
- jerk_mean (trajectory jitter)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from cfm_flowmp.data import FlowMPEnvDataset
from cfm_flowmp.models import create_l2_safety_cfm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate L2 effectiveness without L3")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained L2 checkpoint")
    parser.add_argument("--data_dir", type=str, default="traj_data/cfm_env", help="Directory containing data.npz")
    parser.add_argument("--num_eval", type=int, default=200, help="Number of samples to evaluate")
    parser.add_argument("--num_proposals", type=int, default=32, help="K proposals per sample")
    parser.add_argument("--goal_tol", type=float, default=0.08, help="Goal reach threshold in normalized map [0,1]")
    parser.add_argument("--collision_threshold", type=float, default=0.5, help="Cost-map threshold for collision")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str):
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

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def sample_cost_map(cost_map_2d: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Nearest-neighbor sampling on normalized [0,1]x[0,1]."""
    h, w = cost_map_2d.shape
    x = np.clip((points[:, 0] * (w - 1)).round().astype(int), 0, w - 1)
    y = np.clip((points[:, 1] * (h - 1)).round().astype(int), 0, h - 1)
    return cost_map_2d[y, x]


def trajectory_jerk_mean(traj: np.ndarray) -> float:
    vel = np.diff(traj, axis=0)
    if len(vel) < 2:
        return 0.0
    acc = np.diff(vel, axis=0)
    if len(acc) < 2:
        return float(np.mean(np.linalg.norm(acc, axis=1)))
    jerk = np.diff(acc, axis=0)
    return float(np.mean(np.linalg.norm(jerk, axis=1)))


def collision_flag(traj: np.ndarray, cost_map_2d: np.ndarray, threshold: float) -> bool:
    values = sample_cost_map(cost_map_2d, traj)
    return bool(np.any(values >= threshold))


def proposal_score(
    traj: np.ndarray,
    goal_xy: np.ndarray,
    cost_map_2d: np.ndarray,
    collision_threshold: float,
) -> float:
    final_err = np.linalg.norm(traj[-1] - goal_xy)
    collision_penalty = 1.0 if collision_flag(traj, cost_map_2d, collision_threshold) else 0.0
    jitter = trajectory_jerk_mean(traj)
    # Lower is better
    return float(final_err + 2.0 * collision_penalty + 0.2 * jitter)


def select_best_proposal(
    proposals: np.ndarray,
    goal_xy: np.ndarray,
    cost_map_2d: np.ndarray,
    collision_threshold: float,
) -> np.ndarray:
    scores = [proposal_score(p, goal_xy, cost_map_2d, collision_threshold) for p in proposals]
    return proposals[int(np.argmin(scores))]


def random_proposals(start_xy: np.ndarray, goal_xy: np.ndarray, k: int, horizon: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.linspace(start_xy, goal_xy, horizon)
    proposals = []
    for _ in range(k):
        amp = rng.uniform(0.02, 0.15)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        freq = rng.uniform(1.0, 3.0)

        t = np.linspace(0.0, 1.0, horizon)
        direction = goal_xy - start_xy
        norm = np.linalg.norm(direction) + 1e-8
        tangent = direction / norm
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)

        curve = (amp * np.sin(2.0 * np.pi * freq * t + phase))[:, None] * normal[None, :]
        traj = base + curve
        traj[0] = start_xy
        traj = np.clip(traj, 0.0, 1.0)
        proposals.append(traj.astype(np.float32))

    return np.stack(proposals, axis=0)


def evaluate_one(
    pred: np.ndarray,
    expert: np.ndarray,
    goal_xy: np.ndarray,
    cost_map_2d: np.ndarray,
    goal_tol: float,
    collision_threshold: float,
) -> Dict[str, float]:
    d = np.linalg.norm(pred - expert, axis=-1)
    ade = float(np.mean(d))
    fde = float(np.linalg.norm(pred[-1] - expert[-1]))
    goal_err = float(np.linalg.norm(pred[-1] - goal_xy))
    reached = float(goal_err <= goal_tol)
    coll = float(collision_flag(pred, cost_map_2d, collision_threshold))
    jerk = trajectory_jerk_mean(pred)

    return {
        "goal_reach": reached,
        "collision": coll,
        "ade": ade,
        "fde": fde,
        "jerk": jerk,
    }


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    keys = rows[0].keys()
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std())
    return out


def print_summary(name: str, stats: Dict[str, float]) -> None:
    print(f"[{name}]")
    print(f"  goal_reach_rate: {stats['goal_reach_mean']:.3f} +- {stats['goal_reach_std']:.3f}")
    print(f"  collision_rate : {stats['collision_mean']:.3f} +- {stats['collision_std']:.3f}")
    print(f"  ADE            : {stats['ade_mean']:.4f} +- {stats['ade_std']:.4f}")
    print(f"  FDE            : {stats['fde_mean']:.4f} +- {stats['fde_std']:.4f}")
    print(f"  jerk_mean      : {stats['jerk_mean']:.4f} +- {stats['jerk_std']:.4f}")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    dataset = FlowMPEnvDataset(args.data_dir)
    n_eval = min(args.num_eval, len(dataset))

    model = load_model(str(ckpt), args.device)

    l2_rows: List[Dict[str, float]] = []
    rnd_rows: List[Dict[str, float]] = []

    for i in range(n_eval):
        sample = dataset[i]

        cost_map = sample["cost_map"].unsqueeze(0).to(args.device)      # [1,1,H,W]
        x_curr = sample["start_state"].unsqueeze(0).to(args.device)      # [1,6]
        x_goal = sample["goal_state"].unsqueeze(0).to(args.device)       # [1,4]
        style = sample["style_weights"].unsqueeze(0).to(args.device)     # [1,2]

        out = model.generate_trajectory_anchors(
            cost_map=cost_map,
            x_curr=x_curr,
            x_goal=x_goal,
            w_style=style,
            num_samples=args.num_proposals,
        )

        proposals_l2 = out["trajectories"].detach().cpu().numpy()        # [K,T,2]

        expert = sample["positions"].numpy()
        goal_xy = sample["goal_state"][0:2].numpy()
        start_xy = sample["start_state"][0:2].numpy()
        cost_map_2d = sample["cost_map"][0].numpy()

        best_l2 = select_best_proposal(
            proposals=proposals_l2,
            goal_xy=goal_xy,
            cost_map_2d=cost_map_2d,
            collision_threshold=args.collision_threshold,
        )

        proposals_rnd = random_proposals(
            start_xy=start_xy,
            goal_xy=goal_xy,
            k=args.num_proposals,
            horizon=proposals_l2.shape[1],
            seed=args.seed + i,
        )

        best_rnd = select_best_proposal(
            proposals=proposals_rnd,
            goal_xy=goal_xy,
            cost_map_2d=cost_map_2d,
            collision_threshold=args.collision_threshold,
        )

        l2_rows.append(
            evaluate_one(
                pred=best_l2,
                expert=expert,
                goal_xy=goal_xy,
                cost_map_2d=cost_map_2d,
                goal_tol=args.goal_tol,
                collision_threshold=args.collision_threshold,
            )
        )

        rnd_rows.append(
            evaluate_one(
                pred=best_rnd,
                expert=expert,
                goal_xy=goal_xy,
                cost_map_2d=cost_map_2d,
                goal_tol=args.goal_tol,
                collision_threshold=args.collision_threshold,
            )
        )

        if (i + 1) % 20 == 0 or (i + 1) == n_eval:
            print(f"Evaluated {i + 1}/{n_eval}")

    l2_stats = summarize(l2_rows)
    rnd_stats = summarize(rnd_rows)

    print("\n=== No-L3 L2 Validation on Current Dataset ===")
    print_summary("L2-only (selected from L2 proposals)", l2_stats)
    print_summary("Random-only (selected from random proposals)", rnd_stats)

    # Simple effectiveness verdict
    better_goal = l2_stats["goal_reach_mean"] >= rnd_stats["goal_reach_mean"]
    better_collision = l2_stats["collision_mean"] <= rnd_stats["collision_mean"]
    better_ade = l2_stats["ade_mean"] <= rnd_stats["ade_mean"]

    print("\n=== Verdict ===")
    if better_goal and better_collision and better_ade:
        print("L2 is effective without L3 under this dataset setting (beats random baseline on key metrics).")
    else:
        print("L2 is NOT clearly effective yet without L3 (does not dominate random baseline on key metrics).")


if __name__ == "__main__":
    main()
