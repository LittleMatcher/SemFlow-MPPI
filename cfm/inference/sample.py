from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch

from cfm.models import ConditionalVectorFieldTransformer
from cfm.training.checkpoint import load_checkpoint
from cfm.inference.rk4 import integrate_rk4


def _build_model_from_cfg(cfg: dict[str, Any]) -> ConditionalVectorFieldTransformer:
    m = cfg["model"]
    return ConditionalVectorFieldTransformer(
        in_dim=int(m["in_dim"]),
        out_dim=int(m["out_dim"]),
        cond_dim=int(m["cond_dim"]),
        d_model=int(m["d_model"]),
        n_heads=int(m["n_heads"]),
        n_layers=int(m["n_layers"]),
        d_ff=int(m["d_ff"]),
        dropout=float(m["dropout"]),
        time_embed_dim=int(m["time_embed_dim"]),
        time_embed_kind=m.get("time_embed_kind", "gaussian"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--n", default=8, type=int)
    ap.add_argument("--seq-len", default=50, type=int)
    ap.add_argument("--device", default="auto", type=str)
    ap.add_argument("--steps", default=None, type=int)
    ap.add_argument("--cond", default=None, type=str, help="Optional npz with q_start,dq_start,q_goal arrays.")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)

    # Load checkpoint and config
    model = None
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]
    model = _build_model_from_cfg(cfg).to(device)
    load_checkpoint(args.ckpt, model=model, optim=None, map_location="cpu")
    model.eval()

    steps = args.steps if args.steps is not None else int(cfg.get("inference", {}).get("steps", 20))

    B = int(args.n)
    T = int(args.seq_len)

    # Conditions
    if args.cond is None:
        # Default: zeros (user should provide meaningful start/goal in real use)
        c = torch.zeros((B, int(cfg["model"]["cond_dim"])), device=device)
    else:
        d = np.load(args.cond)
        q_start = d["q_start"].astype(np.float32)
        dq_start = d["dq_start"].astype(np.float32)
        q_goal = d["q_goal"].astype(np.float32)
        if q_start.shape[0] != B:
            raise ValueError("cond batch size must equal --n")
        c = torch.from_numpy(np.concatenate([q_start, dq_start, q_goal], axis=-1)).to(device)

    # Initial state x0 from standard normal
    x0 = torch.randn((B, T, int(cfg["model"]["in_dim"])), device=device)

    # Define vector field: model predicts [u,v,w] for dx/dt
    def vf(x_t, t, c_in):
        return model(x_t, t, c_in)

    x1 = integrate_rk4(vf, x0=x0, c=c, steps=steps)  # (B,T,6)
    x1_np = x1.detach().cpu().numpy()

    q = x1_np[..., 0:2]
    dq = x1_np[..., 2:4]
    ddq = x1_np[..., 4:6]
    np.savez(args.out, q_gen=q, dq_gen=dq, ddq_gen=ddq)


if __name__ == "__main__":
    main()

