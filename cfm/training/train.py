from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cfm.data import NpzTrajectoryDataset
from cfm.models import ConditionalVectorFieldTransformer
from cfm.training.checkpoint import load_checkpoint, save_checkpoint
from cfm.training.interpolation import build_xt_and_targets
from cfm.training.losses import cfm_joint_mse
from cfm.utils.io import RunPaths, ensure_dir, load_yaml
from cfm.utils.seed import seed_all


def _select_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _build_model(cfg: dict[str, Any]) -> ConditionalVectorFieldTransformer:
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
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--data", required=True, type=str)
    ap.add_argument("--resume", default=None, type=str)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed_all(int(cfg.get("seed", 42)))

    device = _select_device(str(cfg["train"].get("device", "auto")))
    run = RunPaths(cfg["run"]["out_dir"])
    ensure_dir(run.out_dir)

    ds = NpzTrajectoryDataset(args.data, seq_len=cfg["data"].get("seq_len", None))
    dl = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    model = _build_model(cfg).to(device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )

    step0 = 0
    epoch0 = 0
    if args.resume is not None and os.path.exists(args.resume):
        ckpt = load_checkpoint(args.resume, model=model, optim=optim, map_location="cpu")
        step0 = int(ckpt.get("step", 0))
        epoch0 = int(ckpt.get("epoch", 0))

    lambda_acc = float(cfg["train"].get("lambda_acc", 1.0))
    lambda_jerk = float(cfg["train"].get("lambda_jerk", 1.0))
    grad_clip = float(cfg["train"].get("grad_clip_norm", 1.0))
    log_every = int(cfg["train"].get("log_every", 50))
    save_every = int(cfg["train"].get("save_every", 500))
    epochs = int(cfg["train"]["epochs"])

    model.train()
    step = step0
    for epoch in range(epoch0, epochs):
        pbar = tqdm(dl, desc=f"epoch {epoch+1}/{epochs}", dynamic_ncols=True)
        for batch in pbar:
            q1 = batch["q1"].to(device)        # (B,T,2)
            dq1 = batch["dq1"].to(device)
            ddq1 = batch["ddq1"].to(device)
            c = batch["c"].to(device)          # (B,cond_dim)

            B, T, _ = q1.shape
            t = torch.rand(B, device=device)
            eps_q = torch.randn_like(q1)
            eps_dq = torch.randn_like(dq1)
            eps_ddq = torch.randn_like(ddq1)

            x_t, u_tgt, v_tgt, w_tgt = build_xt_and_targets(q1, dq1, ddq1, t, eps_q, eps_dq, eps_ddq)
            pred = model(x_t, t, c)  # (B,T,6)
            loss = cfm_joint_mse(pred, u_tgt, v_tgt, w_tgt, lambda_acc=lambda_acc, lambda_jerk=lambda_jerk)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()

            step += 1
            if step % log_every == 0:
                pbar.set_postfix(loss=float(loss.detach().cpu()))

            if step % save_every == 0:
                save_checkpoint(run.step_ckpt(step), step=step, epoch=epoch, model=model, optim=optim, config=cfg)
                save_checkpoint(run.latest_ckpt, step=step, epoch=epoch, model=model, optim=optim, config=cfg)

        # end epoch save
        save_checkpoint(run.latest_ckpt, step=step, epoch=epoch, model=model, optim=optim, config=cfg)


if __name__ == "__main__":
    main()

