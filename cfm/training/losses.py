from __future__ import annotations

import torch


def split_uvw(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    y: (B,T,6) -> (u,v,w) each (B,T,2)
    """
    if y.shape[-1] != 6:
        raise ValueError(f"Expected last dim=6. Got {y.shape[-1]}")
    u, v, w = y.split(2, dim=-1)
    return u, v, w


def cfm_joint_mse(
    pred: torch.Tensor,
    u_target: torch.Tensor,
    v_target: torch.Tensor,
    w_target: torch.Tensor,
    lambda_acc: float = 1.0,
    lambda_jerk: float = 1.0,
) -> torch.Tensor:
    u_hat, v_hat, w_hat = split_uvw(pred)
    loss_u = torch.mean((u_hat - u_target) ** 2)
    loss_v = torch.mean((v_hat - v_target) ** 2)
    loss_w = torch.mean((w_hat - w_target) ** 2)
    return loss_u + lambda_acc * loss_v + lambda_jerk * loss_w

