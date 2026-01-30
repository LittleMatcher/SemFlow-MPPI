from __future__ import annotations

import torch


def linear_bridge_xt(x1: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """
    Default CFM bridge: x_t = (1-t)*eps + t*x1
    x1: (B,T,C)
    t: (B,) in [0,1]
    eps: (B,T,C) ~ N(0, I)
    """
    if t.dim() == 2 and t.shape[-1] == 1:
        t = t[:, 0]
    tt = t[:, None, None]
    return (1.0 - tt) * eps + tt * x1


def target_field_from_linear_bridge(x1: torch.Tensor, t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
    """
    Target field used in your description:
      v_target = (x1 - x_t) / (1 - t)
    Note: With linear_bridge_xt, this simplifies to x1 - eps (constant in t).
    """
    if t.dim() == 2 and t.shape[-1] == 1:
        t = t[:, 0]
    denom = (1.0 - t).clamp_min(1e-5)[:, None, None]
    return (x1 - x_t) / denom


def build_xt_and_targets(
    q1: torch.Tensor,
    dq1: torch.Tensor,
    ddq1: torch.Tensor,
    t: torch.Tensor,
    eps_q: torch.Tensor,
    eps_dq: torch.Tensor,
    eps_ddq: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Implements the "Step 1: sampling & interpolation" in a modular way.

    Returns:
      x_t: (B,T,6) concatenated [q_t, dq_t, ddq_t]
      u_target: (B,T,2)
      v_target: (B,T,2)
      w_target: (B,T,2)
    """
    q_t = linear_bridge_xt(q1, t, eps_q)
    dq_t = linear_bridge_xt(dq1, t, eps_dq)
    ddq_t = linear_bridge_xt(ddq1, t, eps_ddq)

    u_target = target_field_from_linear_bridge(q1, t, q_t)
    v_target = target_field_from_linear_bridge(dq1, t, dq_t)
    w_target = target_field_from_linear_bridge(ddq1, t, ddq_t)

    x_t = torch.cat([q_t, dq_t, ddq_t], dim=-1)
    return x_t, u_target, v_target, w_target

