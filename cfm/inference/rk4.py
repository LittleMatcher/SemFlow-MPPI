from __future__ import annotations

import torch


@torch.no_grad()
def rk4_step(
    f,
    x: torch.Tensor,
    t: torch.Tensor,
    dt: float,
):
    """
    One RK4 step for dx/dt = f(x,t).
    f(x,t) must return tensor with same shape as x.
    """
    k1 = f(x, t)
    k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(x + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(x + dt * k3, t + dt)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@torch.no_grad()
def integrate_rk4(
    f,
    x0: torch.Tensor,
    c: torch.Tensor,
    steps: int = 20,
):
    """
    Integrate from t=0 to t=1 with fixed steps.
    x0: (B,T,6)
    c: (B,cond_dim) forwarded to closure f if needed.
    """
    dt = 1.0 / steps
    x = x0
    B = x0.shape[0]
    for i in range(steps):
        ti = torch.full((B,), float(i) / steps, device=x.device, dtype=x.dtype)

        def _f(x_in, t_in):
            return f(x_in, t_in, c)

        x = rk4_step(_f, x, ti, dt)
    return x

