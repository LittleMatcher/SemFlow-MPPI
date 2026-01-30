from __future__ import annotations

import math

import torch
from torch import nn


class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier features for a scalar t in [0, 1].
    Commonly used in score/flow models.
    """

    def __init__(self, embed_dim: int, scale: float = 1.0):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError("embed_dim must be even for GaussianFourierProjection.")
        self.embed_dim = embed_dim
        self.register_buffer(
            "W",
            torch.randn(embed_dim // 2) * scale,
            persistent=False,
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) or (B,1)
        if t.dim() == 2 and t.shape[-1] == 1:
            t = t[:, 0]
        if t.dim() != 1:
            raise ValueError(f"t must be shape (B,) or (B,1). Got {tuple(t.shape)}")
        x = t[:, None] * self.W[None, :] * 2 * math.pi  # (B, D/2)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # (B, D)


def sinusoidal_time_embedding(t: torch.Tensor, embed_dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """
    Classic sinusoidal embedding for scalar timesteps.
    t: (B,) float tensor in [0,1] (or any positive range).
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even for sinusoidal_time_embedding.")
    if t.dim() == 2 and t.shape[-1] == 1:
        t = t[:, 0]
    half = embed_dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=t.device, dtype=t.dtype) / half)
    args = t[:, None] * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimeEmbedding(nn.Module):
    """
    Wraps a base embedding (Gaussian Fourier by default) and projects to a hidden dimension.
    """

    def __init__(self, base_dim: int, out_dim: int, kind: str = "gaussian"):
        super().__init__()
        self.kind = kind
        if kind == "gaussian":
            self.base = GaussianFourierProjection(base_dim)
        elif kind == "sinusoidal":
            self.base = None
            self.base_dim = base_dim
        else:
            raise ValueError(f"Unknown time embedding kind: {kind}")
        self.mlp = nn.Sequential(
            nn.Linear(base_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if self.kind == "gaussian":
            x = self.base(t)
        else:
            x = sinusoidal_time_embedding(t, self.base_dim)
        return self.mlp(x)

