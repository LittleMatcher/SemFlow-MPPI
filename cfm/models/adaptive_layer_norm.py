from __future__ import annotations

import torch
from torch import nn


class AdaLayerNorm(nn.Module):
    """
    Adaptive LayerNorm / FiLM conditioning:
      y = (1 + gamma) * LN(x) + beta
    where (gamma, beta) are predicted from a conditioning vector.

    This follows the common "AdaLN" pattern used in DiT-style models.
    """

    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(d_model, elementwise_affine=False)
        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * d_model),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        cond: (B, C)
        """
        h = self.ln(x)
        ss = self.to_scale_shift(cond)  # (B, 2D)
        gamma, beta = ss.chunk(2, dim=-1)
        gamma = gamma[:, None, :]
        beta = beta[:, None, :]
        return (1.0 + gamma) * h + beta

