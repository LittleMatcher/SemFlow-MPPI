from __future__ import annotations

import torch
from torch import nn

from cfm.models.adaptive_layer_norm import AdaLayerNorm
from cfm.models.embeddings import TimeEmbedding


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaLNTransformerBlock(nn.Module):
    """
    Transformer encoder block with AdaLN conditioning injected into both attention and FFN sublayers.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, cond_dim: int):
        super().__init__()
        self.adaln1 = AdaLayerNorm(d_model, cond_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.adaln2 = AdaLayerNorm(d_model, cond_dim)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Attention
        h = self.adaln1(x, cond)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop1(attn_out)
        # FFN
        h = self.adaln2(x, cond)
        x = x + self.drop2(self.ff(h))
        return x


class ConditionalVectorFieldTransformer(nn.Module):
    """
    v_theta(x_t, t, c) for trajectory sequences.

    Input:
      x_t: (B, T, in_dim) where in_dim=6: [q(2), dq(2), ddq(2)]
      t: (B,) in [0,1]
      c: (B, cond_dim) e.g. [q_start(2), dq_start(2), q_goal(2)]

    Output:
      (B, T, out_dim=6): [u(2), v(2), w(2)]
    """

    def __init__(
        self,
        in_dim: int = 6,
        out_dim: int = 6,
        cond_dim: int = 6,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.0,
        time_embed_dim: int = 256,
        time_embed_kind: str = "gaussian",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cond_dim = cond_dim
        self.d_model = d_model

        self.x_proj = nn.Linear(in_dim, d_model)
        self.t_embed = TimeEmbedding(base_dim=time_embed_dim, out_dim=time_embed_dim, kind=time_embed_kind)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim + time_embed_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.blocks = nn.ModuleList(
            [
                AdaLNTransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    cond_dim=d_model,
                )
                for _ in range(n_layers)
            ]
        )
        self.out_ln = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, out_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        key_padding_mask: (B, T) bool, True for padded positions.
        """
        if x_t.dim() != 3:
            raise ValueError(f"x_t must be (B,T,C). Got {tuple(x_t.shape)}")
        B, T, _ = x_t.shape
        if t.dim() == 2 and t.shape[-1] == 1:
            t = t[:, 0]
        if t.dim() != 1 or t.shape[0] != B:
            raise ValueError(f"t must be (B,) matching batch. Got {tuple(t.shape)} vs B={B}")
        if c.dim() != 2 or c.shape[0] != B:
            raise ValueError(f"c must be (B,cond_dim). Got {tuple(c.shape)} vs B={B}")

        h = self.x_proj(x_t)  # (B,T,D)
        te = self.t_embed(t)  # (B,Dt)
        cond = self.cond_proj(torch.cat([c, te], dim=-1))  # (B,D)

        for blk in self.blocks:
            h = blk(h, cond, key_padding_mask=key_padding_mask)
        h = self.out_ln(h)
        return self.out(h)

