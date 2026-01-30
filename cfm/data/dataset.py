from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TrajBatch:
    q1: torch.Tensor     # (B,T,2)
    dq1: torch.Tensor    # (B,T,2)
    ddq1: torch.Tensor   # (B,T,2)
    c: torch.Tensor      # (B,cond_dim)


class NpzTrajectoryDataset(Dataset):
    """
    Minimal dataset for expert trajectories.
    Expected arrays:
      q1, dq1, ddq1: (N,T,2)
    Optional:
      q_start: (N,2) default q1[:,0]
      dq_start: (N,2) default dq1[:,0]
      q_goal: (N,2) default q1[:,-1]
    """

    def __init__(self, path: str, seq_len: int | None = None):
        super().__init__()
        data = np.load(path)
        self.q1 = np.asarray(data["q1"], dtype=np.float32)
        self.dq1 = np.asarray(data["dq1"], dtype=np.float32)
        self.ddq1 = np.asarray(data["ddq1"], dtype=np.float32)

        if self.q1.ndim != 3 or self.q1.shape[-1] != 2:
            raise ValueError(f"q1 must be (N,T,2). Got {self.q1.shape}")
        if self.dq1.shape != self.q1.shape or self.ddq1.shape != self.q1.shape:
            raise ValueError("dq1 and ddq1 must match q1 shape (N,T,2).")

        self.N, self.T, _ = self.q1.shape
        self.seq_len = seq_len or self.T
        if self.seq_len > self.T:
            raise ValueError(f"seq_len {self.seq_len} > data T {self.T}")

        self.q_start = np.asarray(data["q_start"], dtype=np.float32) if "q_start" in data else self.q1[:, 0]
        self.dq_start = np.asarray(data["dq_start"], dtype=np.float32) if "dq_start" in data else self.dq1[:, 0]
        self.q_goal = np.asarray(data["q_goal"], dtype=np.float32) if "q_goal" in data else self.q1[:, -1]

        for name, arr in [("q_start", self.q_start), ("dq_start", self.dq_start), ("q_goal", self.q_goal)]:
            if arr.shape != (self.N, 2):
                raise ValueError(f"{name} must be (N,2). Got {arr.shape}")

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # If seq_len < T, take a centered crop (deterministic) for simplicity.
        if self.seq_len == self.T:
            sl = slice(None)
        else:
            start = (self.T - self.seq_len) // 2
            sl = slice(start, start + self.seq_len)

        q1 = torch.from_numpy(self.q1[idx, sl])
        dq1 = torch.from_numpy(self.dq1[idx, sl])
        ddq1 = torch.from_numpy(self.ddq1[idx, sl])

        c = torch.from_numpy(
            np.concatenate([self.q_start[idx], self.dq_start[idx], self.q_goal[idx]], axis=0).astype(np.float32)
        )
        return {"q1": q1, "dq1": dq1, "ddq1": ddq1, "c": c}

