from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class CheckpointState:
    step: int
    epoch: int
    model: dict[str, Any]
    optim: dict[str, Any]
    config: dict[str, Any]


def save_checkpoint(path: str, *, step: int, epoch: int, model: torch.nn.Module, optim: torch.optim.Optimizer, config: dict) -> None:
    ckpt = {
        "step": int(step),
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "config": config,
    }
    torch.save(ckpt, path)


def load_checkpoint(path: str, *, model: torch.nn.Module, optim: torch.optim.Optimizer | None = None, map_location: str = "cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    if optim is not None and "optim" in ckpt:
        optim.load_state_dict(ckpt["optim"])
    return ckpt

