from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import yaml


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@dataclass(frozen=True)
class RunPaths:
    out_dir: str

    @property
    def latest_ckpt(self) -> str:
        return os.path.join(self.out_dir, "latest.pt")

    def step_ckpt(self, step: int) -> str:
        return os.path.join(self.out_dir, f"step_{step:08d}.pt")

