from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from torch import nn


@dataclass
class BCConfig:
    input_dim: int
    hidden_dim: int = 128
    hidden_dim2: int = 128


class BCAuthorityActor(nn.Module):
    """Deterministic actor for authority-rate imitation.

    Input: stacked observation z_k
    Output: action a_k in [-1, 1]
    """

    def __init__(self, cfg: BCConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim2, 1),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def save_checkpoint(path: str | Path, model: nn.Module, extra: Dict | None = None) -> None:
    payload = {"state_dict": model.state_dict(), "extra": extra or {}}
    torch.save(payload, str(path))


def load_checkpoint(path: str | Path, model: nn.Module) -> Dict:
    payload = torch.load(str(path), map_location="cpu")
    model.load_state_dict(payload["state_dict"])
    return payload.get("extra", {})
