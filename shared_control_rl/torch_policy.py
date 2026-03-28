from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch.distributions import Normal


LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0
ACTION_EPS = 1e-6


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.mean_net = MLP(obs_dim, hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.mean_net(obs)
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std).expand_as(mean)
        return Normal(mean, std)

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1.0 - action.pow(2) + ACTION_EPS)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return action, log_prob, entropy

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean = self.mean_net(obs)
        return torch.tanh(mean)

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        clipped_action = torch.clamp(action, -1.0 + ACTION_EPS, 1.0 - ACTION_EPS)
        z = torch.atanh(clipped_action)
        dist = self.distribution(obs)
        log_prob = dist.log_prob(z) - torch.log(1.0 - clipped_action.pow(2) + ACTION_EPS)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy


class ValueCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = MLP(obs_dim, hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.actor = SquashedGaussianActor(obs_dim, action_dim, hidden_dim=hidden_dim)
        self.critic = ValueCritic(obs_dim, hidden_dim=hidden_dim)

    def step(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action, log_prob, entropy = self.actor.sample(obs)
        value = self.critic(obs)
        return action, log_prob, entropy, value

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return self.actor.deterministic(obs)
        action, _, _ = self.actor.sample(obs)
        return action

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_prob, entropy = self.actor.evaluate(obs, actions)
        value = self.critic(obs)
        return log_prob, entropy, value


@dataclass
class PPOCheckpoint:
    obs_dim: int
    action_dim: int
    hidden_dim: int
    state_dict: Dict[str, torch.Tensor]
    metadata: Dict[str, Any]


def load_checkpoint_data(path: str | Path, device: str | torch.device = "cpu") -> PPOCheckpoint:
    ckpt = torch.load(path, map_location=device)
    metadata = ckpt.get("metadata", {}) if isinstance(ckpt, dict) else {}
    return PPOCheckpoint(
        obs_dim=int(ckpt["obs_dim"]),
        action_dim=int(ckpt["action_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        state_dict=ckpt["state_dict"],
        metadata=dict(metadata),
    )


def checkpoint_metadata(path: str | Path, device: str | torch.device = "cpu") -> Dict[str, Any]:
    return load_checkpoint_data(path, device=device).metadata


def save_checkpoint(
    model: ActorCritic,
    obs_dim: int,
    action_dim: int,
    hidden_dim: int,
    path: str | Path,
    metadata: Dict[str, Any] | None = None,
) -> None:
    ckpt = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_dim": hidden_dim,
        "state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(ckpt, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> ActorCritic:
    bundle = load_checkpoint_data(path, device=device)
    model = ActorCritic(
        obs_dim=int(bundle.obs_dim),
        action_dim=int(bundle.action_dim),
        hidden_dim=int(bundle.hidden_dim),
    )
    model.load_state_dict(bundle.state_dict)
    model.to(device)
    model.eval()
    return model
