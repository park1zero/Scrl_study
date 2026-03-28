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
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ObsFeatureEncoder(nn.Module):
    """Encode either a flat observation or a short observation sequence.

    The environment already provides a flattened history stack. When
    `encoder_type='gru'`, the flat vector is reshaped to
    `[batch, history_stack, base_obs_dim]` and processed by a compact GRU.
    Otherwise the encoder falls back to a standard MLP.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        *,
        history_stack: int = 1,
        base_obs_dim: int | None = None,
        encoder_type: str = "mlp",
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.hidden_dim = int(hidden_dim)
        self.history_stack = max(1, int(history_stack))
        inferred_base = self.obs_dim // self.history_stack
        self.base_obs_dim = int(base_obs_dim or inferred_base)
        self.encoder_type = str(encoder_type).lower()

        if self.obs_dim != self.history_stack * self.base_obs_dim:
            raise ValueError(
                f"obs_dim={self.obs_dim} is incompatible with history_stack={self.history_stack} "
                f"and base_obs_dim={self.base_obs_dim}."
            )

        if self.encoder_type == "gru" and self.history_stack > 1:
            self.frame_embed = nn.Sequential(
                nn.Linear(self.base_obs_dim, hidden_dim),
                nn.ReLU(),
            )
            self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
            self.post = nn.Sequential(
                nn.Linear(hidden_dim + self.base_obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        else:
            self.encoder_type = "mlp"
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.encoder_type == "mlp":
            return self.net(obs)

        batch = obs.shape[0]
        seq = obs.reshape(batch, self.history_stack, self.base_obs_dim)
        seq_emb = self.frame_embed(seq)
        _, h_n = self.gru(seq_emb)
        final_state = h_n[-1]
        current_frame = seq[:, -1, :]
        return self.post(torch.cat([final_state, current_frame], dim=-1))


class SACActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        *,
        history_stack: int = 1,
        base_obs_dim: int | None = None,
        encoder_type: str = "mlp",
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.history_stack = max(1, int(history_stack))
        self.base_obs_dim = int(base_obs_dim or (self.obs_dim // self.history_stack))
        self.encoder_type = str(encoder_type).lower()

        self.backbone = ObsFeatureEncoder(
            obs_dim=self.obs_dim,
            hidden_dim=self.hidden_dim,
            history_stack=self.history_stack,
            base_obs_dim=self.base_obs_dim,
            encoder_type=self.encoder_type,
        )
        self.mean = nn.Linear(self.hidden_dim, self.action_dim)
        self.log_std = nn.Linear(self.hidden_dim, self.action_dim)

    def _stats(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean, log_std = self._stats(obs)
        return Normal(mean, torch.exp(log_std))

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1.0 - action.pow(2) + ACTION_EPS)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean, _ = self._stats(obs)
        deterministic_action = torch.tanh(mean)
        return action, log_prob, deterministic_action

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            mean, _ = self._stats(obs)
            return torch.tanh(mean)
        action, _, _ = self.sample(obs)
        return action


class QNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        *,
        history_stack: int = 1,
        base_obs_dim: int | None = None,
        encoder_type: str = "mlp",
    ) -> None:
        super().__init__()
        self.obs_encoder = ObsFeatureEncoder(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            history_stack=history_stack,
            base_obs_dim=base_obs_dim,
            encoder_type=encoder_type,
        )
        self.head = MLP(hidden_dim + action_dim, hidden_dim, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h = self.obs_encoder(obs)
        x = torch.cat([h, action], dim=-1)
        return self.head(x)


class DoubleQCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        *,
        history_stack: int = 1,
        base_obs_dim: int | None = None,
        encoder_type: str = "mlp",
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.history_stack = max(1, int(history_stack))
        self.base_obs_dim = int(base_obs_dim or (self.obs_dim // self.history_stack))
        self.encoder_type = str(encoder_type).lower()
        self.q1 = QNetwork(
            obs_dim,
            action_dim,
            hidden_dim=hidden_dim,
            history_stack=self.history_stack,
            base_obs_dim=self.base_obs_dim,
            encoder_type=self.encoder_type,
        )
        self.q2 = QNetwork(
            obs_dim,
            action_dim,
            hidden_dim=hidden_dim,
            history_stack=self.history_stack,
            base_obs_dim=self.base_obs_dim,
            encoder_type=self.encoder_type,
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs, action), self.q2(obs, action)


@dataclass
class SACCheckpoint:
    obs_dim: int
    action_dim: int
    hidden_dim: int
    actor_state: Dict[str, torch.Tensor]
    critic_state: Dict[str, torch.Tensor] | None
    target_critic_state: Dict[str, torch.Tensor] | None
    log_alpha: float | None
    metadata: Dict[str, Any]


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def load_checkpoint_data(path: str | Path, device: str | torch.device = "cpu") -> SACCheckpoint:
    ckpt = torch.load(path, map_location=device)
    metadata = ckpt.get("metadata", {}) if isinstance(ckpt, dict) else {}
    return SACCheckpoint(
        obs_dim=int(ckpt["obs_dim"]),
        action_dim=int(ckpt["action_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        actor_state=ckpt["actor_state"],
        critic_state=ckpt.get("critic_state"),
        target_critic_state=ckpt.get("target_critic_state"),
        log_alpha=ckpt.get("log_alpha"),
        metadata=dict(metadata),
    )


def checkpoint_metadata(path: str | Path, device: str | torch.device = "cpu") -> Dict[str, Any]:
    return load_checkpoint_data(path, device=device).metadata


def save_checkpoint(
    actor: SACActor,
    critic: DoubleQCritic | None,
    target_critic: DoubleQCritic | None,
    obs_dim: int,
    action_dim: int,
    hidden_dim: int,
    path: str | Path,
    log_alpha: float | None = None,
    metadata: Dict[str, Any] | None = None,
) -> None:
    payload_meta = dict(metadata or {})
    payload_meta.setdefault("history_stack", int(getattr(actor, "history_stack", 1)))
    payload_meta.setdefault("base_obs_dim", int(getattr(actor, "base_obs_dim", obs_dim)))
    payload_meta.setdefault("encoder_type", str(getattr(actor, "encoder_type", "mlp")))

    ckpt = {
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "hidden_dim": int(hidden_dim),
        "actor_state": actor.state_dict(),
        "critic_state": critic.state_dict() if critic is not None else None,
        "target_critic_state": target_critic.state_dict() if target_critic is not None else None,
        "log_alpha": float(log_alpha) if log_alpha is not None else None,
        "metadata": payload_meta,
    }
    torch.save(ckpt, path)


def load_actor(path: str | Path, device: str | torch.device = "cpu") -> SACActor:
    bundle = load_checkpoint_data(path, device=device)
    meta = bundle.metadata or {}
    history_stack = int(meta.get("history_stack", 1))
    base_obs_dim = int(meta.get("base_obs_dim", bundle.obs_dim // max(history_stack, 1)))
    encoder_type = str(meta.get("encoder_type", "mlp"))
    actor = SACActor(
        obs_dim=bundle.obs_dim,
        action_dim=bundle.action_dim,
        hidden_dim=bundle.hidden_dim,
        history_stack=history_stack,
        base_obs_dim=base_obs_dim,
        encoder_type=encoder_type,
    )
    actor.load_state_dict(bundle.actor_state)
    actor.to(device)
    actor.eval()
    return actor
