from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from bc_model import BCConfig, BCAuthorityActor


LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0
EPS = 1e-6


@dataclass
class SACConfig:
    obs_dim: int
    action_dim: int = 1
    hidden_dim: int = 128
    hidden_dim2: int = 128
    gamma: float = 0.99
    tau: float = 0.01
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_alpha: float = 0.10
    target_entropy: float = -1.0
    bc_coef: float = 0.25
    anchor_coef: float = 0.12


class GaussianActor(nn.Module):
    def __init__(self, cfg: SACConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.obs_dim, cfg.hidden_dim)
        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.hidden_dim2)
        self.mu = nn.Linear(cfg.hidden_dim2, cfg.action_dim)
        self.log_std = nn.Linear(cfg.hidden_dim2, cfg.action_dim)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return x

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._features(obs)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(obs)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)
        # tanh squash correction
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + EPS)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean_action = torch.tanh(mu)
        return action, log_prob, mean_action

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mu, log_std = self.forward(obs)
        if deterministic:
            return torch.tanh(mu)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        return torch.tanh(dist.sample())


class QNetwork(nn.Module):
    def __init__(self, cfg: SACConfig):
        super().__init__()
        in_dim = cfg.obs_dim + cfg.action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim2, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


class SACAgent:
    def __init__(self, cfg: SACConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.actor = GaussianActor(cfg).to(device)
        self.q1 = QNetwork(cfg).to(device)
        self.q2 = QNetwork(cfg).to(device)
        self.q1_target = QNetwork(cfg).to(device)
        self.q2_target = QNetwork(cfg).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.anchor_actor: GaussianActor | None = None
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg.critic_lr)
        self.log_alpha = torch.tensor(np.log(cfg.init_alpha), dtype=torch.float32, device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, obs_np, deterministic: bool = False) -> float:
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor.act(obs, deterministic=deterministic).cpu().numpy()[0, 0]
        return float(np.clip(action, -1.0, 1.0))

    def update(self, batch: Dict[str, torch.Tensor], step: int, actor_update_delay: int = 2) -> Dict[str, float]:
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_obs']
        dones = batch['dones']
        is_demo = batch['is_demo']

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_obs)
            q1_t = self.q1_target(next_obs, next_action)
            q2_t = self.q2_target(next_obs, next_action)
            q_t = torch.min(q1_t, q2_t) - self.alpha.detach() * next_log_prob
            target_q = rewards + (1.0 - dones) * self.cfg.gamma * q_t

        q1_pred = self.q1(obs, actions)
        q2_pred = self.q2(obs, actions)
        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        actor_loss_value = 0.0
        alpha_loss_value = 0.0
        bc_loss_value = 0.0
        anchor_loss_value = 0.0
        if step % actor_update_delay == 0:
            new_action, log_prob, mean_action = self.actor.sample(obs)
            q1_new = self.q1(obs, new_action)
            q2_new = self.q2(obs, new_action)
            q_new = torch.min(q1_new, q2_new)
            actor_loss = (self.alpha.detach() * log_prob - q_new).mean()

            demo_mask = (is_demo > 0.5).squeeze(-1)
            if demo_mask.any():
                bc_loss = F.mse_loss(mean_action[demo_mask], actions[demo_mask])
                bc_loss_value = float(bc_loss.item())
                actor_loss = actor_loss + self.cfg.bc_coef * bc_loss
            if self.anchor_actor is not None:
                with torch.no_grad():
                    anchor_mean = self.anchor_actor.act(obs, deterministic=True)
                anchor_loss = F.mse_loss(mean_action, anchor_mean)
                anchor_loss_value = float(anchor_loss.item())
                actor_loss = actor_loss + self.cfg.anchor_coef * anchor_loss

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            actor_loss_value = float(actor_loss.item())

            alpha_loss = -(self.log_alpha * (log_prob + self.cfg.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha_loss_value = float(alpha_loss.item())

            self._soft_update(self.q1, self.q1_target)
            self._soft_update(self.q2, self.q2_target)

        return {
            'q1_loss': float(q1_loss.item()),
            'q2_loss': float(q2_loss.item()),
            'actor_loss': actor_loss_value,
            'alpha_loss': alpha_loss_value,
            'alpha': float(self.alpha.item()),
            'bc_loss': bc_loss_value,
            'anchor_loss': anchor_loss_value,
        }

    def _soft_update(self, src: nn.Module, dst: nn.Module) -> None:
        tau = self.cfg.tau
        for p, p_targ in zip(src.parameters(), dst.parameters()):
            p_targ.data.mul_(1.0 - tau)
            p_targ.data.add_(tau * p.data)

    def load_bc_actor(self, ckpt_path: str | Path) -> Dict:
        payload = torch.load(str(ckpt_path), map_location='cpu')
        state = payload['state_dict']
        # Reconstruct BC model and copy compatible layers.
        bc_model = BCAuthorityActor(BCConfig(input_dim=self.cfg.obs_dim))
        bc_model.load_state_dict(state)
        # BC actor: net[0], net[2], net[4]
        self.actor.fc1.weight.data.copy_(bc_model.net[0].weight.data)
        self.actor.fc1.bias.data.copy_(bc_model.net[0].bias.data)
        self.actor.fc2.weight.data.copy_(bc_model.net[2].weight.data)
        self.actor.fc2.bias.data.copy_(bc_model.net[2].bias.data)
        self.actor.mu.weight.data.copy_(bc_model.net[4].weight.data)
        self.actor.mu.bias.data.copy_(bc_model.net[4].bias.data)
        # Conservative initial exploration around BC mean.
        nn.init.constant_(self.actor.log_std.weight, 0.0)
        nn.init.constant_(self.actor.log_std.bias, -2.2)
        self.anchor_actor = GaussianActor(self.cfg).to(self.device)
        self.anchor_actor.load_state_dict(self.actor.state_dict())
        for p in self.anchor_actor.parameters():
            p.requires_grad = False
        self.anchor_actor.eval()
        return payload.get('extra', {})

    def save(self, path: str | Path, extra: Dict | None = None) -> None:
        payload = {
            'actor': self.actor.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
            'extra': extra or {},
        }
        torch.save(payload, str(path))

    def load(self, path: str | Path) -> Dict:
        payload = torch.load(str(path), map_location=self.device)
        self.actor.load_state_dict(payload['actor'])
        self.q1.load_state_dict(payload['q1'])
        self.q2.load_state_dict(payload['q2'])
        self.q1_target.load_state_dict(payload['q1_target'])
        self.q2_target.load_state_dict(payload['q2_target'])
        self.log_alpha.data.copy_(payload['log_alpha'].to(self.device))
        return payload.get('extra', {})
