from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch


@dataclass
class ReplayConfig:
    capacity: int = 100000
    demo_fraction: float = 0.25


class ReplayBuffer:
    """Simple replay buffer with optional demo-aware sampling.

    Each transition stores:
      obs, action, reward, next_obs, done, is_demo
    """

    def __init__(self, obs_dim: int, action_dim: int, cfg: ReplayConfig | None = None):
        self.cfg = cfg or ReplayConfig()
        cap = self.cfg.capacity
        self.obs = np.zeros((cap, obs_dim), dtype=np.float32)
        self.actions = np.zeros((cap, action_dim), dtype=np.float32)
        self.rewards = np.zeros((cap, 1), dtype=np.float32)
        self.next_obs = np.zeros((cap, obs_dim), dtype=np.float32)
        self.dones = np.zeros((cap, 1), dtype=np.float32)
        self.is_demo = np.zeros((cap, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.capacity = cap

    def add(self, obs, action, reward, next_obs, done, is_demo: bool = False):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        self.is_demo[i] = 1.0 if is_demo else 0.0
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _valid_indices(self):
        return np.arange(self.size)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        idx_all = self._valid_indices()
        demo_idx = idx_all[self.is_demo[idx_all, 0] > 0.5]
        non_demo_idx = idx_all[self.is_demo[idx_all, 0] <= 0.5]

        n_demo = min(int(round(batch_size * self.cfg.demo_fraction)), len(demo_idx))
        n_non = batch_size - n_demo
        if len(non_demo_idx) == 0:
            n_demo = batch_size
            n_non = 0
        elif n_non > len(non_demo_idx):
            n_non = len(non_demo_idx)
            n_demo = min(batch_size - n_non, len(demo_idx))
        if n_demo == 0 and len(idx_all) > 0 and batch_size > 0:
            # Fallback to uniform sampling when demos are unavailable.
            idx = np.random.randint(0, self.size, size=batch_size)
        else:
            parts = []
            if n_demo > 0:
                parts.append(np.random.choice(demo_idx, size=n_demo, replace=len(demo_idx) < n_demo))
            if n_non > 0:
                parts.append(np.random.choice(non_demo_idx, size=n_non, replace=len(non_demo_idx) < n_non))
            idx = np.concatenate(parts, axis=0)
            if len(idx) < batch_size:
                extra = np.random.randint(0, self.size, size=batch_size - len(idx))
                idx = np.concatenate([idx, extra], axis=0)
            np.random.shuffle(idx)

        batch = {
            'obs': torch.as_tensor(self.obs[idx], device=device),
            'actions': torch.as_tensor(self.actions[idx], device=device),
            'rewards': torch.as_tensor(self.rewards[idx], device=device),
            'next_obs': torch.as_tensor(self.next_obs[idx], device=device),
            'dones': torch.as_tensor(self.dones[idx], device=device),
            'is_demo': torch.as_tensor(self.is_demo[idx], device=device),
        }
        return batch
