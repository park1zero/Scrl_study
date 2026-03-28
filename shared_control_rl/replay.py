from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import torch


@dataclass
class TransitionBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor
    demo: torch.Tensor
    hazard: torch.Tensor
    difficulty: torch.Tensor
    hazard_severity: torch.Tensor


class StratifiedReplayBuffer:
    """Replay buffer with lightweight demo / hazard stratification.

    The buffer tracks three useful subsets for this project:
    - demo transitions collected from the heuristic allocator
    - hazard transitions near the obstacle or collisions
    - everything else

    Sampling can then request a fixed fraction from the demo/hazard subsets,
    which stabilizes off-policy learning when easy episodes dominate.
    """

    _DIFFICULTY_TO_ID: Dict[str, int] = {"easy": 0, "medium": 1, "hard": 2}

    def __init__(self, obs_dim: int, action_dim: int, capacity: int) -> None:
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.demo = np.zeros((capacity, 1), dtype=np.float32)
        self.hazard = np.zeros((capacity, 1), dtype=np.float32)
        self.difficulty = np.zeros((capacity, 1), dtype=np.float32)
        self.hazard_severity = np.zeros((capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

        self._demo_flags = np.zeros((capacity,), dtype=bool)
        self._hazard_flags = np.zeros((capacity,), dtype=bool)
        self._demo_indices: set[int] = set()
        self._hazard_indices: set[int] = set()

    def __len__(self) -> int:
        return self.size

    @classmethod
    def difficulty_id(cls, difficulty: str | None) -> int:
        if difficulty is None:
            return 1
        return int(cls._DIFFICULTY_TO_ID.get(str(difficulty), 1))

    def _clear_index(self, idx: int) -> None:
        if self._demo_flags[idx]:
            self._demo_indices.discard(idx)
            self._demo_flags[idx] = False
        if self._hazard_flags[idx]:
            self._hazard_indices.discard(idx)
            self._hazard_flags[idx] = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        *,
        demo: bool = False,
        hazard: bool = False,
        hazard_severity: float = 0.0,
        difficulty: str | None = None,
    ) -> None:
        idx = self.ptr
        self._clear_index(idx)

        self.obs[idx] = np.asarray(obs, dtype=np.float32)
        self.actions[idx] = np.asarray(action, dtype=np.float32)
        self.rewards[idx, 0] = float(reward)
        self.next_obs[idx] = np.asarray(next_obs, dtype=np.float32)
        self.dones[idx, 0] = float(done)
        self.demo[idx, 0] = 1.0 if demo else 0.0
        self.hazard[idx, 0] = 1.0 if hazard else 0.0
        self.difficulty[idx, 0] = float(self.difficulty_id(difficulty))
        self.hazard_severity[idx, 0] = float(max(hazard_severity, 0.0))

        if demo:
            self._demo_flags[idx] = True
            self._demo_indices.add(idx)
        if hazard:
            self._hazard_flags[idx] = True
            self._hazard_indices.add(idx)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def counts(self) -> Dict[str, int]:
        return {
            "size": int(self.size),
            "demo": int(len(self._demo_indices)),
            "hazard": int(len(self._hazard_indices)),
            "regular": int(self.size - len(self._hazard_indices)),
        }

    def _all_indices(self) -> np.ndarray:
        return np.arange(self.size, dtype=np.int64)

    def _sample_from_indices(
        self,
        indices: Iterable[int],
        n: int,
        *,
        weighted: bool = False,
    ) -> np.ndarray:
        if n <= 0:
            return np.empty((0,), dtype=np.int64)
        arr = np.fromiter(indices, dtype=np.int64)
        if arr.size == 0:
            return np.empty((0,), dtype=np.int64)
        replace = bool(arr.size < n)
        if weighted:
            weights = np.asarray(self.hazard_severity[arr, 0], dtype=np.float64)
            weights = np.clip(weights, 1e-6, None)
            weights = weights / np.sum(weights)
            return np.random.choice(arr, size=n, replace=replace, p=weights).astype(np.int64)
        return np.random.choice(arr, size=n, replace=replace).astype(np.int64)

    def sample(
        self,
        batch_size: int,
        device: torch.device,
        *,
        demo_frac: float = 0.0,
        hazard_frac: float = 0.0,
    ) -> TransitionBatch:
        if self.size <= 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")

        batch_size = int(batch_size)
        demo_frac = float(np.clip(demo_frac, 0.0, 1.0))
        hazard_frac = float(np.clip(hazard_frac, 0.0, 1.0))

        n_demo = min(batch_size, int(round(batch_size * demo_frac)))
        n_hazard = min(batch_size - n_demo, int(round(batch_size * hazard_frac)))
        n_rest = batch_size - n_demo - n_hazard

        demo_idx = self._sample_from_indices(self._demo_indices, n_demo)

        hazard_pool = self._hazard_indices.difference(set(demo_idx.tolist()))
        hazard_idx = self._sample_from_indices(hazard_pool, n_hazard, weighted=True)

        used = set(demo_idx.tolist()) | set(hazard_idx.tolist())
        regular_pool = set(self._all_indices().tolist()).difference(used)
        rest_idx = self._sample_from_indices(regular_pool, n_rest)

        idx = np.concatenate([demo_idx, hazard_idx, rest_idx], axis=0)
        if idx.size < batch_size:
            fill_idx = self._sample_from_indices(self._all_indices().tolist(), batch_size - idx.size)
            idx = np.concatenate([idx, fill_idx], axis=0)

        np.random.shuffle(idx)

        return TransitionBatch(
            obs=torch.as_tensor(self.obs[idx], dtype=torch.float32, device=device),
            actions=torch.as_tensor(self.actions[idx], dtype=torch.float32, device=device),
            rewards=torch.as_tensor(self.rewards[idx], dtype=torch.float32, device=device),
            next_obs=torch.as_tensor(self.next_obs[idx], dtype=torch.float32, device=device),
            dones=torch.as_tensor(self.dones[idx], dtype=torch.float32, device=device),
            demo=torch.as_tensor(self.demo[idx], dtype=torch.float32, device=device),
            hazard=torch.as_tensor(self.hazard[idx], dtype=torch.float32, device=device),
            difficulty=torch.as_tensor(self.difficulty[idx], dtype=torch.float32, device=device),
            hazard_severity=torch.as_tensor(self.hazard_severity[idx], dtype=torch.float32, device=device),
        )
