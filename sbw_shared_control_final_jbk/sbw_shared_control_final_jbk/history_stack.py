from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional, Tuple

import numpy as np


class HistoryStackEnv:
    """Wrap the base env and return a stacked observation [o_{t-H+1}, ..., o_t]."""

    def __init__(self, env, history_len: int = 4):
        self.env = env
        self.history_len = int(history_len)
        self._buffer: Deque[np.ndarray] = deque(maxlen=self.history_len)
        self.base_obs_dim: Optional[int] = None

    @property
    def observation_dim(self) -> int:
        if self.base_obs_dim is None:
            raise RuntimeError("reset the environment before querying observation_dim")
        return self.base_obs_dim * self.history_len

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, float]]:
        obs, info = self.env.reset(seed=seed)
        self.base_obs_dim = int(obs.shape[0])
        self._buffer.clear()
        for _ in range(self.history_len):
            self._buffer.append(obs.copy())
        return self._stacked_obs(), info

    def step(self, action: float):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._buffer.append(obs.copy())
        return self._stacked_obs(), reward, terminated, truncated, info

    def _stacked_obs(self) -> np.ndarray:
        return np.concatenate(list(self._buffer), axis=0).astype(np.float32)
