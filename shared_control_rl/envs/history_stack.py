from __future__ import annotations

from collections import deque
from typing import Any, Deque

import numpy as np

try:
    from gymnasium import spaces
except ImportError:
    try:
        from gym import spaces  # type: ignore
    except ImportError:
        class _FallbackBox:
            def __init__(self, low, high, shape, dtype):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype

        class _FallbackSpaces:
            Box = _FallbackBox

        spaces = _FallbackSpaces()  # type: ignore


class ObservationHistoryStack:
    """Lightweight observation-history stack for partially observed policies.

    This wrapper repeats the reset observation to fill the buffer, then appends
    the newest observation every step. The returned observation is the flattened
    concatenation `[o_{t-k+1}, ..., o_t]`.
    """

    def __init__(self, env: Any, n_stack: int = 1) -> None:
        if n_stack < 1:
            raise ValueError("n_stack must be >= 1")
        self.env = env
        self.n_stack = int(n_stack)
        self._buffer: Deque[np.ndarray] = deque(maxlen=self.n_stack)

        base_space = env.observation_space
        if len(base_space.shape) != 1:
            raise ValueError("ObservationHistoryStack only supports 1D observations.")
        self.base_obs_dim = int(base_space.shape[0])

        base_low = np.asarray(base_space.low, dtype=np.float32)
        base_high = np.asarray(base_space.high, dtype=np.float32)
        if base_low.size == 1:
            base_low = np.full((self.base_obs_dim,), float(base_low.reshape(-1)[0]), dtype=np.float32)
        else:
            base_low = np.broadcast_to(base_low.reshape(-1), (self.base_obs_dim,)).astype(np.float32)
        if base_high.size == 1:
            base_high = np.full((self.base_obs_dim,), float(base_high.reshape(-1)[0]), dtype=np.float32)
        else:
            base_high = np.broadcast_to(base_high.reshape(-1), (self.base_obs_dim,)).astype(np.float32)

        low = np.tile(base_low, self.n_stack)
        high = np.tile(base_high, self.n_stack)
        self.observation_space = spaces.Box(low=low, high=high, shape=low.shape, dtype=np.float32)
        self.action_space = env.action_space
        self.metadata = getattr(env, "metadata", {})

    def _stacked_obs(self) -> np.ndarray:
        if len(self._buffer) != self.n_stack:
            raise RuntimeError("ObservationHistoryStack buffer is not initialized.")
        return np.concatenate(list(self._buffer), axis=0).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        self._buffer.clear()
        for _ in range(self.n_stack):
            self._buffer.append(obs_arr.copy())
        stacked = self._stacked_obs()
        info = dict(info)
        info.update({
            "history_stack": int(self.n_stack),
            "base_obs_dim": int(self.base_obs_dim),
            "stacked_obs_dim": int(stacked.shape[0]),
        })
        return stacked, info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        self._buffer.append(obs_arr.copy())
        stacked = self._stacked_obs()
        info = dict(info)
        info.update({
            "history_stack": int(self.n_stack),
            "base_obs_dim": int(self.base_obs_dim),
            "stacked_obs_dim": int(stacked.shape[0]),
        })
        return stacked, reward, terminated, truncated, info

    def __getattr__(self, name: str):
        return getattr(self.env, name)
