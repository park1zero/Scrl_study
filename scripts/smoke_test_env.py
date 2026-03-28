#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from shared_control_rl.config import EnvConfig
from shared_control_rl.envs.shared_control_env import SharedControlEnv


def main() -> None:
    env = SharedControlEnv(EnvConfig())
    obs, info = env.reset()
    total_reward = 0.0
    terminated = truncated = False

    while not (terminated or truncated):
        action = np.array([0.0], dtype=np.float32)  # hold current authority
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    print("Episode finished")
    print(f"total_reward={total_reward:.3f}")
    print(f"final x={info['x']:.2f}, y={info['y']:.2f}, h={info['h']:.2f}, lambda={info['lambda_safe']:.2f}")


if __name__ == "__main__":
    main()
