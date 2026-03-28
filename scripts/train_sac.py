#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path

from shared_control_rl.config import EnvConfig
from shared_control_rl.envs.shared_control_env import SharedControlEnv

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
except ImportError as exc:
    raise SystemExit(
        "stable-baselines3 is required for training. Install requirements.txt first."
    ) from exc


def make_env(config: EnvConfig):
    def _factory():
        env = SharedControlEnv(config)
        return Monitor(env)

    return _factory


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAC for authority allocation.")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="artifacts/sac_authority")
    parser.add_argument("--randomize", action="store_true")
    args = parser.parse_args()

    config = EnvConfig(seed=args.seed)
    config.scenario.domain_randomization = args.randomize
    config.scenario.side_randomization = args.randomize

    vec_env = make_vec_env(
        make_env(config),
        n_envs=1,
        seed=args.seed,
    )

    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        learning_starts=2_000,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        seed=args.seed,
        device="auto",
        tensorboard_log="artifacts/tb_logs",
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
