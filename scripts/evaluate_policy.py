#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from shared_control_rl.config import EnvConfig
from shared_control_rl.controllers.baselines import fixed_lambda_policy, heuristic_hazard_policy
from shared_control_rl.envs.shared_control_env import SharedControlEnv

try:
    from stable_baselines3 import SAC
except ImportError:
    SAC = None


def action_from_lambda_target(lambda_target: float, current_lambda: float, max_lambda_rate: float, dt: float) -> np.ndarray:
    desired_rate = (lambda_target - current_lambda) / max(dt, 1e-6)
    normalized = desired_rate / max(max_lambda_rate, 1e-6)
    return np.array([np.clip(normalized, -1.0, 1.0)], dtype=np.float32)


def run_policy(
    env: SharedControlEnv,
    policy_name: str,
    model_path: str | None = None,
    seed: int | None = None,
    reset_options: dict | None = None,
) -> dict[str, list[float]]:
    obs, info = env.reset(seed=seed, options=reset_options)

    if policy_name == "model":
        if model_path is None or SAC is None:
            raise RuntimeError("For policy_name='model', stable-baselines3 and --model are required.")
        model = SAC.load(model_path)
        predict: Callable[[np.ndarray], np.ndarray] = lambda x: model.predict(x, deterministic=True)[0]  # type: ignore
    elif policy_name == "driver":
        policy = fixed_lambda_policy(1.0)
        predict = lambda x: action_from_lambda_target(policy(info), env.state.lam, env.base_config.max_lambda_rate, env.vehicle_params.dt)
    elif policy_name == "automation":
        policy = fixed_lambda_policy(0.0)
        predict = lambda x: action_from_lambda_target(policy(info), env.state.lam, env.base_config.max_lambda_rate, env.vehicle_params.dt)
    elif policy_name == "shared":
        policy = fixed_lambda_policy(0.5)
        predict = lambda x: action_from_lambda_target(policy(info), env.state.lam, env.base_config.max_lambda_rate, env.vehicle_params.dt)
    elif policy_name == "heuristic":
        predict = lambda x: action_from_lambda_target(heuristic_hazard_policy(info), env.state.lam, env.base_config.max_lambda_rate, env.vehicle_params.dt)
    else:
        raise ValueError(f"Unknown policy_name={policy_name}")

    terminated = truncated = False
    while not (terminated or truncated):
        action = predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

    return env.history


def plot_history(history: dict[str, list[float]], env: SharedControlEnv, title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    axes[0].plot(history["x"], history["y"], label="vehicle path")
    theta = np.linspace(0, 2 * np.pi, 200)
    ox = env.obstacle.x + (env.obstacle.a + env.scenario_cfg.obstacle_margin) * np.cos(theta)
    oy = env.obstacle.y + (env.obstacle.b + env.scenario_cfg.obstacle_margin) * np.sin(theta)
    axes[0].plot(ox, oy, linestyle="--", label="inflated obstacle")
    axes[0].set_xlabel("X [m]")
    axes[0].set_ylabel("Y [m]")
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["t"], history["lambda"], label="lambda_safe")
    axes[1].plot(history["t"], history["lambda_des"], label="lambda_des", linestyle="--")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("authority")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(history["t"], history["delta_drv_rwa"], label="driver equiv RWA")
    axes[2].plot(history["t"], history["delta_auto_rwa"], label="automation RWA")
    axes[2].plot(history["t"], history["delta_cmd_rwa"], label="commanded RWA")
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("steering [rad]")
    axes[2].legend()
    axes[2].grid(True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a baseline or trained policy.")
    parser.add_argument("--policy", type=str, default="heuristic", choices=["model", "driver", "automation", "shared", "heuristic"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--out", type=str, default="artifacts/eval.png")
    args = parser.parse_args()

    config = EnvConfig(seed=args.seed)
    config.scenario.domain_randomization = args.randomize
    config.scenario.side_randomization = args.randomize

    env = SharedControlEnv(config)
    history = run_policy(env, policy_name=args.policy, model_path=args.model, seed=args.seed)
    plot_history(history, env, title=f"Policy: {args.policy}", out_path=Path(args.out))


if __name__ == "__main__":
    main()
