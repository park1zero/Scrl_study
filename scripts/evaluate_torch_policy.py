#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from shared_control_rl.config import EnvConfig
from shared_control_rl.envs import make_env
from shared_control_rl.metrics import summarize_history
from shared_control_rl.torch_policy import checkpoint_metadata, load_checkpoint


def _resolve_history_stack(model_path: str, requested_stack: int | None) -> int:
    if requested_stack is not None and requested_stack > 0:
        return int(requested_stack)
    meta = checkpoint_metadata(model_path)
    return int(meta.get("history_stack", 1)) if meta else 1


def rollout(env, model_path: str, deterministic: bool = True, seed: int | None = None, reset_options: dict | None = None) -> tuple[dict[str, list[float]], dict]:
    device = torch.device("cpu")
    model = load_checkpoint(model_path, device=device)
    obs, info = env.reset(seed=seed, options=reset_options)
    terminated = truncated = False

    while not (terminated or truncated):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_t = model.act(obs_t, deterministic=deterministic)
        action = action_t.squeeze(0).cpu().numpy().astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

    return env.history, info


def plot_history(history: dict[str, list[float]], env, title: str, out_path: Path) -> None:
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
    parser = argparse.ArgumentParser(description="Evaluate a minimal PyTorch PPO checkpoint.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--driver-population", action="store_true")
    parser.add_argument("--curriculum-progress", type=float, default=None)
    parser.add_argument("--history-stack", type=int, default=None)
    parser.add_argument("--out", type=str, default="artifacts/eval_torch_policy.png")
    parser.add_argument("--metrics-json", type=str, default=None)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    history_stack = _resolve_history_stack(args.model, args.history_stack)

    config = EnvConfig(seed=args.seed)
    config.scenario.domain_randomization = args.randomize or args.curriculum or args.driver_population
    config.scenario.side_randomization = args.randomize or args.curriculum or args.driver_population
    config.curriculum.enabled = args.curriculum
    config.driver_population.enabled = args.driver_population

    env = make_env(config, history_stack=history_stack)
    if args.curriculum_progress is not None:
        env.set_curriculum_progress(args.curriculum_progress)
    history, info = rollout(env, args.model, deterministic=not args.stochastic, seed=args.seed)
    plot_history(history, env, title=f"PyTorch PPO policy (stack={history_stack})", out_path=Path(args.out))

    metrics = summarize_history(history, info)
    print("Episode summary:")
    for key in sorted(metrics):
        print(f"  {key}: {metrics[key]:.4f}")

    if args.metrics_json is not None:
        payload = {
            "metrics": metrics,
            "history_stack": int(history_stack),
            "seed": int(args.seed),
            "randomized": bool(args.randomize),
            "curriculum": bool(args.curriculum),
            "driver_population": bool(args.driver_population),
            "curriculum_progress": args.curriculum_progress,
        }
        out_path = Path(args.metrics_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved metrics JSON to {out_path}")


if __name__ == "__main__":
    main()
