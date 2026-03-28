#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from shared_control_rl.config import EnvConfig
from shared_control_rl.controllers.baselines import heuristic_hazard_policy
from shared_control_rl.envs import make_env
from shared_control_rl.metrics import aggregate_episode_metrics, summarize_history
from shared_control_rl.torch_policy import checkpoint_metadata as ppo_checkpoint_metadata, load_checkpoint
from shared_control_rl.torch_sac import checkpoint_metadata as sac_checkpoint_metadata, load_actor


def action_from_lambda_target(lambda_target: float, current_lambda: float, max_lambda_rate: float, dt: float) -> np.ndarray:
    desired_rate = (lambda_target - current_lambda) / max(dt, 1e-6)
    normalized = desired_rate / max(max_lambda_rate, 1e-6)
    return np.array([np.clip(normalized, -1.0, 1.0)], dtype=np.float32)


def _resolve_history_stack(policy: str, model_path: str | None, requested_stack: int | None) -> int:
    if requested_stack is not None and requested_stack > 0:
        return int(requested_stack)
    if model_path is None:
        return 1
    if policy == "torch":
        meta = ppo_checkpoint_metadata(model_path)
        return int(meta.get("history_stack", 1)) if meta else 1
    if policy == "sac":
        meta = sac_checkpoint_metadata(model_path)
        return int(meta.get("history_stack", 1)) if meta else 1
    return 1


def run_episode(env, policy: str, model_path: str | None = None, seed: int | None = None) -> Dict[str, float]:
    obs, info = env.reset(seed=seed)
    ppo_model = None
    sac_actor = None
    if policy == "torch":
        if model_path is None:
            raise ValueError("--model is required when --policy torch")
        ppo_model = load_checkpoint(model_path)
    elif policy == "sac":
        if model_path is None:
            raise ValueError("--model is required when --policy sac")
        sac_actor = load_actor(model_path)

    terminated = truncated = False
    while not (terminated or truncated):
        if policy == "driver":
            action = action_from_lambda_target(1.0, env.state.lam, env.base_config.max_lambda_rate, env.vehicle_params.dt)
        elif policy == "automation":
            action = action_from_lambda_target(0.0, env.state.lam, env.base_config.max_lambda_rate, env.vehicle_params.dt)
        elif policy == "shared":
            action = action_from_lambda_target(0.5, env.state.lam, env.base_config.max_lambda_rate, env.vehicle_params.dt)
        elif policy == "heuristic":
            action = action_from_lambda_target(heuristic_hazard_policy(info), env.state.lam, env.base_config.max_lambda_rate, env.vehicle_params.dt)
        elif policy == "torch":
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = ppo_model.act(obs_t, deterministic=True).squeeze(0).cpu().numpy().astype(np.float32)
        elif policy == "sac":
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = sac_actor.act(obs_t, deterministic=True).squeeze(0).cpu().numpy().astype(np.float32)
        else:
            raise ValueError(f"Unknown policy: {policy}")

        obs, reward, terminated, truncated, info = env.step(action)

    return summarize_history(env.history, info)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep baseline or torch policy over many randomized episodes.")
    parser.add_argument("--policy", type=str, default="heuristic", choices=["driver", "automation", "shared", "heuristic", "torch", "sac"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--history-stack", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--driver-population", action="store_true")
    parser.add_argument("--out", type=str, default="artifacts/policy_sweep.csv")
    args = parser.parse_args()

    history_stack = _resolve_history_stack(args.policy, args.model, args.history_stack)

    config = EnvConfig(seed=args.seed)
    config.scenario.domain_randomization = args.randomize or args.curriculum or args.driver_population
    config.scenario.side_randomization = args.randomize or args.curriculum or args.driver_population
    config.curriculum.enabled = args.curriculum
    config.driver_population.enabled = args.driver_population

    rows: List[Dict[str, Any]] = []
    for ep in range(args.episodes):
        env = make_env(config, history_stack=history_stack)
        if args.curriculum:
            env.set_curriculum_progress(ep / max(args.episodes - 1, 1))
        metrics = run_episode(env, policy=args.policy, model_path=args.model, seed=args.seed + ep)
        metrics["episode"] = float(ep)
        rows.append(metrics)

    aggregate = aggregate_episode_metrics(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(rows[0].keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved episode metrics to {out_path}")
    print(f"History stack: {history_stack}")

    print("Aggregate summary:")
    for key in sorted(aggregate):
        print(f"  {key}: {aggregate[key]:.4f}")


if __name__ == "__main__":
    main()
