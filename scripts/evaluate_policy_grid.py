#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from shared_control_rl.config import EnvConfig
from shared_control_rl.envs import make_env
from shared_control_rl.controllers.baselines import fixed_lambda_policy, heuristic_hazard_policy
from shared_control_rl.metrics import summarize_history
from shared_control_rl.torch_policy import checkpoint_metadata as ppo_checkpoint_metadata
from shared_control_rl.torch_sac import checkpoint_metadata as sac_checkpoint_metadata, load_actor as load_sac_actor
from scripts.evaluate_torch_policy import rollout as rollout_torch


DIFFICULTIES = ["easy", "medium", "hard"]
DRIVER_PROFILES = ["late_linear", "late_aggressive", "frozen", "wrong_initial"]


def action_from_lambda_target(lambda_target: float, current_lambda: float, max_lambda_rate: float, dt: float) -> np.ndarray:
    desired_rate = (lambda_target - current_lambda) / max(dt, 1e-6)
    normalized = desired_rate / max(max_lambda_rate, 1e-6)
    return np.array([np.clip(normalized, -1.0, 1.0)], dtype=np.float32)


def rollout_baseline(env, policy_name: str, seed: int, reset_options: Dict[str, object]):
    obs, info = env.reset(seed=seed, options=reset_options)
    if policy_name == "driver":
        policy = fixed_lambda_policy(1.0)
        predict = lambda: action_from_lambda_target(policy(info), env.state.lam, env.base_config.max_lambda_rate, env.vehicle_params.dt)
    elif policy_name == "automation":
        policy = fixed_lambda_policy(0.0)
        predict = lambda: action_from_lambda_target(policy(info), env.state.lam, env.base_config.max_lambda_rate, env.vehicle_params.dt)
    elif policy_name == "shared":
        policy = fixed_lambda_policy(0.5)
        predict = lambda: action_from_lambda_target(policy(info), env.state.lam, env.base_config.max_lambda_rate, env.vehicle_params.dt)
    elif policy_name == "heuristic":
        predict = lambda: action_from_lambda_target(heuristic_hazard_policy(info), env.state.lam, env.base_config.max_lambda_rate, env.vehicle_params.dt)
    else:
        raise ValueError(policy_name)
    terminated = truncated = False
    while not (terminated or truncated):
        action = predict()
        obs, reward, terminated, truncated, info = env.step(action)
    return env.history, info


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


@torch.no_grad()
def rollout_sac(env, actor, seed: int):
    obs, info = env.reset(seed=seed, options=env._forced_options)  # type: ignore[attr-defined]
    terminated = truncated = False
    while not (terminated or truncated):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        action = actor.act(obs_t, deterministic=True).cpu().numpy().reshape(-1).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
    return env.history, info


def make_forced_env(seed: int, history_stack: int) -> object:
    config = EnvConfig(seed=seed)
    config.scenario.domain_randomization = False
    config.scenario.side_randomization = True
    config.curriculum.enabled = True
    config.driver_population.enabled = True
    env = make_env(config, history_stack=history_stack)
    return env


def build_heatmap(df: pd.DataFrame, value_col: str, out_path: Path, title: str) -> None:
    pivot = df.pivot(index="driver_profile", columns="difficulty", values=value_col).reindex(index=DRIVER_PROFILES, columns=DIFFICULTIES)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    im = ax.imshow(pivot.values, vmin=np.nanmin(pivot.values), vmax=np.nanmax(pivot.values))
    ax.set_xticks(range(len(DIFFICULTIES)), DIFFICULTIES)
    ax.set_yticks(range(len(DRIVER_PROFILES)), DRIVER_PROFILES)
    ax.set_title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            text = "nan" if np.isnan(val) else f"{val:.2f}"
            ax.text(j, i, text, ha="center", va="center", color="white" if not np.isnan(val) and val < np.nanmean(pivot.values) else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a policy over difficulty x driver-profile grid.")
    parser.add_argument("--policy", type=str, choices=["driver", "automation", "shared", "heuristic", "torch", "sac"], required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--episodes-per-cell", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history-stack", type=int, default=None)
    parser.add_argument("--csv-out", type=str, default="artifacts/grid_eval.csv")
    parser.add_argument("--summary-out", type=str, default="artifacts/grid_eval_summary.csv")
    parser.add_argument("--heatmap-prefix", type=str, default="artifacts/grid_eval")
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    history_stack = _resolve_history_stack(args.policy, args.model, args.history_stack)
    env = make_forced_env(seed=args.seed, history_stack=history_stack)
    sac_actor = load_sac_actor(args.model) if args.policy == "sac" else None

    rows: List[Dict[str, float | str | int]] = []
    for difficulty in DIFFICULTIES:
        for profile in DRIVER_PROFILES:
            for rep in range(args.episodes_per_cell):
                seed = int(args.seed + 100 * DIFFICULTIES.index(difficulty) + 10 * DRIVER_PROFILES.index(profile) + rep)
                forced_options = {
                    "difficulty": difficulty,
                    "driver_profile": profile,
                    "curriculum_progress": 1.0,
                }
                env._forced_options = forced_options  # type: ignore[attr-defined]
                if args.policy == "sac":
                    history, info = rollout_sac(env, sac_actor, seed=seed)
                elif args.policy == "torch":
                    history, info = rollout_torch(env, args.model, deterministic=True, seed=seed, reset_options=forced_options)
                else:
                    history, info = rollout_baseline(env, policy_name=args.policy, seed=seed, reset_options=forced_options)
                metrics = summarize_history(history, info)
                row: Dict[str, float | str | int] = {
                    "policy": args.policy,
                    "difficulty": difficulty,
                    "driver_profile": profile,
                    "rep": rep,
                    "seed": seed,
                }
                row.update(metrics)
                rows.append(row)

    episodes_df = pd.DataFrame(rows)
    summary_df = (
        episodes_df.groupby(["policy", "difficulty", "driver_profile"], as_index=False)
        .agg({
            "success": "mean",
            "collision": "mean",
            "min_h": "mean",
            "mean_takeover": "mean",
            "return": "mean",
        })
        .rename(columns={
            "success": "success_mean",
            "collision": "collision_mean",
            "min_h": "min_h_mean",
            "mean_takeover": "mean_takeover_mean",
            "return": "return_mean",
        })
    )

    csv_out = Path(args.csv_out)
    summary_out = Path(args.summary_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    episodes_df.to_csv(csv_out, index=False)
    summary_df.to_csv(summary_out, index=False)

    build_heatmap(summary_df, value_col="success_mean", out_path=Path(f"{args.heatmap_prefix}_success.png"), title=f"{args.policy} success")
    build_heatmap(summary_df, value_col="collision_mean", out_path=Path(f"{args.heatmap_prefix}_collision.png"), title=f"{args.policy} collision")
    build_heatmap(summary_df, value_col="min_h_mean", out_path=Path(f"{args.heatmap_prefix}_min_h.png"), title=f"{args.policy} min h")

    overall = {
        "policy": args.policy,
        "history_stack": int(history_stack),
        "episodes": int(len(episodes_df)),
        "success_mean": float(episodes_df["success"].mean()),
        "collision_mean": float(episodes_df["collision"].mean()),
        "min_h_mean": float(episodes_df["min_h"].mean()),
        "mean_takeover_mean": float(episodes_df["mean_takeover"].mean()),
        "return_mean": float(episodes_df["return"].mean()),
    }
    print(json.dumps(overall, indent=2))

    if args.json_out is not None:
        out_json = Path(args.json_out)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump({"overall": overall}, f, indent=2)


if __name__ == "__main__":
    main()
