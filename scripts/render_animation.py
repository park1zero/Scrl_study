#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json

from shared_control_rl.config import EnvConfig
from shared_control_rl.envs import make_env
from shared_control_rl.metrics import summarize_history
from shared_control_rl.torch_policy import checkpoint_metadata as ppo_checkpoint_metadata
from shared_control_rl.torch_sac import checkpoint_metadata as sac_checkpoint_metadata
from shared_control_rl.visualization import animate_history
from scripts.evaluate_policy import run_policy
from scripts.evaluate_torch_policy import rollout as rollout_torch
from scripts.evaluate_sac_torch import rollout as rollout_sac


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a policy rollout as GIF or MP4.")
    parser.add_argument(
        "--policy",
        type=str,
        default="heuristic",
        choices=["driver", "automation", "shared", "heuristic", "model", "torch", "sac"],
    )
    parser.add_argument("--model", type=str, default=None, help="Path to SB3 SAC model or torch PPO checkpoint.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--driver-population", action="store_true")
    parser.add_argument("--curriculum-progress", type=float, default=None)
    parser.add_argument("--difficulty", type=str, default=None, choices=["easy", "medium", "hard"])
    parser.add_argument("--driver-profile", type=str, default=None, choices=["late_linear", "late_aggressive", "frozen", "wrong_initial"])
    parser.add_argument("--history-stack", type=int, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--out", type=str, default="artifacts/animations/heuristic.gif")
    parser.add_argument("--summary", type=str, default=None, help="Optional JSON summary path.")
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    history_stack = _resolve_history_stack(args.policy, args.model, args.history_stack)

    config = EnvConfig(seed=args.seed)
    config.scenario.domain_randomization = args.randomize or args.curriculum or args.driver_population
    config.scenario.side_randomization = args.randomize or args.curriculum or args.driver_population
    config.curriculum.enabled = args.curriculum
    config.driver_population.enabled = args.driver_population

    env = make_env(config, history_stack=history_stack)
    if args.curriculum_progress is not None:
        env.set_curriculum_progress(args.curriculum_progress)

    reset_options = None
    if args.difficulty is not None or args.driver_profile is not None:
        env.base_config.curriculum.enabled = True
        env.base_config.driver_population.enabled = True
        reset_options = {"curriculum_progress": args.curriculum_progress if args.curriculum_progress is not None else 1.0}
        if args.difficulty is not None:
            reset_options["difficulty"] = args.difficulty
        if args.driver_profile is not None:
            reset_options["driver_profile"] = args.driver_profile

    if args.policy == "torch":
        if args.model is None:
            raise ValueError("--model is required when --policy torch")
        history, info = rollout_torch(env, args.model, deterministic=True, seed=args.seed, reset_options=reset_options)
        title = args.title or f"Torch PPO authority policy (stack={history_stack})"
    elif args.policy == "sac":
        if args.model is None:
            raise ValueError("--model is required when --policy sac")
        history, info = rollout_sac(env, args.model, deterministic=True, seed=args.seed, reset_options=reset_options)
        title = args.title or f"Torch SAC authority policy (stack={history_stack})"
    else:
        history = run_policy(env, policy_name=args.policy, model_path=args.model, seed=args.seed, reset_options=reset_options)
        info = {
            "collision": bool(history["h"][-1] < 0.0),
            "success": bool(history["x"][-1] > env.obstacle.x + env.scenario_cfg.success_x_margin and history["h"][-1] >= 0.0),
        }
        title = args.title or f"Policy: {args.policy}"

    out_path = animate_history(
        history=history,
        obstacle=env.obstacle,
        scenario_cfg=env.scenario_cfg,
        vehicle_params=env.vehicle_params,
        out_path=args.out,
        title=title,
        fps=args.fps,
        stride=args.stride,
        dpi=args.dpi,
    )

    summary = summarize_history(history, info)
    print(f"Saved animation to {out_path}")
    print("Episode summary:")
    for key in sorted(summary):
        print(f"  {key}: {summary[key]:.4f}")

    if args.summary is not None:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": summary,
            "history_stack": int(history_stack),
            "policy": args.policy,
        }
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
