#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

from shared_control_rl.config import EnvConfig
from shared_control_rl.controllers.baselines import heuristic_hazard_policy
from shared_control_rl.envs import make_env
from shared_control_rl.torch_policy import ActorCritic, checkpoint_metadata, load_checkpoint, save_checkpoint


def action_from_lambda_target(lambda_target: float, current_lambda: float, max_lambda_rate: float, dt: float) -> float:
    desired_rate = (lambda_target - current_lambda) / max(dt, 1e-6)
    normalized = desired_rate / max(max_lambda_rate, 1e-6)
    return float(np.clip(normalized, -1.0, 1.0))


def collect_dataset(
    episodes: int,
    seed: int,
    randomize: bool,
    history_stack: int,
    extra_seeds: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    config = EnvConfig(seed=seed)
    config.scenario.domain_randomization = randomize
    config.scenario.side_randomization = randomize
    env = make_env(config, history_stack=history_stack)

    seeds = [seed + i for i in range(episodes)]
    if extra_seeds:
        seeds.extend(extra_seeds)

    obs_rows: list[np.ndarray] = []
    act_rows: list[list[float]] = []

    for ep_seed in seeds:
        obs, info = env.reset(seed=ep_seed)
        terminated = truncated = False
        while not (terminated or truncated):
            lambda_target = heuristic_hazard_policy(info)
            action = action_from_lambda_target(
                lambda_target=lambda_target,
                current_lambda=env.state.lam,
                max_lambda_rate=env.base_config.max_lambda_rate,
                dt=env.vehicle_params.dt,
            )
            obs_rows.append(obs.copy())
            act_rows.append([action])
            obs, reward, terminated, truncated, info = env.step(np.array([action], dtype=np.float32))

    return np.asarray(obs_rows, dtype=np.float32), np.asarray(act_rows, dtype=np.float32), seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="Behavior-clone the heuristic authority allocator as a warm start for PPO.")
    parser.add_argument("--episodes", type=int, default=14)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--history-stack", type=int, default=1)
    parser.add_argument("--extra-seeds", type=str, default="", help="Comma-separated extra rollout seeds to append to the dataset.")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--dataset-out", type=str, default="artifacts/bc_dataset.npz")
    parser.add_argument("--out", type=str, default="artifacts/bc_warmstart_actorcritic.pt")
    parser.add_argument("--curve", type=str, default="artifacts/bc_warmstart_curve.png")
    parser.add_argument("--meta", type=str, default="artifacts/bc_warmstart_meta.json")
    parser.add_argument("--init-model", type=str, default=None, help="Optional checkpoint to continue training from.")
    args = parser.parse_args()

    torch.set_num_threads(max(1, int(args.torch_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    extra_seeds = [int(x) for x in args.extra_seeds.split(",") if x.strip()] if args.extra_seeds else []
    obs_arr, act_arr, used_seeds = collect_dataset(
        episodes=args.episodes,
        seed=args.seed,
        randomize=args.randomize,
        history_stack=args.history_stack,
        extra_seeds=extra_seeds,
    )

    dataset_out = Path(args.dataset_out)
    dataset_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dataset_out, obs=obs_arr, act=act_arr)

    X = torch.as_tensor(obs_arr, dtype=torch.float32)
    Y = torch.as_tensor(act_arr, dtype=torch.float32)

    if args.init_model is not None:
        model = load_checkpoint(args.init_model, device="cpu")
        meta = checkpoint_metadata(args.init_model, device="cpu")
        init_stack = int(meta.get("history_stack", 1)) if meta else 1
        if init_stack != args.history_stack:
            raise ValueError(
                f"--init-model expects history_stack={init_stack}, but got --history-stack={args.history_stack}."
            )
        hidden_dim = model.actor.mean_net.net[0].out_features
    else:
        model = ActorCritic(obs_dim=X.shape[1], action_dim=Y.shape[1], hidden_dim=args.hidden_dim)
        hidden_dim = args.hidden_dim

    optimizer = torch.optim.Adam(model.actor.parameters(), lr=args.lr)
    losses: List[float] = []

    for epoch in range(args.epochs):
        idx = torch.randperm(len(X))
        for i in range(0, len(X), args.batch_size):
            mb = idx[i : i + args.batch_size]
            pred = model.actor.deterministic(X[mb])
            loss = ((pred - Y[mb]) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            mse = float(((model.actor.deterministic(X) - Y) ** 2).mean().item())
        losses.append(mse)
        print(f"[epoch {epoch + 1:02d}] mse={mse:.6f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "algorithm": "behavior_cloning",
        "history_stack": int(args.history_stack),
        "randomized_dataset": bool(args.randomize),
        "seed": int(args.seed),
        "samples": int(len(obs_arr)),
    }
    save_checkpoint(
        model,
        obs_dim=X.shape[1],
        action_dim=Y.shape[1],
        hidden_dim=hidden_dim,
        path=out_path,
        metadata=metadata,
    )

    curve_path = Path(args.curve)
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.2))
    plt.plot(np.arange(1, len(losses) + 1), losses)
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(curve_path, dpi=150)
    plt.close()

    meta = {
        "episodes": args.episodes,
        "seed": args.seed,
        "randomize": args.randomize,
        "history_stack": int(args.history_stack),
        "extra_seeds": extra_seeds,
        "used_seeds": used_seeds,
        "samples": int(len(obs_arr)),
        "hidden_dim": int(hidden_dim),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "final_mse": float(losses[-1] if losses else np.nan),
    }
    meta_path = Path(args.meta)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved dataset to {dataset_out}")
    print(f"Saved warm-start checkpoint to {out_path}")
    print(f"Saved training curve to {curve_path}")
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
