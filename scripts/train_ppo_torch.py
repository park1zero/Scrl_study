#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from shared_control_rl.config import EnvConfig
from shared_control_rl.envs import make_env
from shared_control_rl.metrics import aggregate_episode_metrics, summarize_history
from shared_control_rl.torch_policy import ActorCritic, checkpoint_metadata, load_checkpoint, save_checkpoint


@dataclass
class RolloutBatch:
    obs: List[np.ndarray]
    actions: List[np.ndarray]
    log_probs: List[float]
    rewards: List[float]
    values: List[float]
    dones: List[float]
    next_value: float
    episode_returns: List[float]
    episode_metrics: List[Dict[str, float]]


def set_seed(seed: int, torch_threads: int = 1) -> None:
    np.random.seed(seed)
    torch.set_num_threads(max(1, int(torch_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(seed)


@torch.no_grad()
def collect_rollouts(env, model: ActorCritic, batch_episodes: int, device: torch.device, curriculum: bool = False, progress: float = 0.0) -> RolloutBatch:
    obs_buf: List[np.ndarray] = []
    act_buf: List[np.ndarray] = []
    logp_buf: List[float] = []
    rew_buf: List[float] = []
    val_buf: List[float] = []
    done_buf: List[float] = []
    episode_returns: List[float] = []
    episode_metrics: List[Dict[str, float]] = []

    next_value = 0.0

    for ep in range(batch_episodes):
        if curriculum:
            env.set_curriculum_progress(progress)
        obs, info = env.reset()
        done = False
        truncated = False
        ep_return = 0.0

        while not (done or truncated):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action_t, logp_t, _, value_t = model.step(obs_t)
            action = action_t.squeeze(0).cpu().numpy().astype(np.float32)

            next_obs, reward, done, truncated, info = env.step(action)

            obs_buf.append(obs.astype(np.float32))
            act_buf.append(action.astype(np.float32))
            logp_buf.append(float(logp_t.item()))
            rew_buf.append(float(reward))
            val_buf.append(float(value_t.item()))
            done_buf.append(0.0 if (done or truncated) else 1.0)

            ep_return += reward
            obs = next_obs

        episode_returns.append(float(ep_return))
        episode_metrics.append(summarize_history(env.history, info))

        if ep == batch_episodes - 1 and not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            next_value = float(model.critic(obs_t).item())
        else:
            next_value = 0.0

    return RolloutBatch(
        obs=obs_buf,
        actions=act_buf,
        log_probs=logp_buf,
        rewards=rew_buf,
        values=val_buf,
        dones=done_buf,
        next_value=next_value,
        episode_returns=episode_returns,
        episode_metrics=episode_metrics,
    )


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_adv = 0.0
    next_val = next_value
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_val * dones[t] - values[t]
        last_adv = delta + gamma * gae_lambda * dones[t] * last_adv
        advantages[t] = last_adv
        next_val = values[t]
    returns = advantages + values
    return advantages, returns


def make_minibatches(n: int, batch_size: int) -> List[np.ndarray]:
    idx = np.arange(n)
    np.random.shuffle(idx)
    return [idx[i : i + batch_size] for i in range(0, n, batch_size)]


def _selection_score(return_mean: float, aggregate: Dict[str, float]) -> tuple[float, float, float, float, float]:
    success_rate = aggregate.get("success_mean", 0.0)
    collision_rate = aggregate.get("collision_mean", 0.0)
    min_h_mean = aggregate.get("min_h_mean", -1e9)
    takeover_mean = aggregate.get("mean_takeover_mean", 1e9)
    return (float(success_rate), float(-collision_rate), float(min_h_mean), float(-takeover_mean), float(return_mean))


def train(args: argparse.Namespace) -> tuple[ActorCritic, Dict[str, List[float]], Dict[str, float], int, int, int, int]:
    set_seed(args.seed, torch_threads=args.torch_threads)
    device = torch.device(args.device)

    config = EnvConfig(seed=args.seed)
    config.scenario.domain_randomization = args.randomize or args.curriculum or args.driver_population
    config.scenario.side_randomization = args.randomize or args.curriculum or args.driver_population
    config.curriculum.enabled = args.curriculum
    config.driver_population.enabled = args.driver_population
    env = make_env(config, history_stack=args.history_stack)

    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])

    if args.init_model is not None:
        model = load_checkpoint(args.init_model, device=device)
        init_meta = checkpoint_metadata(args.init_model, device=device)
        init_stack = int(init_meta.get("history_stack", 1)) if init_meta else 1
        if init_stack != args.history_stack:
            raise ValueError(
                f"--init-model expects history_stack={init_stack}, but got --history-stack={args.history_stack}."
            )
        if model.actor.mean_net.net[0].in_features != obs_dim or model.actor.mean_net.net[-1].out_features != action_dim:
            raise ValueError("--init-model checkpoint shape does not match the environment.")
        hidden_dim = model.actor.mean_net.net[0].out_features
    else:
        model = ActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=args.hidden_dim).to(device)
        hidden_dim = args.hidden_dim
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: Dict[str, List[float]] = {
        "update": [],
        "return_mean": [],
        "return_std": [],
        "success_rate": [],
        "collision_rate": [],
        "min_h_mean": [],
        "takeover_mean": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
    }

    total_steps = 0
    update = 0
    last_aggregate: Dict[str, float] = {}
    best_score: tuple[float, float, float, float, float] | None = None

    while total_steps < args.total_steps:
        progress = float(total_steps / max(args.total_steps, 1)) if args.curriculum else 0.0
        batch = collect_rollouts(env, model, batch_episodes=args.batch_episodes, device=device, curriculum=args.curriculum, progress=progress)
        total_steps += len(batch.rewards)

        obs = torch.as_tensor(np.asarray(batch.obs), dtype=torch.float32, device=device)
        actions = torch.as_tensor(np.asarray(batch.actions), dtype=torch.float32, device=device)
        old_log_probs = torch.as_tensor(np.asarray(batch.log_probs), dtype=torch.float32, device=device).unsqueeze(-1)
        rewards = np.asarray(batch.rewards, dtype=np.float32)
        values = np.asarray(batch.values, dtype=np.float32)
        dones = np.asarray(batch.dones, dtype=np.float32)

        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            next_value=batch.next_value,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-6)

        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=device).unsqueeze(-1)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=device).unsqueeze(-1)

        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0

        for _ in range(args.update_epochs):
            for mb_idx in make_minibatches(len(obs), args.minibatch_size):
                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logp = old_log_probs[mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_ret = ret_t[mb_idx]

                new_logp, entropy, value = model.evaluate_actions(mb_obs, mb_actions)
                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * ((value - mb_ret) ** 2).mean()
                entropy_bonus = entropy.mean()

                loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy_bonus

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

                last_policy_loss = float(policy_loss.item())
                last_value_loss = float(value_loss.item())
                last_entropy = float(entropy_bonus.item())

        aggregate = aggregate_episode_metrics(batch.episode_metrics)
        last_aggregate = aggregate
        success_rate = aggregate.get("success_mean", 0.0)
        collision_rate = aggregate.get("collision_mean", 0.0)
        return_mean = float(np.mean(batch.episode_returns))
        return_std = float(np.std(batch.episode_returns))
        min_h_mean = aggregate.get("min_h_mean", np.nan)
        takeover_mean = aggregate.get("mean_takeover_mean", np.nan)

        history["update"].append(update)
        history["return_mean"].append(return_mean)
        history["return_std"].append(return_std)
        history["success_rate"].append(success_rate)
        history["collision_rate"].append(collision_rate)
        history["min_h_mean"].append(min_h_mean)
        history["takeover_mean"].append(takeover_mean)
        history["policy_loss"].append(last_policy_loss)
        history["value_loss"].append(last_value_loss)
        history["entropy"].append(last_entropy)

        if update % args.log_every == 0:
            print(
                f"[update {update:04d}] steps={total_steps:6d} "
                f"ret={return_mean:8.2f}±{return_std:6.2f} "
                f"success={success_rate:5.2f} collision={collision_rate:5.2f} "
                f"min_h={min_h_mean:6.3f} takeover={takeover_mean:5.3f}"
            )

        if args.best_out is not None:
            current_score = _selection_score(return_mean, aggregate)
            if best_score is None or current_score > best_score:
                best_score = current_score
                best_path = Path(args.best_out)
                best_path.parent.mkdir(parents=True, exist_ok=True)
                metadata = {
                    "algorithm": "ppo_torch",
                    "history_stack": int(args.history_stack),
                    "randomized_training": bool(args.randomize),
                    "curriculum": bool(args.curriculum),
                    "driver_population": bool(args.driver_population),
                    "seed": int(args.seed),
                    "best_update": int(update),
                    "selection_score": list(current_score),
                }
                save_checkpoint(
                    model,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    hidden_dim=hidden_dim,
                    path=best_path,
                    metadata=metadata,
                )
                if args.best_metrics_json is not None:
                    best_metrics_path = Path(args.best_metrics_json)
                    best_metrics_path.parent.mkdir(parents=True, exist_ok=True)
                    with best_metrics_path.open("w", encoding="utf-8") as f:
                        json.dump({"aggregate": aggregate, "score": current_score, "update": update, "metadata": metadata}, f, indent=2)
                print(f"  saved best checkpoint to {best_path} at update {update}")

        update += 1

    return model, history, last_aggregate, obs_dim, action_dim, hidden_dim, args.history_stack


def plot_training(history: Dict[str, List[float]], out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    axes[0].plot(history["update"], history["return_mean"], label="mean episode return")
    axes[0].fill_between(
        history["update"],
        np.asarray(history["return_mean"]) - np.asarray(history["return_std"]),
        np.asarray(history["return_mean"]) + np.asarray(history["return_std"]),
        alpha=0.25,
    )
    axes[0].set_xlabel("update")
    axes[0].set_ylabel("return")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(history["update"], history["success_rate"], label="success rate")
    axes[1].plot(history["update"], history["collision_rate"], label="collision rate")
    axes[1].set_xlabel("update")
    axes[1].set_ylabel("rate")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(history["update"], history["min_h_mean"], label="mean min barrier")
    axes[2].plot(history["update"], history["takeover_mean"], label="mean takeover")
    axes[2].set_xlabel("update")
    axes[2].set_ylabel("metric")
    axes[2].grid(True)
    axes[2].legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved training curve to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a minimal PPO agent in PyTorch for authority allocation.")
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--batch-episodes", type=int, default=8)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--driver-population", action="store_true")
    parser.add_argument("--history-stack", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--init-model", type=str, default=None, help="Optional warm-start checkpoint to initialize PPO from.")
    parser.add_argument("--out", type=str, default="artifacts/ppo_authority.pt")
    parser.add_argument("--best-out", type=str, default=None)
    parser.add_argument("--curve", type=str, default="artifacts/ppo_training_curve.png")
    parser.add_argument("--metrics-json", type=str, default=None)
    parser.add_argument("--best-metrics-json", type=str, default=None)
    args = parser.parse_args()

    model, history, aggregate, obs_dim, action_dim, hidden_dim, history_stack = train(args)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "algorithm": "ppo_torch",
        "history_stack": int(history_stack),
        "randomized_training": bool(args.randomize),
        "curriculum": bool(args.curriculum),
        "driver_population": bool(args.driver_population),
        "seed": int(args.seed),
    }
    save_checkpoint(
        model,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        path=out_path,
        metadata=metadata,
    )
    print(f"Saved checkpoint to {out_path}")

    plot_training(history, Path(args.curve))

    if args.metrics_json is not None:
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump({"aggregate": aggregate, "training": history, "metadata": metadata}, f, indent=2)
        print(f"Saved metrics JSON to {metrics_path}")

    print("Final aggregated rollout metrics:")
    for key in sorted(aggregate):
        print(f"  {key}: {aggregate[key]:.4f}")


if __name__ == "__main__":
    main()
