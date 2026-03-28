#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import copy
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from shared_control_rl.config import EnvConfig
from shared_control_rl.controllers.baselines import heuristic_hazard_policy
from shared_control_rl.envs import make_env
from shared_control_rl.metrics import aggregate_episode_metrics, summarize_history
from shared_control_rl.replay import StratifiedReplayBuffer, TransitionBatch
from shared_control_rl.torch_policy import load_checkpoint as load_ppo_checkpoint
from shared_control_rl.torch_sac import (
    DoubleQCritic,
    SACActor,
    load_checkpoint_data,
    save_checkpoint,
    soft_update,
)


@dataclass
class EvalResult:
    metrics: Dict[str, float]
    returns: List[float]


def set_seed(seed: int, torch_threads: int = 1) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, int(torch_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def heuristic_action_from_info(info: Dict[str, float], env) -> np.ndarray:
    lam_target = heuristic_hazard_policy(info)
    lam_now = float(info.get("lambda_safe", info.get("lambda", 1.0)))
    denom = max(env.base_config.max_lambda_rate * env.vehicle_params.dt, 1e-6)
    a = np.clip((lam_target - lam_now) / denom, -1.0, 1.0)
    return np.array([a], dtype=np.float32)


def driver_only_action() -> np.ndarray:
    return np.array([1.0], dtype=np.float32)


def hazard_severity_from_info(info: Dict[str, float], hazard_threshold: float) -> Tuple[bool, float]:
    h = float(info.get("h", np.inf))
    collision = bool(info.get("collision", False))
    ttc = float(info.get("ttc", np.inf))
    takeover = float(1.0 - info.get("lambda_safe", 1.0))
    hazard_margin = max(hazard_threshold - h, 0.0)
    time_urgency = max(2.0 - ttc, 0.0)
    severity = hazard_margin + 0.5 * time_urgency + 0.25 * takeover
    if collision:
        severity += 3.0
    return (collision or h <= hazard_threshold or ttc <= 1.5), float(severity)


@torch.no_grad()
def evaluate_actor(env, actor: SACActor, episodes: int, device: torch.device, curriculum: bool) -> EvalResult:
    episode_metrics: List[Dict[str, float]] = []
    returns: List[float] = []
    for ep in range(episodes):
        progress = float(ep / max(episodes - 1, 1)) if curriculum else 0.0
        env.set_curriculum_progress(progress)
        obs, info = env.reset(seed=1000 + ep)
        terminated = truncated = False
        ep_return = 0.0
        while not (terminated or truncated):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = actor.act(obs_t, deterministic=True).cpu().numpy().reshape(-1).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
        returns.append(float(ep_return))
        episode_metrics.append(summarize_history(env.history, info))
    aggregate = aggregate_episode_metrics(episode_metrics)
    aggregate["return_mean"] = float(np.mean(returns)) if returns else np.nan
    aggregate["return_std"] = float(np.std(returns)) if returns else np.nan
    return EvalResult(metrics=aggregate, returns=returns)


@torch.no_grad()
def seed_replay_with_policy(
    env,
    buffer: StratifiedReplayBuffer,
    *,
    episodes: int,
    curriculum: bool,
    policy_name: str,
    hazard_threshold: float,
) -> Dict[str, float]:
    metrics: List[Dict[str, float]] = []
    for ep in range(episodes):
        progress = float(ep / max(episodes - 1, 1)) if curriculum else 0.0
        env.set_curriculum_progress(progress)
        obs, info = env.reset(seed=2000 + ep if policy_name == "heuristic" else 3000 + ep)
        terminated = truncated = False
        while not (terminated or truncated):
            if policy_name == "heuristic":
                action = heuristic_action_from_info(info, env)
                demo = True
            elif policy_name == "driver":
                action = driver_only_action()
                demo = False
            else:
                raise ValueError(f"unknown seed policy: {policy_name}")
            next_obs, reward, terminated, truncated, info = env.step(action)
            hazard, severity = hazard_severity_from_info(info, hazard_threshold=hazard_threshold)
            buffer.add(
                obs,
                action,
                reward,
                next_obs,
                done=(terminated or truncated),
                demo=demo,
                hazard=hazard,
                hazard_severity=severity,
                difficulty=str(info.get("difficulty", "medium")),
            )
            obs = next_obs
        metrics.append(summarize_history(env.history, info))
    return aggregate_episode_metrics(metrics)


def maybe_init_actor_from_ppo(actor: SACActor, ppo_path: str, device: torch.device) -> bool:
    if actor.encoder_type != "mlp" or actor.history_stack != 1:
        return False
    model = load_ppo_checkpoint(ppo_path, device=device)
    src = model.actor.mean_net.net
    try:
        mapping = [
            (actor.backbone.net[0], src[0]),
            (actor.backbone.net[2], src[2]),
            (actor.mean, src[4]),
        ]
        for dst_layer, src_layer in mapping:
            if dst_layer.weight.shape != src_layer.weight.shape or dst_layer.bias.shape != src_layer.bias.shape:
                return False
            dst_layer.weight.data.copy_(src_layer.weight.data)
            dst_layer.bias.data.copy_(src_layer.bias.data)
        actor.log_std.weight.data.zero_()
        actor.log_std.bias.data.fill_(-0.7)
        return True
    except Exception:
        return False


def freeze_module(module: nn.Module) -> nn.Module:
    for param in module.parameters():
        param.requires_grad_(False)
    module.eval()
    return module


class SACTrainer:
    def __init__(self, args: argparse.Namespace) -> None:
        set_seed(args.seed, torch_threads=args.torch_threads)
        self.args = args
        self.device = torch.device(args.device)

        config = EnvConfig(seed=args.seed)
        config.scenario.domain_randomization = args.randomize or args.curriculum or args.driver_population
        config.scenario.side_randomization = args.randomize or args.curriculum or args.driver_population
        config.curriculum.enabled = args.curriculum
        config.driver_population.enabled = args.driver_population
        self.env = make_env(config, history_stack=args.history_stack)
        self.eval_env = make_env(config, history_stack=args.history_stack)

        self.obs_dim = int(self.env.observation_space.shape[0])
        self.action_dim = int(self.env.action_space.shape[0])
        self.base_obs_dim = int(getattr(self.env, "base_obs_dim", self.obs_dim // max(args.history_stack, 1)))
        self.actor = SACActor(
            self.obs_dim,
            self.action_dim,
            hidden_dim=args.hidden_dim,
            history_stack=args.history_stack,
            base_obs_dim=self.base_obs_dim,
            encoder_type=args.encoder_type,
        ).to(self.device)
        self.critic = DoubleQCritic(
            self.obs_dim,
            self.action_dim,
            hidden_dim=args.hidden_dim,
            history_stack=args.history_stack,
            base_obs_dim=self.base_obs_dim,
            encoder_type=args.encoder_type,
        ).to(self.device)
        self.target_critic = DoubleQCritic(
            self.obs_dim,
            self.action_dim,
            hidden_dim=args.hidden_dim,
            history_stack=args.history_stack,
            base_obs_dim=self.base_obs_dim,
            encoder_type=args.encoder_type,
        ).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        actor_lr = float(args.actor_lr if args.actor_lr is not None else args.lr)
        critic_lr = float(args.critic_lr if args.critic_lr is not None else args.lr)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.log_alpha = torch.tensor(np.log(args.init_alpha), dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr)
        self.target_entropy = -float(self.action_dim)
        self.anchor_actor: SACActor | None = None
        self.anchor_source: str | None = None

        if args.init_ppo is not None:
            ok = maybe_init_actor_from_ppo(self.actor, args.init_ppo, self.device)
            print(f"PPO actor init: {'ok' if ok else 'skipped'} from {args.init_ppo}")

        if args.init_actor_from_sac is not None:
            bundle = load_checkpoint_data(args.init_actor_from_sac, device=self.device)
            if bundle.obs_dim != self.obs_dim or bundle.action_dim != self.action_dim:
                raise ValueError(
                    f"init actor checkpoint dims mismatch: ckpt ({bundle.obs_dim}, {bundle.action_dim}) vs env ({self.obs_dim}, {self.action_dim})"
                )
            self.actor.load_state_dict(bundle.actor_state, strict=True)
            if args.init_critic_from_sac and bundle.critic_state is not None:
                self.critic.load_state_dict(bundle.critic_state, strict=True)
                if bundle.target_critic_state is not None:
                    self.target_critic.load_state_dict(bundle.target_critic_state, strict=True)
                else:
                    self.target_critic.load_state_dict(self.critic.state_dict())
            if args.init_alpha_from_sac and bundle.log_alpha is not None:
                with torch.no_grad():
                    self.log_alpha.copy_(torch.tensor(float(bundle.log_alpha), dtype=torch.float32, device=self.device))
            print(
                f"SAC actor init: loaded from {args.init_actor_from_sac} "
                f"(critic={'yes' if args.init_critic_from_sac and bundle.critic_state is not None else 'no'}, "
                f"alpha={'yes' if args.init_alpha_from_sac and bundle.log_alpha is not None else 'no'})"
            )

        self.replay = StratifiedReplayBuffer(self.obs_dim, self.action_dim, capacity=args.buffer_size)
        self.history: Dict[str, List[float]] = {
            "step": [],
            "episode_return": [],
            "critic_loss": [],
            "actor_loss": [],
            "alpha": [],
            "bc_loss": [],
            "bc_coef": [],
            "anchor_loss": [],
            "anchor_coef": [],
            "hazard_frac": [],
            "demo_frac": [],
            "medium_demo_frac": [],
            "hard_demo_frac": [],
            "medium_bc_weight": [],
            "hard_bc_weight": [],
            "eval_step": [],
            "eval_success": [],
            "eval_collision": [],
            "eval_min_h": [],
            "eval_takeover": [],
        }
        self.best_score: Tuple[float, float, float, float] | None = None

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def bc_coef_at_progress(self, progress: float) -> float:
        progress = float(np.clip(progress, 0.0, 1.0))
        return float((1.0 - progress) * self.args.bc_coef_start + progress * self.args.bc_coef_end)

    def guided_explore_prob_at_progress(self, progress: float) -> float:
        progress = float(np.clip(progress, 0.0, 1.0))
        return float((1.0 - progress) * self.args.guided_explore_prob_start + progress * self.args.guided_explore_prob_end)

    def anchor_coef_at_progress(self, progress: float) -> float:
        progress = float(np.clip(progress, 0.0, 1.0))
        return float((1.0 - progress) * self.args.anchor_coef_start + progress * self.args.anchor_coef_end)

    def set_anchor_from_current(self, *, source: str) -> None:
        self.anchor_actor = freeze_module(copy.deepcopy(self.actor).to(self.device))
        self.anchor_source = str(source)

    def _bc_masks(self, batch: TransitionBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        sev = batch.hazard_severity
        medium_severity = ((sev >= self.args.medium_severity_min) & (sev <= self.args.medium_severity_max)).float()
        medium_difficulty = ((batch.difficulty >= 0.5) & (batch.difficulty < 1.5)).float()
        medium_mask = torch.maximum(medium_severity, medium_difficulty)
        hard_mask = torch.maximum((batch.difficulty >= 1.5).float(), (sev > self.args.medium_severity_max).float())
        return medium_mask, hard_mask

    def pretrain_actor(self) -> Dict[str, float]:
        if self.args.pretrain_bc_steps <= 0 or self.replay.counts()['demo'] <= 0:
            return {'pretrain_bc_loss': 0.0}

        last_loss = 0.0
        for _ in range(int(self.args.pretrain_bc_steps)):
            batch = self.replay.sample(
                int(self.args.pretrain_batch_size),
                device=self.device,
                demo_frac=1.0,
                hazard_frac=0.0,
            )
            medium_mask, hard_mask = self._bc_masks(batch)
            _, _, deterministic_action = self.actor.sample(batch.obs)
            demo_mse = ((deterministic_action - batch.actions) ** 2).sum(dim=-1, keepdim=True)
            bc_weights = batch.demo * (1.0 + self.args.medium_bc_boost * medium_mask + (self.args.hard_bc_scale - 1.0) * hard_mask)
            bc_weights = torch.clamp(bc_weights, min=0.0)
            loss = (demo_mse * bc_weights).sum() / torch.clamp(bc_weights.sum(), min=1.0)
            self.actor_optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
            self.actor_optim.step()
            last_loss = float(loss.item())

        return {'pretrain_bc_loss': last_loss}

    def update(self, batch: TransitionBatch, *, train_progress: float, update_actor: bool = True) -> Dict[str, float]:
        args = self.args
        bc_coef = self.bc_coef_at_progress(train_progress)
        anchor_coef = self.anchor_coef_at_progress(train_progress)
        medium_mask, hard_mask = self._bc_masks(batch)

        with torch.no_grad():
            next_action, next_logp, _ = self.actor.sample(batch.next_obs)
            next_q1, next_q2 = self.target_critic(batch.next_obs, next_action)
            next_q = torch.min(next_q1, next_q2) - self.alpha.detach() * next_logp
            target_q = batch.rewards + args.gamma * (1.0 - batch.dones) * next_q

        q1, q2 = self.critic(batch.obs, batch.actions)
        td = 0.5 * ((q1 - target_q) ** 2 + (q2 - target_q) ** 2)
        severity_term = torch.clamp(batch.hazard_severity, 0.0, 3.0)
        critic_weights = 1.0 + args.hazard_td_weight * (batch.hazard + 0.25 * severity_term)
        critic_loss = (td * critic_weights).mean()

        self.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), args.max_grad_norm)
        self.critic_optim.step()

        new_action, logp, deterministic_action = self.actor.sample(batch.obs)
        q1_pi, q2_pi = self.critic(batch.obs, new_action)
        q_pi = torch.min(q1_pi, q2_pi)

        bc_mask = batch.demo
        bc_loss = torch.tensor(0.0, device=self.device)
        anchor_loss = torch.tensor(0.0, device=self.device)
        medium_bc_weight = torch.tensor(0.0, device=self.device)
        hard_bc_weight = torch.tensor(0.0, device=self.device)
        medium_demo_frac = torch.tensor(0.0, device=self.device)
        hard_demo_frac = torch.tensor(0.0, device=self.device)
        if bc_mask.sum().item() > 0 and bc_coef > 0.0:
            demo_mse = ((deterministic_action - batch.actions) ** 2).sum(dim=-1, keepdim=True)
            bc_weights = bc_mask * (1.0 + args.medium_bc_boost * medium_mask + (args.hard_bc_scale - 1.0) * hard_mask)
            bc_weights = torch.clamp(bc_weights, min=0.0)
            bc_loss = (demo_mse * bc_weights).sum() / torch.clamp(bc_weights.sum(), min=1.0)

            medium_demo_frac = ((bc_mask * medium_mask).sum() / torch.clamp(bc_mask.sum(), min=1.0)).detach()
            hard_demo_frac = ((bc_mask * hard_mask).sum() / torch.clamp(bc_mask.sum(), min=1.0)).detach()
            medium_bc_weight = ((bc_weights * medium_mask).sum() / torch.clamp((bc_mask * medium_mask).sum(), min=1.0)).detach()
            hard_bc_weight = ((bc_weights * hard_mask).sum() / torch.clamp((bc_mask * hard_mask).sum(), min=1.0)).detach()

        if self.anchor_actor is not None and anchor_coef > 0.0:
            with torch.no_grad():
                anchor_action = self.anchor_actor.act(batch.obs, deterministic=True)
            anchor_mse = ((deterministic_action - anchor_action) ** 2).sum(dim=-1, keepdim=True)
            anchor_weights = 1.0 + args.medium_anchor_boost * medium_mask + (args.hard_anchor_scale - 1.0) * hard_mask
            anchor_weights = torch.clamp(anchor_weights, min=0.0)
            anchor_loss = (anchor_mse * anchor_weights).mean()

        actor_loss = (self.alpha.detach() * logp - q_pi).mean() + bc_coef * bc_loss + anchor_coef * anchor_loss

        if update_actor:
            self.actor_optim.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), args.max_grad_norm)
            self.actor_optim.step()

            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optim.step()
        else:
            actor_loss = actor_loss.detach()

        soft_update(self.target_critic, self.critic, tau=args.tau)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
            "bc_loss": float(bc_loss.item()),
            "bc_coef": float(bc_coef),
            "anchor_loss": float(anchor_loss.item()),
            "anchor_coef": float(anchor_coef if self.anchor_actor is not None else 0.0),
            "medium_demo_frac": float(medium_demo_frac.item()),
            "hard_demo_frac": float(hard_demo_frac.item()),
            "medium_bc_weight": float(medium_bc_weight.item()),
            "hard_bc_weight": float(hard_bc_weight.item()),
            "actor_updated": 1.0 if update_actor else 0.0,
        }

    def maybe_save_best(self, step: int, eval_result: EvalResult) -> None:
        if self.args.best_out is None:
            return
        metrics = eval_result.metrics
        score = (
            float(metrics.get("success_mean", 0.0)),
            float(-metrics.get("collision_mean", 0.0)),
            float(-metrics.get("mean_takeover_mean", 1e9)),
            float(metrics.get("min_h_mean", -1e9)),
        )
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            path = Path(self.args.best_out)
            path.parent.mkdir(parents=True, exist_ok=True)
            metadata = {
                "algorithm": "sac_torch_v11_conservative_seq_bc",
                "history_stack": int(self.args.history_stack),
                "base_obs_dim": int(self.base_obs_dim),
                "encoder_type": str(self.args.encoder_type),
                "curriculum": bool(self.args.curriculum),
                "driver_population": bool(self.args.driver_population),
                "randomized_training": bool(self.args.randomize),
                "best_at_step": int(step),
                "eval_metrics": metrics,
                "replay_counts": self.replay.counts(),
                "bc_shaping": {
                    "medium_bc_boost": float(self.args.medium_bc_boost),
                    "hard_bc_scale": float(self.args.hard_bc_scale),
                    "medium_anchor_boost": float(self.args.medium_anchor_boost),
                    "hard_anchor_scale": float(self.args.hard_anchor_scale),
                    "medium_severity_min": float(self.args.medium_severity_min),
                    "medium_severity_max": float(self.args.medium_severity_max),
                    "pretrain_bc_steps": int(self.args.pretrain_bc_steps),
                    "anchor_source": self.anchor_source,
                },
            }
            save_checkpoint(
                actor=self.actor,
                critic=self.critic,
                target_critic=self.target_critic,
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden_dim=self.args.hidden_dim,
                path=path,
                log_alpha=float(self.log_alpha.item()),
                metadata=metadata,
            )

    def train(self) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
        args = self.args
        seed_stats: Dict[str, Dict[str, float]] = {}
        if args.seed_demo_episodes > 0:
            seed_stats["heuristic"] = seed_replay_with_policy(
                self.env,
                self.replay,
                episodes=args.seed_demo_episodes,
                curriculum=args.curriculum,
                policy_name="heuristic",
                hazard_threshold=args.hazard_threshold,
            )
            print(
                "[seed-heuristic] "
                f"success={seed_stats['heuristic'].get('success_mean', np.nan):.2f} "
                f"collision={seed_stats['heuristic'].get('collision_mean', np.nan):.2f} "
                f"min_h={seed_stats['heuristic'].get('min_h_mean', np.nan):.3f}"
            )
        if args.seed_driver_episodes > 0:
            seed_stats["driver"] = seed_replay_with_policy(
                self.env,
                self.replay,
                episodes=args.seed_driver_episodes,
                curriculum=args.curriculum,
                policy_name="driver",
                hazard_threshold=args.hazard_threshold,
            )
            print(
                "[seed-driver] "
                f"success={seed_stats['driver'].get('success_mean', np.nan):.2f} "
                f"collision={seed_stats['driver'].get('collision_mean', np.nan):.2f} "
                f"min_h={seed_stats['driver'].get('min_h_mean', np.nan):.3f}"
            )
        print(f"[replay-init] {self.replay.counts()}")
        pretrain_stats = self.pretrain_actor()
        if args.pretrain_bc_steps > 0:
            print(f"[pretrain-bc] steps={args.pretrain_bc_steps} loss={pretrain_stats.get('pretrain_bc_loss', float('nan')):.4f}")

        if args.anchor_after_pretrain:
            self.set_anchor_from_current(source="post_pretrain")
            print(f"[anchor] frozen anchor actor from {self.anchor_source}")

        obs, info = self.env.reset(seed=args.seed)
        episode_return = 0.0
        episode_metrics: List[Dict[str, float]] = []
        last_train_stats = {
            "critic_loss": np.nan,
            "actor_loss": np.nan,
            "alpha": float(self.alpha.item()),
            "bc_loss": np.nan,
            "bc_coef": np.nan,
            "anchor_loss": np.nan,
            "anchor_coef": np.nan,
            "medium_demo_frac": np.nan,
            "hard_demo_frac": np.nan,
            "medium_bc_weight": np.nan,
            "hard_bc_weight": np.nan,
            "actor_updated": 0.0,
        }

        for step in range(1, args.total_steps + 1):
            progress = float(step / max(args.total_steps, 1))
            if args.curriculum:
                self.env.set_curriculum_progress(progress)

            if step <= args.start_random_steps:
                action = np.random.uniform(-1.0, 1.0, size=(self.action_dim,)).astype(np.float32)
            else:
                guided_prob = self.guided_explore_prob_at_progress(progress)
                if np.random.random() < guided_prob:
                    action = heuristic_action_from_info(info, self.env)
                else:
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        action = self.actor.act(obs_t, deterministic=False).cpu().numpy().reshape(-1).astype(np.float32)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = bool(terminated or truncated)
            hazard, severity = hazard_severity_from_info(info, hazard_threshold=args.hazard_threshold)
            self.replay.add(
                obs,
                action,
                reward,
                next_obs,
                done=done,
                demo=False,
                hazard=hazard,
                hazard_severity=severity,
                difficulty=str(info.get("difficulty", "medium")),
            )
            episode_return += reward
            obs = next_obs

            if step >= args.learning_starts and len(self.replay) >= args.batch_size:
                for _ in range(args.updates_per_step):
                    batch = self.replay.sample(
                        args.batch_size,
                        device=self.device,
                        demo_frac=args.demo_sample_frac,
                        hazard_frac=args.hazard_sample_frac,
                    )
                    update_actor = step >= args.actor_learning_starts
                    last_train_stats = self.update(batch, train_progress=progress, update_actor=update_actor)

            if done:
                episode_metrics.append(summarize_history(self.env.history, info))
                self.history["step"].append(float(step))
                self.history["episode_return"].append(float(episode_return))
                self.history["critic_loss"].append(float(last_train_stats["critic_loss"]))
                self.history["actor_loss"].append(float(last_train_stats["actor_loss"]))
                self.history["alpha"].append(float(last_train_stats["alpha"]))
                self.history["bc_loss"].append(float(last_train_stats["bc_loss"]))
                self.history["bc_coef"].append(float(last_train_stats["bc_coef"]))
                self.history["anchor_loss"].append(float(last_train_stats["anchor_loss"]))
                self.history["anchor_coef"].append(float(last_train_stats["anchor_coef"]))
                self.history["medium_demo_frac"].append(float(last_train_stats["medium_demo_frac"]))
                self.history["hard_demo_frac"].append(float(last_train_stats["hard_demo_frac"]))
                self.history["medium_bc_weight"].append(float(last_train_stats["medium_bc_weight"]))
                self.history["hard_bc_weight"].append(float(last_train_stats["hard_bc_weight"]))
                counts = self.replay.counts()
                hazard_frac = counts["hazard"] / max(counts["size"], 1)
                demo_frac = counts["demo"] / max(counts["size"], 1)
                self.history["hazard_frac"].append(float(hazard_frac))
                self.history["demo_frac"].append(float(demo_frac))
                obs, info = self.env.reset()
                episode_return = 0.0

            if args.eval_every > 0 and step % args.eval_every == 0:
                eval_result = evaluate_actor(
                    self.eval_env,
                    self.actor,
                    episodes=args.eval_episodes,
                    device=self.device,
                    curriculum=args.curriculum,
                )
                metrics = eval_result.metrics
                self.history["eval_step"].append(float(step))
                self.history["eval_success"].append(float(metrics.get("success_mean", np.nan)))
                self.history["eval_collision"].append(float(metrics.get("collision_mean", np.nan)))
                self.history["eval_min_h"].append(float(metrics.get("min_h_mean", np.nan)))
                self.history["eval_takeover"].append(float(metrics.get("mean_takeover_mean", np.nan)))
                print(
                    f"[step {step:05d}] eval success={metrics.get('success_mean', np.nan):.2f} "
                    f"collision={metrics.get('collision_mean', np.nan):.2f} "
                    f"min_h={metrics.get('min_h_mean', np.nan):.3f} "
                    f"takeover={metrics.get('mean_takeover_mean', np.nan):.3f} "
                    f"alpha={self.alpha.item():.3f} replay={self.replay.counts()}"
                )
                self.maybe_save_best(step, eval_result)

        aggregate = aggregate_episode_metrics(episode_metrics)
        aggregate["replay_size"] = float(len(self.replay))
        aggregate["replay_demo_count"] = float(self.replay.counts()["demo"])
        aggregate["replay_hazard_count"] = float(self.replay.counts()["hazard"])
        for name, stats in seed_stats.items():
            aggregate[f"seed_{name}_success_mean"] = float(stats.get("success_mean", np.nan))
            aggregate[f"seed_{name}_collision_mean"] = float(stats.get("collision_mean", np.nan))
            aggregate[f"seed_{name}_min_h_mean"] = float(stats.get("min_h_mean", np.nan))
        return self.history, aggregate


def plot_training(history: Dict[str, List[float]], out_path: Path) -> None:
    if not history["step"]:
        return
    fig, axes = plt.subplots(5, 1, figsize=(10, 18))

    axes[0].plot(history["step"], history["episode_return"])
    axes[0].set_ylabel("episode return")
    axes[0].grid(True)

    axes[1].plot(history["step"], history["critic_loss"], label="critic")
    axes[1].plot(history["step"], history["actor_loss"], label="actor")
    axes[1].plot(history["step"], history["bc_loss"], label="bc", linestyle="--")
    axes[1].plot(history["step"], history["anchor_loss"], label="anchor", linestyle="-.")
    axes[1].plot(history["step"], history["bc_coef"], label="bc coef", linestyle=":")
    axes[1].plot(history["step"], history["anchor_coef"], label="anchor coef", linestyle=(0, (3, 1, 1, 1)))
    axes[1].set_ylabel("loss / coef")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(history["step"], history["alpha"], label="alpha")
    axes[2].plot(history["step"], history["hazard_frac"], label="hazard frac")
    axes[2].plot(history["step"], history["demo_frac"], label="demo frac")
    axes[2].set_ylabel("buffer stats")
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(history["step"], history["medium_demo_frac"], label="medium demo frac")
    axes[3].plot(history["step"], history["hard_demo_frac"], label="hard demo frac")
    axes[3].plot(history["step"], history["medium_bc_weight"], label="medium bc weight")
    axes[3].plot(history["step"], history["hard_bc_weight"], label="hard bc weight")
    axes[3].set_ylabel("imitation shaping")
    axes[3].legend()
    axes[3].grid(True)

    if history["eval_step"]:
        axes[4].plot(history["eval_step"], history["eval_success"], label="eval success")
        axes[4].plot(history["eval_step"], history["eval_collision"], label="eval collision")
        axes[4].plot(history["eval_step"], history["eval_min_h"], label="eval min_h")
        axes[4].plot(history["eval_step"], history["eval_takeover"], label="eval takeover")
    axes[4].set_xlabel("environment step")
    axes[4].set_ylabel("evaluation")
    axes[4].legend()
    axes[4].grid(True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PyTorch SAC authority allocator with sequence-aware encoder and risk-shaped imitation.")
    parser.add_argument("--total-steps", type=int, default=7000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--encoder-type", type=str, default="gru", choices=["mlp", "gru"])
    parser.add_argument("--history-stack", type=int, default=4)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--driver-population", action="store_true")
    parser.add_argument("--buffer-size", type=int, default=60_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--start-random-steps", type=int, default=150)
    parser.add_argument("--learning-starts", type=int, default=250)
    parser.add_argument("--actor-learning-starts", type=int, default=250)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--seed-demo-episodes", type=int, default=16)
    parser.add_argument("--seed-driver-episodes", type=int, default=10)
    parser.add_argument("--demo-sample-frac", type=float, default=0.22)
    parser.add_argument("--hazard-sample-frac", type=float, default=0.35)
    parser.add_argument("--hazard-threshold", type=float, default=0.55)
    parser.add_argument("--guided-explore-prob-start", type=float, default=0.28)
    parser.add_argument("--guided-explore-prob-end", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--actor-lr", type=float, default=None)
    parser.add_argument("--critic-lr", type=float, default=None)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--init-alpha", type=float, default=0.18)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--bc-coef-start", type=float, default=0.16)
    parser.add_argument("--bc-coef-end", type=float, default=0.02)
    parser.add_argument("--anchor-coef-start", type=float, default=0.18)
    parser.add_argument("--anchor-coef-end", type=float, default=0.06)
    parser.add_argument("--medium-bc-boost", type=float, default=0.75)
    parser.add_argument("--hard-bc-scale", type=float, default=0.55)
    parser.add_argument("--medium-anchor-boost", type=float, default=0.60)
    parser.add_argument("--hard-anchor-scale", type=float, default=0.45)
    parser.add_argument("--medium-severity-min", type=float, default=0.30)
    parser.add_argument("--medium-severity-max", type=float, default=1.10)
    parser.add_argument("--hazard-td-weight", type=float, default=0.35)
    parser.add_argument("--pretrain-bc-steps", type=int, default=150)
    parser.add_argument("--pretrain-batch-size", type=int, default=128)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--out", type=str, default="artifacts/v10_sac_last.pt")
    parser.add_argument("--best-out", type=str, default="artifacts/v10_sac_best.pt")
    parser.add_argument("--curve-out", type=str, default="artifacts/v10_sac_curve.png")
    parser.add_argument("--metrics-json", type=str, default="artifacts/v10_sac_metrics.json")
    parser.add_argument("--init-ppo", type=str, default=None, help="Optional PPO checkpoint to initialize the SAC actor when encoder-type=mlp and history-stack=1.")
    parser.add_argument("--init-actor-from-sac", type=str, default=None, help="Optional SAC checkpoint to initialize the actor from.")
    parser.add_argument("--init-critic-from-sac", action="store_true", help="Also initialize critic/target critic from the SAC checkpoint when available.")
    parser.add_argument("--init-alpha-from-sac", action="store_true", help="Also initialize entropy temperature from the SAC checkpoint when available.")
    parser.add_argument("--anchor-after-pretrain", action="store_true", help="Freeze a copy of the current actor after BC pretrain / actor init and regularize against it during RL updates.")
    args = parser.parse_args()

    trainer = SACTrainer(args)
    history, metrics = trainer.train()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "algorithm": "sac_torch_v11_conservative_seq_bc",
        "history_stack": int(args.history_stack),
        "base_obs_dim": int(trainer.base_obs_dim),
        "encoder_type": str(args.encoder_type),
        "curriculum": bool(args.curriculum),
        "driver_population": bool(args.driver_population),
        "randomized_training": bool(args.randomize),
        "final_metrics": metrics,
        "replay_counts": trainer.replay.counts(),
        "sampling": {
            "demo_sample_frac": float(args.demo_sample_frac),
            "hazard_sample_frac": float(args.hazard_sample_frac),
            "hazard_threshold": float(args.hazard_threshold),
            "actor_learning_starts": int(args.actor_learning_starts),
        },
        "bc_shaping": {
            "medium_bc_boost": float(args.medium_bc_boost),
            "hard_bc_scale": float(args.hard_bc_scale),
            "medium_anchor_boost": float(args.medium_anchor_boost),
            "hard_anchor_scale": float(args.hard_anchor_scale),
            "medium_severity_min": float(args.medium_severity_min),
            "medium_severity_max": float(args.medium_severity_max),
            "anchor_source": trainer.anchor_source,
        },
    }
    save_checkpoint(
        actor=trainer.actor,
        critic=trainer.critic,
        target_critic=trainer.target_critic,
        obs_dim=trainer.obs_dim,
        action_dim=trainer.action_dim,
        hidden_dim=args.hidden_dim,
        path=out_path,
        log_alpha=float(trainer.log_alpha.item()),
        metadata=metadata,
    )
    plot_training(history, Path(args.curve_out))

    payload = {
        "metrics": metrics,
        "history": history,
        "checkpoint": str(out_path),
        "replay_counts": trainer.replay.counts(),
    }
    metrics_path = Path(args.metrics_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved last checkpoint to {out_path}")
    print(f"Saved metrics JSON to {metrics_path}")


if __name__ == "__main__":
    main()
