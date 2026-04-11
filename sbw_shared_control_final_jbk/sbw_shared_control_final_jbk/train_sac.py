from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from bc_model import BCConfig, BCAuthorityActor
from env import AutomationParams, EnvParams, SharedControlEnv
from history_stack import HistoryStackEnv
from metrics import aggregate_episode_summaries, summarize_episode
from policies import heuristic_authority_policy
from replay_buffer import ReplayBuffer, ReplayConfig
from sac_model import SACAgent, SACConfig


def build_env(driver_model: str = 'jbk', randomize: bool = True, seed: int | None = None) -> HistoryStackEnv:
    env = SharedControlEnv(
        driver_model=driver_model,
        automation_params=AutomationParams(),
        env_params=EnvParams(randomize_on_reset=randomize, randomize_pass_side=True, randomize_driver=(driver_model == 'jbk')),
    )
    if seed is not None:
        env.rng.seed(seed)
    return HistoryStackEnv(env, history_len=4)


class BCPolicy:
    def __init__(self, ckpt_path: Path):
        payload = torch.load(ckpt_path, map_location='cpu')
        input_dim = int(payload['extra']['input_dim'])
        self.model = BCAuthorityActor(BCConfig(input_dim=input_dim))
        self.model.load_state_dict(payload['state_dict'])
        self.model.eval()

    def __call__(self, obs, info, env) -> float:
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            y = self.model(x).squeeze().item()
        return float(np.clip(y, -1.0, 1.0))


def seed_replay_with_teacher(replay: ReplayBuffer, env: HistoryStackEnv, episodes: int, seed0: int = 2000) -> int:
    transitions = 0
    for ep in range(episodes):
        obs, info = env.reset(seed=seed0 + ep)
        done = False
        while not done:
            action = float(heuristic_authority_policy(obs, info, env.env))
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            replay.add(obs, np.array([action], dtype=np.float32), reward, next_obs, done, is_demo=True)
            transitions += 1
            obs, info = next_obs, next_info
    return transitions


def evaluate_agent(agent: SACAgent, driver_model: str = 'jbk', episodes: int = 10, seed0: int = 9000) -> Tuple[List[Dict], Dict[str, float]]:
    env = build_env(driver_model=driver_model, randomize=True)
    summaries = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed0 + ep)
        done = False
        logs: List[Dict] = []
        while not done:
            action = agent.act(obs, deterministic=True)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            logs.append({
                't': next_info['t'], 'x': next_info['x'], 'y': next_info['y'], 'lam': next_info['lam'], 'h': next_info['h'], 'reward': reward,
                'success': next_info['success'], 'collision': next_info['collision'], 'road_departure': next_info['road_departure'],
            })
            obs = next_obs
            done = terminated or truncated
        summaries.append(summarize_episode('sac', seed0 + ep, logs))
    agg = aggregate_episode_summaries(summaries)
    row = agg.iloc[0].to_dict() if len(agg) else {}
    return summaries, row


def _to_python_dict(d: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for k, v in d.items():
        try:
            out[k] = float(v)
        except Exception:
            out[k] = v
    return out


def train(
    bc_ckpt: Path,
    best_out: Path,
    last_out: Path,
    curve_png: Path,
    metrics_json: Path,
    driver_model: str = 'jbk',
    total_steps: int = 2500,
    demo_episodes: int = 24,
    batch_size: int = 128,
    seed: int = 0,
) -> None:
    torch.set_num_threads(1)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = build_env(driver_model=driver_model, randomize=True, seed=seed)
    obs0, _ = env.reset(seed=seed)
    obs_dim = int(obs0.shape[0])
    device = torch.device('cpu')

    agent = SACAgent(SACConfig(obs_dim=obs_dim), device=device)
    bc_extra = agent.load_bc_actor(bc_ckpt)

    replay = ReplayBuffer(obs_dim=obs_dim, action_dim=1, cfg=ReplayConfig(capacity=50000, demo_fraction=0.40))
    demo_transitions = seed_replay_with_teacher(replay, env, episodes=demo_episodes, seed0=2000)

    obs, info = env.reset(seed=seed)
    episode_idx = 0
    history = {'step': [], 'q1_loss': [], 'q2_loss': [], 'actor_loss': [], 'alpha': [], 'bc_loss': [], 'eval_success': [], 'eval_collision': [], 'eval_takeover': [], 'eval_min_h': []}
    best_score = (-1.0, -1.0, -1.0, -1e9, -1e9)  # success, -collision, min_h, -takeover, return
    best_meta = {}
    update_after = max(200, batch_size)
    actor_learning_starts = 800
    eval_interval = 400

    _, row0 = evaluate_agent(agent, driver_model=driver_model, episodes=12, seed0=8000)
    best_score = (
        float(row0.get('success', 0.0)),
        -float(row0.get('collision', 1.0)),
        float(row0.get('min_h', -1e9)),
        -float(row0.get('mean_takeover', 1e9)),
        float(row0.get('episode_return', -1e9)),
    )
    best_meta = {
        'step': 0,
        'driver_model': driver_model,
        'eval_row': _to_python_dict(row0),
        'bc_extra': {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in bc_extra.items()},
        'obs_dim': int(obs_dim),
        'total_steps': int(total_steps),
        'demo_episodes': int(demo_episodes),
        'seed': int(seed),
    }
    best_out.parent.mkdir(parents=True, exist_ok=True)
    agent.save(best_out, extra=best_meta)

    for step in range(1, total_steps + 1):
        action = agent.act(obs, deterministic=False)
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        replay.add(obs, np.array([action], dtype=np.float32), reward, next_obs, done, is_demo=False)
        obs = next_obs
        info = next_info

        if replay.size >= update_after:
            batch = replay.sample(batch_size, device=device)
            delay = 999999 if step < actor_learning_starts else 2
            upd = agent.update(batch, step=step, actor_update_delay=delay)
            history['step'].append(step)
            history['q1_loss'].append(upd['q1_loss'])
            history['q2_loss'].append(upd['q2_loss'])
            history['actor_loss'].append(upd['actor_loss'])
            history['alpha'].append(upd['alpha'])
            history['bc_loss'].append(upd['bc_loss'])

        if done:
            obs, info = env.reset(seed=seed + 100 + episode_idx)
            episode_idx += 1

        if step % eval_interval == 0 or step == total_steps:
            _, row = evaluate_agent(agent, driver_model=driver_model, episodes=12, seed0=9000 + step)
            success = float(row.get('success', 0.0))
            collision = float(row.get('collision', 1.0))
            min_h = float(row.get('min_h', -1e9))
            takeover = float(row.get('mean_takeover', 1e9))
            ret = float(row.get('episode_return', -1e9))
            history['eval_success'].append(success)
            history['eval_collision'].append(collision)
            history['eval_takeover'].append(takeover)
            history['eval_min_h'].append(min_h)
            score = (success, -collision, min_h, -takeover, ret)
            print(f'step {step:05d} | eval success={success:.3f} collision={collision:.3f} min_h={min_h:.3f} takeover={takeover:.3f}')
            if score > best_score:
                best_score = score
                best_meta = {
                    'step': int(step),
                    'driver_model': driver_model,
                    'eval_row': _to_python_dict(row),
                    'bc_extra': {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in bc_extra.items()},
                    'obs_dim': int(obs_dim),
                    'total_steps': int(total_steps),
                    'demo_episodes': int(demo_episodes),
                    'seed': int(seed),
                }
                best_out.parent.mkdir(parents=True, exist_ok=True)
                agent.save(best_out, extra=best_meta)

    last_out.parent.mkdir(parents=True, exist_ok=True)
    agent.save(last_out, extra={'step': total_steps, 'obs_dim': obs_dim, 'demo_episodes': demo_episodes, 'driver_model': driver_model, 'seed': seed})

    curve_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    if history['step']:
        plt.plot(history['step'], history['q1_loss'], label='q1 loss')
        plt.plot(history['step'], history['q2_loss'], label='q2 loss')
        plt.plot(history['step'], history['bc_loss'], label='bc loss')
    plt.xlabel('training step')
    plt.ylabel('loss')
    plt.title(f'Step 8 SAC training traces ({driver_model})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_png, dpi=150)
    plt.close()

    metrics = {
        'driver_model': driver_model,
        'replay_size_final': int(replay.size),
        'demo_transitions': int(demo_transitions),
        'best_score': [float(x) for x in best_score],
        'best_meta': best_meta,
        'history': {k: [float(x) for x in v] if isinstance(v, list) else v for k, v in history.items()},
    }
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_json, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bc-ckpt', type=str, default='artifacts/jbk_bc_actor.pt')
    parser.add_argument('--best-out', type=str, default='artifacts/jbk_sac_best.pt')
    parser.add_argument('--last-out', type=str, default='artifacts/jbk_sac_last.pt')
    parser.add_argument('--curve-png', type=str, default='artifacts/step8_sac_training_curve.png')
    parser.add_argument('--metrics-json', type=str, default='artifacts/step8_sac_metrics.json')
    parser.add_argument('--driver-model', type=str, default='jbk')
    parser.add_argument('--total-steps', type=int, default=2500)
    parser.add_argument('--demo-episodes', type=int, default=24)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    train(Path(args.bc_ckpt), Path(args.best_out), Path(args.last_out), Path(args.curve_png), Path(args.metrics_json), driver_model=args.driver_model, total_steps=args.total_steps, demo_episodes=args.demo_episodes, batch_size=args.batch_size, seed=args.seed)
