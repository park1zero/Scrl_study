from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from bc_model import BCConfig, BCAuthorityActor
from env import AutomationParams, EnvParams, SharedControlEnv
from history_stack import HistoryStackEnv
from metrics import aggregate_episode_summaries, summarize_episode
from policies import driver_hold_policy, heuristic_authority_policy
from sac_model import SACAgent, SACConfig


def build_env(driver_model: str = 'jbk', randomize: bool = True) -> HistoryStackEnv:
    env = SharedControlEnv(
        driver_model=driver_model,
        automation_params=AutomationParams(),
        env_params=EnvParams(randomize_on_reset=randomize, randomize_pass_side=True, randomize_driver=(driver_model == 'jbk')),
    )
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


class SACPolicy:
    def __init__(self, ckpt_path: Path):
        payload = torch.load(ckpt_path, map_location='cpu')
        obs_dim = int(payload['extra']['obs_dim'])
        self.agent = SACAgent(SACConfig(obs_dim=obs_dim), device=torch.device('cpu'))
        self.agent.load(ckpt_path)

    def __call__(self, obs, info, env) -> float:
        return self.agent.act(obs, deterministic=True)


def rollout(env: HistoryStackEnv, policy_name: str, policy_fn: Callable, seed: int) -> List[Dict]:
    obs, info = env.reset(seed=seed)
    done = False
    logs: List[Dict] = []
    while not done:
        action = float(policy_fn(obs, info, env.env))
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        logs.append({
            't': next_info['t'], 'x': next_info['x'], 'y': next_info['y'], 'lam': next_info['lam'], 'h': next_info['h'],
            'reward': reward, 'success': next_info['success'], 'collision': next_info['collision'], 'road_departure': next_info['road_departure'],
            'policy': policy_name,
        })
        obs, info = next_obs, next_info
        done = terminated or truncated
    return logs


def evaluate(bc_ckpt: Path, sac_ckpt: Path, driver_model: str = 'jbk', episodes: int = 24, summary_csv: Path = Path('artifacts/final_policy_summary.csv'), preview_png: Path = Path('artifacts/final_policy_preview.png'), compare_png: Path = Path('artifacts/final_rollout_compare.png')) -> None:
    env = build_env(driver_model=driver_model, randomize=True)
    policies = {
        'driver_hold': driver_hold_policy,
        'heuristic': heuristic_authority_policy,
        'bc_warmstart': BCPolicy(bc_ckpt),
        'sac_finetune': SACPolicy(sac_ckpt),
    }
    summaries = []
    rollout_bank = {}
    same_seed = 1203
    for name, fn in policies.items():
        for i in range(episodes):
            seed = 1200 + i
            logs = rollout(env, name, fn, seed)
            summaries.append(summarize_episode(name, seed, logs))
        rollout_bank[name] = pd.DataFrame(rollout(env, name, fn, same_seed))
    agg = aggregate_episode_summaries(summaries)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(summary_csv, index=False)

    x = np.arange(len(agg))
    width = 0.25
    plt.figure(figsize=(8.5, 4.8))
    plt.bar(x - width, agg['success'], width=width, label='success')
    plt.bar(x, agg['collision'], width=width, label='collision')
    plt.bar(x + width, agg['mean_takeover'], width=width, label='mean takeover')
    plt.xticks(x, agg['policy'], rotation=15)
    plt.ylim(0.0, 1.05)
    plt.title(f'Final JBK authority-allocation policy comparison ({driver_model})')
    plt.tight_layout()
    plt.legend()
    plt.savefig(preview_png, dpi=150)
    plt.close()

    plt.figure(figsize=(8.5, 5.2))
    for name, df in rollout_bank.items():
        plt.plot(df['t'], 1.0 - df['lam'], label=f'{name} takeover')
    plt.xlabel('time [s]')
    plt.ylabel('1 - lambda')
    plt.title(f'Same-seed takeover comparison (seed={same_seed})')
    plt.tight_layout()
    plt.legend()
    plt.savefig(compare_png, dpi=150)
    plt.close()
    print(agg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bc-ckpt', type=str, default='artifacts/jbk_bc_actor.pt')
    parser.add_argument('--sac-ckpt', type=str, default='artifacts/jbk_sac_best.pt')
    parser.add_argument('--driver-model', type=str, default='jbk')
    parser.add_argument('--episodes', type=int, default=24)
    parser.add_argument('--summary-csv', type=str, default='artifacts/final_policy_summary.csv')
    parser.add_argument('--preview-png', type=str, default='artifacts/final_policy_preview.png')
    parser.add_argument('--compare-png', type=str, default='artifacts/final_rollout_compare.png')
    args = parser.parse_args()
    evaluate(Path(args.bc_ckpt), Path(args.sac_ckpt), driver_model=args.driver_model, episodes=args.episodes, summary_csv=Path(args.summary_csv), preview_png=Path(args.preview_png), compare_png=Path(args.compare_png))
