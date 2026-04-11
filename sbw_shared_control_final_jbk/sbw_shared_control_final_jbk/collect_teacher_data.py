from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from env import AutomationParams, EnvParams, SharedControlEnv
from history_stack import HistoryStackEnv
from policies import heuristic_authority_policy


def build_env(driver_model: str = 'jbk') -> HistoryStackEnv:
    env = SharedControlEnv(
        driver_model=driver_model,
        automation_params=AutomationParams(),
        env_params=EnvParams(randomize_on_reset=True, randomize_pass_side=True, randomize_driver=(driver_model == 'jbk')),
    )
    return HistoryStackEnv(env, history_len=4)


def collect_dataset(episodes: int, output_path: Path, driver_model: str = 'jbk', seed0: int = 0) -> None:
    env = build_env(driver_model=driver_model)
    obs_list = []
    act_list = []
    meta = []
    for ep in range(episodes):
        seed = seed0 + ep
        obs, info = env.reset(seed=seed)
        done = False
        while not done:
            action = heuristic_authority_policy(obs, info, env.env)
            obs_list.append(obs.copy())
            act_list.append([float(action)])
            meta.append([seed, info['lam'], info['h'], info['ttc']])
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    observations = np.asarray(obs_list, dtype=np.float32)
    actions = np.asarray(act_list, dtype=np.float32)
    meta_arr = np.asarray(meta, dtype=np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, observations=observations, actions=actions, meta=meta_arr)
    print(f'saved dataset to {output_path} | samples={len(observations)} | obs_dim={observations.shape[1]} | driver_model={driver_model}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--out', type=str, default='artifacts/jbk_teacher_dataset.npz')
    parser.add_argument('--driver-model', type=str, default='jbk')
    args = parser.parse_args()
    collect_dataset(args.episodes, Path(args.out), driver_model=args.driver_model)
