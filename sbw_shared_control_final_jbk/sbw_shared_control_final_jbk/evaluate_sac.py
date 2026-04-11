from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
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


def rollout(env: HistoryStackEnv, policy_name: str, policy_fn: Callable, seed: int) -> Tuple[List[Dict], Dict[str, float]]:
    obs, info = env.reset(seed=seed)
    meta = {
        'seed': seed,
        'road_limit_abs_y': float(env.env.env_params.road_limit_abs_y),
        'success_x_margin': float(env.env.env_params.success_x_margin),
        'lane_center_y': float(getattr(env.env.driver_params, 'lane_center_y', 0.0)),
        'obstacle_x': float(env.env.obstacle.x),
        'obstacle_y': float(env.env.obstacle.y),
        'obstacle_a': float(env.env.obstacle.a),
        'obstacle_b': float(env.env.obstacle.b),
        'obstacle_margin': float(env.env.obstacle.margin),
    }
    done = False
    logs: List[Dict] = []
    while not done:
        action = float(policy_fn(obs, info, env.env))
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        logs.append({
            't': next_info['t'], 'x': next_info['x'], 'y': next_info['y'], 'lam': next_info['lam'], 'h': next_info['h'],
            'reward': reward, 'success': next_info['success'], 'collision': next_info['collision'], 'road_departure': next_info['road_departure'],
            'action': action, 'auto_y_ref': next_info.get('auto_y_ref', 0.0), 'driver_y_ref_core': next_info.get('driver_y_ref_core', 0.0),
            'delta_cmd': next_info.get('delta_cmd', 0.0), 'delta_drv_rwa': next_info.get('delta_drv_rwa', 0.0), 'delta_auto_rwa': next_info.get('delta_auto_rwa', 0.0),
            'policy': policy_name,
        })
        obs, info = next_obs, next_info
        done = terminated or truncated
    return logs, meta


def _save_animation(anim: animation.FuncAnimation, out_path: Path, fps: int = 20) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix == '.gif':
        try:
            anim.save(out_path, writer=animation.PillowWriter(fps=fps))
            return out_path
        except Exception:
            html_path = out_path.with_suffix('.html')
            html_path.write_text(anim.to_jshtml(), encoding='utf-8')
            return html_path
    if suffix == '.mp4':
        try:
            anim.save(out_path, writer=animation.FFMpegWriter(fps=fps))
            return out_path
        except Exception:
            html_path = out_path.with_suffix('.html')
            html_path.write_text(anim.to_jshtml(), encoding='utf-8')
            return html_path
    if suffix == '.html':
        out_path.write_text(anim.to_jshtml(), encoding='utf-8')
        return out_path
    raise ValueError(f'unsupported animation suffix: {out_path.suffix}')


def create_compare_animation(
    driver_df: pd.DataFrame,
    sac_df: pd.DataFrame,
    meta: Dict[str, float],
    out_path: Path,
    title: str,
) -> Path:
    if driver_df.empty or sac_df.empty:
        raise ValueError('animation requires non-empty rollouts')

    road = float(meta['road_limit_abs_y'])
    lane_center_y = float(meta['lane_center_y'])
    obs_x = float(meta['obstacle_x'])
    obs_y = float(meta['obstacle_y'])
    obs_a = float(meta['obstacle_a'])
    obs_b = float(meta['obstacle_b'])
    obs_margin = float(meta['obstacle_margin'])
    xmax = max(
        float(driver_df['x'].max()),
        float(sac_df['x'].max()),
        obs_x + obs_a + float(meta['success_x_margin']) + 4.0,
    )
    xmin = -1.0
    ymin = -road - 0.7
    ymax = road + 0.7

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), sharex=True, sharey=True)
    fig.suptitle(title)

    panel_cfg = [
        (axes[0], driver_df, 'driver_y_ref_core', 'Driver Only', 'driver ref', '#2563eb'),
        (axes[1], sac_df, 'auto_y_ref', 'Shared Control (SAC)', 'automation ref', '#ea580c'),
    ]
    artists = []
    panels = []

    for ax, df, ref_key, panel_title, ref_label, color in panel_cfg:
        ax.set_title(panel_title)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('x [m]')
        ax.axhline(lane_center_y, color='black', linestyle='--', linewidth=1.0, alpha=0.6, label='lane center')
        ax.axhline(road, color='grey', linestyle=':', linewidth=1.0, alpha=0.7)
        ax.axhline(-road, color='grey', linestyle=':', linewidth=1.0, alpha=0.7)
        obstacle_patch = Ellipse((obs_x, obs_y), width=2.0 * obs_a, height=2.0 * obs_b, facecolor='#9ca3af', edgecolor='#374151', alpha=0.75)
        safety_patch = Ellipse((obs_x, obs_y), width=2.0 * (obs_a + obs_margin), height=2.0 * (obs_b + obs_margin), fill=False, edgecolor='#dc2626', linestyle='--', linewidth=1.3)
        ax.add_patch(obstacle_patch)
        ax.add_patch(safety_patch)
        traj_line, = ax.plot([], [], color=color, linewidth=2.2, label='trajectory')
        ref_line, = ax.plot([], [], color=color, linestyle='--', linewidth=1.6, alpha=0.8, label=ref_label)
        car_pt, = ax.plot([], [], marker='o', color=color, markersize=7)
        text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', ha='left', fontsize=9, family='monospace', bbox={'facecolor': 'white', 'alpha': 0.72, 'edgecolor': 'none'})
        ax.legend(loc='lower right')
        panels.append((df.reset_index(drop=True), ref_key, traj_line, ref_line, car_pt, text))
        artists.extend([traj_line, ref_line, car_pt, text])

    axes[0].set_ylabel('y [m]')
    fig.tight_layout()

    n_frames = max(len(driver_df), len(sac_df))

    def update(frame_idx: int):
        for df, ref_key, traj_line, ref_line, car_pt, text in panels:
            idx = min(frame_idx, len(df) - 1)
            hist = df.iloc[: idx + 1]
            row = df.iloc[idx]
            traj_line.set_data(hist['x'], hist['y'])
            ref_line.set_data(hist['x'], hist[ref_key])
            car_pt.set_data([row['x']], [row['y']])
            status = 'RUN'
            if bool(row['success']):
                status = 'SUCCESS'
            elif bool(row['collision']):
                status = 'COLLISION'
            elif bool(row['road_departure']):
                status = 'ROAD'
            text.set_text(
                f"t={row['t']:.2f}s\n"
                f"lambda={row['lam']:.2f}\n"
                f"h={row['h']:.2f}\n"
                f"y={row['y']:.2f}\n"
                f"status={status}"
            )
        return artists

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=50, blit=False, repeat=False)
    saved_path = _save_animation(anim, out_path, fps=20)
    plt.close(fig)
    return saved_path


def evaluate(
    bc_ckpt: Path,
    sac_ckpt: Path,
    driver_model: str = 'jbk',
    episodes: int = 24,
    summary_csv: Path = Path('artifacts/final_policy_summary.csv'),
    preview_png: Path = Path('artifacts/final_policy_preview.png'),
    compare_png: Path = Path('artifacts/final_rollout_compare.png'),
    animation_path: Path | None = Path('artifacts/final_policy_compare_animation.gif'),
) -> None:
    env = build_env(driver_model=driver_model, randomize=True)
    policies = {
        'driver_hold': driver_hold_policy,
        'heuristic': heuristic_authority_policy,
        'bc_warmstart': BCPolicy(bc_ckpt),
        'sac_finetune': SACPolicy(sac_ckpt),
    }
    summaries = []
    rollout_bank = {}
    rollout_meta = {}
    same_seed = 1203
    for name, fn in policies.items():
        for i in range(episodes):
            seed = 1200 + i
            logs, _ = rollout(env, name, fn, seed)
            summaries.append(summarize_episode(name, seed, logs))
        logs, meta = rollout(env, name, fn, same_seed)
        rollout_bank[name] = pd.DataFrame(logs)
        rollout_meta[name] = meta
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
    if animation_path is not None and 'driver_hold' in rollout_bank and 'sac_finetune' in rollout_bank:
        saved_animation = create_compare_animation(
            rollout_bank['driver_hold'],
            rollout_bank['sac_finetune'],
            rollout_meta['driver_hold'],
            animation_path,
            title=f'Driver vs Shared Control (seed={same_seed})',
        )
        print(f'saved animation to {saved_animation}')
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
    parser.add_argument('--animation-path', type=str, default='artifacts/final_policy_compare_animation.gif')
    args = parser.parse_args()
    animation_path = Path(args.animation_path) if args.animation_path else None
    evaluate(Path(args.bc_ckpt), Path(args.sac_ckpt), driver_model=args.driver_model, episodes=args.episodes, summary_csv=Path(args.summary_csv), preview_png=Path(args.preview_png), compare_png=Path(args.compare_png), animation_path=animation_path)
