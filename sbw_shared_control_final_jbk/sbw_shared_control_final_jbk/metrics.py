from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def summarize_episode(policy_name: str, seed: int, logs: List[Dict]) -> Dict:
    if not logs:
        raise ValueError('logs must not be empty')
    rewards = np.asarray([float(row['reward']) for row in logs], dtype=float)
    hs = np.asarray([float(row['h']) for row in logs], dtype=float)
    lams = np.asarray([float(row['lam']) for row in logs], dtype=float)
    ys = np.asarray([float(row['y']) for row in logs], dtype=float)
    last = logs[-1]
    return {
        'policy': policy_name,
        'seed': int(seed),
        'episode_return': float(rewards.sum()),
        'success': float(bool(last.get('success', False))),
        'collision': float(bool(last.get('collision', False))),
        'road_departure': float(bool(last.get('road_departure', False))),
        'min_h': float(hs.min()),
        'mean_takeover': float(np.mean(1.0 - lams)),
        'min_lambda': float(lams.min()),
        'max_abs_y': float(np.max(np.abs(ys))),
    }


def aggregate_episode_summaries(summaries: Iterable[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(list(summaries))
    if df.empty:
        return df
    grouped = (
        df.groupby('policy', as_index=False)
        .agg({
            'episode_return': 'mean',
            'success': 'mean',
            'collision': 'mean',
            'road_departure': 'mean',
            'min_h': 'mean',
            'mean_takeover': 'mean',
            'min_lambda': 'mean',
            'max_abs_y': 'mean',
        })
    )
    return grouped.sort_values('policy').reset_index(drop=True)
