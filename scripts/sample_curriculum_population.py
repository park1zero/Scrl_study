#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv
from collections import Counter

import matplotlib.pyplot as plt

from shared_control_rl.config import EnvConfig
from shared_control_rl.envs import make_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample curriculum/population episodes and export the reset distribution.")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history-stack", type=int, default=1)
    parser.add_argument("--out-csv", type=str, default="artifacts/v8_curriculum_population_samples.csv")
    parser.add_argument("--out-plot", type=str, default="artifacts/v8_curriculum_population_counts.png")
    args = parser.parse_args()

    config = EnvConfig(seed=args.seed)
    config.scenario.domain_randomization = True
    config.scenario.side_randomization = True
    config.curriculum.enabled = True
    config.driver_population.enabled = True
    env = make_env(config, history_stack=args.history_stack)

    rows = []
    for ep in range(args.episodes):
        progress = ep / max(args.episodes - 1, 1)
        env.set_curriculum_progress(progress)
        obs, info = env.reset(seed=args.seed + ep)
        rows.append({
            "episode": ep,
            "progress": progress,
            "difficulty": info["difficulty"],
            "driver_profile": info["driver_profile"],
            "late_start": int(bool(info["late_start"])),
            "speed": float(info["speed"]),
            "obstacle_x": float(info["obstacle_x"]),
            "obstacle_b": float(info["obstacle_b"]),
            "driver_total_delay": float(info["driver_total_delay"]),
            "wrong_way_duration": float(info["driver_wrong_way_duration"]),
        })

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved sample CSV to {out_csv}")

    difficulty_counts = Counter(row["difficulty"] for row in rows)
    profile_counts = Counter(row["driver_profile"] for row in rows)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    axes[0].bar(list(difficulty_counts.keys()), list(difficulty_counts.values()))
    axes[0].set_ylabel("episodes")
    axes[0].set_title("Curriculum difficulty samples")
    axes[0].grid(True, axis="y")

    axes[1].bar(list(profile_counts.keys()), list(profile_counts.values()))
    axes[1].set_ylabel("episodes")
    axes[1].set_title("Driver profile samples")
    axes[1].grid(True, axis="y")
    fig.tight_layout()

    out_plot = Path(args.out_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=150)
    plt.close(fig)
    print(f"Saved count plot to {out_plot}")


if __name__ == "__main__":
    main()
