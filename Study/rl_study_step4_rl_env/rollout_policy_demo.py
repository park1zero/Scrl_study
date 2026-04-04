from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Dict, List, Tuple
import random

import matplotlib.pyplot as plt

from env import SharedControlEnv
from policies import (
    driver_hold_policy,
    full_takeover_policy,
    heuristic_authority_policy,
    random_policy,
)


PolicyFn = Callable[[object, Dict[str, float], SharedControlEnv], float]


def rollout(
    env: SharedControlEnv,
    policy_name: str,
    policy: PolicyFn,
    seed: int,
    rng: random.Random | None = None,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    obs, info = env.reset(seed=seed)
    rows: List[Dict[str, float]] = []
    total_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        if policy_name == "random":
            assert rng is not None
            action = random_policy(obs, info, env, rng)
        else:
            action = policy(obs, info, env)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        row = {
            "policy": policy_name,
            "t": info["t"],
            "x": info["x"],
            "y": info["y"],
            "psi": info["psi"],
            "delta_rwa": info["delta_rwa"],
            "delta_swa": info["delta_swa"],
            "delta_drv_rwa": info["delta_drv_rwa"],
            "delta_auto_rwa": info["delta_auto_rwa"],
            "delta_cmd": info["delta_cmd"],
            "lam": info["lam"],
            "action": info["action"],
            "h": info["h"],
            "ttc": info["ttc"],
            "reward": reward,
            "reward_barrier": info["reward_barrier"],
            "reward_takeover": info["reward_takeover"],
            "collision": float(info["collision"]),
            "road_departure": float(info["road_departure"]),
            "success": float(info["success"]),
        }
        rows.append(row)

    summary = {
        "policy": policy_name,
        "return": total_reward,
        "success": float(info["success"]),
        "collision": float(info["collision"]),
        "road_departure": float(info["road_departure"]),
        "final_x": info["x"],
        "final_y": info["y"],
        "min_h": min(r["h"] for r in rows) if rows else float("nan"),
        "mean_lambda": sum(r["lam"] for r in rows) / max(len(rows), 1),
        "mean_takeover": sum(1.0 - r["lam"] for r in rows) / max(len(rows), 1),
        "steps": len(rows),
    }
    return rows, summary


def save_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_plot(output_path: Path, obstacle_x: float, obstacle_y: float, logs: Dict[str, List[Dict[str, float]]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    for name, rows in logs.items():
        ax.plot([r["x"] for r in rows], [r["y"] for r in rows], label=name)
    ax.axvline(obstacle_x, linestyle="--")
    ax.axhline(obstacle_y, linestyle=":")
    ax.set_title("Trajectory (x-y)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()

    ax = axes[0, 1]
    for name, rows in logs.items():
        ax.plot([r["t"] for r in rows], [r["lam"] for r in rows], label=name)
    ax.set_title("Authority lambda")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("lambda")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()

    ax = axes[1, 0]
    for name, rows in logs.items():
        ax.plot([r["t"] for r in rows], [r["h"] for r in rows], label=name)
    ax.axhline(0.0, linestyle="--")
    ax.set_title("Barrier h")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("h")
    ax.legend()

    ax = axes[1, 1]
    for name, rows in logs.items():
        ax.plot([r["t"] for r in rows], [r["action"] for r in rows], label=name)
    ax.set_title("Authority-rate action")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("action")
    ax.set_ylim(-1.05, 1.05)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = SharedControlEnv()
    base_seed = 7
    policy_table = {
        "driver_hold": driver_hold_policy,
        "full_takeover": full_takeover_policy,
        "heuristic": heuristic_authority_policy,
        "random": random_policy,
    }

    rng = random.Random(123)
    all_logs: Dict[str, List[Dict[str, float]]] = {}
    summary_rows: List[Dict[str, float]] = []

    for name, policy in policy_table.items():
        logs, summary = rollout(env, name, policy, seed=base_seed, rng=rng)
        all_logs[name] = logs
        summary_rows.append(summary)
        save_csv(output_dir / f"step4_{name}_log.csv", logs)

    save_csv(output_dir / "step4_policy_summary.csv", summary_rows)
    build_plot(output_dir / "step4_policy_preview.png", env.obstacle.x, env.obstacle.y, all_logs)

    print("Saved logs and preview plot to", output_dir)
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
