#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathlib import Path

from shared_control_rl.config import EnvConfig
from shared_control_rl.envs.shared_control_env import SharedControlEnv
from scripts.evaluate_policy import run_policy, plot_history


def main() -> None:
    out_dir = Path("artifacts/baselines")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = EnvConfig(seed=42)
    policies = ["driver", "automation", "shared", "heuristic"]

    for policy in policies:
        env = SharedControlEnv(config)
        history = run_policy(env, policy_name=policy)
        plot_history(history, env, title=f"Baseline: {policy}", out_path=out_dir / f"{policy}.png")


if __name__ == "__main__":
    main()
