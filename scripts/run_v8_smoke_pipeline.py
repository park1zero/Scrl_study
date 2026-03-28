#!/usr/bin/env python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    run([PYTHON, "scripts/sample_curriculum_population.py"])
    run([PYTHON, "scripts/sweep_policies.py", "--policy", "driver", "--episodes", "12", "--curriculum", "--driver-population", "--out", "artifacts/v8_driver_curriculum_sweep.csv"])
    run([PYTHON, "scripts/sweep_policies.py", "--policy", "heuristic", "--episodes", "12", "--curriculum", "--driver-population", "--out", "artifacts/v8_heuristic_curriculum_sweep.csv"])
    run([
        PYTHON,
        "scripts/train_sac_torch.py",
        "--total-steps", "2500",
        "--history-stack", "4",
        "--curriculum",
        "--driver-population",
        "--seed-demo-episodes", "8",
        "--out", "artifacts/v8_sac_last.pt",
        "--best-out", "artifacts/v8_sac_best.pt",
        "--curve-out", "artifacts/v8_sac_curve.png",
        "--metrics-json", "artifacts/v8_sac_metrics.json",
    ])
    run([
        PYTHON,
        "scripts/evaluate_sac_torch.py",
        "--model", "artifacts/v8_sac_best.pt",
        "--curriculum",
        "--driver-population",
        "--curriculum-progress", "1.0",
        "--out", "artifacts/v8_eval_sac_best.png",
        "--metrics-json", "artifacts/v8_eval_sac_best.json",
    ])


if __name__ == "__main__":
    main()
