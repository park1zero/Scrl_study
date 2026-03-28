#!/usr/bin/env python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    art = ROOT / "artifacts"
    anim = art / "animations_v10"
    anim.mkdir(parents=True, exist_ok=True)

    run([
        sys.executable,
        "scripts/train_sac_torch.py",
        "--total-steps", "0",
        "--encoder-type", "gru",
        "--history-stack", "4",
        "--curriculum",
        "--driver-population",
        "--pretrain-bc-steps", "300",
        "--out", "artifacts/v10_pretrain_only.pt",
        "--metrics-json", "artifacts/v10_pretrain_only_metrics.json",
    ])

    for policy, model, stack, prefix in [
        ("driver", None, None, "v10_grid_driver"),
        ("heuristic", None, None, "v10_grid_heuristic"),
        ("sac", "artifacts/v10_pretrain_only.pt", "4", "v10_grid_pretrain"),
    ]:
        cmd = [
            sys.executable,
            "scripts/evaluate_policy_grid.py",
            "--policy", policy,
            "--episodes-per-cell", "1",
            "--csv-out", f"artifacts/{prefix}_episodes.csv",
            "--summary-out", f"artifacts/{prefix}_summary.csv",
            "--heatmap-prefix", f"artifacts/{prefix}",
            "--json-out", f"artifacts/{prefix}_overall.json",
        ]
        if model is not None:
            cmd += ["--model", model, "--history-stack", stack]
        run(cmd)

    run([
        sys.executable,
        "scripts/render_animation.py",
        "--policy", "sac",
        "--model", "artifacts/v10_pretrain_only.pt",
        "--history-stack", "4",
        "--difficulty", "hard",
        "--driver-profile", "frozen",
        "--curriculum-progress", "1.0",
        "--out", "artifacts/animations_v10/hard_frozen_pretrain.gif",
        "--summary", "artifacts/animations_v10/hard_frozen_pretrain.json",
    ])


if __name__ == "__main__":
    main()
