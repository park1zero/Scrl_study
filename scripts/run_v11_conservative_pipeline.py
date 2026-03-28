#!/usr/bin/env python
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
PYTHON = sys.executable


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    run([
        PYTHON,
        "scripts/train_sac_torch.py",
        "--total-steps", "80",
        "--encoder-type", "gru",
        "--history-stack", "4",
        "--curriculum",
        "--driver-population",
        "--seed-demo-episodes", "8",
        "--seed-driver-episodes", "6",
        "--pretrain-bc-steps", "0",
        "--init-actor-from-sac", "artifacts/v10_pretrain_only.pt",
        "--anchor-after-pretrain",
        "--actor-lr", "8e-5",
        "--critic-lr", "3e-4",
        "--alpha-lr", "2e-4",
        "--learning-starts", "40",
        "--actor-learning-starts", "60",
        "--start-random-steps", "0",
        "--guided-explore-prob-start", "0.16",
        "--guided-explore-prob-end", "0.05",
        "--demo-sample-frac", "0.30",
        "--hazard-sample-frac", "0.40",
        "--bc-coef-start", "0.26",
        "--bc-coef-end", "0.16",
        "--anchor-coef-start", "0.30",
        "--anchor-coef-end", "0.20",
        "--medium-bc-boost", "1.0",
        "--hard-bc-scale", "0.40",
        "--medium-anchor-boost", "0.90",
        "--hard-anchor-scale", "0.25",
        "--eval-every", "20",
        "--eval-episodes", "4",
        "--best-out", "artifacts/v11_conservative_best.pt",
        "--out", "artifacts/v11_conservative_last.pt",
        "--curve-out", "artifacts/v11_conservative_curve.png",
        "--metrics-json", "artifacts/v11_conservative_metrics.json",
    ])

    eval_specs = [
        ("driver", None, "v11_grid_driver"),
        ("heuristic", None, "v11_grid_heuristic"),
        ("sac", "artifacts/v10_pretrain_only.pt", "v11_grid_pretrain"),
        ("sac", "artifacts/v11_conservative_best.pt", "v11_grid_sac"),
    ]
    for policy, model, stem in eval_specs:
        cmd = [
            PYTHON,
            "scripts/evaluate_policy_grid.py",
            "--policy", policy,
            "--episodes-per-cell", "1",
            "--csv-out", f"artifacts/{stem}_episodes.csv",
            "--summary-out", f"artifacts/{stem}_summary.csv",
            "--heatmap-prefix", f"artifacts/{stem}",
            "--json-out", f"artifacts/{stem}_overall.json",
        ]
        if model is not None:
            cmd += ["--model", model]
        run(cmd)

    run([
        PYTHON,
        "scripts/render_animation.py",
        "--seed", "262",
        "--policy", "sac",
        "--model", "artifacts/v11_conservative_best.pt",
        "--difficulty", "hard",
        "--driver-profile", "frozen",
        "--curriculum-progress", "1.0",
        "--out", "artifacts/animations_v11/hard_frozen_seed262_v11.gif",
        "--summary", "artifacts/animations_v11/hard_frozen_seed262_v11_summary.json",
        "--title", "v11 conservative SAC — hard/frozen, seed 262",
    ])

    summary = {
        "checkpoint": "artifacts/v11_conservative_best.pt",
        "note": "Use the root-level README / NOTES_v11 and the generated CSV/PNG artifacts for the rest of the comparison.",
    }
    with (ART / "run_v11_conservative_pipeline_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nDone. See artifacts/v11_policy_overall_comparison.csv and artifacts/animations_v11/index.html")


if __name__ == "__main__":
    main()
