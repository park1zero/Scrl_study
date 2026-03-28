#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv
import json

from shared_control_rl.config import EnvConfig
from shared_control_rl.envs.shared_control_env import SharedControlEnv
from shared_control_rl.metrics import summarize_history
from shared_control_rl.visualization import animate_history
from scripts.evaluate_policy import run_policy


def _write_html(index_path: Path, rows: list[dict[str, str]]) -> None:
    cards = []
    for row in rows:
        cards.append(
            f"""
            <div class=\"card\">
              <h2>{row['policy']}</h2>
              <p>{row['status']}</p>
              <img src=\"{row['file']}\" alt=\"{row['policy']} animation\" />
              <pre>{row['metrics']}</pre>
            </div>
            """.strip()
        )
    html = f"""
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>SBW shared control baseline animations</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 20px; }}
    .card {{ border: 1px solid #ccc; border-radius: 12px; padding: 16px; }}
    img {{ width: 100%; height: auto; border-radius: 8px; }}
    pre {{ white-space: pre-wrap; background: #f8f8f8; padding: 12px; border-radius: 8px; }}
  </style>
</head>
<body>
  <h1>SBW shared control baseline animations</h1>
  <p>Deterministic obstacle-avoidance scenario with the drowsy-driver model.</p>
  <div class=\"grid\">
    {''.join(cards)}
  </div>
</body>
</html>
""".strip()
    index_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render baseline policies to GIFs and build an HTML index.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--out-dir", type=str, default="artifacts/animations")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = EnvConfig(seed=args.seed)
    policies = ["driver", "automation", "shared", "heuristic"]
    summary_rows: list[dict[str, str]] = []
    csv_rows: list[dict[str, float | str]] = []

    for policy in policies:
        env = SharedControlEnv(config)
        history = run_policy(env, policy_name=policy)
        info = {
            "collision": bool(history["h"][-1] < 0.0),
            "success": bool(history["x"][-1] > env.obstacle.x + env.scenario_cfg.success_x_margin and history["h"][-1] >= 0.0),
        }
        gif_name = f"{policy}.gif"
        out_path = out_dir / gif_name
        animate_history(
            history=history,
            obstacle=env.obstacle,
            scenario_cfg=env.scenario_cfg,
            vehicle_params=env.vehicle_params,
            out_path=out_path,
            title=f"Baseline: {policy}",
            fps=args.fps,
            stride=args.stride,
            dpi=args.dpi,
        )
        metrics = summarize_history(history, info)
        metrics_text = "\n".join(f"{k}: {metrics[k]:.4f}" for k in sorted(metrics))
        status = "collision" if metrics.get("collision", 0.0) >= 0.5 else "success"
        summary_rows.append(
            {
                "policy": policy,
                "status": status,
                "file": gif_name,
                "metrics": metrics_text,
            }
        )
        csv_row = {"policy": policy}
        csv_row.update(metrics)
        csv_rows.append(csv_row)
        with (out_dir / f"{policy}_summary.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved {out_path}")

    index_path = out_dir / "index.html"
    _write_html(index_path, summary_rows)

    csv_path = out_dir / "baseline_animation_summary.csv"
    fieldnames = list(csv_rows[0].keys()) if csv_rows else ["policy"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Saved HTML index to {index_path}")
    print(f"Saved summary CSV to {csv_path}")


if __name__ == "__main__":
    main()
