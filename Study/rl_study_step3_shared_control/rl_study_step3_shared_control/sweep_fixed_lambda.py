from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from simulate_shared_baselines import run_episode


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "artifacts"
    out_dir.mkdir(exist_ok=True)

    lambdas = [round(0.1 * i, 1) for i in range(11)]
    rows = []
    for lam in lambdas:
        records, summary = run_episode(mode="shared", fixed_lambda=lam)
        summary = dict(summary)
        summary["mode"] = "shared"
        rows.append(summary)

    csv_path = out_dir / "step3_lambda_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig = plt.figure(figsize=(11, 7))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot([r["fixed_lambda"] for r in rows], [r["success"] for r in rows], marker="o", label="success")
    ax1.plot([r["fixed_lambda"] for r in rows], [r["collision"] for r in rows], marker="s", label="collision")
    ax1.set_ylabel("event flag")
    ax1.set_title("Step 3: fixed-lambda sweep")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot([r["fixed_lambda"] for r in rows], [r["min_h"] for r in rows], marker="o", label="min h")
    ax2.plot([r["fixed_lambda"] for r in rows], [r["max_abs_y"] for r in rows], marker="s", label="max |y|")
    ax2.set_xlabel("fixed lambda")
    ax2.set_ylabel("metric")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    png_path = out_dir / "step3_lambda_sweep.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {csv_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
