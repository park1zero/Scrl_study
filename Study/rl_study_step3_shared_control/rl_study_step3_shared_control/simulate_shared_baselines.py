from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from automation import AutomationParams, step_automation
from driver import DriverParams, initialize_driver_state, step_driver
from geometry import Obstacle, approximate_ttc, barrier_h
from vehicle import (
    VehicleParams,
    VehicleState,
    blend_rwa_command,
    equivalent_rwa_from_sfa,
    step_vehicle,
)


def build_default_scenario() -> Tuple[VehicleParams, DriverParams, AutomationParams, Obstacle]:
    vehicle_params = VehicleParams()
    driver_params = DriverParams(
        ky=0.55,
        kpsi=1.40,
        obstacle_gain=3.5,
        burst_gain=1.8,
        burst_trigger_h=12.0,
        lpf_tau=0.12,
        perception_delay_s=0.25,
        motor_delay_s=0.15,
        detection_range=30.0,
        preferred_side=1.0,
    )
    auto_params = AutomationParams(
        ky=0.55,
        kpsi=1.30,
        kvy=0.06,
        kr=0.12,
        detection_range=35.0,
        pass_side=1.0,
        desired_clearance=2.2,
        return_decay=8.0,
        obstacle_gain=0.05,
    )
    obstacle = Obstacle(x=40.0, y=0.0, a=3.5, b=1.2, margin=0.5)
    return vehicle_params, driver_params, auto_params, obstacle


def ellipse_points(obstacle: Obstacle, num: int = 200):
    theta = np.linspace(0.0, 2.0 * np.pi, num)
    x = obstacle.x + (obstacle.a + obstacle.margin) * np.cos(theta)
    y = obstacle.y + (obstacle.b + obstacle.margin) * np.sin(theta)
    return x, y


def run_episode(mode: str, fixed_lambda: float = 0.7, horizon_s: float = 6.0) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """Run one baseline episode.

    Parameters
    ----------
    mode : {'driver', 'automation', 'shared'}
        Which baseline to simulate.
    fixed_lambda : float
        Shared-control authority. Used only when mode == 'shared'.
    horizon_s : float
        Maximum episode length [s].
    """
    if mode not in {"driver", "automation", "shared"}:
        raise ValueError(f"Unsupported mode: {mode}")

    vp, dp, ap, obstacle = build_default_scenario()
    state = VehicleState(lam=1.0 if mode == "driver" else 0.0 if mode == "automation" else fixed_lambda)
    driver_state = initialize_driver_state(dp, vp.dt)

    records: List[Dict[str, float]] = []
    collision = False
    success = False
    first_sfa_cmd_time = None
    first_drv_rwa_time = None

    for _ in range(int(horizon_s / vp.dt)):
        sfa_cmd, ddbg = step_driver(driver_state, state, obstacle, dp, vp.dt)
        delta_drv_rwa = equivalent_rwa_from_sfa(state.delta_sfa, vp)
        delta_auto_rwa, adbg = step_automation(state, obstacle, ap)

        if mode == "driver":
            lam = 1.0
        elif mode == "automation":
            lam = 0.0
        else:
            lam = fixed_lambda
        state.lam = lam

        delta_cmd = blend_rwa_command(lam, delta_drv_rwa, delta_auto_rwa)
        real_h = barrier_h(state.x, state.y, obstacle)
        ttc = approximate_ttc(state.x, vp.vx, obstacle)

        if first_sfa_cmd_time is None and abs(sfa_cmd) > 1e-3:
            first_sfa_cmd_time = state.t
        if first_drv_rwa_time is None and abs(delta_drv_rwa) > 1e-3:
            first_drv_rwa_time = state.t

        records.append(
            {
                "t": state.t,
                "x": state.x,
                "y": state.y,
                "psi": state.psi,
                "vy": state.vy,
                "r": state.r,
                "delta_sfa": state.delta_sfa,
                "delta_rwa": state.delta_rwa,
                "delta_drv_rwa": delta_drv_rwa,
                "delta_auto_rwa": delta_auto_rwa,
                "delta_cmd": delta_cmd,
                "lambda": lam,
                "real_h": real_h,
                "ttc": ttc,
                "perceived_h": ddbg["perceived_h"],
                "raw_cmd": ddbg["raw_cmd"],
                "lpf_cmd": ddbg["lpf_cmd"],
                "sfa_cmd": sfa_cmd,
                "driver_hazard": ddbg["hazard"],
                "driver_obstacle_term": ddbg["obstacle_term"],
                "driver_burst_mult": ddbg["burst_mult"],
                "auto_hazard": adbg["hazard"],
                "auto_y_ref": adbg["y_ref"],
                "auto_repulsive": adbg["repulsive"],
            }
        )

        next_state = step_vehicle(state, rwa_cmd=delta_cmd, sfa_cmd=sfa_cmd, params=vp)
        next_state.lam = lam
        state = next_state

        next_h = barrier_h(state.x, state.y, obstacle)
        if next_h < 0.0:
            collision = True
            break
        if state.x >= obstacle.x + obstacle.a + 5.0:
            success = True
            break

    min_h = min(r["real_h"] for r in records)
    max_abs_y = max(abs(r["y"]) for r in records)
    max_abs_delta_cmd = max(abs(r["delta_cmd"]) for r in records)

    summary = {
        "mode": mode,
        "fixed_lambda": 1.0 if mode == "driver" else 0.0 if mode == "automation" else fixed_lambda,
        "collision": float(collision),
        "success": float(success),
        "min_h": min_h,
        "max_abs_y": max_abs_y,
        "max_abs_delta_cmd": max_abs_delta_cmd,
        "final_x": state.x,
        "final_y": state.y,
        "first_sfa_cmd_time": -1.0 if first_sfa_cmd_time is None else first_sfa_cmd_time,
        "first_drv_rwa_time": -1.0 if first_drv_rwa_time is None else first_drv_rwa_time,
    }
    return records, summary


def save_records_csv(records: List[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def save_summary_csv(summaries: List[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


def build_comparison_plot(all_records: Dict[str, List[Dict[str, float]]], summaries: List[Dict[str, float]], obstacle: Obstacle, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ox, oy = ellipse_points(obstacle)

    fig = plt.figure(figsize=(13, 9))

    # XY path --------------------------------------------------------------
    ax1 = fig.add_subplot(2, 2, 1)
    for mode, records in all_records.items():
        ax1.plot([r["x"] for r in records], [r["y"] for r in records], label=mode)
    ax1.plot(ox, oy, linestyle="--", linewidth=1.2, label="safety ellipse")
    ax1.axhline(0.0, linestyle=":", linewidth=1.0)
    ax1.set_title("vehicle path")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # Barrier history ------------------------------------------------------
    ax2 = fig.add_subplot(2, 2, 2)
    for mode, records in all_records.items():
        ax2.plot([r["t"] for r in records], [r["real_h"] for r in records], label=mode)
    ax2.axhline(0.0, linestyle="--", linewidth=1.0)
    ax2.set_title("barrier h")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("h")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    # Command history ------------------------------------------------------
    ax3 = fig.add_subplot(2, 2, 3)
    for mode, records in all_records.items():
        ax3.plot([r["t"] for r in records], [r["delta_cmd"] for r in records], label=f"{mode} delta_cmd")
    ax3.set_title("final road-wheel command")
    ax3.set_xlabel("time [s]")
    ax3.set_ylabel("delta_cmd [rad]")
    ax3.legend(loc="best")
    ax3.grid(True, alpha=0.3)

    # Summary text ---------------------------------------------------------
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")
    lines = ["Step 3 baseline comparison", ""]
    for row in summaries:
        lines.append(
            f"{row['mode']:>10s} | lambda={row['fixed_lambda']:.2f} | success={int(row['success'])} | "
            f"collision={int(row['collision'])} | min_h={row['min_h']:.3f} | max|y|={row['max_abs_y']:.3f}"
        )
    ax4.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "artifacts"
    out_dir.mkdir(exist_ok=True)

    vp, dp, ap, obstacle = build_default_scenario()

    all_records: Dict[str, List[Dict[str, float]]] = {}
    summaries: List[Dict[str, float]] = []

    baseline_modes = [
        ("driver", 1.0),
        ("automation", 0.0),
        ("shared", 0.7),
    ]

    for mode, lam in baseline_modes:
        records, summary = run_episode(mode=mode, fixed_lambda=lam)
        all_records[mode] = records
        summaries.append(summary)
        save_records_csv(records, out_dir / f"step3_{mode}_log.csv")

    save_summary_csv(summaries, out_dir / "step3_baseline_summary.csv")
    build_comparison_plot(all_records, summaries, obstacle, out_dir / "step3_baseline_comparison.png")

    print("Saved baseline artifacts to:")
    print(f"  {out_dir}")
    for row in summaries:
        print(
            f"{row['mode']:>10s} | lambda={row['fixed_lambda']:.2f} | success={int(row['success'])} | "
            f"collision={int(row['collision'])} | min_h={row['min_h']:.3f} | max|y|={row['max_abs_y']:.3f}"
        )


if __name__ == "__main__":
    main()
