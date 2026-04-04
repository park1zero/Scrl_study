from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from driver import DriverParams, initialize_driver_state, step_driver
from geometry import Obstacle, barrier_h
from vehicle import (
    VehicleParams,
    VehicleState,
    equivalent_rwa_from_sfa,
    step_vehicle,
)


def run_driver_only_simulation() -> Path:
    params = VehicleParams()
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
    obstacle = Obstacle(x=40.0, y=0.0, a=3.5, b=1.2, margin=0.5)

    state = VehicleState()
    driver_state = initialize_driver_state(driver_params, params.dt)

    records = []
    first_sfa_cmd_time = None
    first_rwa_reaction_time = None
    collision_time = None

    horizon_steps = int(6.0 / params.dt)

    for _ in range(horizon_steps):
        sfa_cmd, debug = step_driver(driver_state, state, obstacle, driver_params, params.dt)

        # Driver-only case: the current steering-wheel state is converted
        # to an equivalent road-wheel command.
        delta_drv_rwa = equivalent_rwa_from_sfa(state.delta_sfa, params)
        delta_cmd = delta_drv_rwa

        real_h = barrier_h(state.x, state.y, obstacle)

        if first_sfa_cmd_time is None and abs(sfa_cmd) > 1e-3:
            first_sfa_cmd_time = state.t
        if first_rwa_reaction_time is None and abs(delta_drv_rwa) > 1e-3:
            first_rwa_reaction_time = state.t

        records.append(
            {
                "t": state.t,
                "x": state.x,
                "y": state.y,
                "psi": state.psi,
                "vy": state.vy,
                "r": state.r,
                "delta_sfa": state.delta_sfa,
                "delta_drv_rwa": delta_drv_rwa,
                "sfa_cmd": sfa_cmd,
                "delta_cmd": delta_cmd,
                "real_h": real_h,
                "perceived_x": debug["perceived_x"],
                "perceived_y": debug["perceived_y"],
                "perceived_h": debug["perceived_h"],
                "dx_delayed": debug["dx"],
                "hazard": debug["hazard"],
                "lane_term": debug["lane_term"],
                "obstacle_term": debug["obstacle_term"],
                "burst_mult": debug["burst_mult"],
                "raw_cmd": debug["raw_cmd"],
                "lpf_cmd": debug["lpf_cmd"],
            }
        )

        next_state = step_vehicle(state, rwa_cmd=delta_cmd, sfa_cmd=sfa_cmd, params=params)
        next_real_h = barrier_h(next_state.x, next_state.y, obstacle)

        state = next_state

        if next_real_h < 0.0:
            collision_time = state.t
            break

    out_dir = Path(__file__).resolve().parent / "artifacts"
    out_dir.mkdir(exist_ok=True)

    csv_path = out_dir / "driver_step2_log.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    times = [row["t"] for row in records]
    ys = [row["y"] for row in records]
    real_hs = [row["real_h"] for row in records]
    perceived_hs = [row["perceived_h"] for row in records]
    sfa_cmds = [row["sfa_cmd"] for row in records]
    delta_sfas = [row["delta_sfa"] for row in records]
    delta_drv_rwas = [row["delta_drv_rwa"] for row in records]
    raw_cmds = [row["raw_cmd"] for row in records]
    lpf_cmds = [row["lpf_cmd"] for row in records]

    fig = plt.figure(figsize=(11, 8))

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(times, ys, label="y [m]")
    ax1.axhline(0.0, linestyle="--", linewidth=1.0, label="lane centre")
    ax1.set_ylabel("lateral position")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(times, real_hs, label="real h")
    ax2.plot(times, perceived_hs, label="perceived h")
    ax2.axhline(0.0, linestyle="--", linewidth=1.0, label="collision boundary")
    ax2.set_ylabel("barrier")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(times, raw_cmds, label="raw cmd")
    ax3.plot(times, lpf_cmds, label="LPF cmd")
    ax3.plot(times, sfa_cmds, label="sfa cmd")
    ax3.plot(times, delta_sfas, label="delta_sfa")
    ax3.plot(times, delta_drv_rwas, label="delta_drv_rwa")
    ax3.set_xlabel("time [s]")
    ax3.set_ylabel("steering / command")
    ax3.legend(loc="best", ncol=3)
    ax3.grid(True, alpha=0.3)

    title_bits = []
    if first_sfa_cmd_time is not None:
        title_bits.append(f"first sfa cmd = {first_sfa_cmd_time:.2f}s")
    if first_rwa_reaction_time is not None:
        title_bits.append(f"first RWA reaction = {first_rwa_reaction_time:.2f}s")
    if collision_time is not None:
        title_bits.append(f"collision = {collision_time:.2f}s")
    fig.suptitle("Step 2: drowsy-driver-only response | " + ", ".join(title_bits))
    fig.tight_layout()

    png_path = out_dir / "driver_step2_preview.png"
    fig.savefig(png_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(f"  CSV : {csv_path}")
    print(f"  PNG : {png_path}")
    print()
    print("Quick summary")
    print("-------------")
    print(f"first steering-wheel command time : {first_sfa_cmd_time}")
    print(f"first equivalent road-wheel reaction time : {first_rwa_reaction_time}")
    print(f"collision time : {collision_time}")
    print(f"final y before stop : {records[-1]['y']:.4f} m")
    print(f"final delta_sfa : {records[-1]['delta_sfa']:.4f} rad")
    print(f"final delta_drv_rwa : {records[-1]['delta_drv_rwa']:.4f} rad")

    return png_path


if __name__ == "__main__":
    run_driver_only_simulation()
