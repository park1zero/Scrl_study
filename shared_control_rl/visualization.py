from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Ellipse, Polygon
import numpy as np

from shared_control_rl.config import ScenarioConfig, VehicleParams
from shared_control_rl.utils.geometry import EllipseObstacle


def _vehicle_polygon(
    x: float,
    y: float,
    psi: float,
    length: float,
    width: float,
) -> np.ndarray:
    half_l = 0.5 * length
    half_w = 0.5 * width
    body = np.array(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ],
        dtype=float,
    )
    c = math.cos(psi)
    s = math.sin(psi)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    verts = body @ rot.T
    verts[:, 0] += x
    verts[:, 1] += y
    return verts


def _event_status(
    x: float,
    h: float,
    obstacle_x: float,
    success_x_margin: float,
) -> str:
    if h < 0.0:
        return "collision"
    if x > obstacle_x + success_x_margin:
        return "success"
    return "running"


def animate_history(
    history: dict[str, list[float]],
    obstacle: EllipseObstacle,
    scenario_cfg: ScenarioConfig,
    vehicle_params: VehicleParams,
    out_path: str | Path,
    title: str = "Shared control episode",
    fps: int = 15,
    stride: int = 2,
    dpi: int = 120,
    lane_half_width: float = 3.5,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t = np.asarray(history["t"], dtype=float)
    x = np.asarray(history["x"], dtype=float)
    y = np.asarray(history["y"], dtype=float)
    psi = np.asarray(history["psi"], dtype=float)
    delta_drv = np.asarray(history["delta_drv_rwa"], dtype=float)
    delta_auto = np.asarray(history["delta_auto_rwa"], dtype=float)
    delta_cmd = np.asarray(history["delta_cmd_rwa"], dtype=float)
    lam = np.asarray(history["lambda"], dtype=float)
    lam_des = np.asarray(history["lambda_des"], dtype=float)
    h = np.asarray(history["h"], dtype=float)
    reward = np.asarray(history["reward"], dtype=float)
    driver_perceived_h = np.asarray(history.get("driver_perceived_h", np.full_like(t, np.nan)), dtype=float)
    driver_motor_cmd = np.asarray(history.get("driver_output_swa_cmd", np.full_like(t, np.nan)), dtype=float)
    driver_hazard = np.asarray(history.get("driver_hazard_active", np.zeros_like(t)), dtype=float)

    if len(t) == 0:
        raise ValueError("history is empty")

    frames = list(range(0, len(t), max(int(stride), 1)))
    if frames[-1] != len(t) - 1:
        frames.append(len(t) - 1)

    x_margin_left = 4.0
    x_margin_right = 10.0
    x_min = min(-1.0, float(np.min(x)) - x_margin_left)
    x_max = max(float(np.max(x)) + 2.0, obstacle.x + scenario_cfg.success_x_margin + x_margin_right)
    y_span = max(
        lane_half_width + 0.8,
        float(np.max(np.abs(y))) + 1.2,
        abs(obstacle.y) + obstacle.b + scenario_cfg.obstacle_margin + 1.2,
    )
    y_lim = max(4.0, y_span)

    vehicle_length = vehicle_params.wheelbase + 1.6
    vehicle_width = 1.82

    fig = plt.figure(figsize=(12, 7.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.55, 1.0], height_ratios=[1.0, 1.0, 1.0])
    ax_scene = fig.add_subplot(gs[:, 0])
    ax_lambda = fig.add_subplot(gs[0, 1])
    ax_delta = fig.add_subplot(gs[1, 1])
    ax_h = fig.add_subplot(gs[2, 1])

    fig.suptitle(title)

    ax_scene.set_xlim(x_min, x_max)
    ax_scene.set_ylim(-y_lim, y_lim)
    ax_scene.set_xlabel("X [m]")
    ax_scene.set_ylabel("Y [m]")
    ax_scene.grid(True)
    ax_scene.axhline(0.0, linestyle="--", linewidth=1.0)
    ax_scene.axhline(lane_half_width, linewidth=1.0)
    ax_scene.axhline(-lane_half_width, linewidth=1.0)

    obstacle_patch = Ellipse((obstacle.x, obstacle.y), width=2.0 * obstacle.a, height=2.0 * obstacle.b, fill=False, linewidth=2.0)
    inflated_patch = Ellipse(
        (obstacle.x, obstacle.y),
        width=2.0 * (obstacle.a + scenario_cfg.obstacle_margin),
        height=2.0 * (obstacle.b + scenario_cfg.obstacle_margin),
        fill=False,
        linestyle="--",
        linewidth=1.8,
    )
    ax_scene.add_patch(obstacle_patch)
    ax_scene.add_patch(inflated_patch)

    path_line, = ax_scene.plot([], [], linewidth=2.0, label="vehicle path")
    ego_line, = ax_scene.plot([], [], linewidth=1.2, linestyle=":", label="ego trace")
    vehicle_patch = Polygon(_vehicle_polygon(x[0], y[0], psi[0], vehicle_length, vehicle_width), closed=True, alpha=0.30)
    ax_scene.add_patch(vehicle_patch)
    current_point, = ax_scene.plot([x[0]], [y[0]], marker="o")
    ax_scene.legend(loc="upper right")
    status_text = ax_scene.text(
        0.02,
        0.98,
        "",
        transform=ax_scene.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )

    ax_lambda.set_xlim(float(t[0]), float(t[-1]) + 1e-6)
    ax_lambda.set_ylim(-0.05, 1.05)
    ax_lambda.set_xlabel("time [s]")
    ax_lambda.set_ylabel("authority")
    ax_lambda.grid(True)
    ax_lambda.plot(t, lam, alpha=0.20)
    ax_lambda.plot(t, lam_des, alpha=0.20, linestyle="--")
    lam_line, = ax_lambda.plot([], [], linewidth=2.0, label="lambda_safe")
    lam_des_line, = ax_lambda.plot([], [], linewidth=1.6, linestyle="--", label="lambda_des")
    lam_marker, = ax_lambda.plot([], [], marker="o")
    ax_lambda.legend(loc="lower left")

    delta_limit = 1.1 * max(vehicle_params.max_rwa, float(np.max(np.abs(delta_cmd))) + 1e-6)
    ax_delta.set_xlim(float(t[0]), float(t[-1]) + 1e-6)
    ax_delta.set_ylim(-delta_limit, delta_limit)
    ax_delta.set_xlabel("time [s]")
    ax_delta.set_ylabel("RWA [rad]")
    ax_delta.grid(True)
    ax_delta.plot(t, delta_drv, alpha=0.20)
    ax_delta.plot(t, delta_auto, alpha=0.20)
    ax_delta.plot(t, delta_cmd, alpha=0.20)
    drv_line, = ax_delta.plot([], [], linewidth=1.6, label="driver equiv RWA")
    auto_line, = ax_delta.plot([], [], linewidth=1.6, label="automation RWA")
    cmd_line, = ax_delta.plot([], [], linewidth=2.0, label="commanded RWA")
    delta_marker, = ax_delta.plot([], [], marker="o")
    ax_delta.legend(loc="upper left")

    h_min = float(min(np.min(h), -0.2))
    h_max = float(max(np.max(h), 1.5))
    ax_h.set_xlim(float(t[0]), float(t[-1]) + 1e-6)
    ax_h.set_ylim(h_min - 0.1, h_max + 0.1)
    ax_h.set_xlabel("time [s]")
    ax_h.set_ylabel("barrier h")
    ax_h.grid(True)
    ax_h.axhline(0.0, linestyle="--", linewidth=1.0)
    ax_h.plot(t, h, alpha=0.20)
    h_line, = ax_h.plot([], [], linewidth=2.0, label="h")
    h_marker, = ax_h.plot([], [], marker="o")
    reward_axis = ax_h.twinx()
    reward_axis.set_ylabel("reward")
    reward_lim = float(max(np.max(np.abs(reward)), 1.0))
    reward_axis.set_ylim(-1.05 * reward_lim, 1.05 * reward_lim)
    reward_axis.plot(t, reward, alpha=0.12, linestyle=":")
    reward_line, = reward_axis.plot([], [], linestyle=":", linewidth=1.3, label="reward")
    reward_marker, = reward_axis.plot([], [], marker="o", markersize=4)

    fig.tight_layout()

    def _update(frame_idx: int):
        j = int(frame_idx)

        path_line.set_data(x[: j + 1], y[: j + 1])
        ego_line.set_data(x[max(0, j - 8): j + 1], y[max(0, j - 8): j + 1])
        current_point.set_data([x[j]], [y[j]])
        vehicle_patch.set_xy(_vehicle_polygon(x[j], y[j], psi[j], vehicle_length, vehicle_width))

        lam_line.set_data(t[: j + 1], lam[: j + 1])
        lam_des_line.set_data(t[: j + 1], lam_des[: j + 1])
        lam_marker.set_data([t[j]], [lam[j]])

        drv_line.set_data(t[: j + 1], delta_drv[: j + 1])
        auto_line.set_data(t[: j + 1], delta_auto[: j + 1])
        cmd_line.set_data(t[: j + 1], delta_cmd[: j + 1])
        delta_marker.set_data([t[j]], [delta_cmd[j]])

        h_line.set_data(t[: j + 1], h[: j + 1])
        h_marker.set_data([t[j]], [h[j]])
        reward_line.set_data(t[: j + 1], reward[: j + 1])
        reward_marker.set_data([t[j]], [reward[j]])

        status = _event_status(x[j], h[j], obstacle.x, scenario_cfg.success_x_margin)
        status_text.set_text(
            "\n".join(
                [
                    f"t = {t[j]:.2f} s",
                    f"x = {x[j]:.1f} m, y = {y[j]:.2f} m",
                    f"lambda = {lam[j]:.2f}",
                    f"h = {h[j]:.3f}, drv_h = {driver_perceived_h[j]:.2f}",
                    f"drv_out = {driver_motor_cmd[j]:.2f} rad SWA",
                    f"burst = {'on' if driver_hazard[j] > 0.5 else 'off'}",
                    f"status = {status}",
                ]
            )
        )

        return (
            path_line,
            ego_line,
            current_point,
            vehicle_patch,
            lam_line,
            lam_des_line,
            lam_marker,
            drv_line,
            auto_line,
            cmd_line,
            delta_marker,
            h_line,
            h_marker,
            reward_line,
            reward_marker,
            status_text,
        )

    ani = animation.FuncAnimation(fig, _update, frames=frames, interval=1000 / max(fps, 1), blit=False)

    suffix = out_path.suffix.lower()
    if suffix == ".gif":
        writer = animation.PillowWriter(fps=fps)
    elif suffix == ".mp4":
        writer = animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=1800)
    else:
        raise ValueError(f"Unsupported animation format: {out_path.suffix}. Use .gif or .mp4")

    ani.save(str(out_path), writer=writer, dpi=dpi)
    plt.close(fig)
    return out_path
