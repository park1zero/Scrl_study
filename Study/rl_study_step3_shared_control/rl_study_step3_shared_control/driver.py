from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Deque, Dict, Tuple

from geometry import Obstacle, barrier_h


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class DriverParams:
    """Split-delay drowsy-driver model.

    Structure
    ---------
    delayed perception -> raw steering intent -> low-pass filter -> motor delay
    -> steering-wheel command

    The raw intent is composed of
    1) a lane-keeping term
    2) an obstacle-avoidance term
    3) a burst multiplier that activates when the perceived barrier gets small
    """

    lane_center_y: float = 0.0

    # Lane tracking gains
    ky: float = 0.55
    kpsi: float = 1.40

    # Obstacle avoidance behaviour
    preferred_side: float = 1.0
    detection_range: float = 30.0
    obstacle_gain: float = 3.5
    burst_gain: float = 1.8
    burst_trigger_h: float = 12.0

    # Smoothing / delays
    lpf_tau: float = 0.12
    perception_delay_s: float = 0.25
    motor_delay_s: float = 0.15

    # Steering limits / shaping
    max_sfa_cmd: float = 6.0
    side_deadzone: float = 0.05
    dx_softening: float = 10.0
    dy_softening: float = 0.80


@dataclass
class DriverState:
    perception_buffer: Deque[Dict[str, float]] = field(default_factory=deque)
    motor_buffer: Deque[float] = field(default_factory=deque)
    lpf_cmd: float = 0.0
    last_raw_cmd: float = 0.0


def initialize_driver_state(params: DriverParams, dt: float) -> DriverState:
    perception_steps = max(1, int(round(params.perception_delay_s / dt)))
    motor_steps = max(1, int(round(params.motor_delay_s / dt)))
    return DriverState(
        perception_buffer=deque(maxlen=perception_steps + 1),
        motor_buffer=deque(maxlen=motor_steps + 1),
        lpf_cmd=0.0,
        last_raw_cmd=0.0,
    )


def _fill_delay_buffer_if_needed(buffer: Deque, sample) -> None:
    while len(buffer) < buffer.maxlen:
        buffer.appendleft(sample)


def step_driver(
    driver_state: DriverState,
    vehicle_state,
    obstacle: Obstacle,
    params: DriverParams,
    dt: float,
) -> Tuple[float, Dict[str, float]]:
    """Advance the drowsy-driver model by one sample.

    Returns
    -------
    sfa_cmd : float
        Driver steering-wheel command [rad].
    debug : dict
        Intermediate variables useful for study and plotting.
    """

    # 1) Perception delay --------------------------------------------------
    snapshot = {
        "x": vehicle_state.x,
        "y": vehicle_state.y,
        "psi": vehicle_state.psi,
        "t": vehicle_state.t,
    }
    driver_state.perception_buffer.append(snapshot)
    _fill_delay_buffer_if_needed(driver_state.perception_buffer, snapshot)
    perceived = driver_state.perception_buffer[0]

    e_y = params.lane_center_y - perceived["y"]
    e_psi = -perceived["psi"]
    lane_term = params.ky * e_y + params.kpsi * e_psi

    # 2) Obstacle reaction on delayed perception --------------------------
    dx = obstacle.x - perceived["x"]
    dy = perceived["y"] - obstacle.y
    perceived_h = barrier_h(perceived["x"], perceived["y"], obstacle)

    hazard = 0.0
    obstacle_term = 0.0
    burst_mult = 1.0

    if 0.0 <= dx <= params.detection_range:
        hazard = 1.0 - dx / max(params.detection_range, 1e-6)

        if abs(dy) > params.side_deadzone:
            side = math.copysign(1.0, dy)
        else:
            side = params.preferred_side

        proximity = (params.detection_range - dx) / max(params.dx_softening + dx, 1e-6)
        lateral_amp = 1.0 / (abs(dy) + params.dy_softening)
        obstacle_term = side * params.obstacle_gain * hazard * proximity * lateral_amp

        if perceived_h <= params.burst_trigger_h:
            burst_mult = params.burst_gain

    raw_cmd = lane_term + burst_mult * obstacle_term
    driver_state.last_raw_cmd = raw_cmd

    # 3) Low-pass filter ---------------------------------------------------
    alpha = dt / max(params.lpf_tau, dt)
    alpha = clamp(alpha, 0.0, 1.0)
    driver_state.lpf_cmd = (1.0 - alpha) * driver_state.lpf_cmd + alpha * raw_cmd

    # 4) Motor delay -------------------------------------------------------
    driver_state.motor_buffer.append(driver_state.lpf_cmd)
    _fill_delay_buffer_if_needed(driver_state.motor_buffer, driver_state.lpf_cmd)
    sfa_cmd = clamp(driver_state.motor_buffer[0], -params.max_sfa_cmd, params.max_sfa_cmd)

    debug = {
        "perceived_t": perceived["t"],
        "perceived_x": perceived["x"],
        "perceived_y": perceived["y"],
        "perceived_psi": perceived["psi"],
        "dx": dx,
        "dy": dy,
        "perceived_h": perceived_h,
        "hazard": hazard,
        "lane_term": lane_term,
        "obstacle_term": obstacle_term,
        "burst_mult": burst_mult,
        "raw_cmd": raw_cmd,
        "lpf_cmd": driver_state.lpf_cmd,
        "sfa_cmd": sfa_cmd,
        "perception_steps": driver_state.perception_buffer.maxlen - 1,
        "motor_steps": driver_state.motor_buffer.maxlen - 1,
    }
    return sfa_cmd, debug
