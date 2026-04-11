from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Deque, Dict, Tuple

from geometry import Obstacle, barrier_h, signed_side


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class JBKDriverParams:
    """JBK-style preview driver with drowsiness overlay.

    This is a structured driver model built from three layers:

    1) nominal preview/cybernetic steering core
       alpha_1 -> G1 (compensation / lead-lag)
       alpha_2 -> G2 (predictive / far-preview)
       [alpha_1 G1 + alpha_2 G2] G3 G4

    2) drowsiness degradation
       - delayed perception
       - reduced attention / steering gain
       - motor delay

    3) hazard burst mode
       - late repulsive steering when an obstacle is perceived late
       - stronger steering when the perceived barrier becomes small

    The implementation is intentionally pragmatic. It follows the structure of the
    two-point preview model used in the paper, but adapts the far-preview term for
    the straight-road obstacle-avoidance study code.
    """

    lane_center_y: float = 0.0

    # JBK/two-point preview structure
    near_preview_m: float = 2.0
    far_preview_m: float = 6.34
    K1: float = 30.0
    K2: float = 5.0
    K3: float = 1.0
    T1: float = 3.0
    T2: float = 1.0
    T3: float = 0.10

    # Drowsiness overlay
    perception_delay_s: float = 0.22
    motor_delay_s: float = 0.10
    attention_scale: float = 0.62
    microsleep_phase_s: float = 1.4
    microsleep_duty: float = 0.16
    microsleep_depth: float = 0.35

    # Hazard burst layer
    preferred_side: float = 1.0
    detection_range: float = 30.0
    hazard_onset: float = 0.35
    obstacle_gain: float = 2.00
    burst_gain: float = 2.20
    burst_trigger_h: float = 8.0
    desired_clearance: float = 1.8
    side_deadzone: float = 0.05
    dx_softening: float = 8.0
    dy_softening: float = 0.85

    max_swa_cmd: float = 6.0


@dataclass
class JBKDriverState:
    perception_buffer: Deque[Dict[str, float]] = field(default_factory=deque)
    motor_buffer: Deque[float] = field(default_factory=deque)
    g1_state: float = 0.0
    neuromuscular_state: float = 0.0
    prev_alpha1: float = 0.0
    prev_alpha2: float = 0.0
    last_core_cmd: float = 0.0
    last_hazard_cmd: float = 0.0
    last_raw_cmd: float = 0.0



def initialize_jbk_driver_state(params: JBKDriverParams, dt: float) -> JBKDriverState:
    perception_steps = max(1, int(round(params.perception_delay_s / dt)))
    motor_steps = max(1, int(round(params.motor_delay_s / dt)))
    return JBKDriverState(
        perception_buffer=deque(maxlen=perception_steps + 1),
        motor_buffer=deque(maxlen=motor_steps + 1),
    )



def _fill_delay_buffer_if_needed(buffer: Deque, sample) -> None:
    while len(buffer) < buffer.maxlen:
        buffer.appendleft(sample)



def _attention_gate(t: float, params: JBKDriverParams) -> float:
    """Periodic attention drop to mimic short microsleep / inattention moments."""
    if params.microsleep_phase_s <= 1e-6 or params.microsleep_duty <= 1e-6:
        return 1.0
    phase = (t % params.microsleep_phase_s) / params.microsleep_phase_s
    if phase < params.microsleep_duty:
        return max(0.0, 1.0 - params.microsleep_depth)
    return 1.0



def _compute_preview_angles(perceived: Dict[str, float], params: JBKDriverParams, vx: float, y_ref: float) -> Tuple[float, float, float]:
    beta = math.atan2(perceived['vy'], max(vx, 0.1))
    e_near = y_ref - perceived['y']
    alpha1 = e_near / max(params.near_preview_m, 1e-6) + beta

    # Straight-road adaptation of the far-preview term:
    # use the direction from the delayed position to a far preview point on the
    # desired path, measured relative to current heading, then add slip angle.
    alpha2 = math.atan2(y_ref - perceived['y'], max(params.far_preview_m, 1e-6)) - perceived['psi'] + beta
    return alpha1, alpha2, beta



def step_jbk_driver(driver_state: JBKDriverState, vehicle_state, obstacle: Obstacle, params: JBKDriverParams, dt: float, vx: float = 20.0) -> Tuple[float, Dict[str, float]]:
    snapshot = {
        'x': vehicle_state.x,
        'y': vehicle_state.y,
        'psi': vehicle_state.psi,
        'vy': vehicle_state.vy,
        'r': vehicle_state.r,
        't': vehicle_state.t,
    }
    driver_state.perception_buffer.append(snapshot)
    _fill_delay_buffer_if_needed(driver_state.perception_buffer, snapshot)
    perceived = driver_state.perception_buffer[0]

    # ------------------------------------------------------------------
    # 1) JBK-style nominal preview steering core
    # ------------------------------------------------------------------
    y_ref_core = params.lane_center_y
    alpha1, alpha2, beta = _compute_preview_angles(perceived, params, vx=vx, y_ref=y_ref_core)
    alpha1_dot = (alpha1 - driver_state.prev_alpha1) / max(dt, 1e-6)
    driver_state.prev_alpha1 = alpha1
    driver_state.prev_alpha2 = alpha2

    lead_signal = alpha1 + params.T1 * alpha1_dot
    driver_state.g1_state += dt * (lead_signal - driver_state.g1_state) / max(params.T2, 1e-6)
    compensation = (params.K1 / max(vx, 0.5)) * driver_state.g1_state
    prediction = params.K2 * alpha2

    attention = params.attention_scale * _attention_gate(perceived['t'], params)
    core_cmd = attention * (compensation + prediction)
    driver_state.last_core_cmd = core_cmd

    # ------------------------------------------------------------------
    # 2) Late hazard burst overlay
    # ------------------------------------------------------------------
    dx = obstacle.x - perceived['x']
    dy = perceived['y'] - obstacle.y
    perceived_h = barrier_h(perceived['x'], perceived['y'], obstacle)

    hazard = 0.0
    hazard_gate = 0.0
    hazard_cmd = 0.0
    burst_mult = 1.0
    y_ref_hazard = 0.0

    if 0.0 <= dx <= params.detection_range:
        hazard = 1.0 - dx / max(params.detection_range, 1e-6)
        hazard_gate = clamp((hazard - params.hazard_onset) / max(1.0 - params.hazard_onset, 1e-6), 0.0, 1.0)
        side = signed_side(perceived['y'], obstacle, deadzone=params.side_deadzone, preferred=params.preferred_side)
        proximity = (params.detection_range - dx) / max(params.dx_softening + dx, 1e-6)
        lateral_amp = 1.0 / (abs(dy) + params.dy_softening)
        y_ref_hazard = side * params.desired_clearance * hazard_gate
        hazard_cmd = side * params.obstacle_gain * hazard_gate * hazard * proximity * lateral_amp
        if perceived_h <= params.burst_trigger_h:
            burst_mult = params.burst_gain
            hazard_cmd *= burst_mult

    driver_state.last_hazard_cmd = hazard_cmd
    raw_cmd = core_cmd + hazard_cmd
    driver_state.last_raw_cmd = raw_cmd

    # ------------------------------------------------------------------
    # 3) Neuromuscular lag G3 and motor delay
    # ------------------------------------------------------------------
    driver_state.neuromuscular_state += dt * (raw_cmd - driver_state.neuromuscular_state) / max(params.T3, 1e-6)
    predelay_cmd = params.K3 * driver_state.neuromuscular_state

    driver_state.motor_buffer.append(predelay_cmd)
    _fill_delay_buffer_if_needed(driver_state.motor_buffer, predelay_cmd)
    swa_cmd = clamp(driver_state.motor_buffer[0], -params.max_swa_cmd, params.max_swa_cmd)

    debug = {
        'model': 'jbk',
        'perceived_t': perceived['t'],
        'perceived_x': perceived['x'],
        'perceived_y': perceived['y'],
        'perceived_psi': perceived['psi'],
        'perceived_vy': perceived['vy'],
        'perceived_r': perceived['r'],
        'dx': dx,
        'dy': dy,
        'perceived_h': perceived_h,
        'beta': beta,
        'alpha1': alpha1,
        'alpha2': alpha2,
        'compensation': compensation,
        'prediction': prediction,
        'attention': attention,
        'core_cmd': core_cmd,
        'hazard': hazard,
        'hazard_gate': hazard_gate,
        'y_ref_hazard': y_ref_hazard,
        'hazard_cmd': hazard_cmd,
        'burst_mult': burst_mult,
        'raw_cmd': raw_cmd,
        'neuromuscular_cmd': driver_state.neuromuscular_state,
        'predelay_cmd': predelay_cmd,
        'swa_cmd': swa_cmd,
        'perception_steps': driver_state.perception_buffer.maxlen - 1,
        'motor_steps': driver_state.motor_buffer.maxlen - 1,
    }
    return swa_cmd, debug
