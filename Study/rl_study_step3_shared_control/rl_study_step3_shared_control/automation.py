from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Tuple

from geometry import Obstacle


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class AutomationParams:
    """Simple automation steering law for the study baseline.

    This is intentionally *not* a full MPC yet. It is a lightweight controller that
    is easy to understand before moving to the RL environment. Its job is just to
    create a plausible obstacle-avoidance road-wheel command.

    The controller has two ideas:
    1) create a temporary lateral offset reference while approaching the obstacle
    2) add a small repulsive steering term as the obstacle gets closer

    Resulting control law
    ---------------------
    delta_auto = K_y (y_ref - y) + K_psi (-psi) - K_vy vy - K_r r + u_rep
    """

    ky: float = 0.55
    kpsi: float = 1.30
    kvy: float = 0.06
    kr: float = 0.12

    detection_range: float = 35.0
    pass_side: float = 1.0
    desired_clearance: float = 2.2
    return_decay: float = 8.0

    obstacle_gain: float = 0.05
    dx_softening: float = 8.0
    dy_softening: float = 0.8

    max_rwa_cmd: float = 0.35


def step_automation(vehicle_state, obstacle: Obstacle, params: AutomationParams) -> Tuple[float, Dict[str, float]]:
    """Compute a simple automation road-wheel command.

    Returns
    -------
    delta_auto_rwa : float
        Automation steering command on the road-wheel channel [rad].
    debug : dict
        Intermediate variables for plotting / interpretation.
    """
    dx = obstacle.x - vehicle_state.x
    dy = vehicle_state.y - obstacle.y

    hazard = 0.0
    y_ref = 0.0
    repulsive = 0.0

    if 0.0 <= dx <= params.detection_range:
        # Approaching obstacle: smoothly build a side offset reference.
        hazard = 1.0 - dx / max(params.detection_range, 1e-6)
        y_ref = params.pass_side * params.desired_clearance * hazard

        proximity = (params.detection_range - dx) / max(params.dx_softening + dx, 1e-6)
        lateral_amp = 1.0 / (abs(dy) + params.dy_softening)
        repulsive = params.pass_side * params.obstacle_gain * hazard * proximity * lateral_amp
    elif dx < 0.0:
        # After passing the obstacle: decay the side offset back to zero.
        y_ref = params.pass_side * params.desired_clearance * math.exp(dx / max(params.return_decay, 1e-6))

    e_y = y_ref - vehicle_state.y
    e_psi = -vehicle_state.psi

    delta_auto = (
        params.ky * e_y
        + params.kpsi * e_psi
        - params.kvy * vehicle_state.vy
        - params.kr * vehicle_state.r
        + repulsive
    )
    delta_auto = clamp(delta_auto, -params.max_rwa_cmd, params.max_rwa_cmd)

    debug = {
        "dx": dx,
        "dy": dy,
        "hazard": hazard,
        "y_ref": y_ref,
        "repulsive": repulsive,
        "delta_auto": delta_auto,
    }
    return delta_auto, debug
