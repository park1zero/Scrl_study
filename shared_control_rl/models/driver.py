from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import Deque, Tuple

import numpy as np

from shared_control_rl.config import DrowsyDriverConfig, VehicleParams
from shared_control_rl.models.vehicle import VehicleState
from shared_control_rl.utils.geometry import EllipseObstacle, ellipse_barrier, wrap_to_pi


class DrowsyDriverModel:
    """Drowsy driver with split perception delay, motor delay, and bursty reaction.

    Output is a *steering wheel angle* command in radians at the steering-wheel side.
    The state.delta_sfa actuator in the plant adds one more physical lag on top of the
    driver's own perception and motor delays.

    v8 change
    ---------
    Supports a short initial wrong-way reaction via `wrong_way_*`, which is useful for
    sleepy / startled driver populations.
    """

    def __init__(
        self,
        cfg: DrowsyDriverConfig,
        vehicle_params: VehicleParams,
    ) -> None:
        self.cfg = replace(cfg)
        self.params = vehicle_params

        self.perception_steps: int = 1
        self.motor_steps: int = 1
        self.burst_duration_steps: int = 1
        self.wrong_way_duration_steps: int = 0
        self.perception_buffer: Deque[Tuple[float, float, float]] = deque(maxlen=2)
        self.motor_buffer: Deque[float] = deque(maxlen=2)

        self.filtered_cmd: float = 0.0
        self.raw_cmd: float = 0.0
        self.output_cmd: float = 0.0
        self.perceived_h: float = np.inf
        self.perceived_dx: float = np.inf
        self.hazard_active: bool = False
        self.burst_timer_steps: int = 0
        self.detected_prev: bool = False
        self.wrong_way_timer_steps: int = 0
        self._update_steps()

    def _update_steps(self) -> None:
        self.perception_steps = max(1, int(round(self.cfg.perception_delay_seconds / self.params.dt)))
        self.motor_steps = max(1, int(round(self.cfg.motor_delay_seconds / self.params.dt)))
        self.burst_duration_steps = max(1, int(round(self.cfg.burst_duration_seconds / self.params.dt)))
        self.wrong_way_duration_steps = max(0, int(round(self.cfg.wrong_way_duration_seconds / self.params.dt)))
        self.perception_buffer = deque(maxlen=self.perception_steps + 1)
        self.motor_buffer = deque(maxlen=self.motor_steps + 1)

    def reset(
        self,
        initial_state: VehicleState,
        perception_delay_seconds: float | None = None,
        motor_delay_seconds: float | None = None,
        preferred_side: float | None = None,
        delay_seconds: float | None = None,
    ) -> None:
        """Reset internal driver state.

        delay_seconds is accepted for backward compatibility. If only total delay is
        provided, it is split 60/40 into perception/motor delay.
        """
        if delay_seconds is not None and perception_delay_seconds is None and motor_delay_seconds is None:
            perception_delay_seconds = 0.60 * float(delay_seconds)
            motor_delay_seconds = 0.40 * float(delay_seconds)

        if perception_delay_seconds is not None:
            self.cfg.perception_delay_seconds = float(perception_delay_seconds)
        if motor_delay_seconds is not None:
            self.cfg.motor_delay_seconds = float(motor_delay_seconds)
        if preferred_side is not None:
            self.cfg.preferred_side = float(np.sign(preferred_side) or 1.0)

        self._update_steps()
        for _ in range(self.perception_steps + 1):
            self.perception_buffer.append((initial_state.x, initial_state.y, initial_state.psi))
        for _ in range(self.motor_steps + 1):
            self.motor_buffer.append(0.0)

        self.filtered_cmd = 0.0
        self.raw_cmd = 0.0
        self.output_cmd = 0.0
        self.perceived_h = np.inf
        self.perceived_dx = np.inf
        self.hazard_active = False
        self.burst_timer_steps = 0
        self.detected_prev = False
        self.wrong_way_timer_steps = 0

    def _delayed_measurement(self) -> Tuple[float, float, float]:
        return self.perception_buffer[0]

    def _side_sign(self, y_d: float, obstacle_y: float) -> float:
        if abs(y_d - obstacle_y) < 0.15:
            return self.cfg.preferred_side
        side = np.sign(y_d - obstacle_y)
        return float(side if abs(side) > 1e-6 else self.cfg.preferred_side)

    def compute_command(
        self,
        state: VehicleState,
        ref_y: float,
        ref_psi: float,
        obstacle: EllipseObstacle,
        margin: float,
    ) -> float:
        self.perception_buffer.append((state.x, state.y, state.psi))
        x_d, y_d, psi_d = self._delayed_measurement()

        e_y = y_d - ref_y
        e_psi = wrap_to_pi(psi_d - ref_psi)

        inflated = obstacle.inflated(margin)
        dx = obstacle.x - x_d
        h_d = ellipse_barrier(x_d, y_d, inflated)
        self.perceived_h = float(h_d)
        self.perceived_dx = float(dx)

        detected = bool(0.0 < dx <= self.cfg.detection_range)
        imminent = bool(detected and h_d <= self.cfg.burst_trigger_h)
        obstacle_term = 0.0

        if imminent and not self.hazard_active:
            self.burst_timer_steps = self.burst_duration_steps
        self.hazard_active = imminent

        if detected and not self.detected_prev and self.wrong_way_duration_steps > 0:
            self.wrong_way_timer_steps = self.wrong_way_duration_steps
        self.detected_prev = detected

        if detected:
            side = self._side_sign(y_d, obstacle.y)
            proximity = float(np.clip(1.0 - dx / max(self.cfg.detection_range, 1e-6), 0.0, 1.0))
            risk = float(np.clip((self.cfg.burst_trigger_h - h_d) / max(self.cfg.burst_trigger_h, 1e-6), 0.0, 1.0))

            base_scale = 0.20 + 0.80 * proximity
            risk_scale = 1.0 + 0.60 * risk

            if self.wrong_way_timer_steps > 0:
                side *= -1.0
                risk_scale *= self.cfg.wrong_way_gain
                self.wrong_way_timer_steps -= 1

            obstacle_term = side * self.cfg.obstacle_gain * base_scale * risk_scale

            if self.burst_timer_steps > 0:
                phase = self.burst_timer_steps / max(self.burst_duration_steps, 1)
                burst_scale = 1.0 + (self.cfg.burst_gain - 1.0) * phase
                obstacle_term *= burst_scale
                self.burst_timer_steps -= 1

        raw_cmd = self.cfg.ky * e_y + self.cfg.kpsi * e_psi + obstacle_term
        alpha = self.params.dt / max(self.cfg.lpf_tau, self.params.dt)
        self.filtered_cmd = (1.0 - alpha) * self.filtered_cmd + alpha * raw_cmd
        self.filtered_cmd = float(np.clip(self.filtered_cmd, -self.cfg.max_sfa_cmd, self.cfg.max_sfa_cmd))

        self.motor_buffer.append(self.filtered_cmd)
        self.output_cmd = float(np.clip(self.motor_buffer[0], -self.cfg.max_sfa_cmd, self.cfg.max_sfa_cmd))
        self.raw_cmd = float(np.clip(raw_cmd, -self.cfg.max_sfa_cmd, self.cfg.max_sfa_cmd))
        return self.output_cmd
