
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Tuple

import numpy as np

from shared_control_rl.config import SafetyFilterConfig, VehicleParams
from shared_control_rl.models.vehicle import VehicleState, blend_rwa_command, step_vehicle
from shared_control_rl.utils.geometry import EllipseObstacle, ellipse_barrier


class LambdaSafetyFilter:
    """Discrete one-step safety projection on lambda.

    This is a starter implementation. It does not solve the analytic CBF-QP;
    it checks a scalar lambda grid and keeps the closest safe value.
    """

    def __init__(self, cfg: SafetyFilterConfig, vehicle_params: VehicleParams) -> None:
        self.cfg = replace(cfg)
        self.params = vehicle_params

    def project_lambda(
        self,
        state: VehicleState,
        lambda_des: float,
        delta_drv_rwa: float,
        delta_auto_rwa: float,
        obstacle: EllipseObstacle,
        margin: float,
    ) -> Tuple[float, Dict[str, float]]:
        inflated = obstacle.inflated(margin)
        h_now = ellipse_barrier(state.x, state.y, inflated)

        grid = np.linspace(0.0, 1.0, self.cfg.lambda_grid_points)
        grid = sorted(grid.tolist(), key=lambda lam: abs(lam - lambda_des))

        best_lam = None
        best_h_next = -np.inf
        best_delta = None

        target_h = max(self.cfg.min_h_next, (1.0 - self.cfg.cbf_gamma) * h_now - self.cfg.relax)

        for lam in grid:
            delta_cmd = blend_rwa_command(lam, delta_drv_rwa, delta_auto_rwa)
            next_state = step_vehicle(
                state=state,
                rwa_cmd=delta_cmd,
                sfa_cmd=state.delta_sfa,
                params=self.params,
            )
            h_next = ellipse_barrier(next_state.x, next_state.y, inflated)

            if h_next > best_h_next:
                best_h_next = h_next
                best_lam = lam
                best_delta = delta_cmd

            if h_next >= target_h:
                return float(lam), {
                    "h_now": float(h_now),
                    "h_next": float(h_next),
                    "target_h": float(target_h),
                    "fallback_used": 0.0,
                    "delta_cmd": float(delta_cmd),
                }

        # Fallback: choose the lambda that maximizes next-step barrier.
        assert best_lam is not None
        return float(best_lam), {
            "h_now": float(h_now),
            "h_next": float(best_h_next),
            "target_h": float(target_h),
            "fallback_used": 1.0,
            "delta_cmd": float(best_delta if best_delta is not None else 0.0),
        }
