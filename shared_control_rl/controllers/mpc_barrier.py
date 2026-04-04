
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from shared_control_rl.config import ShootingMPCConfig, VehicleParams
from shared_control_rl.models.vehicle import VehicleState, step_vehicle
from shared_control_rl.utils.geometry import EllipseObstacle, avoidance_target_y, ellipse_barrier, wrap_to_pi


class ShootingBarrierMPC:
    """Very small shooting-based MPC surrogate.

    It evaluates piecewise-constant RWA command sequences over a short horizon,
    using a hand-written cost that combines tracking and a log barrier term.
    """

    def __init__(self, cfg: ShootingMPCConfig, vehicle_params: VehicleParams) -> None:
        self.cfg = replace(cfg)
        self.params = vehicle_params
        self.last_cmd: float = 0.0

    def reset(self) -> None:
        self.last_cmd = 0.0

    def _candidate_cost(
        self,
        initial_state: VehicleState,
        u0: float,
        u1: float,
        obstacle: EllipseObstacle,
        side: float,
        margin: float,
    ) -> float:
        sim_state = initial_state.copy()
        cost = 0.0
        prev_u = self.last_cmd
        inflated = obstacle.inflated(margin)

        for k in range(self.cfg.horizon_steps):
            u = u0 if k < self.cfg.segment_split else u1
            sim_state = step_vehicle(
                sim_state,
                rwa_cmd=u,
                sfa_cmd=sim_state.delta_sfa,
                params=self.params,
            )

            target_y = avoidance_target_y(
                x=sim_state.x,
                obstacle=obstacle,
                side=side,
                clearance=self.cfg.lateral_clearance,
                pre_turn_distance=self.cfg.pre_turn_distance,
                post_turn_distance=self.cfg.post_turn_distance,
            )
            e_y = sim_state.y - target_y
            e_psi = wrap_to_pi(sim_state.psi - 0.0)

            cost += self.cfg.wy * (e_y ** 2)
            cost += self.cfg.wpsi * (e_psi ** 2)
            cost += self.cfg.wvy * (sim_state.vy ** 2)
            cost += self.cfg.wr * (sim_state.r ** 2)
            cost += self.cfg.wu * (u ** 2)
            cost += self.cfg.wdu * ((u - prev_u) ** 2)

            h = ellipse_barrier(sim_state.x, sim_state.y, inflated)
            if h <= self.cfg.barrier_eps:
                cost += self.cfg.hard_barrier_penalty * (1.0 + abs(self.cfg.barrier_eps - h))
            else:
                cost += -self.cfg.w_barrier * math.log(h + self.cfg.barrier_eps)

            prev_u = u

        return cost

    def compute_command(
        self,
        state: VehicleState,
        obstacle: EllipseObstacle,
        side: float,
        margin: float,
    ) -> float:
        u0_candidates = np.linspace(-self.params.max_rwa, self.params.max_rwa, self.cfg.u0_grid)
        u1_candidates = np.linspace(-self.params.max_rwa, self.params.max_rwa, self.cfg.u1_grid)

        best_cost = np.inf
        best_first_cmd = 0.0

        for u0 in u0_candidates:
            for u1 in u1_candidates:
                candidate_cost = self._candidate_cost(
                    initial_state=state,
                    u0=float(u0),
                    u1=float(u1),
                    obstacle=obstacle,
                    side=side,
                    margin=margin,
                )
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_first_cmd = float(u0)

        self.last_cmd = float(best_first_cmd)
        return self.last_cmd
