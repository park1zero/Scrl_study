
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from shared_control_rl.config import VehicleParams


@dataclass
class VehicleState:
    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0
    vy: float = 0.0
    r: float = 0.0
    delta_rwa: float = 0.0
    delta_swa: float = 0.0
    lam: float = 1.0
    t: float = 0.0

    def copy(self) -> "VehicleState":
        return replace(self)


def _apply_first_order_rate_limited(
    current: float,
    target: float,
    tau: float,
    rate_limit: float,
    dt: float,
    lower: float,
    upper: float,
) -> float:
    target = float(np.clip(target, lower, upper))
    dot = (target - current) / max(tau, 1e-6)
    dot = float(np.clip(dot, -rate_limit, rate_limit))
    next_value = current + dt * dot
    return float(np.clip(next_value, lower, upper))


def step_vehicle(
    state: VehicleState,
    rwa_cmd: float,
    swa_cmd: float,
    params: VehicleParams,
) -> VehicleState:
    """One Euler step of constant-speed bicycle dynamics with separate SBW states."""
    dt = params.dt
    vx = max(params.vx, 1.0)

    next_rwa = _apply_first_order_rate_limited(
        current=state.delta_rwa,
        target=rwa_cmd,
        tau=params.tau_rwa,
        rate_limit=params.max_rwa_rate,
        dt=dt,
        lower=-params.max_rwa,
        upper=params.max_rwa,
    )
    next_swa = _apply_first_order_rate_limited(
        current=state.delta_swa,
        target=swa_cmd,
        tau=params.tau_swa,
        rate_limit=params.max_swa_rate,
        dt=dt,
        lower=-params.max_swa,
        upper=params.max_swa,
    )

    alpha_f = (state.vy + params.lf * state.r) / vx - next_rwa
    alpha_r = (state.vy - params.lr * state.r) / vx

    fyf = -params.cf * alpha_f
    fyr = -params.cr * alpha_r

    vy_dot = (fyf + fyr) / params.m - vx * state.r
    r_dot = (params.lf * fyf - params.lr * fyr) / params.iz

    x_dot = vx * math.cos(state.psi) - state.vy * math.sin(state.psi)
    y_dot = vx * math.sin(state.psi) + state.vy * math.cos(state.psi)
    psi_dot = state.r

    return VehicleState(
        x=state.x + dt * x_dot,
        y=state.y + dt * y_dot,
        psi=state.psi + dt * psi_dot,
        vy=state.vy + dt * vy_dot,
        r=state.r + dt * r_dot,
        delta_rwa=next_rwa,
        delta_swa=next_swa,
        lam=state.lam,
        t=state.t + dt,
    )


def equivalent_rwa_from_swa(delta_swa: float, params: VehicleParams) -> float:
    return float(np.clip(delta_swa / params.steering_ratio, -params.max_rwa, params.max_rwa))


def blend_rwa_command(lam: float, delta_drv_rwa: float, delta_auto_rwa: float) -> float:
    lam = float(np.clip(lam, 0.0, 1.0))
    return lam * delta_drv_rwa + (1.0 - lam) * delta_auto_rwa
