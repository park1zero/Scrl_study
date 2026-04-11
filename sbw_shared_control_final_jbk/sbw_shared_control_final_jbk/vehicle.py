from __future__ import annotations

from dataclasses import dataclass
import math


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class VehicleParams:
    """Minimal SBW lateral vehicle model.

    The model intentionally keeps only the states that matter for the shared-control
    study:

        x, y, psi, vy, r, delta_rwa, delta_swa, lambda

    where
    - delta_rwa is the actual road-wheel angle
    - delta_swa is the steering-wheel angle on the driver/feel side
    - lambda is the authority state (kept externally fixed in this step)
    """

    m: float = 1600.0
    iz: float = 2500.0
    lf: float = 1.25
    lr: float = 1.55
    cf: float = 80000.0
    cr: float = 90000.0
    vx: float = 20.0
    dt: float = 0.05

    tau_rwa: float = 0.12
    tau_swa: float = 0.18
    steering_ratio: float = 15.0

    max_rwa: float = 0.35      # road-wheel angle [rad]
    max_swa: float = 6.0       # steering-wheel angle [rad]
    max_rwa_rate: float = 2.5  # [rad/s]
    max_swa_rate: float = 12.0 # [rad/s]


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


def equivalent_rwa_from_swa(delta_swa: float, params: VehicleParams) -> float:
    """Convert steering-wheel angle to an equivalent road-wheel angle."""
    delta_rwa_eq = delta_swa / max(params.steering_ratio, 1e-6)
    return clamp(delta_rwa_eq, -params.max_rwa, params.max_rwa)


def blend_rwa_command(lam: float, delta_drv_rwa: float, delta_auto_rwa: float) -> float:
    """Convex authority blending.

    delta_cmd = lambda * delta_drv_rwa + (1-lambda) * delta_auto_rwa
    """
    lam = clamp(lam, 0.0, 1.0)
    return lam * delta_drv_rwa + (1.0 - lam) * delta_auto_rwa


def step_vehicle(
    state: VehicleState,
    rwa_cmd: float,
    swa_cmd: float,
    params: VehicleParams,
) -> VehicleState:
    """One forward-Euler step of the minimal SBW lateral model."""
    dt = params.dt
    vx = max(params.vx, 0.1)

    # 1) Actuator dynamics -------------------------------------------------
    rwa_cmd = clamp(rwa_cmd, -params.max_rwa, params.max_rwa)
    delta_rwa_dot = (rwa_cmd - state.delta_rwa) / max(params.tau_rwa, 1e-6)
    delta_rwa_dot = clamp(delta_rwa_dot, -params.max_rwa_rate, params.max_rwa_rate)
    delta_rwa_next = clamp(state.delta_rwa + dt * delta_rwa_dot, -params.max_rwa, params.max_rwa)

    swa_cmd = clamp(swa_cmd, -params.max_swa, params.max_swa)
    delta_swa_dot = (swa_cmd - state.delta_swa) / max(params.tau_swa, 1e-6)
    delta_swa_dot = clamp(delta_swa_dot, -params.max_swa_rate, params.max_swa_rate)
    delta_swa_next = clamp(state.delta_swa + dt * delta_swa_dot, -params.max_swa, params.max_swa)

    # 2) Tire lateral forces (linear bicycle model) -----------------------
    alpha_f = delta_rwa_next - (state.vy + params.lf * state.r) / vx
    alpha_r = -(state.vy - params.lr * state.r) / vx

    fy_f = 2.0 * params.cf * alpha_f
    fy_r = 2.0 * params.cr * alpha_r

    # 3) Lateral and yaw dynamics -----------------------------------------
    vy_dot = (fy_f + fy_r) / params.m - vx * state.r
    r_dot = (params.lf * fy_f - params.lr * fy_r) / params.iz

    vy_next = state.vy + dt * vy_dot
    r_next = state.r + dt * r_dot

    # 4) Global kinematics -------------------------------------------------
    x_dot = vx * math.cos(state.psi) - state.vy * math.sin(state.psi)
    y_dot = vx * math.sin(state.psi) + state.vy * math.cos(state.psi)
    psi_next = state.psi + dt * state.r
    x_next = state.x + dt * x_dot
    y_next = state.y + dt * y_dot

    return VehicleState(
        x=x_next,
        y=y_next,
        psi=psi_next,
        vy=vy_next,
        r=r_next,
        delta_rwa=delta_rwa_next,
        delta_swa=delta_swa_next,
        lam=state.lam,
        t=state.t + dt,
    )
