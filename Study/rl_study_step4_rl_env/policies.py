from __future__ import annotations

from dataclasses import dataclass
import random


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class HeuristicAuthorityParams:
    """Simple handcrafted authority policy used as a baseline inside the RL env."""

    h_soft: float = 1.0
    ttc_soft: float = 1.4
    disagreement_gain: float = 0.25
    min_lambda: float = 0.05


def driver_hold_policy(obs, info, env) -> float:
    return 0.0


def full_takeover_policy(obs, info, env) -> float:
    return -1.0


def random_policy(obs, info, env, rng: random.Random) -> float:
    return rng.uniform(-1.0, 1.0)


def heuristic_authority_policy(obs, info, env, params: HeuristicAuthorityParams | None = None) -> float:
    params = params or HeuristicAuthorityParams()
    lam = float(info["lam"])
    h = float(info["h"])
    ttc = float(info["ttc"])
    dx = float(info["dx"])
    delta_drv_rwa = float(info["delta_drv_rwa"])
    delta_auto_rwa = float(info["delta_auto_rwa"])

    if dx < 0.0:
        lambda_target = 1.0
    else:
        risk_h = clip((params.h_soft - h) / max(params.h_soft, 1e-6), 0.0, 1.0)
        risk_ttc = clip((params.ttc_soft - min(ttc, params.ttc_soft)) / max(params.ttc_soft, 1e-6), 0.0, 1.0)
        disagreement = abs(delta_auto_rwa - delta_drv_rwa) / max(env.vehicle_params.max_rwa, 1e-6)
        risk = max(risk_h, risk_ttc)
        lambda_target = clip(1.0 - risk - params.disagreement_gain * risk * disagreement, params.min_lambda, 1.0)

    max_lambda_step = env.env_params.lambda_rate_max * env.vehicle_params.dt
    action = (lambda_target - lam) / max(max_lambda_step, 1e-6)
    return clip(action, -1.0, 1.0)
