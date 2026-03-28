
from __future__ import annotations

import math
from typing import Callable

import numpy as np


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def fixed_lambda_policy(value: float) -> Callable[[dict], float]:
    value = float(np.clip(value, 0.0, 1.0))

    def _policy(info: dict) -> float:
        return value

    return _policy


def heuristic_hazard_policy(info: dict) -> float:
    """More danger -> less driver authority."""
    ttc = float(info.get("ttc", np.inf))
    h = float(info.get("h", 10.0))
    delta_gap = abs(float(info.get("delta_drv_rwa", 0.0)) - float(info.get("delta_auto_rwa", 0.0)))

    ttc_term = _sigmoid(2.0 - ttc)
    h_term = _sigmoid(1.0 - h)
    conflict_term = np.clip(delta_gap / 0.25, 0.0, 1.0)

    hazard = 0.55 * ttc_term + 0.35 * h_term + 0.10 * conflict_term
    lam = 1.0 - hazard
    return float(np.clip(lam, 0.0, 1.0))
