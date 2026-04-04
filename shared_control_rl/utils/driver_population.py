from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from shared_control_rl.config import DrowsyDriverConfig, DriverPopulationConfig


PROFILE_ORDER = ("late_linear", "late_aggressive", "frozen", "wrong_initial")


def _normalize(weights: Sequence[float]) -> np.ndarray:
    arr = np.asarray(weights, dtype=float)
    arr = np.clip(arr, 0.0, None)
    if arr.sum() <= 0.0:
        arr = np.ones_like(arr, dtype=float)
    arr = arr / arr.sum()
    return arr


def profile_weights(pop_cfg: DriverPopulationConfig, difficulty: str) -> np.ndarray:
    if difficulty == "easy":
        return _normalize(pop_cfg.easy_weights)
    if difficulty == "hard":
        return _normalize(pop_cfg.hard_weights)
    return _normalize(pop_cfg.medium_weights)


def sample_profile_name(rng: np.random.Generator, pop_cfg: DriverPopulationConfig, difficulty: str) -> str:
    names = pop_cfg.profile_names or PROFILE_ORDER
    probs = profile_weights(pop_cfg, difficulty)
    if len(probs) != len(names):
        probs = np.ones(len(names), dtype=float) / max(len(names), 1)
    return str(rng.choice(np.asarray(names, dtype=object), p=probs))


def sample_driver_config(
    rng: np.random.Generator,
    base_cfg: DrowsyDriverConfig,
    profile_name: str,
    avoid_side: float,
) -> DrowsyDriverConfig:
    cfg = replace(base_cfg)
    cfg.profile_name = str(profile_name)
    cfg.preferred_side = float(np.sign(avoid_side) or 1.0)
    cfg.wrong_way_duration_seconds = 0.0
    cfg.wrong_way_gain = 1.0

    if profile_name == "late_linear":
        cfg.perception_delay_seconds = float(rng.uniform(0.18, 0.34))
        cfg.motor_delay_seconds = float(rng.uniform(0.08, 0.16))
        cfg.lpf_tau = float(rng.uniform(0.08, 0.18))
        cfg.obstacle_gain = float(rng.uniform(2.2, 4.0))
        cfg.burst_gain = float(rng.uniform(1.0, 1.35))
        cfg.burst_trigger_h = float(rng.uniform(7.0, 10.0))
        cfg.burst_duration_seconds = float(rng.uniform(0.18, 0.30))
        cfg.detection_range = float(rng.uniform(26.0, 34.0))
        cfg.ky = float(rng.uniform(-0.10, -0.05))
        cfg.kpsi = float(rng.uniform(-0.75, -0.45))
    elif profile_name == "frozen":
        cfg.perception_delay_seconds = float(rng.uniform(0.30, 0.58))
        cfg.motor_delay_seconds = float(rng.uniform(0.18, 0.35))
        cfg.lpf_tau = float(rng.uniform(0.16, 0.32))
        cfg.obstacle_gain = float(rng.uniform(0.6, 2.4))
        cfg.burst_gain = float(rng.uniform(1.0, 1.30))
        cfg.burst_trigger_h = float(rng.uniform(5.5, 8.5))
        cfg.burst_duration_seconds = float(rng.uniform(0.10, 0.22))
        cfg.detection_range = float(rng.uniform(18.0, 28.0))
        cfg.ky = float(rng.uniform(-0.08, -0.03))
        cfg.kpsi = float(rng.uniform(-0.55, -0.25))
    elif profile_name == "wrong_initial":
        cfg.perception_delay_seconds = float(rng.uniform(0.22, 0.42))
        cfg.motor_delay_seconds = float(rng.uniform(0.10, 0.22))
        cfg.lpf_tau = float(rng.uniform(0.10, 0.20))
        cfg.obstacle_gain = float(rng.uniform(3.6, 5.8))
        cfg.burst_gain = float(rng.uniform(1.6, 2.8))
        cfg.burst_trigger_h = float(rng.uniform(6.0, 9.5))
        cfg.burst_duration_seconds = float(rng.uniform(0.25, 0.40))
        cfg.detection_range = float(rng.uniform(24.0, 34.0))
        cfg.ky = float(rng.uniform(-0.10, -0.05))
        cfg.kpsi = float(rng.uniform(-0.80, -0.50))
        cfg.wrong_way_duration_seconds = float(rng.uniform(0.18, 0.40))
        cfg.wrong_way_gain = float(rng.uniform(0.7, 1.1))
    else:  # late_aggressive and fallback
        cfg.perception_delay_seconds = float(rng.uniform(0.24, 0.50))
        cfg.motor_delay_seconds = float(rng.uniform(0.12, 0.26))
        cfg.lpf_tau = float(rng.uniform(0.10, 0.18))
        cfg.obstacle_gain = float(rng.uniform(4.0, 6.4))
        cfg.burst_gain = float(rng.uniform(1.8, 3.5))
        cfg.burst_trigger_h = float(rng.uniform(6.5, 10.0))
        cfg.burst_duration_seconds = float(rng.uniform(0.24, 0.38))
        cfg.detection_range = float(rng.uniform(26.0, 34.0))
        cfg.ky = float(rng.uniform(-0.10, -0.05))
        cfg.kpsi = float(rng.uniform(-0.85, -0.55))

    cfg.max_sfa_cmd = float(rng.uniform(5.0, 6.5))
    return cfg
