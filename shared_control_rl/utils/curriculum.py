from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared_control_rl.config import CurriculumConfig, ScenarioConfig, VehicleParams


@dataclass
class EpisodeSample:
    difficulty: str
    late_start: bool
    hazard_focus: bool


def _lerp(a: float, b: float, w: float) -> float:
    return (1.0 - w) * float(a) + w * float(b)


def interpolate_weights(cfg: CurriculumConfig, progress: float) -> np.ndarray:
    progress = float(np.clip(progress, 0.0, 1.0))
    start = np.asarray(cfg.start_weights, dtype=float)
    end = np.asarray(cfg.end_weights, dtype=float)
    weights = (1.0 - progress) * start + progress * end
    weights = np.clip(weights, 0.0, None)
    if weights.sum() <= 0.0:
        weights = np.ones_like(weights, dtype=float)
    return weights / weights.sum()


def sample_episode(
    rng: np.random.Generator,
    cfg: CurriculumConfig,
    progress: float,
    force_difficulty: str | None = None,
) -> EpisodeSample:
    progress = float(np.clip(progress, 0.0, 1.0))
    if force_difficulty is not None:
        difficulty = str(force_difficulty)
    else:
        names = np.asarray(cfg.difficulty_names, dtype=object)
        weights = interpolate_weights(cfg, progress)
        difficulty = str(rng.choice(names, p=weights))

    late_start_prob = _lerp(cfg.late_start_prob_start, cfg.late_start_prob_end, progress)
    late_start = bool(rng.random() < late_start_prob)
    hazard_focus = bool(rng.random() < cfg.hazard_focus_prob)
    return EpisodeSample(difficulty=difficulty, late_start=late_start, hazard_focus=hazard_focus)


def apply_difficulty(
    rng: np.random.Generator,
    difficulty: str,
    scenario: ScenarioConfig,
    vehicle: VehicleParams,
) -> None:
    """Mutate scenario/vehicle in place according to curriculum difficulty."""
    if difficulty == "easy":
        scenario.obstacle_x = float(rng.uniform(42.0, 52.0))
        scenario.obstacle_b = float(rng.uniform(0.8, 1.15))
        scenario.y0 = float(rng.uniform(-0.12, 0.12))
        scenario.psi0 = float(rng.uniform(-0.015, 0.015))
        vehicle.vx = float(rng.uniform(15.0, 19.0))
    elif difficulty == "hard":
        scenario.obstacle_x = float(rng.uniform(32.0, 42.0))
        scenario.obstacle_b = float(rng.uniform(1.25, 1.85))
        scenario.y0 = float(rng.uniform(-0.28, 0.28))
        scenario.psi0 = float(rng.uniform(-0.04, 0.04))
        vehicle.vx = float(rng.uniform(21.0, 26.0))
    else:
        scenario.obstacle_x = float(rng.uniform(36.0, 46.0))
        scenario.obstacle_b = float(rng.uniform(1.0, 1.5))
        scenario.y0 = float(rng.uniform(-0.20, 0.20))
        scenario.psi0 = float(rng.uniform(-0.03, 0.03))
        vehicle.vx = float(rng.uniform(18.0, 22.0))


def apply_late_start(
    rng: np.random.Generator,
    sample: EpisodeSample,
    scenario: ScenarioConfig,
) -> None:
    if not sample.late_start:
        scenario.x0 = 0.0
        return

    if sample.difficulty == "easy":
        dist = float(rng.uniform(16.0, 22.0))
    elif sample.difficulty == "hard":
        dist = float(rng.uniform(8.0, 14.0))
    else:
        dist = float(rng.uniform(12.0, 18.0))
    scenario.x0 = float(max(0.0, scenario.obstacle_x - dist))

    if sample.hazard_focus:
        scenario.y0 = float(np.clip(scenario.y0 + rng.uniform(-0.10, 0.10), -0.35, 0.35))
        scenario.psi0 = float(np.clip(scenario.psi0 + rng.uniform(-0.015, 0.015), -0.06, 0.06))
