
from __future__ import annotations

from dataclasses import dataclass

from shared_control_rl.config import ScenarioConfig
from shared_control_rl.utils.geometry import EllipseObstacle


@dataclass
class Scenario:
    obstacle: EllipseObstacle
    avoid_side: float
    ref_y: float = 0.0
    ref_psi: float = 0.0


def build_scenario(config: ScenarioConfig) -> Scenario:
    obstacle = EllipseObstacle(
        x=config.obstacle_x,
        y=config.obstacle_y,
        a=config.obstacle_a,
        b=config.obstacle_b,
    )
    return Scenario(
        obstacle=obstacle,
        avoid_side=config.avoid_side,
        ref_y=0.0,
        ref_psi=0.0,
    )
