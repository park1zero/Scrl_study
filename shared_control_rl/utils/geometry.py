
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class EllipseObstacle:
    x: float
    y: float
    a: float
    b: float

    def inflated(self, margin: float) -> "EllipseObstacle":
        return EllipseObstacle(
            x=self.x,
            y=self.y,
            a=self.a + margin,
            b=self.b + margin,
        )


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def ellipse_barrier(x: float, y: float, obstacle: EllipseObstacle) -> float:
    """Positive outside, zero on boundary, negative inside."""
    dx = x - obstacle.x
    dy = y - obstacle.y
    return (dx * dx) / (obstacle.a * obstacle.a) + (dy * dy) / (obstacle.b * obstacle.b) - 1.0


def approximate_ttc(x: float, vx: float, obstacle_x: float) -> float:
    dx = obstacle_x - x
    if vx <= 1e-6:
        return np.inf
    return max(dx / vx, 0.0)


def avoidance_target_y(
    x: float,
    obstacle: EllipseObstacle,
    side: float,
    clearance: float,
    pre_turn_distance: float,
    post_turn_distance: float,
) -> float:
    """Simple hand-made lateral target around the obstacle."""
    y_peak = obstacle.y + side * (obstacle.b + clearance)
    x_ramp_up_start = obstacle.x - pre_turn_distance
    x_peak = obstacle.x - 0.25 * obstacle.a
    x_ramp_down_end = obstacle.x + post_turn_distance

    if x <= x_ramp_up_start or x >= x_ramp_down_end:
        return 0.0
    if x < x_peak:
        alpha = (x - x_ramp_up_start) / max(x_peak - x_ramp_up_start, 1e-6)
        return alpha * y_peak

    alpha = 1.0 - (x - x_peak) / max(x_ramp_down_end - x_peak, 1e-6)
    return max(alpha, 0.0) * y_peak
