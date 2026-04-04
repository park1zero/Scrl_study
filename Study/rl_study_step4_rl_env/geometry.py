from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Obstacle:
    """Axis-aligned elliptical obstacle with an added safety margin."""

    x: float = 40.0
    y: float = 0.0
    a: float = 3.5
    b: float = 1.2
    margin: float = 0.5


def barrier_h(x: float, y: float, obstacle: Obstacle) -> float:
    """Barrier function used for collision and reward shaping.

    h(x, y) = ((x-x_o)^2 / (a+m)^2) + ((y-y_o)^2 / (b+m)^2) - 1

    h > 0 : outside the safety ellipse
    h = 0 : on the boundary
    h < 0 : inside the collision set
    """
    ax = obstacle.a + obstacle.margin
    by = obstacle.b + obstacle.margin
    return ((x - obstacle.x) ** 2) / (ax ** 2) + ((y - obstacle.y) ** 2) / (by ** 2) - 1.0


def approximate_ttc(x: float, vx: float, obstacle: Obstacle) -> float:
    """Simple x-distance based time-to-collision approximation."""
    gap = obstacle.x - x
    if vx <= 1e-6:
        return float("inf")
    return gap / vx
