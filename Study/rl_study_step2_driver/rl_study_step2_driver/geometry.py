from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Obstacle:
    """Axis-aligned elliptical obstacle.

    Parameters
    ----------
    x, y : float
        Obstacle centre in global coordinates [m].
    a, b : float
        Semi-axes of the geometric obstacle [m].
    margin : float
        Extra safety buffer added to both axes when computing the barrier.
    """

    x: float = 40.0
    y: float = 0.0
    a: float = 3.5
    b: float = 1.2
    margin: float = 0.5


def barrier_h(x: float, y: float, obstacle: Obstacle) -> float:
    """Barrier-style obstacle function.

    h(x, y) = ((x-x_o)^2 / (a+m)^2) + ((y-y_o)^2 / (b+m)^2) - 1

    Interpretation
    --------------
    h > 0 : outside the safety ellipse
    h = 0 : on the boundary
    h < 0 : inside the obstacle / collision set
    """
    ax = obstacle.a + obstacle.margin
    by = obstacle.b + obstacle.margin
    return ((x - obstacle.x) ** 2) / (ax ** 2) + ((y - obstacle.y) ** 2) / (by ** 2) - 1.0


def approximate_ttc(x: float, vx: float, obstacle: Obstacle) -> float:
    """Very simple time-to-collision approximation using obstacle centre distance.

    This is only for debugging / intuition in the study code.
    """
    gap = obstacle.x - x
    if vx <= 1e-6:
        return float("inf")
    return gap / vx
