from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Obstacle:
    x: float = 40.0
    y: float = 0.0
    a: float = 3.5
    b: float = 1.2
    margin: float = 0.5


def barrier_h(x: float, y: float, obstacle: Obstacle) -> float:
    ax = obstacle.a + obstacle.margin
    by = obstacle.b + obstacle.margin
    return ((x - obstacle.x) ** 2) / (ax ** 2) + ((y - obstacle.y) ** 2) / (by ** 2) - 1.0


def approximate_ttc(x: float, vx: float, obstacle: Obstacle) -> float:
    gap = obstacle.x - x
    if vx <= 1e-6:
        return float("inf")
    return gap / vx


def signed_side(y: float, obstacle: Obstacle, deadzone: float = 0.05, preferred: float = 1.0) -> float:
    dy = y - obstacle.y
    if abs(dy) > deadzone:
        return 1.0 if dy > 0.0 else -1.0
    return 1.0 if preferred >= 0.0 else -1.0
