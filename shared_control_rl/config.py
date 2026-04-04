from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class VehicleParams:
    """Constant-speed bicycle model plus SBW actuator states."""

    dt: float = 0.05
    m: float = 1500.0
    iz: float = 2500.0
    lf: float = 1.2
    lr: float = 1.6
    cf: float = 80000.0
    cr: float = 90000.0
    vx: float = 20.0

    tau_rwa: float = 0.12
    tau_sfa: float = 0.10

    max_rwa: float = 0.35          # rad
    max_rwa_rate: float = 0.70     # rad/s
    max_sfa: float = 7.5           # rad at steering wheel
    max_sfa_rate: float = 8.0      # rad/s at steering wheel
    steering_ratio: float = 16.0   # sfa / RWA

    @property
    def wheelbase(self) -> float:
        return self.lf + self.lr


@dataclass
class ScenarioConfig:
    """Single-obstacle scenario on a straight road."""

    x0: float = 0.0
    y0: float = 0.0
    psi0: float = 0.0
    obstacle_x: float = 40.0
    obstacle_y: float = 0.0
    obstacle_a: float = 3.5
    obstacle_b: float = 1.2
    obstacle_margin: float = 0.8
    avoid_side: float = 1.0       # +1 left, -1 right
    episode_seconds: float = 8.0

    domain_randomization: bool = False
    obstacle_x_range: Tuple[float, float] = (35.0, 48.0)
    obstacle_b_range: Tuple[float, float] = (0.8, 1.6)
    speed_range: Tuple[float, float] = (16.0, 24.0)
    cf_scale_range: Tuple[float, float] = (0.85, 1.15)
    cr_scale_range: Tuple[float, float] = (0.85, 1.15)
    tau_rwa_scale_range: Tuple[float, float] = (0.85, 1.20)
    tau_sfa_scale_range: Tuple[float, float] = (0.85, 1.20)
    steering_ratio_scale_range: Tuple[float, float] = (0.92, 1.08)
    # Split delay ranges for the drowsy-driver model.
    perception_delay_range: Tuple[float, float] = (0.25, 0.50)
    motor_delay_range: Tuple[float, float] = (0.12, 0.28)
    # Backward-compatible total delay range, retained for older scripts / notes.
    delay_range: Tuple[float, float] = (0.45, 0.85)
    sfay_range: Tuple[float, float] = (-0.2, 0.2)
    psi0_range: Tuple[float, float] = (-0.03, 0.03)
    late_start_distance_range: Tuple[float, float] = (10.0, 20.0)
    side_randomization: bool = False

    success_x_margin: float = 8.0


@dataclass
class DrowsyDriverConfig:
    """Drowsy-driver model with separate perception and motor delays.

    Interpretation
    --------------
    - perception_delay_seconds: obstacle/state are perceived late.
    - motor_delay_seconds: even after deciding, the command is released late.
    - lpf_tau: sluggish hand-wheel build-up.
    - burst_*: once hazard becomes salient, the driver overreacts for a short time.
    - wrong_way_*: a subset of sleepy drivers initially steer in the wrong direction.
    """

    profile_name: str = "late_aggressive"
    perception_delay_seconds: float = 0.26
    motor_delay_seconds: float = 0.14
    lpf_tau: float = 0.12
    ky: float = -0.08
    kpsi: float = -0.70
    obstacle_gain: float = 4.6
    burst_gain: float = 2.4
    burst_trigger_h: float = 8.0
    burst_duration_seconds: float = 0.30
    detection_range: float = 30.0
    preferred_side: float = 1.0
    max_sfa_cmd: float = 6.0
    wrong_way_duration_seconds: float = 0.0
    wrong_way_gain: float = 1.0

    @property
    def total_delay_seconds(self) -> float:
        return float(self.perception_delay_seconds + self.motor_delay_seconds)


@dataclass
class DriverPopulationConfig:
    """Driver-profile mixture used for curriculum/domain randomization."""

    enabled: bool = False
    profile_names: Tuple[str, ...] = ("late_linear", "late_aggressive", "frozen", "wrong_initial")
    easy_weights: Tuple[float, ...] = (0.50, 0.25, 0.20, 0.05)
    medium_weights: Tuple[float, ...] = (0.25, 0.45, 0.15, 0.15)
    hard_weights: Tuple[float, ...] = (0.10, 0.40, 0.30, 0.20)


@dataclass
class CurriculumConfig:
    """Episode-mixture curriculum.

    The environment interpolates difficulty weights from `start_weights` to
    `end_weights` as `curriculum_progress` moves from 0 to 1. `late_start_prob`
    is also increased with progress to emphasize the hazardous part of the episode.
    """

    enabled: bool = False
    difficulty_names: Tuple[str, ...] = ("easy", "medium", "hard")
    start_weights: Tuple[float, float, float] = (0.45, 0.40, 0.15)
    end_weights: Tuple[float, float, float] = (0.20, 0.35, 0.45)
    late_start_prob_start: float = 0.05
    late_start_prob_end: float = 0.35
    hazard_focus_prob: float = 0.20


@dataclass
class ShootingMPCConfig:
    """Small shooting-based MPC surrogate.

    This is intentionally simple and dependency-light.
    Replace with a true CasADi/do-mpc implementation later if desired.
    """

    horizon_steps: int = 12
    segment_split: int = 4
    u0_grid: int = 11
    u1_grid: int = 7

    wy: float = 8.0
    wpsi: float = 4.0
    wvy: float = 0.4
    wr: float = 0.4
    wu: float = 0.2
    wdu: float = 0.4
    w_barrier: float = 2.2
    hard_barrier_penalty: float = 2.0e4

    pre_turn_distance: float = 18.0
    post_turn_distance: float = 8.0
    lateral_clearance: float = 1.0
    barrier_eps: float = 1.0e-3


@dataclass
class SafetyFilterConfig:
    """Scalar lambda projection via one-step discrete barrier screening."""

    lambda_grid_points: int = 51
    cbf_gamma: float = 0.35
    min_h_next: float = 0.02
    relax: float = 0.02


@dataclass
class RewardConfig:
    """Reward weights for RL authority allocation."""

    w_progress: float = 0.06
    w_track_y: float = 0.6
    w_track_psi: float = 0.15
    w_stability_vy: float = 0.02
    w_stability_r: float = 0.02
    w_safety_margin: float = 2.5
    safety_margin_target: float = 0.7
    w_takeover: float = 0.25
    w_lambda_slew: float = 0.12
    w_conflict: float = 0.08
    collision_penalty: float = 200.0
    success_bonus: float = 60.0


@dataclass
class EnvConfig:
    vehicle: VehicleParams = field(default_factory=VehicleParams)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    driver: DrowsyDriverConfig = field(default_factory=DrowsyDriverConfig)
    driver_population: DriverPopulationConfig = field(default_factory=DriverPopulationConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    mpc: ShootingMPCConfig = field(default_factory=ShootingMPCConfig)
    safety: SafetyFilterConfig = field(default_factory=SafetyFilterConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    max_lambda_rate: float = 2.0
    seed: int | None = 42

    def max_steps(self) -> int:
        return int(self.scenario.episode_seconds / self.vehicle.dt)
