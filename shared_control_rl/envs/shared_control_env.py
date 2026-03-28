from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    try:
        import gym  # type: ignore
        from gym import spaces  # type: ignore
    except ImportError:
        class _FallbackEnv:
            metadata = {}

            def reset(self, *, seed=None, options=None):
                return None

        class _FallbackBox:
            def __init__(self, low, high, shape, dtype):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype

        class _FallbackSpaces:
            Box = _FallbackBox

        class _FallbackGymModule:
            Env = _FallbackEnv

        gym = _FallbackGymModule()  # type: ignore
        spaces = _FallbackSpaces()  # type: ignore

import numpy as np

from shared_control_rl.config import EnvConfig, VehicleParams, ScenarioConfig, DrowsyDriverConfig
from shared_control_rl.controllers.mpc_barrier import ShootingBarrierMPC
from shared_control_rl.controllers.safety_filter import LambdaSafetyFilter
from shared_control_rl.models.driver import DrowsyDriverModel
from shared_control_rl.models.vehicle import (
    VehicleState,
    blend_rwa_command,
    equivalent_rwa_from_swa,
    step_vehicle,
)
from shared_control_rl.utils.curriculum import apply_difficulty, apply_late_start, sample_episode
from shared_control_rl.utils.driver_population import sample_driver_config, sample_profile_name
from shared_control_rl.utils.geometry import EllipseObstacle, approximate_ttc, ellipse_barrier
from shared_control_rl.utils.scenario import build_scenario


class SharedControlEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None) -> None:
        super().__init__()
        self.base_config = replace(config) if config is not None else EnvConfig()
        self.rng = np.random.default_rng(self.base_config.seed)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(13,),
            dtype=np.float32,
        )

        self.vehicle_params: VehicleParams
        self.scenario_cfg: ScenarioConfig
        self.driver_cfg: DrowsyDriverConfig
        self.obstacle: EllipseObstacle
        self.state: VehicleState
        self.driver: DrowsyDriverModel
        self.mpc: ShootingBarrierMPC
        self.safety_filter: LambdaSafetyFilter

        self.current_driver_swa_cmd: float = 0.0
        self.current_driver_rwa: float = 0.0
        self.current_auto_rwa: float = 0.0
        self.current_cmd_rwa: float = 0.0
        self.current_lambda_des: float = 1.0
        self.step_count: int = 0
        self.history: dict[str, list[float]] = {}

        self.curriculum_progress: float = 0.0
        self.current_difficulty: str = "medium"
        self.current_driver_profile: str = self.base_config.driver.profile_name
        self.current_late_start: bool = False
        self.current_hazard_focus: bool = False
        self.last_reset_options: dict[str, Any] = {}

        self.reset(seed=self.base_config.seed)

    def set_curriculum_progress(self, progress: float) -> None:
        self.curriculum_progress = float(np.clip(progress, 0.0, 1.0))

    def _randomize(self, options: dict[str, Any] | None = None) -> None:
        self.vehicle_params = replace(self.base_config.vehicle)
        self.scenario_cfg = replace(self.base_config.scenario)
        self.driver_cfg = replace(self.base_config.driver)

        options = dict(options or {})
        self.last_reset_options = dict(options)
        force_difficulty = options.get("difficulty")
        force_driver_profile = options.get("driver_profile")
        progress_override = options.get("curriculum_progress")
        if progress_override is not None:
            self.set_curriculum_progress(float(progress_override))

        use_curriculum = bool(self.base_config.curriculum.enabled)
        use_population = bool(self.base_config.driver_population.enabled)
        use_randomization = bool(self.scenario_cfg.domain_randomization or use_curriculum or use_population)

        if self.scenario_cfg.side_randomization or use_curriculum:
            self.scenario_cfg.avoid_side = float(self.rng.choice([-1.0, 1.0]))

        if use_curriculum:
            sample = sample_episode(
                rng=self.rng,
                cfg=self.base_config.curriculum,
                progress=self.curriculum_progress,
                force_difficulty=str(force_difficulty) if force_difficulty is not None else None,
            )
            self.current_difficulty = sample.difficulty
            self.current_late_start = bool(sample.late_start)
            self.current_hazard_focus = bool(sample.hazard_focus)
            apply_difficulty(self.rng, sample.difficulty, self.scenario_cfg, self.vehicle_params)
            apply_late_start(self.rng, sample, self.scenario_cfg)
        else:
            self.current_difficulty = str(force_difficulty) if force_difficulty is not None else "medium"
            self.current_late_start = bool(options.get("late_start", False))
            self.current_hazard_focus = bool(options.get("hazard_focus", False))

        if use_randomization and not use_curriculum and self.scenario_cfg.domain_randomization:
            self.scenario_cfg.obstacle_x = float(self.rng.uniform(*self.scenario_cfg.obstacle_x_range))
            self.scenario_cfg.obstacle_b = float(self.rng.uniform(*self.scenario_cfg.obstacle_b_range))
            self.vehicle_params.vx = float(self.rng.uniform(*self.scenario_cfg.speed_range))
            self.scenario_cfg.y0 = float(self.rng.uniform(*self.scenario_cfg.sway_range))
            self.scenario_cfg.psi0 = float(self.rng.uniform(*self.scenario_cfg.psi0_range))
            if self.current_late_start:
                dist = float(self.rng.uniform(*self.scenario_cfg.late_start_distance_range))
                self.scenario_cfg.x0 = float(max(0.0, self.scenario_cfg.obstacle_x - dist))

        if use_randomization:
            self.vehicle_params.cf *= float(self.rng.uniform(*self.scenario_cfg.cf_scale_range))
            self.vehicle_params.cr *= float(self.rng.uniform(*self.scenario_cfg.cr_scale_range))
            self.vehicle_params.tau_rwa *= float(self.rng.uniform(*self.scenario_cfg.tau_rwa_scale_range))
            self.vehicle_params.tau_swa *= float(self.rng.uniform(*self.scenario_cfg.tau_swa_scale_range))
            self.vehicle_params.steering_ratio *= float(self.rng.uniform(*self.scenario_cfg.steering_ratio_scale_range))

        if use_population:
            self.current_driver_profile = (
                str(force_driver_profile)
                if force_driver_profile is not None
                else sample_profile_name(self.rng, self.base_config.driver_population, self.current_difficulty)
            )
            self.driver_cfg = sample_driver_config(
                rng=self.rng,
                base_cfg=self.base_config.driver,
                profile_name=self.current_driver_profile,
                avoid_side=self.scenario_cfg.avoid_side,
            )
        else:
            self.current_driver_profile = self.driver_cfg.profile_name
            if use_randomization:
                self.driver_cfg.perception_delay_seconds = float(self.rng.uniform(*self.scenario_cfg.perception_delay_range))
                self.driver_cfg.motor_delay_seconds = float(self.rng.uniform(*self.scenario_cfg.motor_delay_range))

        self.driver_cfg.preferred_side = float(self.scenario_cfg.avoid_side)

    def _build_modules(self) -> None:
        scenario = build_scenario(self.scenario_cfg)
        self.obstacle = scenario.obstacle
        self.driver = DrowsyDriverModel(self.driver_cfg, self.vehicle_params)
        self.mpc = ShootingBarrierMPC(self.base_config.mpc, self.vehicle_params)
        self.safety_filter = LambdaSafetyFilter(self.base_config.safety, self.vehicle_params)
        self.mpc.reset()

    def _initial_state(self) -> VehicleState:
        return VehicleState(
            x=self.scenario_cfg.x0,
            y=self.scenario_cfg.y0,
            psi=self.scenario_cfg.psi0,
            vy=0.0,
            r=0.0,
            delta_rwa=0.0,
            delta_swa=0.0,
            lam=1.0,
            t=0.0,
        )

    def _build_obs(self) -> np.ndarray:
        dx_obs = self.obstacle.x - self.state.x
        dy_obs = self.state.y - self.obstacle.y
        h = ellipse_barrier(
            self.state.x,
            self.state.y,
            self.obstacle.inflated(self.scenario_cfg.obstacle_margin),
        )
        ttc = approximate_ttc(self.state.x, self.vehicle_params.vx, self.obstacle.x)

        obs = np.array(
            [
                np.clip(self.state.y / 5.0, -5.0, 5.0),
                np.clip(self.state.psi / 0.6, -5.0, 5.0),
                np.clip(self.state.vy / 5.0, -5.0, 5.0),
                np.clip(self.state.r / 1.5, -5.0, 5.0),
                np.clip(self.state.delta_rwa / max(self.vehicle_params.max_rwa, 1e-6), -5.0, 5.0),
                np.clip(self.state.delta_swa / max(self.vehicle_params.max_swa, 1e-6), -5.0, 5.0),
                np.clip(self.current_driver_rwa / max(self.vehicle_params.max_rwa, 1e-6), -5.0, 5.0),
                np.clip(self.current_auto_rwa / max(self.vehicle_params.max_rwa, 1e-6), -5.0, 5.0),
                np.clip(self.state.lam, 0.0, 1.0),
                np.clip(h / 5.0, -5.0, 5.0),
                np.clip(ttc / 5.0, 0.0, 5.0),
                np.clip(dx_obs / 50.0, -5.0, 5.0),
                np.clip(dy_obs / 5.0, -5.0, 5.0),
            ],
            dtype=np.float32,
        )
        return obs

    def _make_info(self, filter_info: Dict[str, float] | None = None) -> Dict[str, Any]:
        h = ellipse_barrier(
            self.state.x,
            self.state.y,
            self.obstacle.inflated(self.scenario_cfg.obstacle_margin),
        )
        ttc = approximate_ttc(self.state.x, self.vehicle_params.vx, self.obstacle.x)
        info: Dict[str, Any] = {
            "x": float(self.state.x),
            "y": float(self.state.y),
            "psi": float(self.state.psi),
            "vy": float(self.state.vy),
            "r": float(self.state.r),
            "delta_swa": float(self.state.delta_swa),
            "delta_rwa": float(self.state.delta_rwa),
            "driver_swa_cmd": float(self.current_driver_swa_cmd),
            "driver_raw_swa_cmd": float(self.driver.raw_cmd),
            "driver_filtered_swa_cmd": float(self.driver.filtered_cmd),
            "driver_output_swa_cmd": float(self.driver.output_cmd),
            "driver_perceived_h": float(self.driver.perceived_h),
            "driver_perceived_dx": float(self.driver.perceived_dx),
            "driver_hazard_active": float(self.driver.hazard_active),
            "driver_burst_timer_steps": float(self.driver.burst_timer_steps),
            "driver_profile": str(self.current_driver_profile),
            "driver_perception_delay": float(self.driver_cfg.perception_delay_seconds),
            "driver_motor_delay": float(self.driver_cfg.motor_delay_seconds),
            "driver_total_delay": float(self.driver_cfg.total_delay_seconds),
            "driver_wrong_way_duration": float(self.driver_cfg.wrong_way_duration_seconds),
            "driver_wrong_way_gain": float(self.driver_cfg.wrong_way_gain),
            "delta_drv_rwa": float(self.current_driver_rwa),
            "delta_auto_rwa": float(self.current_auto_rwa),
            "delta_cmd_rwa": float(self.current_cmd_rwa),
            "lambda_des": float(self.current_lambda_des),
            "lambda_safe": float(self.state.lam),
            "h": float(h),
            "ttc": float(ttc),
            "obstacle_x": float(self.obstacle.x),
            "obstacle_y": float(self.obstacle.y),
            "obstacle_a": float(self.obstacle.a),
            "obstacle_b": float(self.obstacle.b),
            "avoid_side": float(self.scenario_cfg.avoid_side),
            "speed": float(self.vehicle_params.vx),
            "difficulty": str(self.current_difficulty),
            "late_start": bool(self.current_late_start),
            "hazard_focus": bool(self.current_hazard_focus),
            "curriculum_progress": float(self.curriculum_progress),
        }
        if filter_info is not None:
            info.update(filter_info)
        return info

    def _reward(
        self,
        prev_state: VehicleState,
        next_state: VehicleState,
        collision: bool,
        success: bool,
        delta_cmd: float,
        delta_drv_rwa: float,
    ) -> float:
        cfg = self.base_config.reward
        h = ellipse_barrier(
            next_state.x,
            next_state.y,
            self.obstacle.inflated(self.scenario_cfg.obstacle_margin),
        )
        margin_deficit = max(cfg.safety_margin_target - h, 0.0)
        lambda_slew_norm = (next_state.lam - prev_state.lam) / max(
            self.base_config.max_lambda_rate * self.vehicle_params.dt, 1e-6
        )

        reward = 0.0
        reward += cfg.w_progress * (next_state.x - prev_state.x)
        reward -= cfg.w_track_y * (next_state.y ** 2)
        reward -= cfg.w_track_psi * (next_state.psi ** 2)
        reward -= cfg.w_stability_vy * (next_state.vy ** 2)
        reward -= cfg.w_stability_r * (next_state.r ** 2)
        reward -= cfg.w_safety_margin * (margin_deficit ** 2)
        reward -= cfg.w_takeover * ((1.0 - next_state.lam) ** 2)
        reward -= cfg.w_lambda_slew * (lambda_slew_norm ** 2)
        reward -= cfg.w_conflict * (((delta_cmd - delta_drv_rwa) / max(self.vehicle_params.max_rwa, 1e-6)) ** 2)

        if collision:
            reward -= cfg.collision_penalty
        if success:
            reward += cfg.success_bonus
        return float(reward)

    def _record(self, reward: float, info: Dict[str, Any]) -> None:
        if not self.history:
            self.history = {
                "t": [],
                "x": [],
                "y": [],
                "psi": [],
                "delta_swa": [],
                "delta_rwa": [],
                "delta_drv_rwa": [],
                "delta_auto_rwa": [],
                "delta_cmd_rwa": [],
                "lambda": [],
                "lambda_des": [],
                "h": [],
                "reward": [],
                "driver_perceived_h": [],
                "driver_perceived_dx": [],
                "driver_raw_swa_cmd": [],
                "driver_filtered_swa_cmd": [],
                "driver_output_swa_cmd": [],
                "driver_hazard_active": [],
            }

        self.history["t"].append(self.state.t)
        self.history["x"].append(self.state.x)
        self.history["y"].append(self.state.y)
        self.history["psi"].append(self.state.psi)
        self.history["delta_swa"].append(self.state.delta_swa)
        self.history["delta_rwa"].append(self.state.delta_rwa)
        self.history["delta_drv_rwa"].append(info["delta_drv_rwa"])
        self.history["delta_auto_rwa"].append(info["delta_auto_rwa"])
        self.history["delta_cmd_rwa"].append(info["delta_cmd_rwa"])
        self.history["lambda"].append(self.state.lam)
        self.history["lambda_des"].append(info["lambda_des"])
        self.history["h"].append(info["h"])
        self.history["reward"].append(reward)
        self.history["driver_perceived_h"].append(info["driver_perceived_h"])
        self.history["driver_perceived_dx"].append(info["driver_perceived_dx"])
        self.history["driver_raw_swa_cmd"].append(info["driver_raw_swa_cmd"])
        self.history["driver_filtered_swa_cmd"].append(info["driver_filtered_swa_cmd"])
        self.history["driver_output_swa_cmd"].append(info["driver_output_swa_cmd"])
        self.history["driver_hazard_active"].append(info["driver_hazard_active"])

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._randomize(options=options)
        self._build_modules()
        self.state = self._initial_state()
        self.driver.reset(
            initial_state=self.state,
            perception_delay_seconds=self.driver_cfg.perception_delay_seconds,
            motor_delay_seconds=self.driver_cfg.motor_delay_seconds,
            preferred_side=self.scenario_cfg.avoid_side,
        )

        self.current_driver_swa_cmd = 0.0
        self.current_driver_rwa = 0.0
        self.current_auto_rwa = 0.0
        self.current_cmd_rwa = 0.0
        self.current_lambda_des = self.state.lam
        self.step_count = 0
        self.history = {}

        obs = self._build_obs()
        info = self._make_info()
        self._record(0.0, info)
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_scalar = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        dt = self.vehicle_params.dt

        driver_swa_cmd = self.driver.compute_command(
            state=self.state,
            ref_y=0.0,
            ref_psi=0.0,
            obstacle=self.obstacle,
            margin=self.scenario_cfg.obstacle_margin,
        )
        delta_drv_rwa = equivalent_rwa_from_swa(self.state.delta_swa, self.vehicle_params)
        delta_auto_rwa = self.mpc.compute_command(
            state=self.state,
            obstacle=self.obstacle,
            side=self.scenario_cfg.avoid_side,
            margin=self.scenario_cfg.obstacle_margin,
        )

        lambda_des = float(
            np.clip(
                self.state.lam + self.base_config.max_lambda_rate * dt * action_scalar,
                0.0,
                1.0,
            )
        )
        lambda_safe, filter_info = self.safety_filter.project_lambda(
            state=self.state,
            lambda_des=lambda_des,
            delta_drv_rwa=delta_drv_rwa,
            delta_auto_rwa=delta_auto_rwa,
            obstacle=self.obstacle,
            margin=self.scenario_cfg.obstacle_margin,
        )
        delta_cmd = blend_rwa_command(lambda_safe, delta_drv_rwa, delta_auto_rwa)

        prev_state = self.state
        next_state = step_vehicle(
            state=self.state,
            rwa_cmd=delta_cmd,
            swa_cmd=driver_swa_cmd,
            params=self.vehicle_params,
        )
        next_state.lam = lambda_safe
        self.state = next_state

        h = ellipse_barrier(
            self.state.x,
            self.state.y,
            self.obstacle.inflated(self.scenario_cfg.obstacle_margin),
        )
        collision = bool(h < 0.0)
        success = bool(self.state.x > self.obstacle.x + self.scenario_cfg.success_x_margin and not collision)
        self.step_count += 1
        truncated = self.step_count >= self.base_config.max_steps()
        terminated = collision or success

        self.current_driver_swa_cmd = driver_swa_cmd
        self.current_driver_rwa = delta_drv_rwa
        self.current_auto_rwa = delta_auto_rwa
        self.current_cmd_rwa = delta_cmd
        self.current_lambda_des = lambda_des

        reward = self._reward(
            prev_state=prev_state,
            next_state=next_state,
            collision=collision,
            success=success,
            delta_cmd=delta_cmd,
            delta_drv_rwa=delta_drv_rwa,
        )
        obs = self._build_obs()
        info = self._make_info(filter_info=filter_info)
        info.update({
            "collision": bool(collision),
            "success": bool(success),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "step_count": int(self.step_count),
        })
        self._record(reward, info)
        return obs, reward, terminated, truncated, info
