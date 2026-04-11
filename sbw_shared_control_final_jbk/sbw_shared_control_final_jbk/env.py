from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import random

import numpy as np

from automation import AutomationParams, step_automation
from driver_factory import default_driver_params, initialize_driver_state, step_driver
from geometry import Obstacle, approximate_ttc, barrier_h
from vehicle import VehicleParams, VehicleState, blend_rwa_command, equivalent_rwa_from_swa, step_vehicle


OBSERVATION_NAMES = [
    'y_norm', 'psi_norm', 'vy_norm', 'r_norm', 'delta_rwa_norm', 'delta_swa_norm',
    'delta_drv_rwa_norm', 'delta_auto_rwa_norm', 'lambda', 'h_norm', 'ttc_norm', 'dx_norm', 'dy_norm'
]


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class EnvParams:
    episode_seconds: float = 4.0
    lambda_rate_max: float = 2.0
    initial_lambda: float = 1.0

    progress_weight: float = 0.06
    lateral_weight: float = 0.60
    yaw_weight: float = 0.15
    vy_weight: float = 0.02
    r_weight: float = 0.02
    barrier_weight: float = 2.50
    barrier_target: float = 0.70
    takeover_weight: float = 0.25
    lambda_smooth_weight: float = 0.12
    conflict_weight: float = 0.08

    collision_penalty: float = 200.0
    success_bonus: float = 60.0
    road_departure_penalty: float = 100.0

    success_x_margin: float = 8.0
    road_limit_abs_y: float = 4.5

    randomize_on_reset: bool = False
    initial_y_range: float = 0.15
    obstacle_x_jitter: float = 2.0
    obstacle_b_jitter: float = 0.20
    speed_jitter: float = 1.5
    randomize_pass_side: bool = False
    randomize_driver: bool = False


class SharedControlEnv:
    observation_names = OBSERVATION_NAMES
    action_low = -1.0
    action_high = 1.0

    def __init__(
        self,
        vehicle_params: Optional[VehicleParams] = None,
        driver_model: str = 'jbk',
        driver_params = None,
        automation_params: Optional[AutomationParams] = None,
        env_params: Optional[EnvParams] = None,
        obstacle: Optional[Obstacle] = None,
    ) -> None:
        self.base_vehicle_params = vehicle_params or VehicleParams()
        self.driver_model = driver_model
        self.base_driver_params = driver_params or default_driver_params(driver_model)
        self.base_automation_params = automation_params or AutomationParams()
        self.env_params = env_params or EnvParams()
        self.base_obstacle = obstacle or Obstacle()

        self.rng = random.Random(0)
        self.vehicle_params = self.base_vehicle_params
        self.driver_params = self.base_driver_params
        self.automation_params = self.base_automation_params
        self.obstacle = self.base_obstacle
        self.state = VehicleState(lam=self.env_params.initial_lambda)
        self.driver_state = initialize_driver_state(self.driver_model, self.driver_params, self.base_vehicle_params.dt)
        self.step_count = 0
        self.max_steps = max(1, int(round(self.env_params.episode_seconds / self.base_vehicle_params.dt)))

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, float]]:
        if seed is not None:
            self.rng.seed(seed)
        self.vehicle_params = VehicleParams(**self.base_vehicle_params.__dict__)
        self.driver_params = type(self.base_driver_params)(**self.base_driver_params.__dict__)
        self.automation_params = AutomationParams(**self.base_automation_params.__dict__)
        self.obstacle = Obstacle(**self.base_obstacle.__dict__)

        y0 = 0.0
        if self.env_params.randomize_on_reset:
            y0 = self.rng.uniform(-self.env_params.initial_y_range, self.env_params.initial_y_range)
            self.obstacle.x += self.rng.uniform(-self.env_params.obstacle_x_jitter, self.env_params.obstacle_x_jitter)
            self.obstacle.b = max(0.8, self.obstacle.b + self.rng.uniform(-self.env_params.obstacle_b_jitter, self.env_params.obstacle_b_jitter))
            self.vehicle_params.vx = max(10.0, self.vehicle_params.vx + self.rng.uniform(-self.env_params.speed_jitter, self.env_params.speed_jitter))
            if self.env_params.randomize_pass_side:
                sign = -1.0 if self.rng.random() < 0.5 else 1.0
                if hasattr(self.driver_params, 'preferred_side'):
                    self.driver_params.preferred_side = sign
                self.automation_params.pass_side = sign
            if self.env_params.randomize_driver and self.driver_model == 'jbk':
                self.driver_params.perception_delay_s = max(0.12, self.driver_params.perception_delay_s + self.rng.uniform(-0.05, 0.08))
                self.driver_params.motor_delay_s = max(0.06, self.driver_params.motor_delay_s + self.rng.uniform(-0.03, 0.05))
                self.driver_params.attention_scale = clip(self.driver_params.attention_scale + self.rng.uniform(-0.08, 0.08), 0.45, 0.75)
                self.driver_params.obstacle_gain = max(1.2, self.driver_params.obstacle_gain + self.rng.uniform(-0.35, 0.45))

        self.state = VehicleState(x=0.0, y=y0, psi=0.0, vy=0.0, r=0.0, delta_rwa=0.0, delta_swa=0.0, lam=self.env_params.initial_lambda, t=0.0)
        self.driver_state = initialize_driver_state(self.driver_model, self.driver_params, self.vehicle_params.dt)
        self.step_count = 0
        obs, obs_info = self._build_observation(self.state)
        info = self._base_info(obs_info)
        return obs, info

    def step(self, action: float):
        action = float(np.asarray(action).reshape(()))
        action = clip(action, self.action_low, self.action_high)

        prev_state = self.state
        lam_prev = prev_state.lam
        lam_next = clip(lam_prev + self.env_params.lambda_rate_max * self.vehicle_params.dt * action, 0.0, 1.0)

        delta_drv_rwa = equivalent_rwa_from_swa(prev_state.delta_swa, self.vehicle_params)
        delta_auto_rwa, auto_debug = step_automation(prev_state, self.obstacle, self.automation_params)
        swa_cmd, driver_debug = step_driver(
            self.driver_model,
            self.driver_state,
            prev_state,
            self.obstacle,
            self.driver_params,
            self.vehicle_params.dt,
            vx=self.vehicle_params.vx,
        )
        delta_cmd = blend_rwa_command(lam_next, delta_drv_rwa, delta_auto_rwa)

        next_state = step_vehicle(prev_state, delta_cmd, swa_cmd, self.vehicle_params)
        next_state.lam = lam_next
        self.state = next_state
        self.step_count += 1

        obs, obs_info = self._build_observation(next_state)
        reward, reward_terms = self._compute_reward(prev_state, next_state, delta_drv_rwa, delta_cmd, lam_prev, lam_next, obs_info)

        collision = obs_info['h'] <= 0.0
        road_departure = abs(next_state.y) >= self.env_params.road_limit_abs_y
        passed_obstacle = next_state.x >= self.obstacle.x + self.obstacle.a + self.env_params.success_x_margin
        success = bool(passed_obstacle and obs_info['h'] > 0.0 and not road_departure)

        terminated = collision or road_departure or success
        truncated = self.step_count >= self.max_steps and not terminated
        if collision:
            reward -= self.env_params.collision_penalty
        if road_departure:
            reward -= self.env_params.road_departure_penalty
        if success:
            reward += self.env_params.success_bonus

        info = self._base_info(obs_info)
        info.update(
            {
                'driver_model': self.driver_model,
                'action': action,
                'lam_prev': lam_prev,
                'lam_next': lam_next,
                'delta_drv_rwa': delta_drv_rwa,
                'delta_auto_rwa': delta_auto_rwa,
                'delta_cmd': delta_cmd,
                'swa_cmd': swa_cmd,
                'collision': collision,
                'road_departure': road_departure,
                'success': success,
                'terminated': terminated,
                'truncated': truncated,
                **{f'driver_{k}': v for k, v in driver_debug.items()},
                **{f'auto_{k}': v for k, v in auto_debug.items()},
                **reward_terms,
            }
        )
        return obs, float(reward), terminated, truncated, info

    def _build_observation(self, state: VehicleState) -> Tuple[np.ndarray, Dict[str, float]]:
        delta_drv_rwa = equivalent_rwa_from_swa(state.delta_swa, self.vehicle_params)
        delta_auto_rwa, _ = step_automation(state, self.obstacle, self.automation_params)
        h = barrier_h(state.x, state.y, self.obstacle)
        ttc = approximate_ttc(state.x, self.vehicle_params.vx, self.obstacle)
        dx = self.obstacle.x - state.x
        dy = state.y - self.obstacle.y
        obs = np.array(
            [
                clip(state.y / 5.0, -3.0, 3.0),
                clip(state.psi / 0.6, -3.0, 3.0),
                clip(state.vy / 5.0, -3.0, 3.0),
                clip(state.r / 1.5, -3.0, 3.0),
                clip(state.delta_rwa / max(self.vehicle_params.max_rwa, 1e-6), -1.0, 1.0),
                clip(state.delta_swa / max(self.vehicle_params.max_swa, 1e-6), -1.0, 1.0),
                clip(delta_drv_rwa / max(self.vehicle_params.max_rwa, 1e-6), -1.0, 1.0),
                clip(delta_auto_rwa / max(self.vehicle_params.max_rwa, 1e-6), -1.0, 1.0),
                clip(state.lam, 0.0, 1.0),
                clip(h / 5.0, -3.0, 3.0),
                clip(min(ttc, 10.0) / 5.0, 0.0, 2.0),
                clip(dx / 50.0, -2.0, 2.0),
                clip(dy / 5.0, -3.0, 3.0),
            ],
            dtype=np.float32,
        )
        return obs, {'delta_drv_rwa': delta_drv_rwa, 'delta_auto_rwa': delta_auto_rwa, 'h': h, 'ttc': ttc, 'dx': dx, 'dy': dy}

    def _compute_reward(self, prev_state, next_state, delta_drv_rwa, delta_cmd, lam_prev, lam_next, obs_info: Dict[str, float]):
        progress = next_state.x - prev_state.x
        barrier_shortfall = max(0.0, self.env_params.barrier_target - obs_info['h'])
        normalized_lambda_step = (lam_next - lam_prev) / max(self.env_params.lambda_rate_max * self.vehicle_params.dt, 1e-6)
        conflict = (delta_cmd - delta_drv_rwa) / max(self.vehicle_params.max_rwa, 1e-6)
        terms = {
            'reward_progress': self.env_params.progress_weight * progress,
            'reward_lateral': -self.env_params.lateral_weight * (next_state.y ** 2),
            'reward_yaw': -self.env_params.yaw_weight * (next_state.psi ** 2),
            'reward_vy': -self.env_params.vy_weight * (next_state.vy ** 2),
            'reward_r': -self.env_params.r_weight * (next_state.r ** 2),
            'reward_barrier': -self.env_params.barrier_weight * (barrier_shortfall ** 2),
            'reward_takeover': -self.env_params.takeover_weight * ((1.0 - lam_next) ** 2),
            'reward_lambda_smooth': -self.env_params.lambda_smooth_weight * (normalized_lambda_step ** 2),
            'reward_conflict': -self.env_params.conflict_weight * (conflict ** 2),
        }
        total = sum(terms.values())
        terms['reward_total_shaping'] = total
        terms['barrier_shortfall'] = barrier_shortfall
        return total, terms

    def _base_info(self, obs_info: Dict[str, float]) -> Dict[str, float]:
        return {
            't': self.state.t,
            'x': self.state.x,
            'y': self.state.y,
            'psi': self.state.psi,
            'vy': self.state.vy,
            'r': self.state.r,
            'delta_rwa': self.state.delta_rwa,
            'delta_swa': self.state.delta_swa,
            'lam': self.state.lam,
            **obs_info,
        }
