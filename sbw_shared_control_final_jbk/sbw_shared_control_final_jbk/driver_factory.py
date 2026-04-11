from __future__ import annotations

from typing import Any, Tuple

from driver_simple import SimpleDriverParams, initialize_simple_driver_state, step_simple_driver
from driver_jbk import JBKDriverParams, initialize_jbk_driver_state, step_jbk_driver


DriverParamsType = SimpleDriverParams | JBKDriverParams


def default_driver_params(driver_model: str) -> DriverParamsType:
    if driver_model == 'simple':
        return SimpleDriverParams()
    if driver_model == 'jbk':
        return JBKDriverParams()
    raise ValueError(f'unknown driver_model={driver_model!r}')



def initialize_driver_state(driver_model: str, params: DriverParamsType, dt: float) -> Any:
    if driver_model == 'simple':
        return initialize_simple_driver_state(params, dt)
    if driver_model == 'jbk':
        return initialize_jbk_driver_state(params, dt)
    raise ValueError(f'unknown driver_model={driver_model!r}')



def step_driver(driver_model: str, driver_state: Any, vehicle_state, obstacle, params: DriverParamsType, dt: float, vx: float) -> Tuple[float, dict]:
    if driver_model == 'simple':
        return step_simple_driver(driver_state, vehicle_state, obstacle, params, dt, vx=vx)
    if driver_model == 'jbk':
        return step_jbk_driver(driver_state, vehicle_state, obstacle, params, dt, vx=vx)
    raise ValueError(f'unknown driver_model={driver_model!r}')
