from __future__ import annotations

from shared_control_rl.config import EnvConfig
from shared_control_rl.envs.history_stack import ObservationHistoryStack
from shared_control_rl.envs.shared_control_env import SharedControlEnv


def make_env(config: EnvConfig | None = None, history_stack: int = 1):
    env = SharedControlEnv(config)
    if history_stack > 1:
        env = ObservationHistoryStack(env, n_stack=history_stack)
    return env
