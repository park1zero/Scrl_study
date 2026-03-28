from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np


def _first_threshold_time(signal: np.ndarray, t: np.ndarray, threshold: float) -> float:
    if len(signal) == 0 or len(t) == 0:
        return np.nan
    idx = np.flatnonzero(np.abs(signal) >= threshold)
    return float(t[idx[0]]) if len(idx) else np.nan


def summarize_history(history: Dict[str, list[float]], info: Dict[str, Any] | None = None) -> Dict[str, float]:
    """Summarize one episode history into scalar metrics.

    Parameters
    ----------
    history:
        The `env.history` dictionary collected during a rollout.
    info:
        Final `info` dict from the environment, if available.
    """
    if not history:
        return {
            "steps": 0.0,
            "episode_time": 0.0,
            "final_x": 0.0,
            "final_y": 0.0,
            "min_h": np.nan,
            "mean_lambda": np.nan,
            "min_lambda": np.nan,
            "mean_takeover": np.nan,
            "rms_conflict": np.nan,
            "rms_rwa": np.nan,
            "first_driver_reaction_time": np.nan,
            "first_motor_release_time": np.nan,
            "peak_driver_rwa": np.nan,
            "peak_driver_swa_cmd": np.nan,
            "collision": np.nan,
            "success": np.nan,
            "return": np.nan,
        }

    x = np.asarray(history.get("x", []), dtype=float)
    y = np.asarray(history.get("y", []), dtype=float)
    h = np.asarray(history.get("h", []), dtype=float)
    lam = np.asarray(history.get("lambda", []), dtype=float)
    delta_cmd = np.asarray(history.get("delta_cmd_rwa", []), dtype=float)
    delta_drv = np.asarray(history.get("delta_drv_rwa", []), dtype=float)
    reward = np.asarray(history.get("reward", []), dtype=float)
    t = np.asarray(history.get("t", []), dtype=float)
    driver_out = np.asarray(history.get("driver_output_swa_cmd", []), dtype=float)

    conflict = delta_cmd - delta_drv if len(delta_cmd) == len(delta_drv) and len(delta_cmd) > 0 else np.array([np.nan])

    collision = float(bool(info.get("collision", False))) if info is not None else float(np.min(h) < 0.0)
    success = float(bool(info.get("success", False))) if info is not None else 0.0

    return {
        "steps": float(len(x)),
        "episode_time": float(t[-1]) if len(t) else 0.0,
        "final_x": float(x[-1]) if len(x) else 0.0,
        "final_y": float(y[-1]) if len(y) else 0.0,
        "min_h": float(np.min(h)) if len(h) else np.nan,
        "mean_lambda": float(np.mean(lam)) if len(lam) else np.nan,
        "min_lambda": float(np.min(lam)) if len(lam) else np.nan,
        "mean_takeover": float(np.mean(1.0 - lam)) if len(lam) else np.nan,
        "rms_conflict": float(np.sqrt(np.nanmean(conflict ** 2))),
        "rms_rwa": float(np.sqrt(np.mean(delta_cmd ** 2))) if len(delta_cmd) else np.nan,
        "first_driver_reaction_time": _first_threshold_time(delta_drv, t, threshold=0.01),
        "first_motor_release_time": _first_threshold_time(driver_out, t, threshold=0.25),
        "peak_driver_rwa": float(np.max(np.abs(delta_drv))) if len(delta_drv) else np.nan,
        "peak_driver_swa_cmd": float(np.max(np.abs(driver_out))) if len(driver_out) else np.nan,
        "collision": collision,
        "success": success,
        "return": float(np.sum(reward)) if len(reward) else np.nan,
    }


def aggregate_episode_metrics(metrics: Iterable[Dict[str, float]]) -> Dict[str, float]:
    metrics = list(metrics)
    if not metrics:
        return {}

    keys = sorted(metrics[0].keys())
    out: Dict[str, float] = {}
    for key in keys:
        values = np.asarray([m[key] for m in metrics], dtype=float)
        out[f"{key}_mean"] = float(np.nanmean(values))
        out[f"{key}_std"] = float(np.nanstd(values))
    return out
