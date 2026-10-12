"""
EchoLocate — Spatial Response Analysis Module (Experiment 3)
============================================================
Diagnostic tools to extract spatial map profiles, peak competition landscapes,
and grid resolution ablation statistics.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from src.localization.srp_phat import SRPGrid, localize_srp_phat

logger = logging.getLogger(__name__)


def run_grid_resolution_ablation(
    mic_signals_list: list[np.ndarray],
    true_positions_list: list[np.ndarray],
    mic_positions_2d: np.ndarray,
    room_dimensions: list[float],
    fs: int,
    resolutions: list[float] = [0.10, 0.05, 0.025],
    boundary_margin_m: float = 0.10,
    speed_of_sound: float = 343.0,
) -> pd.DataFrame:
    """
    Evaluate localization accuracy and runtime across multiple spatial grid resolutions.
    """
    import time
    records = []

    for res in resolutions:
        logger.info("Running Resolution Ablation for grid spacing: %.3f m", res)
        # Precompute grid
        t0_grid = time.perf_counter()
        grid = SRPGrid(
            room_dimensions=room_dimensions,
            mic_positions_2d=mic_positions_2d,
            grid_resolution_m=res,
            boundary_margin_m=boundary_margin_m,
            speed_of_sound=speed_of_sound,
        )
        grid_build_time_ms = (time.perf_counter() - t0_grid) * 1000.0

        raw_errors = []
        refined_errors = []
        runtimes_ms = []

        for signals, true_pos in zip(mic_signals_list, true_positions_list):
            res_obj = localize_srp_phat(
                mic_signals=signals,
                mic_positions_2d=mic_positions_2d,
                fs=fs,
                room_dimensions=room_dimensions,
                grid=grid,
                grid_resolution_m=res,
                boundary_margin_m=boundary_margin_m,
                speed_of_sound=speed_of_sound,
                true_position_2d=true_pos,
            )
            raw_errors.append(res_obj.raw_grid_error)
            refined_errors.append(res_obj.refined_error)
            runtimes_ms.append(res_obj.eval_runtime_ms)

        records.append({
            "grid_resolution_m": res,
            "grid_points_total": grid.n_points,
            "grid_nx": grid.n_x,
            "grid_ny": grid.n_y,
            "grid_build_time_ms": grid_build_time_ms,
            "mean_raw_error_m": float(np.mean(raw_errors)),
            "median_raw_error_m": float(np.median(raw_errors)),
            "mean_refined_error_m": float(np.mean(refined_errors)),
            "median_refined_error_m": float(np.median(refined_errors)),
            "pct_within_0_25m": float(np.mean(np.array(refined_errors) <= 0.25) * 100.0),
            "pct_within_0_5m": float(np.mean(np.array(refined_errors) <= 0.50) * 100.0),
            "pct_within_1_0m": float(np.mean(np.array(refined_errors) <= 1.0) * 100.0),
            "mean_runtime_per_frame_ms": float(np.mean(runtimes_ms)),
            "median_runtime_per_frame_ms": float(np.median(runtimes_ms)),
            "p95_runtime_per_frame_ms": float(np.percentile(runtimes_ms, 95)),
        })

    return pd.DataFrame(records)
