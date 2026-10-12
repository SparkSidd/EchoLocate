"""
EchoLocate — Experiment 3: Spatial Likelihood SRP-PHAT Baseline Runner
======================================================================
Executes strict paired evaluations between frozen classical GCC-PHAT + NLS
and spatial-likelihood SRP-PHAT on identical multi-channel acoustic signals.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyroomacoustics as pra
import yaml

from src.evaluation.comparison_metrics import (
    compute_paired_comparison_summary,
    compute_per_source_comparison,
)
from src.evaluation.spatial_response_analysis import run_grid_resolution_ablation
from src.experiments.environment_robustness import (
    generate_exact_snr_noise,
    safe_inverse_sabine,
)
from src.experiments.transition_zone import measure_single_rir_rt60
from src.localization.gcc_phat import compute_max_tau_from_geometry, estimate_all_tdoas
from src.localization.multilateration import localize_with_multiple_starts
from src.localization.srp_phat import SRPGrid, localize_srp_phat
from src.simulation.source_generator import generate_chirp

logger = logging.getLogger(__name__)


def run_experiment_03_spatial_likelihood(
    cfg: dict[str, Any] | None = None,
    config_path: str | Path = "configs/experiment_03.yaml",
    output_dir: str | Path = "results/experiment_03",
    smoke_test: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Run full Experiment 3 paired evaluation sweep.
    """
    if cfg is None:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    figures_dir = out_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = out_path / "spatial_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    # 1. Global Setup
    base_seed = cfg["reproducibility"]["base_seed"]
    room_dims = cfg["room"]["dimensions"]
    c = cfg["room"]["speed_of_sound"]
    fs = cfg["room"]["fs"]
    max_order = cfg["room"]["max_order"]
    air_abs = cfg["room"]["air_absorption"]

    # Microphones: Baseline Geometry A
    mics_2d = np.array(cfg["microphones"]["positions_2d"], dtype=np.float64)
    mic_height = cfg["microphones"]["height"]
    mics_3d = np.column_stack([mics_2d, np.full(len(mics_2d), mic_height)]).T  # (3, n_mics)

    # Source Signal
    sig_cfg = cfg["source"]["signal_parameters"]
    clean_chirp = generate_chirp(
        fs=fs,
        duration=sig_cfg["duration"],
        f_low=sig_cfg["f_start"],
        f_high=sig_cfg["f_end"],
        amplitude=sig_cfg["amplitude"],
    )
    src_height = cfg["source"]["height"]

    # Canonical Sources
    canonical_sources = cfg["source"]["canonical_positions_2d"]
    if smoke_test:
        canonical_sources = canonical_sources[:3]  # S01, S02, S03
        n_trials = 2
        conditions = cfg["conditions"][:2]
    else:
        n_trials = cfg["trials_per_condition"]
        conditions = cfg["conditions"]

    # Solver & SRP Settings
    max_tau_samples = compute_max_tau_from_geometry(
        mic_positions_2d=mics_2d,
        speed_of_sound=c,
        fs=fs,
        safety_factor=1.3,
    )

    srp_cfg = cfg["srp_phat"]
    srp_res_m = srp_cfg["grid_resolution_m"]
    srp_margin_m = srp_cfg["boundary_margin_m"]
    interp_method = srp_cfg["interpolation_method"]
    subgrid_refine = srp_cfg["subgrid_refinement"]
    min_peak_sep = srp_cfg["min_peak_separation_m"]
    top_k_peaks = srp_cfg["top_k_peaks"]

    # Pre-instantiate standard benchmark SRP Grid
    grid_fine = SRPGrid(
        room_dimensions=room_dims,
        mic_positions_2d=mics_2d,
        grid_resolution_m=srp_res_m,
        boundary_margin_m=srp_margin_m,
        speed_of_sound=c,
        source_height=src_height,
        mic_height=mic_height,
    )

    logger.info("============================================================")
    logger.info("Starting Experiment 3: Spatial Likelihood SRP-PHAT Baseline")
    logger.info("Conditions: %d | Sources: %d | Trials: %d | Total Evals per method: %d",
                len(conditions), len(canonical_sources), n_trials, len(conditions) * len(canonical_sources) * n_trials)
    logger.info("============================================================")

    # 2. Main Simulation & Paired Evaluation Loop
    eval_records = []
    all_rir_measurements = []
    rt60_summary_rows = []
    saved_diagnostic_maps = {}

    ablation_signals = []
    ablation_true_pos = []

    cond_idx = 0
    for cond in conditions:
        cond_id = cond["id"]
        rt60_tgt = float(cond["target_rt60"])
        snr_db = float(cond["snr_target_db"])

        logger.info("------------------------------------------------------------")
        logger.info("Condition: %s (Target RT60=%.3fs, SNR=%.1fdB)", cond_id, rt60_tgt, snr_db)

        # Build Pyroomacoustics Room
        absorption, max_order_room = safe_inverse_sabine(
            rt60_tgt=rt60_tgt,
            room_dims=room_dims,
        )
        room = pra.ShoeBox(
            room_dims,
            fs=fs,
            materials=pra.Material(absorption),
            max_order=max_order_room,
            air_absorption=air_abs,
            ray_tracing=False,
        )
        room.add_microphone_array(pra.MicrophoneArray(mics_3d, fs=fs))

        # Add all sources
        for src in canonical_sources:
            s_pos_3d = [src["xy"][0], src["xy"][1], src_height]
            room.add_source(s_pos_3d, signal=clean_chirp)

        room.compute_rir()

        # Measure Schroeder RT60 on all 45 paths
        rir_vals = []
        for s_idx, src in enumerate(canonical_sources):
            for m_idx in range(len(mics_2d)):
                rir_signal = room.rir[m_idx][s_idx]
                m_rt60, m_method = measure_single_rir_rt60(rir_signal, fs=fs)
                m_valid = bool(m_method != "measurement_failed" and m_rt60 > 0.0)
                if m_valid:
                    rir_vals.append(m_rt60)
                all_rir_measurements.append({
                    "condition_id": cond_id,
                    "target_rt60_s": rt60_tgt,
                    "snr_target_db": snr_db,
                    "source_id": src["id"],
                    "mic_index": m_idx,
                    "measured_rt60_s": m_rt60 if m_valid else np.nan,
                    "measurement_method": m_method,
                    "is_valid": m_valid,
                })

        m_mean = float(np.mean(rir_vals)) if rir_vals else rt60_tgt
        m_median = float(np.median(rir_vals)) if rir_vals else rt60_tgt
        m_std = float(np.std(rir_vals)) if rir_vals else 0.0

        rt60_summary_rows.append({
            "condition_id": cond_id,
            "target_rt60_s": rt60_tgt,
            "measured_mean_rt60_s": m_mean,
            "measured_median_rt60_s": m_median,
            "measured_std_rt60_s": m_std,
            "num_valid_measurements": len(rir_vals),
        })

        # Evaluate all sources across trials
        for s_idx, src in enumerate(canonical_sources):
            s_id = src["id"]
            s_xy = src["xy"]
            s_pos_3d = [s_xy[0], s_xy[1], src_height]

            # Generate individual source room for clean multi-channel simulation
            src_room = pra.ShoeBox(
                room_dims,
                fs=fs,
                materials=pra.Material(absorption),
                max_order=max_order_room,
                air_absorption=air_abs,
                ray_tracing=False,
            )
            src_room.add_microphone_array(pra.MicrophoneArray(mics_3d, fs=fs))
            src_room.add_source(s_pos_3d, signal=clean_chirp)
            src_room.simulate()

            clean_mic_signals = src_room.mic_array.signals.copy()

            # Trials loop
            for trial_idx in range(n_trials):
                trial_seed = base_seed + cond_idx * 1000 + s_idx * 50 + trial_idx
                noisy_signals, achieved_snr = generate_exact_snr_noise(
                    signals=clean_mic_signals,
                    target_snr_db=snr_db,
                    seed=trial_seed,
                )

                if cond_id == "Condition_B_Transition_Edge" and len(ablation_signals) < 15:
                    ablation_signals.append(noisy_signals.copy())
                    ablation_true_pos.append(np.array(s_xy))

                # --- 1. Classical Baseline: GCC-PHAT + NLS ---
                t0_nls = time.perf_counter()
                tdoa_results = estimate_all_tdoas(
                    mic_signals=noisy_signals,
                    fs=fs,
                    max_tau_samples=max_tau_samples,
                )
                tdoa_dict = {pair: res.tau_seconds for pair, res in tdoa_results.items()}
                nls_res = localize_with_multiple_starts(
                    mic_positions_2d=mics_2d,
                    tdoa_dict=tdoa_dict,
                    speed_of_sound=c,
                    room_dimensions=room_dims,
                    n_restarts=9,
                    rng_seed=trial_seed,
                    true_position=np.array(s_xy),
                )
                t_nls_ms = (time.perf_counter() - t0_nls) * 1000.0

                nls_pred_x = float(nls_res.predicted_position[0])
                nls_pred_y = float(nls_res.predicted_position[1])
                nls_loc_err = float(nls_res.localization_error)

                # --- 2. Spatial Likelihood: SRP-PHAT ---
                srp_res = localize_srp_phat(
                    mic_signals=noisy_signals,
                    mic_positions_2d=mics_2d,
                    fs=fs,
                    room_dimensions=room_dims,
                    grid=grid_fine,
                    grid_resolution_m=srp_res_m,
                    boundary_margin_m=srp_margin_m,
                    speed_of_sound=c,
                    interpolation_method=interp_method,
                    subgrid_refinement=subgrid_refine,
                    min_peak_separation_m=min_peak_sep,
                    top_k_peaks=top_k_peaks,
                    true_position_2d=np.array(s_xy),
                )

                # Save diagnostic map for first trial of representative sources
                if trial_idx == 0 and s_id in ["S06", "S10", "S04"]:
                    map_key = f"{cond_id}_{s_id}"
                    saved_diagnostic_maps[map_key] = {
                        "condition_id": cond_id,
                        "source_id": s_id,
                        "true_xy": s_xy,
                        "nls_pred_xy": [nls_pred_x, nls_pred_y],
                        "nls_error": nls_loc_err,
                        "srp_raw_xy": srp_res.raw_grid_position.tolist(),
                        "srp_refined_xy": srp_res.refined_position.tolist(),
                        "srp_error": srp_res.refined_error,
                        "srp_map": srp_res.srp_map_2d.tolist(),
                        "x_coords": srp_res.x_coords.tolist(),
                        "y_coords": srp_res.y_coords.tolist(),
                        "top_k_peaks": [
                            {"x": p.grid_x, "y": p.grid_y, "power": p.power, "rank": p.rank}
                            for p in srp_res.top_k_peaks
                        ],
                    }

                eval_records.append({
                    "condition_id": cond_id,
                    "target_rt60_s": rt60_tgt,
                    "measured_mean_rt60_s": m_mean,
                    "measured_median_rt60_s": m_median,
                    "measured_std_rt60_s": m_std,
                    "snr_target_db": snr_db,
                    "achieved_snr_db": achieved_snr,
                    "trial_idx": trial_idx,
                    "trial_seed": trial_seed,
                    "source_id": s_id,
                    "true_x": s_xy[0],
                    "true_y": s_xy[1],
                    # NLS Predictions
                    "nls_pred_x": nls_pred_x,
                    "nls_pred_y": nls_pred_y,
                    "nls_localization_error": nls_loc_err,
                    "nls_converged": nls_res.converged,
                    "nls_runtime_ms": t_nls_ms,
                    # SRP Predictions
                    "srp_raw_x": float(srp_res.raw_grid_position[0]),
                    "srp_raw_y": float(srp_res.raw_grid_position[1]),
                    "srp_raw_error": float(srp_res.raw_grid_error),
                    "srp_refined_x": float(srp_res.refined_position[0]),
                    "srp_refined_y": float(srp_res.refined_position[1]),
                    "srp_refined_error": float(srp_res.refined_error),
                    "srp_primary_peak_power": srp_res.primary_peak_power,
                    "srp_secondary_peak_power": srp_res.secondary_peak_power,
                    "srp_peak_ratio": srp_res.peak_ratio,
                    "srp_peak_separation_m": srp_res.peak_separation_m,
                    "srp_top_k_proximity": srp_res.top_k_proximity,
                    "srp_runtime_ms": srp_res.eval_runtime_ms,
                    # Paired Delta
                    "delta_error_m": float(srp_res.refined_error - nls_loc_err),
                    "srp_wins": bool(srp_res.refined_error < nls_loc_err),
                })

        cond_idx += 1

    df_evals = pd.DataFrame(eval_records)
    df_rir_meas = pd.DataFrame(all_rir_measurements)
    df_rt60_sum = pd.DataFrame(rt60_summary_rows)

    # 3. Compute Summary Metrics
    df_summary = compute_paired_comparison_summary(
        df_evals=df_evals,
        catastrophic_threshold=cfg["evaluation"]["catastrophic_failure_threshold_m"],
    )

    df_per_source = compute_per_source_comparison(
        df_evals=df_evals,
        catastrophic_threshold=cfg["evaluation"]["catastrophic_failure_threshold_m"],
    )

    # 4. Resolution Ablation Study (if not smoke test)
    if not smoke_test and ablation_signals:
        logger.info("Executing Grid Resolution Ablation Study...")
        res_list = cfg["resolution_ablation_study"]["resolutions_m"]
        df_resolution = run_grid_resolution_ablation(
            mic_signals_list=ablation_signals,
            true_positions_list=ablation_true_pos,
            mic_positions_2d=mics_2d,
            room_dimensions=room_dims,
            fs=fs,
            resolutions=res_list,
            boundary_margin_m=srp_margin_m,
            speed_of_sound=c,
        )
    else:
        df_resolution = pd.DataFrame()

    # 5. Export CSVs
    df_summary.to_csv(out_path / "summary.csv", index=False)
    df_evals.to_csv(out_path / "per_evaluation.csv", index=False)
    df_per_source.to_csv(out_path / "per_source_summary.csv", index=False)
    df_rir_meas.to_csv(out_path / "rt60_measurements.csv", index=False)
    df_rt60_sum.to_csv(out_path / "rt60_summary.csv", index=False)
    if not df_resolution.empty:
        df_resolution.to_csv(out_path / "resolution_study.csv", index=False)

    # Export diagnostic maps JSON
    import json
    with open(out_path / "diagnostic_maps.json", "w", encoding="utf-8") as f:
        json.dump(saved_diagnostic_maps, f, indent=2)

    logger.info("============================================================")
    logger.info("Experiment 3 Execution Complete! Results saved to: %s", out_path)
    logger.info("============================================================")

    return df_summary, df_evals, df_per_source, df_resolution, df_rt60_sum, saved_diagnostic_maps
