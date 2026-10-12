"""
EchoLocate — Experiment 2.5: Transition-Zone Verification & Threshold Refinement
================================================================================
Fine-grained sweep across nominal RT60 targets from 0.200s to 0.350s to locate
and characterize the acoustic localization failure boundary using measured Schroeder RT60.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyroomacoustics as pra

from src.evaluation.failure_analysis import (
    FailureClass,
    classify_evaluation_record,
    compute_per_source_sensitivity,
    analyze_failure_mechanisms,
)
from src.evaluation.threshold_analysis import analyze_reverberation_thresholds
from src.experiments.environment_robustness import (
    generate_exact_snr_noise,
    safe_inverse_sabine,
)
from src.localization.gcc_phat import compute_max_tau_from_geometry, estimate_all_tdoas
from src.localization.multilateration import localize_with_multiple_starts
from src.simulation.source_generator import generate_chirp

logger = logging.getLogger(__name__)


def measure_single_rir_rt60(
    rir: np.ndarray,
    fs: int,
) -> tuple[float, str]:
    """
    Estimate RT60 from an individual room impulse response using Schroeder integration
    with fallback extrapolation (RT30, RT20) and method tracking.
    """
    if len(rir) == 0 or np.all(rir == 0):
        return 0.0, "measurement_failed"
        
    try:
        # Energy decay curve
        energy = np.cumsum(rir[::-1] ** 2)[::-1]
        energy_db = 10.0 * np.log10(energy / np.max(energy) + 1e-12)
        
        # Try full 60 dB direct
        idx_5 = np.where(energy_db <= -5.0)[0]
        idx_65 = np.where(energy_db <= -65.0)[0]
        idx_35 = np.where(energy_db <= -35.0)[0]
        idx_25 = np.where(energy_db <= -25.0)[0]
        
        if len(idx_5) > 0 and len(idx_65) > 0 and idx_65[0] > idx_5[0]:
            t5 = idx_5[0] / fs
            t65 = idx_65[0] / fs
            rt60 = float(t65 - t5)
            if 0.01 < rt60 < 5.0:
                return rt60, "RT60_direct"
                
        # Try T30 extrapolated
        if len(idx_5) > 0 and len(idx_35) > 0 and idx_35[0] > idx_5[0]:
            t5 = idx_5[0] / fs
            t35 = idx_35[0] / fs
            t30 = float(t35 - t5)
            rt60 = float(2.0 * t30)
            if 0.01 < rt60 < 5.0:
                return rt60, "RT30_extrapolated"
                
        # Try T20 extrapolated
        if len(idx_5) > 0 and len(idx_25) > 0 and idx_25[0] > idx_5[0]:
            t5 = idx_5[0] / fs
            t25 = idx_25[0] / fs
            t20 = float(t25 - t5)
            rt60 = float(3.0 * t20)
            if 0.01 < rt60 < 5.0:
                return rt60, "RT20_extrapolated"
                
        # Fallback to pra built-in measure_rt60
        val = pra.experimental.measure_rt60(rir, fs=fs, decay_db=30)
        if np.isfinite(val) and 0.01 < val < 5.0:
            return float(val), "RT30_extrapolated"
            
        return 0.0, "measurement_failed"
    except Exception as e:
        logger.debug("RIR RT60 measurement exception: %s", e)
        return 0.0, "measurement_failed"


def run_transition_zone_experiment(
    cfg: dict,
    output_dir: Path,
    smoke_test: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Execute Experiment 2.5 Transition-Zone Sweep.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = output_dir / "diagnostics" / "gcc_phat"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # 1. Unpack Config
    room_dims = cfg["room"]["dimensions"]
    fs = cfg["room"]["sampling_rate"]
    speed_of_sound = cfg["room"]["speed_of_sound"]
    base_seed = cfg["experiment"]["base_seed"]
    mic_height = cfg["microphones"]["height"]
    mic_positions_cfg = cfg["microphones"]["positions"]
    
    # 2. Source positions
    src_file = Path(cfg["source"]["positions_file"])
    if not src_file.is_absolute():
        src_file = Path(__file__).resolve().parent.parent.parent / src_file
    df_sources = pd.read_csv(src_file)
    
    # 3. Microphones
    mics_2d = np.array([
        mic_positions_cfg["M1"],
        mic_positions_cfg["M2"],
        mic_positions_cfg["M3"],
    ])  # Shape (3, 2)
    mics_3d = np.array([
        [mic_positions_cfg["M1"][0], mic_positions_cfg["M1"][1], mic_height],
        [mic_positions_cfg["M2"][0], mic_positions_cfg["M2"][1], mic_height],
        [mic_positions_cfg["M3"][0], mic_positions_cfg["M3"][1], mic_height],
    ]).T  # Shape (3, 3) for pyroomacoustics

    # 4. Source Signal
    sig_cfg = cfg["source"]["signal_parameters"]
    clean_chirp = generate_chirp(
        fs=fs,
        duration=sig_cfg["duration"],
        f_low=sig_cfg["f_start"],
        f_high=sig_cfg["f_end"],
        amplitude=sig_cfg["amplitude"],
    )

    # 5. Solver & GCC-PHAT Settings
    max_tau_samples = compute_max_tau_from_geometry(
        mic_positions_2d=mics_2d,
        speed_of_sound=speed_of_sound,
        fs=fs,
        safety_factor=1.3,
    )

    # 6. Conditions Sweep
    if smoke_test:
        target_rt60_list = [0.200, 0.275, 0.350]
        source_ids = ["S01", "S08", "S15"]
        source_positions = df_sources[df_sources["source_id"].isin(source_ids)][["x", "y"]].values
        num_trials = 2
        snr_db = cfg["environment"]["primary_snr_db"]
        logger.info("Running Experiment 2.5 in SMOKE TEST mode (3 RT60 x 3 Sources x 2 Trials)")
    else:
        target_rt60_list = cfg["environment"]["target_rt60_values"]
        source_ids = df_sources["source_id"].tolist()
        source_positions = df_sources[["x", "y"]].values
        num_trials = cfg["trials"]["num_trials"]
        snr_db = cfg["environment"]["primary_snr_db"]
        logger.info("Running Experiment 2.5 FULL SWEEP (%d RT60 x %d Sources x %d Trials = %d Evals)",
                    len(target_rt60_list), len(source_ids), num_trials, len(target_rt60_list) * len(source_ids) * num_trials)

    # Output storage
    rir_measurement_records = []
    rt60_summary_records = []
    per_evaluation_records = []
    condition_summary_records = []

    # Iterate through nominal RT60 targets
    for cond_idx, rt60_tgt in enumerate(target_rt60_list):
        logger.info("------------------------------------------------------------")
        logger.info("Simulating Room for Nominal RT60 Target: %.3f s", rt60_tgt)
        
        e_abs, used_order = safe_inverse_sabine(rt60_tgt, room_dims)
        
        # Build room for RIR measurement and clean simulation
        room = pra.ShoeBox(
            room_dims,
            fs=fs,
            materials=pra.Material(e_abs),
            max_order=used_order,
            air_absorption=False,
            ray_tracing=False,
        )
        room.add_microphone_array(pra.MicrophoneArray(mics_3d, fs=fs))
        
        # Add all sources to measure 45 RIRs
        for s_id, s_xy in zip(source_ids, source_positions):
            room.add_source([s_xy[0], s_xy[1], mic_height], signal=clean_chirp)
            
        room.compute_rir()
        
        # Measure RIR RT60 across all paths
        measured_vals = []
        for s_idx, s_id in enumerate(source_ids):
            for m_idx in range(3):
                rir = room.rir[m_idx][s_idx]
                meas_rt60, meas_method = measure_single_rir_rt60(rir, fs)
                
                is_valid = (meas_rt60 > 0.0 and meas_method != "measurement_failed")
                if is_valid:
                    measured_vals.append(meas_rt60)
                    
                rir_measurement_records.append({
                    "target_rt60": rt60_tgt,
                    "source_id": s_id,
                    "microphone_id": f"M{m_idx+1}",
                    "measured_rt60": meas_rt60,
                    "measurement_status": "valid" if is_valid else "failed",
                    "rt60_measurement_method": meas_method,
                })
                
        # Aggregate RIR statistics for this condition
        if len(measured_vals) > 0:
            m_mean = float(np.mean(measured_vals))
            m_median = float(np.median(measured_vals))
            m_std = float(np.std(measured_vals))
            m_min = float(np.min(measured_vals))
            m_max = float(np.max(measured_vals))
        else:
            m_mean = rt60_tgt
            m_median = rt60_tgt
            m_std = 0.0
            m_min = rt60_tgt
            m_max = rt60_tgt
            
        num_valid = len(measured_vals)
        num_failed = len(source_ids) * 3 - num_valid
        
        rt60_summary_records.append({
            "target_rt60": rt60_tgt,
            "mean_measured_rt60": m_mean,
            "median_measured_rt60": m_median,
            "std_measured_rt60": m_std,
            "min_measured_rt60": m_min,
            "max_measured_rt60": m_max,
            "num_valid_measurements": num_valid,
            "num_failed_measurements": num_failed,
        })
        logger.info("  Measured RT60 Distribution: Mean=%.4fs, Med=%.4fs, Std=%.4fs (Min=%.4fs, Max=%.4fs)",
                    m_mean, m_median, m_std, m_min, m_max)

        # Pre-simulate clean mic signals for each source individually
        clean_signals_cache = {}
        for s_id, s_xy in zip(source_ids, source_positions):
            single_room = pra.ShoeBox(
                room_dims,
                fs=fs,
                materials=pra.Material(e_abs),
                max_order=used_order,
                air_absorption=False,
                ray_tracing=False,
            )
            single_room.add_microphone_array(pra.MicrophoneArray(mics_3d, fs=fs))
            single_room.add_source([s_xy[0], s_xy[1], mic_height], signal=clean_chirp)
            single_room.simulate()
            clean_signals_cache[s_id] = single_room.mic_array.signals.copy()

        # Run multi-trial noise evaluations
        cond_eval_records = []
        for trial_idx in range(num_trials):
            for s_idx, (s_id, s_xy) in enumerate(zip(source_ids, source_positions)):
                trial_seed = base_seed + int(rt60_tgt * 10000) + trial_idx * 100 + s_idx * 7
                clean_sig = clean_signals_cache[s_id]
                
                # Add AWGN with exact power scaling
                noisy_sig, achieved_snr = generate_exact_snr_noise(clean_sig, target_snr_db=snr_db, seed=trial_seed)
                
                # Estimate TDOA
                tdoa_results = estimate_all_tdoas(noisy_sig, fs=fs, max_tau_samples=max_tau_samples)
                
                # Theoretical TDOAs
                d0 = np.linalg.norm(s_xy - mics_2d[0])
                d1 = np.linalg.norm(s_xy - mics_2d[1])
                d2 = np.linalg.norm(s_xy - mics_2d[2])
                theo_01 = float((d0 - d1) / speed_of_sound)
                theo_02 = float((d0 - d2) / speed_of_sound)
                theo_12 = float((d1 - d2) / speed_of_sound)
                
                t_err_01 = abs(tdoa_results[(0, 1)].tau_seconds - theo_01)
                t_err_02 = abs(tdoa_results[(0, 2)].tau_seconds - theo_02)
                t_err_12 = abs(tdoa_results[(1, 2)].tau_seconds - theo_12)
                
                mean_t_err_s = float(np.mean([t_err_01, t_err_02, t_err_12]))
                mean_t_err_ms = mean_t_err_s * 1000.0
                mean_t_err_samples = mean_t_err_s * fs
                max_t_err_ms = float(np.max([t_err_01, t_err_02, t_err_12])) * 1000.0
                rmse_t_err_ms = float(np.sqrt(np.mean([t_err_01**2, t_err_02**2, t_err_12**2]))) * 1000.0
                
                # Localize using NLS with multiple random starts
                tdoa_dict = {pair: res.tau_seconds for pair, res in tdoa_results.items()}
                loc_res = localize_with_multiple_starts(
                    mic_positions_2d=mics_2d,
                    tdoa_dict=tdoa_dict,
                    speed_of_sound=speed_of_sound,
                    room_dimensions=room_dims,
                    n_restarts=9,
                    rng_seed=trial_seed,
                    true_position=np.array(s_xy),
                )
                
                pred_x = float(loc_res.predicted_position[0])
                pred_y = float(loc_res.predicted_position[1])
                loc_err = float(loc_res.localization_error)
                
                fail_cat = classify_evaluation_record(
                    pred_x=pred_x,
                    pred_y=pred_y,
                    loc_error=loc_err,
                    converged=loc_res.converged,
                    room_dims=room_dims,
                    catastrophic_thresh=cfg["evaluation"]["catastrophic_failure_threshold"],
                )
                
                rec = {
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
                    "pred_x": pred_x,
                    "pred_y": pred_y,
                    "x_error": pred_x - s_xy[0],
                    "y_error": pred_y - s_xy[1],
                    "localization_error": loc_err,
                    "solver_converged": loc_res.converged,
                    "solver_status": loc_res.solver_message,
                    "failure_classification": fail_cat.value,
                    "is_catastrophic_failure": bool(fail_cat == FailureClass.CATASTROPHIC_ERROR),
                    "mean_tdoa_error_s": mean_t_err_s,
                    "mean_tdoa_error_ms": mean_t_err_ms,
                    "mean_tdoa_error_samples": mean_t_err_samples,
                    "max_tdoa_error_ms": max_t_err_ms,
                    "rmse_tdoa_error_ms": rmse_t_err_ms,
                    "tdoa_01_estimated_s": tdoa_results[(0, 1)].tau_seconds,
                    "tdoa_01_theoretical_s": theo_01,
                    "tdoa_02_estimated_s": tdoa_results[(0, 2)].tau_seconds,
                    "tdoa_02_theoretical_s": theo_02,
                    "tdoa_12_estimated_s": tdoa_results[(1, 2)].tau_seconds,
                    "tdoa_12_theoretical_s": theo_12,
                }
                per_evaluation_records.append(rec)
                cond_eval_records.append(rec)
                
                # Save sample GCC-PHAT correlations for diagnostics (S08 center, trial 0)
                if s_id == "S08" and trial_idx == 0 and cond_idx in [0, len(target_rt60_list)//2, len(target_rt60_list)-1]:
                    np.savez(
                        diag_dir / f"gcc_phat_rt60_{rt60_tgt:.3f}_{s_id}.npz",
                        corr_01=tdoa_results[(0, 1)].gcc_phat_response,
                        tau_axis_01=tdoa_results[(0, 1)].tau_axis_seconds,
                        tau_est_01=tdoa_results[(0, 1)].tau_seconds,
                        tau_theo_01=theo_01,
                    )

        # Aggregate Condition Metrics
        df_c = pd.DataFrame(cond_eval_records)
        loc_errs = df_c["localization_error"].values
        tdoa_errs_ms = df_c["mean_tdoa_error_ms"].values
        
        n_evals = len(df_c)
        cat_count = df_c["is_catastrophic_failure"].sum()
        
        c_mean = float(np.mean(loc_errs))
        c_std = float(np.std(loc_errs, ddof=1)) if n_evals > 1 else 0.0
        c_median = float(np.median(loc_errs))
        c_rmse = float(np.sqrt(np.mean(loc_errs**2)))
        c_min = float(np.min(loc_errs))
        c_max = float(np.max(loc_errs))
        c_p90 = float(np.percentile(loc_errs, 90))
        c_p95 = float(np.percentile(loc_errs, 95))
        
        # 95% Confidence interval
        sem = c_std / np.sqrt(n_evals) if n_evals > 1 else 0.0
        ci_low = max(0.0, c_mean - 1.96 * sem)
        ci_high = c_mean + 1.96 * sem
        
        condition_summary_records.append({
            "rt60_target_s": rt60_tgt,
            "rt60_measured_mean_s": m_mean,
            "rt60_measured_median_s": m_median,
            "rt60_measured_std_s": m_std,
            "rt60_measured_min_s": m_min,
            "rt60_measured_max_s": m_max,
            "snr_target_db": snr_db,
            "total_evaluations": n_evals,
            "mean_localization_error_m": c_mean,
            "std_localization_error_m": c_std,
            "median_localization_error_m": c_median,
            "rmse_localization_error_m": c_rmse,
            "min_localization_error_m": c_min,
            "max_localization_error_m": c_max,
            "p90_localization_error_m": c_p90,
            "p95_localization_error_m": c_p95,
            "ci95_low_m": ci_low,
            "ci95_high_m": ci_high,
            "pct_within_0_25m": float(np.sum(loc_errs <= 0.25) / n_evals * 100.0),
            "pct_within_0_5m": float(np.sum(loc_errs <= 0.50) / n_evals * 100.0),
            "pct_within_1_0m": float(np.sum(loc_errs <= 1.00) / n_evals * 100.0),
            "pct_within_2_0m": float(np.sum(loc_errs <= 2.00) / n_evals * 100.0),
            "pct_catastrophic_failure": float(cat_count / n_evals * 100.0),
            "pct_solver_failure": float((df_c["failure_classification"] == "solver_failed").sum() / n_evals * 100.0),
            "pct_nan_inf": float((df_c["failure_classification"] == "nan_inf").sum() / n_evals * 100.0),
            "mean_tdoa_error_ms": float(np.mean(tdoa_errs_ms)),
            "median_tdoa_error_ms": float(np.median(tdoa_errs_ms)),
            "rmse_tdoa_error_ms": float(np.sqrt(np.mean(tdoa_errs_ms**2))),
            "max_tdoa_error_ms": float(np.max(tdoa_errs_ms)),
            "x_bias_m": float(df_c["x_error"].mean()),
            "y_bias_m": float(df_c["y_error"].mean()),
        })
        logger.info("  Target RT60=%.3fs (Meas=%.4fs) -> Mean Err: %.3fm | Med Err: %.3fm | <=1m: %.1f%% | Catastrophic: %.1f%%",
                    rt60_tgt, m_mean, c_mean, c_median, float(np.sum(loc_errs <= 1.00) / n_evals * 100.0), float(cat_count / n_evals * 100.0))

    # 7. Convert and Export DataFrames
    df_rt60_meas = pd.DataFrame(rir_measurement_records)
    df_rt60_summary = pd.DataFrame(rt60_summary_records)
    df_evals = pd.DataFrame(per_evaluation_records)
    df_summary = pd.DataFrame(condition_summary_records)
    
    df_rt60_meas.to_csv(output_dir / "rt60_measurements.csv", index=False)
    df_rt60_summary.to_csv(output_dir / "rt60_summary.csv", index=False)
    df_evals.to_csv(output_dir / "per_evaluation.csv", index=False)
    df_summary.to_csv(output_dir / "summary.csv", index=False)
    
    # 8. Threshold & Failure Mechanism Analysis
    threshold_results, df_slopes = analyze_reverberation_thresholds(df_summary)
    df_slopes.to_csv(output_dir / "threshold_analysis.csv", index=False)
    
    df_per_source = compute_per_source_sensitivity(
        df_evals=df_evals,
        df_summary=df_summary,
        catastrophic_thresh=cfg["evaluation"]["catastrophic_failure_threshold"],
    )
    df_per_source.to_csv(output_dir / "per_source_summary.csv", index=False)
    
    df_mechanisms = analyze_failure_mechanisms(df_summary, speed_of_sound=speed_of_sound)
    df_mechanisms.to_csv(output_dir / "failure_analysis.csv", index=False)
    
    logger.info("============================================================")
    logger.info("Experiment 2.5 Transition-Zone Sweep Complete!")
    logger.info("Interpretation: %s", threshold_results.interpretation_note)
    logger.info("Results successfully saved to: %s", output_dir)
    logger.info("============================================================")
    
    return df_summary, df_evals, df_rt60_summary, df_per_source
