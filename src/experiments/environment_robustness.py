"""
EchoLocate — Part A: Environmental Robustness Study
====================================================
Executes the systematic 42-condition (7 RT60 x 6 SNR x 5 trials) acoustic
robustness grid evaluation using GCC-PHAT and NLS multilateration.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyroomacoustics as pra

from src.simulation.room_simulator import build_mic_array_3d
from src.simulation.source_generator import generate_source_signal
from src.localization.gcc_phat import (
    estimate_all_tdoas,
    compute_max_tau_from_geometry,
    TDOAResult,
)
from src.localization.multilateration import localize_with_multiple_starts
from src.evaluation.robustness_metrics import compute_condition_summary, ConditionSummary

logger = logging.getLogger(__name__)


def generate_exact_snr_noise(
    signals: np.ndarray,
    target_snr_db: float,
    seed: int,
) -> tuple[np.ndarray, float]:
    """
    Inject additive white Gaussian noise (AWGN) according to exact target SNR.

    Formula:
        SNR_dB = 10 * log10(P_signal / P_noise)
        P_noise = P_signal / (10^(target_snr_db / 10))

    Parameters
    ----------
    signals : np.ndarray
        Shape (n_channels, n_samples) multichannel audio array.
    target_snr_db : float
        Target signal-to-noise ratio in decibels.
    seed : int
        Deterministic random seed for noise realization.

    Returns
    -------
    noisy_signals : np.ndarray
        Signals with injected AWGN.
    achieved_snr_db : float
        Actual measured SNR achieved in the generated noise.
    """
    rng = np.random.default_rng(seed)
    signal_power = float(np.mean(signals ** 2))
    
    if signal_power < 1e-30:
        return signals.copy(), target_snr_db
        
    snr_linear = 10.0 ** (target_snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power)
    
    noise = rng.normal(0.0, noise_std, size=signals.shape)
    actual_noise_power = float(np.mean(noise ** 2))
    
    if actual_noise_power > 1e-30:
        achieved_snr = 10.0 * np.log10(signal_power / actual_noise_power)
    else:
        achieved_snr = target_snr_db
        
    return signals + noise, float(achieved_snr)


def measure_room_rt60_distribution(
    room_dims: list[float],
    e_absorption: float,
    max_order: int,
    mic_3d: np.ndarray,
    source_positions: list[list[float]],
    mic_height: float,
    fs: int,
) -> dict[str, float]:
    """
    Measure the empirical RT60 distribution across all source-mic RIRs for a given room configuration.
    """
    rt60_values = []
    
    for s_idx, src_xy in enumerate(source_positions):
        src_3d = np.array([src_xy[0], src_xy[1], mic_height])
        room = pra.ShoeBox(
            room_dims,
            fs=fs,
            materials=pra.Material(e_absorption),
            max_order=max_order,
            air_absorption=False,
            ray_tracing=False,
        )
        room.add_microphone(mic_3d)
        room.add_source(src_3d)
        room.compute_rir()
        
        for m_idx in range(mic_3d.shape[1]):
            rir = room.rir[m_idx][0]
            try:
                measured = float(pra.experimental.measure_rt60(rir, fs=fs))
                if not np.isnan(measured) and measured > 0:
                    rt60_values.append(measured)
            except Exception as exc:
                logger.debug("RT60 measure failed for S%02d-M%d: %s", s_idx+1, m_idx+1, exc)
                
    if len(rt60_values) == 0:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
    arr = np.array(rt60_values)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": len(arr),
    }


def safe_inverse_sabine(rt60_tgt: float, room_dims: list[float]) -> tuple[float, int]:
    """
    Compute wall absorption and max reflection order via inverse Sabine,
    clamping to maximum physical absorption (alpha=0.99) if room volume
    is too large for the requested short RT60.
    """
    try:
        e_abs, max_ord = pra.inverse_sabine(rt60_tgt, room_dims)
        return float(np.clip(e_abs, 0.01, 0.99)), max(1, max_ord)
    except ValueError:
        logger.info("Target RT60=%.2fs below room's physical Sabine floor (alpha=1.0). Clamping absorption to 0.99.", rt60_tgt)
        return 0.99, 17


def run_environmental_robustness_study(
    cfg: dict,
    output_dir: Path,
    smoke_test: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run Part A: Environmental Robustness Study over the full (or smoke test) RT60 x SNR grid.

    Parameters
    ----------
    cfg : dict
        Experiment 2 configuration dictionary.
    output_dir : Path
        Base output directory (results/experiment_02/environmental_robustness).
    smoke_test : bool
        If True, runs a reduced 3-condition x 1-trial smoke test.

    Returns
    -------
    df_summary : pd.DataFrame
        Condition-level summary table (42 rows or 3 rows for smoke test).
    df_all_trials : pd.DataFrame
        Detailed source-level evaluations across all conditions and trials.
    df_rt60_calib : pd.DataFrame
        RT60 calibration table across target levels.
    """
    logger.info("============================================================")
    logger.info("EchoLocate — Starting Part A: Environmental Robustness Study")
    logger.info("============================================================")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    per_condition_dir = output_dir / "per_condition"
    per_condition_dir.mkdir(parents=True, exist_ok=True)
    
    room_dims = cfg["room"]["dimensions"]
    speed_of_sound = cfg["room"]["speed_of_sound"]
    fs = cfg["room"]["sampling_rate"]
    base_seed = cfg["experiment"]["base_seed"]
    
    # Load canonical source positions
    src_file = Path(cfg["source"]["positions_file"])
    if not src_file.is_absolute():
        src_file = Path(__file__).resolve().parent.parent.parent / src_file
    df_sources = pd.read_csv(src_file)
    source_positions = df_sources[["x", "y"]].values.tolist()
    source_ids = df_sources["source_id"].tolist()
    
    # Source signal
    source_signal, _ = generate_source_signal({
        "sampling_rate": fs,
        "source": cfg["source"],
    })
    
    # Microphone geometry (Baseline Geometry A)
    base_geom_key = cfg["environmental_robustness"]["baseline_geometry"]
    geom_cfg = cfg["geometry_study"]["geometries"][base_geom_key]
    mic_height = cfg["geometry_study"]["mic_height"]
    mic_3d, mic_labels = build_mic_array_3d(geom_cfg["positions"], mic_height)
    mic_2d = mic_3d[:2, :].T
    
    # Compute max tau constraint
    max_tau_samples = compute_max_tau_from_geometry(mic_2d, speed_of_sound, fs, safety_factor=1.3)
    
    # Grid definition
    if smoke_test:
        rt60_list = [0.0, 0.3, 0.6]
        snr_list = [60.0, 20.0, 0.0]
        num_trials = 1
        logger.info("Running in SMOKE TEST mode: 3 conditions x 1 trial")
    else:
        rt60_list = cfg["environmental_robustness"]["rt60_values"]
        snr_list = cfg["environmental_robustness"]["snr_values"]
        num_trials = cfg["experiment"].get("num_noise_trials", 5)
        logger.info("Running FULL grid: %d RT60 x %d SNR x %d trials (%d conditions, %d evals)",
                    len(rt60_list), len(snr_list), num_trials, len(rt60_list)*len(snr_list),
                    len(rt60_list)*len(snr_list)*num_trials*len(source_positions))

    rt60_calib_records = []
    all_trial_records = []
    summary_records = []
    
    # Loop over RT60 targets
    for rt60_target in rt60_list:
        logger.info("------------------------------------------------------------")
        logger.info("Simulating Room Acoustics for RT60 target: %.2f s", rt60_target)
        
        is_anechoic = (rt60_target == 0.0)
        
        if is_anechoic:
            e_absorption = 1.0
            used_order = 0
            rt60_stats = {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        else:
            e_absorption, used_order = safe_inverse_sabine(rt60_target, room_dims)
            # Measure empirical RT60 distribution across all 45 source-mic paths
            rt60_stats = measure_room_rt60_distribution(
                room_dims=room_dims,
                e_absorption=e_absorption,
                max_order=used_order,
                mic_3d=mic_3d,
                source_positions=source_positions,
                mic_height=mic_height,
                fs=fs,
            )
            logger.info("  Measured RT60 distribution: Mean=%.4fs, Med=%.4fs, Std=%.4fs (Min=%.4fs, Max=%.4fs)",
                        rt60_stats["mean"], rt60_stats["median"], rt60_stats["std"], rt60_stats["min"], rt60_stats["max"])
            
        rt60_calib_records.append({
            "target_rt60_s": rt60_target,
            "measured_mean_rt60_s": rt60_stats["mean"],
            "measured_median_rt60_s": rt60_stats["median"],
            "measured_std_rt60_s": rt60_stats["std"],
            "measured_min_rt60_s": rt60_stats["min"],
            "measured_max_rt60_s": rt60_stats["max"],
            "is_anechoic": is_anechoic,
        })
        
        # Pre-synthesize clean multichannel signals for all 15 sources in this room
        clean_signals_cache = {}
        for s_idx, (src_id, src_xy) in enumerate(zip(source_ids, source_positions)):
            src_3d = np.array([src_xy[0], src_xy[1], mic_height])
            room = pra.ShoeBox(
                room_dims,
                fs=fs,
                materials=pra.Material(e_absorption),
                max_order=used_order,
                air_absorption=False,
                ray_tracing=False,
            )
            room.add_microphone(mic_3d)
            room.add_source(src_3d, signal=source_signal)
            room.simulate()
            clean_signals_cache[src_id] = room.mic_array.signals.copy()
            
        # Loop over SNR levels
        for snr_target in snr_list:
            cond_trial_records = []
            
            # Loop over independent noise trials
            for trial_idx in range(num_trials):
                trial_seed = int(base_seed + trial_idx * 1000 + int(rt60_target * 100) + int(snr_target))
                
                for src_id, src_xy in zip(source_ids, source_positions):
                    src_pos_2d = np.array(src_xy)
                    clean_sigs = clean_signals_cache[src_id]
                    
                    # Generate exact SNR noise
                    noisy_sigs, achieved_snr = generate_exact_snr_noise(
                        signals=clean_sigs,
                        target_snr_db=snr_target,
                        seed=trial_seed + int(src_xy[0] * 10 + src_xy[1]),
                    )
                    
                    # Estimate TDOAs via GCC-PHAT
                    tdoa_results = estimate_all_tdoas(
                        noisy_sigs,
                        fs=fs,
                        max_tau_samples=max_tau_samples,
                    )
                    
                    # Theoretical TDOAs
                    d0 = np.linalg.norm(src_pos_2d - mic_2d[0])
                    d1 = np.linalg.norm(src_pos_2d - mic_2d[1])
                    d2 = np.linalg.norm(src_pos_2d - mic_2d[2])
                    theo_01 = float((d0 - d1) / speed_of_sound)
                    theo_02 = float((d0 - d2) / speed_of_sound)
                    theo_12 = float((d1 - d2) / speed_of_sound)
                    
                    # Multilateration
                    tdoa_dict = {
                        (0, 1): tdoa_results[(0, 1)].tau_seconds,
                        (0, 2): tdoa_results[(0, 2)].tau_seconds,
                        (1, 2): tdoa_results[(1, 2)].tau_seconds,
                    }
                    
                    loc_res = localize_with_multiple_starts(
                        mic_positions_2d=mic_2d,
                        tdoa_dict=tdoa_dict,
                        speed_of_sound=speed_of_sound,
                        room_dimensions=room_dims,
                        n_restarts=cfg["localization"].get("n_restarts", 9),
                        true_position=src_pos_2d,
                    )
                    
                    rec = {
                        "rt60_target_s": rt60_target,
                        "snr_target_db": snr_target,
                        "trial_idx": trial_idx,
                        "trial_seed": trial_seed,
                        "source_id": src_id,
                        "true_x": src_xy[0],
                        "true_y": src_xy[1],
                        "pred_x": float(loc_res.predicted_position[0]),
                        "pred_y": float(loc_res.predicted_position[1]),
                        "localization_error": float(loc_res.localization_error),
                        "solver_converged": loc_res.converged,
                        "solver_cost": float(np.sum(loc_res.tdoa_residuals ** 2)),
                        "achieved_snr_db": achieved_snr,
                        "tdoa_01_estimated_s": tdoa_results[(0, 1)].tau_seconds,
                        "tdoa_01_theoretical_s": theo_01,
                        "tdoa_02_estimated_s": tdoa_results[(0, 2)].tau_seconds,
                        "tdoa_02_theoretical_s": theo_02,
                        "tdoa_12_estimated_s": tdoa_results[(1, 2)].tau_seconds,
                        "tdoa_12_theoretical_s": theo_12,
                    }
                    cond_trial_records.append(rec)
                    all_trial_records.append(rec)
                    
            # Aggregate condition summary across trials and sources
            df_cond = pd.DataFrame(cond_trial_records)
            cond_summary = compute_condition_summary(
                df_trials=df_cond,
                rt60_target=rt60_target,
                snr_target=snr_target,
                rt60_measured_stats=rt60_stats,
                fs=fs,
            )
            summary_records.append(asdict(cond_summary))
            
            # Save per-condition CSV
            cond_csv_name = f"condition_rt60_{rt60_target:.2f}s_snr_{snr_target:.1f}dB.csv"
            df_cond.to_csv(per_condition_dir / cond_csv_name, index=False)
            
            logger.info("  Cond RT60=%.2fs | SNR=%4.1fdB -> Mean Err: %.3fm | Med Err: %.3fm | <=1m: %5.1f%%",
                        rt60_target, snr_target, cond_summary.mean_localization_error_m,
                        cond_summary.median_localization_error_m, cond_summary.pct_within_1_0m)

    df_summary = pd.DataFrame(summary_records)
    df_all_trials = pd.DataFrame(all_trial_records)
    df_rt60_calib = pd.DataFrame(rt60_calib_records)
    
    # Save main outputs
    df_summary.to_csv(output_dir / "summary.csv", index=False)
    df_all_trials.to_csv(output_dir / "all_trial_evaluations.csv", index=False)
    df_rt60_calib.to_csv(output_dir / "rt60_calibration.csv", index=False)
    
    logger.info("Part A Environmental Robustness Complete. Summary saved to: %s", output_dir / "summary.csv")
    return df_summary, df_all_trials, df_rt60_calib
