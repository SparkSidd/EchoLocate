"""
EchoLocate — Part B: Microphone Geometry Study
===============================================
Evaluates 4 distinct microphone array topologies across 5 representative
acoustic conditions (Anechoic Clean, Mild Reverb, Moderate Reverb, High Reverb,
Strong Reverb) over 5 independent noise trials per condition.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import pyroomacoustics as pra

from src.simulation.room_simulator import build_mic_array_3d
from src.simulation.source_generator import generate_source_signal
from src.localization.gcc_phat import (
    estimate_all_tdoas,
    compute_max_tau_from_geometry,
)
from src.localization.multilateration import localize_with_multiple_starts
from src.evaluation.geometry_metrics import (
    compute_geometry_properties,
    evaluate_source_geometry_metrics,
)
from src.evaluation.robustness_metrics import compute_condition_summary
from src.experiments.environment_robustness import generate_exact_snr_noise, safe_inverse_sabine

logger = logging.getLogger(__name__)


def run_microphone_geometry_study(
    cfg: dict,
    output_dir: Path,
    smoke_test: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Execute Part B: Microphone Geometry Study.

    Parameters
    ----------
    cfg : dict
        Experiment 2 configuration dictionary.
    output_dir : Path
        Base output directory (results/experiment_02/geometry_study).
    smoke_test : bool
        If True, runs reduced 2 geometries x 2 conditions x 1 trial.

    Returns
    -------
    df_geometry_summary : pd.DataFrame
        Aggregated summary table across geometries and conditions.
    df_pre_analysis : pd.DataFrame
        Structural and conditioning metrics for each geometry and source.
    df_rankings : pd.DataFrame
        Overall and per-condition ranking of the geometries.
    """
    logger.info("============================================================")
    logger.info("EchoLocate — Starting Part B: Microphone Geometry Study")
    logger.info("============================================================")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    per_geom_dir = output_dir / "per_geometry"
    per_geom_dir.mkdir(parents=True, exist_ok=True)
    
    room_dims = cfg["room"]["dimensions"]
    speed_of_sound = cfg["room"]["speed_of_sound"]
    fs = cfg["room"]["sampling_rate"]
    base_seed = cfg["experiment"]["base_seed"]
    mic_height = cfg["geometry_study"]["mic_height"]
    
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
    
    # Geometry definitions
    geometries_cfg = cfg["geometry_study"]["geometries"]
    if smoke_test:
        geom_keys = ["geometry_a", "geometry_c"]
        conditions_list = [
            {"name": "Condition 1: Anechoic Clean", "rt60": 0.0, "snr": 60.0},
            {"name": "Condition 3: Moderate Reverb / Medium SNR", "rt60": 0.3, "snr": 20.0},
        ]
        num_trials = 1
        logger.info("Running Part B in SMOKE TEST mode: 2 geometries x 2 conditions x 1 trial")
    else:
        geom_keys = list(geometries_cfg.keys())
        conditions_list = cfg["geometry_study"]["representative_conditions"]
        num_trials = cfg["experiment"].get("num_noise_trials", 5)
        logger.info("Running Part B FULL: %d geometries x %d conditions x %d trials",
                    len(geom_keys), len(conditions_list), num_trials)

    # --------------------------------------------------------------------------
    # Step 1: Pre-Analysis of Geometries (Aperture, Triangle Area, GDOP, Conditioning)
    # --------------------------------------------------------------------------
    pre_analysis_records = []
    geometry_props_dict = {}
    
    for g_key in geom_keys:
        g_info = geometries_cfg[g_key]
        m_3d, _ = build_mic_array_3d(g_info["positions"], mic_height)
        m_2d = m_3d[:2, :].T
        
        g_props = compute_geometry_properties(g_key, g_info["name"], m_2d)
        geometry_props_dict[g_key] = g_props
        
        logger.info("Geometry '%s' (%s):", g_key, g_info["name"])
        logger.info("  Aperture: Min Sep=%.2fm, Max Sep=%.2fm, Triangle Area=%.2f m^2",
                    g_props.min_separation_m, g_props.max_separation_m, g_props.triangle_area_m2)
        
        for src_id, src_xy in zip(source_ids, source_positions):
            s_metrics = evaluate_source_geometry_metrics(np.array(src_xy), m_2d, speed_of_sound)
            pre_analysis_records.append({
                "geometry_id": g_key,
                "geometry_name": g_info["name"],
                "triangle_area_m2": g_props.triangle_area_m2,
                "min_mic_separation_m": g_props.min_separation_m,
                "max_mic_separation_m": g_props.max_separation_m,
                "source_id": src_id,
                "source_x": src_xy[0],
                "source_y": src_xy[1],
                **s_metrics,
            })
            
    df_pre_analysis = pd.DataFrame(pre_analysis_records)
    df_pre_analysis.to_csv(output_dir / "geometry_pre_analysis.csv", index=False)
    logger.info("Saved geometry pre-analysis to: %s", output_dir / "geometry_pre_analysis.csv")

    # --------------------------------------------------------------------------
    # Step 2: Evaluate All Geometries across Representative Conditions
    # --------------------------------------------------------------------------
    summary_records = []
    all_geom_evaluations = []
    
    for g_key in geom_keys:
        g_info = geometries_cfg[g_key]
        m_3d, _ = build_mic_array_3d(g_info["positions"], mic_height)
        m_2d = m_3d[:2, :].T
        max_tau_samples = compute_max_tau_from_geometry(m_2d, speed_of_sound, fs, safety_factor=1.3)
        g_area = geometry_props_dict[g_key].triangle_area_m2
        
        geom_eval_records = []
        
        for cond_idx, cond in enumerate(conditions_list):
            cond_name = cond["name"]
            rt60_tgt = cond["rt60"]
            snr_tgt = cond["snr"]
            is_anechoic = (rt60_tgt == 0.0)
            
            # Setup room
            if is_anechoic:
                e_absorption = 1.0
                used_order = 0
            else:
                e_absorption, used_order = safe_inverse_sabine(rt60_tgt, room_dims)
                
            # Pre-simulate clean signals for this room & geometry
            clean_cache = {}
            for src_id, src_xy in zip(source_ids, source_positions):
                src_3d = np.array([src_xy[0], src_xy[1], mic_height])
                room = pra.ShoeBox(
                    room_dims,
                    fs=fs,
                    materials=pra.Material(e_absorption),
                    max_order=used_order,
                    air_absorption=False,
                    ray_tracing=False,
                )
                room.add_microphone(m_3d)
                room.add_source(src_3d, signal=source_signal)
                room.simulate()
                clean_cache[src_id] = room.mic_array.signals.copy()
                
            # Evaluate across noise trials
            cond_records = []
            for trial_idx in range(num_trials):
                trial_seed = int(base_seed + trial_idx * 1000 + int(rt60_tgt * 100) + int(snr_tgt) + cond_idx * 50)
                
                for src_id, src_xy in zip(source_ids, source_positions):
                    src_pos_2d = np.array(src_xy)
                    clean_sigs = clean_cache[src_id]
                    
                    noisy_sigs, achieved_snr = generate_exact_snr_noise(
                        clean_sigs,
                        target_snr_db=snr_tgt,
                        seed=trial_seed + int(src_xy[0] * 10 + src_xy[1]),
                    )
                    
                    tdoa_results = estimate_all_tdoas(noisy_sigs, fs=fs, max_tau_samples=max_tau_samples)
                    
                    # Theoretical TDOAs
                    d0 = np.linalg.norm(src_pos_2d - m_2d[0])
                    d1 = np.linalg.norm(src_pos_2d - m_2d[1])
                    d2 = np.linalg.norm(src_pos_2d - m_2d[2])
                    theo_01 = float((d0 - d1) / speed_of_sound)
                    theo_02 = float((d0 - d2) / speed_of_sound)
                    theo_12 = float((d1 - d2) / speed_of_sound)
                    
                    tdoa_dict = {
                        (0, 1): tdoa_results[(0, 1)].tau_seconds,
                        (0, 2): tdoa_results[(0, 2)].tau_seconds,
                        (1, 2): tdoa_results[(1, 2)].tau_seconds,
                    }
                    
                    loc_res = localize_with_multiple_starts(
                        mic_positions_2d=m_2d,
                        tdoa_dict=tdoa_dict,
                        speed_of_sound=speed_of_sound,
                        room_dimensions=room_dims,
                        n_restarts=cfg["localization"].get("n_restarts", 9),
                        true_position=src_pos_2d,
                    )
                    
                    # Find inside_triangle for this source & geometry
                    s_geom_row = df_pre_analysis[(df_pre_analysis["geometry_id"] == g_key) & (df_pre_analysis["source_id"] == src_id)].iloc[0]
                    
                    rec = {
                        "geometry_id": g_key,
                        "geometry_name": g_info["name"],
                        "triangle_area_m2": g_area,
                        "condition_name": cond_name,
                        "rt60_target_s": rt60_tgt,
                        "snr_target_db": snr_tgt,
                        "trial_idx": trial_idx,
                        "source_id": src_id,
                        "true_x": src_xy[0],
                        "true_y": src_xy[1],
                        "pred_x": float(loc_res.predicted_position[0]),
                        "pred_y": float(loc_res.predicted_position[1]),
                        "localization_error": float(loc_res.localization_error),
                        "solver_converged": loc_res.converged,
                        "inside_triangle": s_geom_row["inside_triangle"],
                        "jacobian_cond": s_geom_row["jacobian_condition_number"],
                        "gdop": s_geom_row["gdop"],
                        "tdoa_01_estimated_s": tdoa_results[(0, 1)].tau_seconds,
                        "tdoa_01_theoretical_s": theo_01,
                        "tdoa_02_estimated_s": tdoa_results[(0, 2)].tau_seconds,
                        "tdoa_02_theoretical_s": theo_02,
                        "tdoa_12_estimated_s": tdoa_results[(1, 2)].tau_seconds,
                        "tdoa_12_theoretical_s": theo_12,
                    }
                    cond_records.append(rec)
                    geom_eval_records.append(rec)
                    all_geom_evaluations.append(rec)
                    
            # Compute summary for this geometry and condition
            df_g_cond = pd.DataFrame(cond_records)
            cond_sum = compute_condition_summary(
                df_trials=df_g_cond,
                rt60_target=rt60_tgt,
                snr_target=snr_tgt,
                fs=fs,
            )
            
            # Hypothesis H5 test: inside vs outside triangle localization error
            inside_errs = df_g_cond[df_g_cond["inside_triangle"]]["localization_error"].values
            outside_errs = df_g_cond[~df_g_cond["inside_triangle"]]["localization_error"].values
            mean_in = float(np.mean(inside_errs)) if len(inside_errs) > 0 else np.nan
            mean_out = float(np.mean(outside_errs)) if len(outside_errs) > 0 else np.nan
            std_in = float(np.std(inside_errs, ddof=1)) if len(inside_errs) > 1 else np.nan
            std_out = float(np.std(outside_errs, ddof=1)) if len(outside_errs) > 1 else np.nan
            
            summary_records.append({
                "geometry_id": g_key,
                "geometry_name": g_info["name"],
                "triangle_area_m2": g_area,
                "condition_name": cond_name,
                "rt60_target_s": rt60_tgt,
                "snr_target_db": snr_tgt,
                "num_trials": num_trials,
                "mean_localization_error_m": cond_sum.mean_localization_error_m,
                "std_localization_error_m": cond_sum.std_localization_error_m,
                "median_localization_error_m": cond_sum.median_localization_error_m,
                "rmse_localization_error_m": cond_sum.rmse_localization_error_m,
                "max_localization_error_m": cond_sum.max_localization_error_m,
                "mean_tdoa_error_ms": cond_sum.mean_tdoa_error_ms,
                "pct_within_0_5m": cond_sum.pct_within_0_5m,
                "pct_within_1_0m": cond_sum.pct_within_1_0m,
                "pct_within_2_0m": cond_sum.pct_within_2_0m,
                "mean_error_inside_triangle_m": mean_in,
                "mean_error_outside_triangle_m": mean_out,
                "std_error_inside_triangle_m": std_in,
                "std_error_outside_triangle_m": std_out,
                "x_bias_m": cond_sum.x_bias_m,
                "y_bias_m": cond_sum.y_bias_m,
            })
            
            logger.info("  %s | %s -> Mean Err: %.3fm | Med Err: %.3fm | <=1m: %5.1f%%",
                        g_key, cond_name, cond_sum.mean_localization_error_m,
                        cond_sum.median_localization_error_m, cond_sum.pct_within_1_0m)
            
        # Save per-geometry detailed CSV
        df_g_all = pd.DataFrame(geom_eval_records)
        df_g_all.to_csv(per_geom_dir / f"{g_key}_evaluations.csv", index=False)
        
    df_geometry_summary = pd.DataFrame(summary_records)
    df_geometry_summary.to_csv(output_dir / "summary.csv", index=False)
    
    # --------------------------------------------------------------------------
    # Step 3: Compute Overall Geometry Rankings
    # --------------------------------------------------------------------------
    rankings = []
    for g_key in geom_keys:
        df_sub = df_geometry_summary[df_geometry_summary["geometry_id"] == g_key]
        g_name = df_sub["geometry_name"].iloc[0]
        g_area = df_sub["triangle_area_m2"].iloc[0]
        
        # Anechoic (Cond 1)
        anechoic_mean = df_sub[df_sub["rt60_target_s"] == 0.0]["mean_localization_error_m"].mean()
        # Reverberant (Cond 2-5)
        reverb_mean = df_sub[df_sub["rt60_target_s"] > 0.0]["mean_localization_error_m"].mean()
        overall_mean = df_sub["mean_localization_error_m"].mean()
        overall_rmse = df_sub["rmse_localization_error_m"].mean()
        worst_case = df_sub["max_localization_error_m"].max()
        success_1m = df_sub["pct_within_1_0m"].mean()
        
        rankings.append({
            "geometry_id": g_key,
            "geometry_name": g_name,
            "triangle_area_m2": g_area,
            "overall_mean_error_m": overall_mean,
            "overall_rmse_m": overall_rmse,
            "anechoic_mean_error_m": anechoic_mean,
            "reverberant_mean_error_m": reverb_mean,
            "worst_case_error_m": worst_case,
            "overall_success_rate_1m_pct": success_1m,
        })
        
    df_rankings = pd.DataFrame(rankings).sort_values("overall_mean_error_m")
    df_rankings.to_csv(output_dir / "geometry_rankings.csv", index=False)
    logger.info("Saved geometry rankings to: %s", output_dir / "geometry_rankings.csv")
    
    return df_geometry_summary, df_pre_analysis, df_rankings
