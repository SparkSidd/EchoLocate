"""
EchoLocate — Experiment 4 Master Simulation & Benchmark Orchestrator
===================================================================
Executes paired multi-model evaluations across Methods M1–M5, benchmark conditions
A–D, voiced speech signals, OOD test suites, and mechanistic analyses.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyroomacoustics as pra
import torch
import yaml

from src.evaluation.acoustic_mechanism_analysis import extract_correlation_peaks_and_rank
from src.evaluation.confidence_analysis import compute_confidence_calibration_summary
from src.evaluation.experiment_04_analysis import (
    compute_method_condition_summary,
    compute_paired_statistical_comparison,
)
from src.evaluation.neural_leakage_audit import run_full_neural_leakage_audit
from src.evaluation.speech_mechanism_analysis import analyze_speech_pitch_correlation_peaks
from src.experiments.environment_robustness import (
    generate_exact_snr_noise,
    safe_inverse_sabine,
)
from src.localization.enhanced_gcc import localize_neural_gcc_nls
from src.localization.multilateration import localize_source_2d
from src.localization.neural_srp import NeuralSRPLocalizer
from src.localization.gcc_phat import estimate_all_tdoas
from src.localization.srp_phat import compute_phat_cross_correlations
from src.localization.blind_localizer import BlindSRPLocalizer
from src.models.acoustic_frontend import generate_experiment_04_signal
from src.models.inference import BlindNeuralGCCPredictor
from src.models.training import train_deep_gcc_model

logger = logging.getLogger(__name__)


def run_experiment_04_pipeline(
    config_path: str | Path = "configs/experiment_04.yaml",
    output_dir: str | Path = "results/experiment_04",
    model_checkpoint_path: str | Path | None = None,
    smoke_test: bool = False,
) -> dict[str, Any]:
    """
    Master execution pipeline for Experiment 4.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    (out_p / "figures").mkdir(parents=True, exist_ok=True)

    base_seed = cfg["reproducibility"]["eval_seed"]
    true_dims = cfg["physical_world"]["room_dimensions"]
    true_c = cfg["physical_world"]["speed_of_sound"]
    fs = cfg["physical_world"]["fs"]
    true_mics_2d = np.array(cfg["physical_world"]["mic_positions_2d"], dtype=np.float64)
    mic_height = cfg["physical_world"]["mic_height"]
    src_height = cfg["physical_world"]["source_height"]
    true_mics_3d = np.column_stack([true_mics_2d, np.full(len(true_mics_2d), mic_height)]).T

    # Load canonical sources from Experiment 3 config
    with open("configs/experiment_03.yaml", "r", encoding="utf-8") as f:
        exp03_cfg = yaml.safe_load(f)
    canonical_sources = exp03_cfg["source"]["canonical_positions_2d"]
    benchmark_conds = cfg["benchmark_conditions"]
    eval_signals = cfg["source_signals"]["eval_signals"]

    if smoke_test:
        canonical_sources = canonical_sources[:3]
        benchmark_conds = benchmark_conds[:2]
        eval_signals = ["linear_chirp", "speech_male_vowel"]
        n_trials = 2
    else:
        n_trials = cfg["trials_per_condition"]

    # 1. Train or Load DeepGCC Model
    ckpt_path = Path(model_checkpoint_path) if model_checkpoint_path else out_p / "deep_gcc_model.pt"
    if not ckpt_path.exists():
        logger.info("DeepGCC checkpoint not found. Starting training...")
        model, df_hist = train_deep_gcc_model(
            config_path=config_path,
            save_checkpoint_path=ckpt_path,
            history_csv_path=out_p / "training_history.csv",
            use_ipd_loss=True,
            smoke_test=smoke_test,
        )
    else:
        logger.info("Loading pre-trained DeepGCC model from: %s", ckpt_path)
        predictor_temp = BlindNeuralGCCPredictor(model_checkpoint_path=ckpt_path)
        model = predictor_temp.model

    predictor = BlindNeuralGCCPredictor(model=model)

    # 2. Initialize Localizers
    raw_srp_localizer = BlindSRPLocalizer(
        assumed_room_dimensions=true_dims,
        assumed_mic_positions_2d=true_mics_2d,
        grid_resolution_m=0.05,
        assumed_speed_of_sound=true_c,
    )
    neural_srp_localizer = NeuralSRPLocalizer(
        assumed_room_dimensions=true_dims,
        assumed_mic_positions_2d=true_mics_2d,
        predictor=predictor,
        grid_resolution_m=0.05,
        assumed_speed_of_sound=true_c,
    )

    pairs = []
    for i in range(len(true_mics_2d)):
        for j in range(i + 1, len(true_mics_2d)):
            pairs.append((i, j))

    # 3. Leakage Audit
    logger.info("--- Executing Neural Leakage Audit ---")
    df_leakage = run_full_neural_leakage_audit(out_p / "neural_leakage_audit.csv")

    # 4. Master Benchmark Execution
    logger.info("--- Executing Master Benchmark Across Methods M1, M2, M3, M4 ---")
    eval_records = []
    pair_records = []
    speech_records = []

    eval_counter = 0

    for cond_idx, cond in enumerate(benchmark_conds):
        cond_id = cond["id"]
        rt60_tgt = float(cond["target_rt60"])
        snr_db = float(cond["snr_target_db"])
        abs_mat, max_ord = safe_inverse_sabine(rt60_tgt, true_dims)

        for src_idx, src in enumerate(canonical_sources):
            src_xy = src["xy"]
            src_id = src["id"]

            # Compute true theoretical delays for all pairs
            true_tdoas = {}
            for i, j in pairs:
                d_i = np.linalg.norm(true_mics_2d[i] - np.array(src_xy))
                d_j = np.linalg.norm(true_mics_2d[j] - np.array(src_xy))
                true_tdoas[(i, j)] = (d_i - d_j) / true_c

            for sig_idx, sig_type in enumerate(eval_signals):
                sig_data, sig_meta = generate_experiment_04_signal(
                    sig_type, fs=fs, duration=0.1, seed=base_seed + sig_idx * 100 + src_idx
                )

                # Acoustic simulation
                room = pra.ShoeBox(true_dims, fs=fs, materials=pra.Material(abs_mat), max_order=max_ord)
                room.add_microphone_array(pra.MicrophoneArray(true_mics_3d, fs=fs))
                room.add_source([src_xy[0], src_xy[1], src_height], signal=sig_data)
                room.simulate()
                clean_sig = room.mic_array.signals.copy()

                for t_idx in range(n_trials):
                    eval_counter += 1
                    eval_seed = base_seed + cond_idx * 5000 + src_idx * 100 + sig_idx * 10 + t_idx
                    noisy_sig, ach_snr = generate_exact_snr_noise(clean_sig, snr_db, seed=eval_seed)

                    # --- Method M1: GCC-PHAT + NLS ---
                    t1_start = time.perf_counter()
                    raw_tdoas_res = estimate_all_tdoas(noisy_sig, fs=fs)
                    raw_tdoas = {pair: res.tau_seconds for pair, res in raw_tdoas_res.items()}
                    m1_res = localize_source_2d(
                        mic_positions_2d=true_mics_2d,
                        tdoa_dict=raw_tdoas,
                        speed_of_sound=true_c,
                        room_dimensions=true_dims,
                    )
                    m1_time = (time.perf_counter() - t1_start) * 1000.0
                    m1_err = float(np.linalg.norm(m1_res.predicted_position - np.array(src_xy)))

                    # --- Method M2: Raw SRP-PHAT ---
                    m2_res = raw_srp_localizer.localize(noisy_sig, fs=fs)
                    m2_err = float(np.linalg.norm(np.array([m2_res.pred_x, m2_res.pred_y]) - np.array(src_xy)))

                    # --- Method M3: Neural GCC + NLS ---
                    m3_res = localize_neural_gcc_nls(
                        noisy_sig, true_mics_2d, predictor=predictor, fs=fs, speed_of_sound=true_c,
                        room_dimensions=true_dims,
                    )
                    m3_err = float(np.linalg.norm(np.array([m3_res.pred_x, m3_res.pred_y]) - np.array(src_xy)))

                    # --- Method M4: Neural SRP (Primary Proposed Method) ---
                    m4_res = neural_srp_localizer.localize(noisy_sig, fs=fs)
                    m4_err = float(np.linalg.norm(np.array([m4_res.pred_x, m4_res.pred_y]) - np.array(src_xy)))

                    # Collect Mechanistic metrics on Pair (0, 1) and average across pairs
                    pair_tdoa_errs_raw = []
                    pair_tdoa_errs_neu = []
                    pair_entropy_raw = []
                    pair_entropy_neu = []
                    pair_pcr_raw = []
                    pair_pcr_neu = []

                    for pair in pairs:
                        true_tau = true_tdoas[pair]
                        raw_curve = m4_res.raw_gcc_dict[pair]
                        neu_curve = m4_res.enhanced_gcc_dict[pair]
                        tau_axis = m4_res.tau_axis

                        raw_mech = extract_correlation_peaks_and_rank(raw_curve, tau_axis, true_tau)
                        neu_mech = extract_correlation_peaks_and_rank(neu_curve, tau_axis, true_tau)

                        pair_tdoa_errs_raw.append(raw_mech["tdoa_error_s"])
                        pair_tdoa_errs_neu.append(neu_mech["tdoa_error_s"])
                        pair_entropy_raw.append(raw_mech["correlation_entropy"])
                        pair_entropy_neu.append(neu_mech["correlation_entropy"])
                        pair_pcr_raw.append(raw_mech["peak_competition_ratio"])
                        pair_pcr_neu.append(neu_mech["peak_competition_ratio"])

                        # Speech pitch analysis
                        if sig_meta["f0_hz"] > 0.0:
                            raw_sp = analyze_speech_pitch_correlation_peaks(raw_curve, tau_axis, true_tau, sig_meta["f0_hz"])
                            neu_sp = analyze_speech_pitch_correlation_peaks(neu_curve, tau_axis, true_tau, sig_meta["f0_hz"])
                            speech_records.append({
                                "eval_id": eval_counter,
                                "condition_id": cond_id,
                                "signal_type": sig_type,
                                "pair": str(pair),
                                "f0_hz": sig_meta["f0_hz"],
                                "raw_harmonic_ratio": raw_sp["pitch_harmonic_peak_power_ratio"],
                                "neural_harmonic_ratio": neu_sp["pitch_harmonic_peak_power_ratio"],
                                "raw_pseudo_peaks": raw_sp["pitch_pseudo_peaks_detected"],
                                "neural_pseudo_peaks": neu_sp["pitch_pseudo_peaks_detected"],
                            })

                    # Record Method Results
                    base_row = {
                        "eval_id": eval_counter,
                        "condition_id": cond_id,
                        "source_id": src_id,
                        "signal_type": sig_type,
                        "trial_idx": t_idx,
                        "true_x": src_xy[0],
                        "true_y": src_xy[1],
                        "f0_hz": sig_meta.get("f0_hz", 0.0),
                    }

                    eval_records.append({
                        **base_row,
                        "method_id": "M1_Classical_NLS",
                        "pred_x": m1_res.predicted_position[0],
                        "pred_y": m1_res.predicted_position[1],
                        "localization_error_m": m1_err,
                        "runtime_ms": m1_time,
                        "mean_tdoa_error_us": float(np.mean(pair_tdoa_errs_raw) * 1e6),
                        "mean_correlation_entropy": float(np.mean(pair_entropy_raw)),
                        "mean_peak_competition_ratio": float(np.mean(pair_pcr_raw)),
                        "spatial_peak_contrast": 1.0,
                    })

                    eval_records.append({
                        **base_row,
                        "method_id": "M2_Raw_SRP",
                        "pred_x": m2_res.pred_x,
                        "pred_y": m2_res.pred_y,
                        "localization_error_m": m2_err,
                        "runtime_ms": m2_res.runtime_ms,
                        "mean_tdoa_error_us": float(np.mean(pair_tdoa_errs_raw) * 1e6),
                        "mean_correlation_entropy": float(np.mean(pair_entropy_raw)),
                        "mean_peak_competition_ratio": float(np.mean(pair_pcr_raw)),
                        "spatial_peak_contrast": m2_res.peak_ratio,
                    })

                    eval_records.append({
                        **base_row,
                        "method_id": "M3_Neural_NLS",
                        "pred_x": m3_res.pred_x,
                        "pred_y": m3_res.pred_y,
                        "localization_error_m": m3_err,
                        "runtime_ms": m3_res.runtime_ms,
                        "mean_tdoa_error_us": float(np.mean(pair_tdoa_errs_neu) * 1e6),
                        "mean_correlation_entropy": float(np.mean(pair_entropy_neu)),
                        "mean_peak_competition_ratio": float(np.mean(pair_pcr_neu)),
                        "spatial_peak_contrast": 1.0,
                    })

                    eval_records.append({
                        **base_row,
                        "method_id": "M4_Neural_SRP",
                        "pred_x": m4_res.pred_x,
                        "pred_y": m4_res.pred_y,
                        "localization_error_m": m4_err,
                        "runtime_ms": m4_res.runtime_ms,
                        "mean_tdoa_error_us": float(np.mean(pair_tdoa_errs_neu) * 1e6),
                        "mean_correlation_entropy": float(np.mean(pair_entropy_neu)),
                        "mean_peak_competition_ratio": float(np.mean(pair_pcr_neu)),
                        "spatial_peak_contrast": m4_res.spatial_peak_contrast,
                    })

    df_evals = pd.DataFrame(eval_records)
    df_evals.to_csv(out_p / "per_evaluation.csv", index=False)

    df_summary = compute_method_condition_summary(df_evals)
    df_summary.to_csv(out_p / "summary.csv", index=False)

    df_stats = compute_paired_statistical_comparison(df_evals, primary_method="M4_Neural_SRP", baseline_method="M2_Raw_SRP")
    df_stats.to_csv(out_p / "statistical_comparison.csv", index=False)

    df_speech = pd.DataFrame(speech_records)
    if not df_speech.empty:
        df_speech.to_csv(out_p / "pitch_peak_analysis.csv", index=False)

    # Confidence calibration
    df_m4 = df_evals[df_evals["method_id"] == "M4_Neural_SRP"]
    conf_summary, df_calib = compute_confidence_calibration_summary(df_m4)
    df_calib.to_csv(out_p / "confidence_analysis.csv", index=False)

    logger.info("================================================================================")
    logger.info("Experiment 4 Execution Complete! Results saved to: %s", out_p)
    logger.info("================================================================================")

    return {
        "summary": df_summary,
        "per_evaluation": df_evals,
        "statistical_comparison": df_stats,
        "confidence_calibration": df_calib,
        "confidence_summary": conf_summary,
        "pitch_analysis": df_speech,
        "leakage_audit": df_leakage,
    }
