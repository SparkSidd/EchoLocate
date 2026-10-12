"""
EchoLocate — Experiment 3.5: Reality Check & Adversarial Audit Runner
====================================================================
Orchestrates Studies A through N to evaluate model mismatch, grid quantization,
signal generalization, room topologies, and blind pipeline integrity.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyroomacoustics as pra
from scipy.signal import chirp as scipy_chirp
import yaml

from src.evaluation.generalization_metrics import (
    compute_room_generalization_summary,
    compute_signal_generalization_summary,
)
from src.evaluation.leakage_audit import run_full_leakage_audit
from src.evaluation.mismatch_metrics import compute_mismatch_sensitivity_table
from src.experiments.environment_robustness import (
    generate_exact_snr_noise,
    safe_inverse_sabine,
)
from src.localization.blind_localizer import BlindSRPLocalizer
from src.localization.srp_phat import SRPGrid
from src.simulation.source_generator import generate_chirp

logger = logging.getLogger(__name__)


def generate_diverse_signal(signal_type: str, fs: int = 16000, duration: float = 0.1, seed: int = 42) -> np.ndarray:
    """Generate diverse acoustic source signals for generalization testing."""
    n_samples = int(fs * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    rng = np.random.default_rng(seed)

    if signal_type == "linear_chirp":
        return generate_chirp(fs=fs, duration=duration, f_low=300.0, f_high=7000.0)
    elif signal_type == "log_chirp":
        sig = scipy_chirp(t, f0=300.0, t1=duration, f1=7000.0, method="logarithmic")
        # Apply 5% Tukey taper
        window = np.hanning(len(sig))
        return sig * window
    elif signal_type == "broadband_noise":
        noise = rng.normal(0.0, 1.0, size=n_samples)
        # Apply smooth cosine taper
        window = np.sin(np.pi * np.linspace(0, 1, n_samples))
        return noise * window
    elif signal_type == "speech_synthetic":
        # Synthesize voiced vowel /a/ with pitch f0=150Hz and formants at 700, 1200, 2500 Hz
        f0 = 150.0
        n_harmonics = int(7000 / f0)
        signal = np.zeros(n_samples)
        for h in range(1, n_harmonics + 1):
            fh = h * f0
            # Formant shaping
            a1 = np.exp(-((fh - 700)**2) / (2 * 100**2))
            a2 = np.exp(-((fh - 1200)**2) / (2 * 150**2))
            a3 = np.exp(-((fh - 2500)**2) / (2 * 200**2))
            weight = a1 + 0.6 * a2 + 0.3 * a3 + 0.05
            signal += weight * np.sin(2 * np.pi * fh * t)
        env = np.sin(np.pi * np.linspace(0, 1, n_samples)) ** 2
        return signal * env / (np.max(np.abs(signal)) + 1e-12)
    elif signal_type == "impulsive_transient":
        # Exponentially decaying impulsive click
        decay = np.exp(-t / 0.008)
        carrier = np.sin(2 * np.pi * 1500 * t) + 0.5 * np.sin(2 * np.pi * 3500 * t)
        return decay * carrier
    elif signal_type == "narrowband_low":
        return generate_chirp(fs=fs, duration=duration, f_low=300.0, f_high=1000.0)
    elif signal_type == "narrowband_mid":
        return generate_chirp(fs=fs, duration=duration, f_low=1000.0, f_high=3000.0)
    elif signal_type == "narrowband_high":
        return generate_chirp(fs=fs, duration=duration, f_low=3000.0, f_high=7000.0)
    else:
        return generate_chirp(fs=fs, duration=duration, f_low=300.0, f_high=7000.0)


def generate_offgrid_sources(count: int, room_dims: list[float], margin: float = 0.50, seed: int = 42) -> list[dict[str, Any]]:
    """Generate random continuous 2D coordinates guaranteed to be off-grid."""
    rng = np.random.default_rng(seed)
    sources = []
    for i in range(count):
        while True:
            # Generate random coordinate
            x = rng.uniform(margin, room_dims[0] - margin)
            y = rng.uniform(margin, room_dims[1] - margin)
            # Ensure not a multiple of 0.025m (within 2mm)
            if not (np.isclose(x % 0.025, 0, atol=0.002) or np.isclose(x % 0.025, 0.025, atol=0.002)):
                if not (np.isclose(y % 0.025, 0, atol=0.002) or np.isclose(y % 0.025, 0.025, atol=0.002)):
                    break
        sources.append({"id": f"R{i+1:03d}", "xy": [float(x), float(y)]})
    return sources


def run_experiment_03_5_reality_check(
    cfg: dict[str, Any] | None = None,
    config_path: str | Path = "configs/experiment_03_5.yaml",
    output_dir: str | Path = "results/experiment_03_5",
    smoke_test: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Run full Experiment 3.5 adversarial validation suite.
    """
    if cfg is None:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "figures").mkdir(parents=True, exist_ok=True)

    base_seed = cfg["reproducibility"]["base_seed"]
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

    if smoke_test:
        canonical_sources = canonical_sources[:3]
        benchmark_conds = benchmark_conds[:2]
        num_offgrid = 5
        mismatch_realizations = 2
        mic_err_cm_list = [0.0, 2.0]
        c_assumed_list = [335.0, 343.0, 350.0]
        sync_offsets_us = [0.0, 50.0, 250.0]
        signal_types = ["linear_chirp", "broadband_noise", "speech_synthetic"]
        blind_scenes_count = 5
    else:
        num_offgrid = cfg["study_b_grid_alignment"]["num_offgrid_sources"]
        mismatch_realizations = cfg["study_d_mic_mismatch"]["realizations"]
        mic_err_cm_list = cfg["study_d_mic_mismatch"]["errors_cm"]
        c_assumed_list = cfg["study_e_speed_of_sound"]["assumed_c_list"]
        sync_offsets_us = cfg["study_f_synchronization"]["offsets_us"]
        signal_types = cfg["study_i_signals"]["signal_types"]
        blind_scenes_count = cfg["study_n_blind_validation"]["scenes_count"]

    results_dict: dict[str, pd.DataFrame] = {}

    # =========================================================================
    # STUDY C / AUDIT: Automated Leakage Audit
    # =========================================================================
    logger.info("--- Running Automated Leakage Audit ---")
    df_leakage = run_full_leakage_audit(canonical_sources, out_path / "leakage_audit.csv")
    results_dict["leakage_audit"] = df_leakage

    # =========================================================================
    # STUDY A: Experiment 3 Reproducibility Control
    # =========================================================================
    logger.info("--- Running Study A: Experiment 3 Reproducibility Control ---")
    repro_records = []
    default_localizer = BlindSRPLocalizer(
        assumed_room_dimensions=true_dims,
        assumed_mic_positions_2d=true_mics_2d,
        grid_resolution_m=0.05,
        boundary_margin_m=0.10,
        assumed_speed_of_sound=true_c,
    )

    clean_chirp = generate_chirp(fs=fs, duration=0.1, f_low=300.0, f_high=7000.0)

    for c_idx, cond in enumerate(benchmark_conds):
        cond_id = cond["id"]
        rt60_tgt = float(cond["target_rt60"])
        snr_db = float(cond["snr_target_db"])

        abs_mat, max_ord = safe_inverse_sabine(rt60_tgt, true_dims)

        for s_idx, src in enumerate(canonical_sources):
            s_xy = src["xy"]
            room = pra.ShoeBox(true_dims, fs=fs, materials=pra.Material(abs_mat), max_order=max_ord)
            room.add_microphone_array(pra.MicrophoneArray(true_mics_3d, fs=fs))
            room.add_source([s_xy[0], s_xy[1], src_height], signal=clean_chirp)
            room.simulate()
            clean_sig = room.mic_array.signals.copy()

            n_trials = 2 if smoke_test else 5
            for t_idx in range(n_trials):
                seed = base_seed + c_idx * 1000 + s_idx * 50 + t_idx
                noisy_sig, ach_snr = generate_exact_snr_noise(clean_sig, snr_db, seed)

                out = default_localizer.localize(noisy_sig, fs=fs)
                err = float(np.linalg.norm(np.array([out.pred_x, out.pred_y]) - np.array(s_xy)))
                raw_err = float(np.linalg.norm(np.array([out.raw_x, out.raw_y]) - np.array(s_xy)))

                repro_records.append({
                    "condition_id": cond_id,
                    "source_id": src["id"],
                    "trial_idx": t_idx,
                    "true_x": s_xy[0],
                    "true_y": s_xy[1],
                    "pred_x": out.pred_x,
                    "pred_y": out.pred_y,
                    "localization_error_m": err,
                    "raw_grid_error_m": raw_err,
                    "peak_ratio": out.peak_ratio,
                    "runtime_ms": out.runtime_ms,
                })

    df_repro = pd.DataFrame(repro_records)
    df_repro.to_csv(out_path / "reproducibility.csv", index=False)
    results_dict["reproducibility"] = df_repro

    # =========================================================================
    # STUDY B: Grid Alignment & Off-Grid Quantization Audit
    # =========================================================================
    logger.info("--- Running Study B: Grid Alignment & Off-Grid Audit ---")
    offgrid_sources = generate_offgrid_sources(count=num_offgrid, room_dims=true_dims, seed=base_seed)
    offgrid_records = []

    # Evaluate Condition B (Transition Edge) on 100 offgrid sources
    abs_mat_b, max_ord_b = safe_inverse_sabine(0.250, true_dims)
    for s_idx, src in enumerate(offgrid_sources):
        s_xy = src["xy"]
        room = pra.ShoeBox(true_dims, fs=fs, materials=pra.Material(abs_mat_b), max_order=max_ord_b)
        room.add_microphone_array(pra.MicrophoneArray(true_mics_3d, fs=fs))
        room.add_source([s_xy[0], s_xy[1], src_height], signal=clean_chirp)
        room.simulate()
        clean_sig = room.mic_array.signals.copy()

        noisy_sig, _ = generate_exact_snr_noise(clean_sig, 40.0, seed=base_seed + s_idx)
        out = default_localizer.localize(noisy_sig, fs=fs)
        err = float(np.linalg.norm(np.array([out.pred_x, out.pred_y]) - np.array(s_xy)))
        raw_err = float(np.linalg.norm(np.array([out.raw_x, out.raw_y]) - np.array(s_xy)))

        # Distance to nearest grid node
        grid_d = float(np.min(np.linalg.norm(default_localizer.grid.points_flat - np.array(s_xy), axis=1)))

        offgrid_records.append({
            "source_id": src["id"],
            "true_x": s_xy[0],
            "true_y": s_xy[1],
            "grid_alignment_distance_m": grid_d,
            "raw_grid_error_m": raw_err,
            "refined_error_m": err,
        })

    df_offgrid = pd.DataFrame(offgrid_records)
    df_offgrid.to_csv(out_path / "offgrid_sources.csv", index=False)
    results_dict["offgrid_sources"] = df_offgrid

    # =========================================================================
    # STUDY D: Microphone Position Calibration Mismatch
    # =========================================================================
    logger.info("--- Running Study D: Microphone Position Calibration Mismatch ---")
    mic_mismatch_records = []

    for c_idx, cond in enumerate(benchmark_conds):
        cond_id = cond["id"]
        rt60_tgt = float(cond["target_rt60"])
        snr_db = float(cond["snr_target_db"])
        abs_mat, max_ord = safe_inverse_sabine(rt60_tgt, true_dims)

        for s_idx, src in enumerate(canonical_sources):
            s_xy = src["xy"]
            room = pra.ShoeBox(true_dims, fs=fs, materials=pra.Material(abs_mat), max_order=max_ord)
            room.add_microphone_array(pra.MicrophoneArray(true_mics_3d, fs=fs))
            room.add_source([s_xy[0], s_xy[1], src_height], signal=clean_chirp)
            room.simulate()
            clean_sig = room.mic_array.signals.copy()

            noisy_sig, _ = generate_exact_snr_noise(clean_sig, snr_db, seed=base_seed + s_idx)

            for err_cm in mic_err_cm_list:
                err_m = err_cm / 100.0
                for r_idx in range(mismatch_realizations):
                    rng_mic = np.random.default_rng(base_seed + int(err_cm * 100) + r_idx)
                    # Random 2D perturbation
                    delta_mics = rng_mic.normal(0.0, err_m / np.sqrt(2), size=true_mics_2d.shape) if err_m > 0 else np.zeros_like(true_mics_2d)
                    assumed_mics_2d = true_mics_2d + delta_mics

                    mismatched_localizer = BlindSRPLocalizer(
                        assumed_room_dimensions=true_dims,
                        assumed_mic_positions_2d=assumed_mics_2d,
                        grid_resolution_m=0.05,
                        assumed_speed_of_sound=true_c,
                    )
                    out = mismatched_localizer.localize(noisy_sig, fs=fs)
                    loc_err = float(np.linalg.norm(np.array([out.pred_x, out.pred_y]) - np.array(s_xy)))

                    mic_mismatch_records.append({
                        "condition_id": cond_id,
                        "mic_error_cm": err_cm,
                        "realization_idx": r_idx,
                        "source_id": src["id"],
                        "true_x": s_xy[0],
                        "true_y": s_xy[1],
                        "pred_x": out.pred_x,
                        "pred_y": out.pred_y,
                        "x_error_m": out.pred_x - s_xy[0],
                        "y_error_m": out.pred_y - s_xy[1],
                        "localization_error_m": loc_err,
                    })

    df_mic_evals = pd.DataFrame(mic_mismatch_records)
    df_mic_mismatch = compute_mismatch_sensitivity_table(df_mic_evals, mismatch_param_col="mic_error_cm", control_val=0.0)
    df_mic_mismatch.to_csv(out_path / "mic_mismatch_summary.csv", index=False)
    results_dict["mic_mismatch"] = df_mic_mismatch

    # =========================================================================
    # STUDY E: Speed of Sound Mismatch
    # =========================================================================
    logger.info("--- Running Study E: Speed of Sound Mismatch ---")
    sos_records = []

    for c_idx, cond in enumerate(benchmark_conds):
        cond_id = cond["id"]
        rt60_tgt = float(cond["target_rt60"])
        snr_db = float(cond["snr_target_db"])
        abs_mat, max_ord = safe_inverse_sabine(rt60_tgt, true_dims)

        for s_idx, src in enumerate(canonical_sources):
            s_xy = src["xy"]
            room = pra.ShoeBox(true_dims, fs=fs, materials=pra.Material(abs_mat), max_order=max_ord)
            room.add_microphone_array(pra.MicrophoneArray(true_mics_3d, fs=fs))
            room.add_source([s_xy[0], s_xy[1], src_height], signal=clean_chirp)
            room.simulate()
            clean_sig = room.mic_array.signals.copy()

            noisy_sig, _ = generate_exact_snr_noise(clean_sig, snr_db, seed=base_seed + s_idx)

            for assumed_c in c_assumed_list:
                sos_localizer = BlindSRPLocalizer(
                    assumed_room_dimensions=true_dims,
                    assumed_mic_positions_2d=true_mics_2d,
                    grid_resolution_m=0.05,
                    assumed_speed_of_sound=assumed_c,
                )
                out = sos_localizer.localize(noisy_sig, fs=fs)
                loc_err = float(np.linalg.norm(np.array([out.pred_x, out.pred_y]) - np.array(s_xy)))

                sos_records.append({
                    "condition_id": cond_id,
                    "assumed_speed_of_sound": assumed_c,
                    "delta_c_mps": assumed_c - true_c,
                    "source_id": src["id"],
                    "true_x": s_xy[0],
                    "true_y": s_xy[1],
                    "pred_x": out.pred_x,
                    "pred_y": out.pred_y,
                    "x_error_m": out.pred_x - s_xy[0],
                    "y_error_m": out.pred_y - s_xy[1],
                    "localization_error_m": loc_err,
                })

    df_sos_evals = pd.DataFrame(sos_records)
    df_sos_mismatch = compute_mismatch_sensitivity_table(df_sos_evals, mismatch_param_col="assumed_speed_of_sound", control_val=343.0)
    df_sos_mismatch.to_csv(out_path / "speed_of_sound_summary.csv", index=False)
    results_dict["speed_of_sound"] = df_sos_mismatch

    # =========================================================================
    # STUDY F: Microphone Clock Synchronization Jitter
    # =========================================================================
    logger.info("--- Running Study F: Microphone Synchronization Error ---")
    sync_records = []

    for c_idx, cond in enumerate(benchmark_conds):
        cond_id = cond["id"]
        rt60_tgt = float(cond["target_rt60"])
        snr_db = float(cond["snr_target_db"])
        abs_mat, max_ord = safe_inverse_sabine(rt60_tgt, true_dims)

        for s_idx, src in enumerate(canonical_sources):
            s_xy = src["xy"]
            room = pra.ShoeBox(true_dims, fs=fs, materials=pra.Material(abs_mat), max_order=max_ord)
            room.add_microphone_array(pra.MicrophoneArray(true_mics_3d, fs=fs))
            room.add_source([s_xy[0], s_xy[1], src_height], signal=clean_chirp)
            room.simulate()
            clean_sig = room.mic_array.signals.copy()

            noisy_sig, _ = generate_exact_snr_noise(clean_sig, snr_db, seed=base_seed + s_idx)

            for offset_us in sync_offsets_us:
                # Apply timing offset: shift mic 1 by +offset_us, mic 2 by -offset_us
                offset_samples = int(round((offset_us * 1e-6) * fs))
                shifted_sig = noisy_sig.copy()
                if offset_samples != 0:
                    shifted_sig[1] = np.roll(shifted_sig[1], offset_samples)
                    shifted_sig[2] = np.roll(shifted_sig[2], -offset_samples)

                out = default_localizer.localize(shifted_sig, fs=fs)
                loc_err = float(np.linalg.norm(np.array([out.pred_x, out.pred_y]) - np.array(s_xy)))

                sync_records.append({
                    "condition_id": cond_id,
                    "sync_offset_us": offset_us,
                    "source_id": src["id"],
                    "true_x": s_xy[0],
                    "true_y": s_xy[1],
                    "pred_x": out.pred_x,
                    "pred_y": out.pred_y,
                    "x_error_m": out.pred_x - s_xy[0],
                    "y_error_m": out.pred_y - s_xy[1],
                    "localization_error_m": loc_err,
                })

    df_sync_evals = pd.DataFrame(sync_records)
    df_sync_mismatch = compute_mismatch_sensitivity_table(df_sync_evals, mismatch_param_col="sync_offset_us", control_val=0.0)
    df_sync_mismatch.to_csv(out_path / "synchronization_summary.csv", index=False)
    results_dict["synchronization"] = df_sync_mismatch

    # =========================================================================
    # STUDY I: Source Signal Generalization
    # =========================================================================
    logger.info("--- Running Study I: Source Signal Generalization ---")
    sig_records = []

    for sig_type in signal_types:
        sig_data = generate_diverse_signal(sig_type, fs=fs, duration=0.1, seed=base_seed)

        for c_idx, cond in enumerate(benchmark_conds):
            cond_id = cond["id"]
            rt60_tgt = float(cond["target_rt60"])
            snr_db = float(cond["snr_target_db"])
            abs_mat, max_ord = safe_inverse_sabine(rt60_tgt, true_dims)

            for s_idx, src in enumerate(canonical_sources):
                s_xy = src["xy"]
                room = pra.ShoeBox(true_dims, fs=fs, materials=pra.Material(abs_mat), max_order=max_ord)
                room.add_microphone_array(pra.MicrophoneArray(true_mics_3d, fs=fs))
                room.add_source([s_xy[0], s_xy[1], src_height], signal=sig_data)
                room.simulate()
                clean_sig = room.mic_array.signals.copy()

                noisy_sig, _ = generate_exact_snr_noise(clean_sig, snr_db, seed=base_seed + s_idx)
                out = default_localizer.localize(noisy_sig, fs=fs)
                loc_err = float(np.linalg.norm(np.array([out.pred_x, out.pred_y]) - np.array(s_xy)))

                sig_records.append({
                    "signal_type": sig_type,
                    "condition_id": cond_id,
                    "source_id": src["id"],
                    "localization_error_m": loc_err,
                })

    df_sig_evals = pd.DataFrame(sig_records)
    df_sig_summary = compute_signal_generalization_summary(df_sig_evals)
    df_sig_summary.to_csv(out_path / "signal_generalization_summary.csv", index=False)
    results_dict["signal_generalization"] = df_sig_summary

    # =========================================================================
    # STUDY K: Unseen Room Topologies Generalization
    # =========================================================================
    logger.info("--- Running Study K: Unseen Room Topologies ---")
    room_records = []
    unseen_rooms = cfg["study_k_unseen_rooms"]["rooms"]
    if smoke_test:
        unseen_rooms = unseen_rooms[:2]

    for room_cfg in unseen_rooms:
        r_id = room_cfg["id"]
        r_dims = room_cfg["dimensions"]
        r_mics_2d = np.array(room_cfg["mics_2d"], dtype=np.float64)
        r_mics_3d = np.column_stack([r_mics_2d, np.full(len(r_mics_2d), mic_height)]).T

        room_localizer = BlindSRPLocalizer(
            assumed_room_dimensions=r_dims,
            assumed_mic_positions_2d=r_mics_2d,
            grid_resolution_m=0.05,
            assumed_speed_of_sound=true_c,
        )

        # 5 random test sources in the new room
        r_sources = generate_offgrid_sources(count=5 if smoke_test else 10, room_dims=r_dims, margin=0.5, seed=base_seed)

        for c_idx, cond in enumerate(benchmark_conds):
            cond_id = cond["id"]
            rt60_tgt = float(cond["target_rt60"])
            snr_db = float(cond["snr_target_db"])
            abs_mat, max_ord = safe_inverse_sabine(rt60_tgt, r_dims)

            for s_idx, src in enumerate(r_sources):
                s_xy = src["xy"]
                room = pra.ShoeBox(r_dims, fs=fs, materials=pra.Material(abs_mat), max_order=max_ord)
                room.add_microphone_array(pra.MicrophoneArray(r_mics_3d, fs=fs))
                room.add_source([s_xy[0], s_xy[1], src_height], signal=clean_chirp)
                room.simulate()
                clean_sig = room.mic_array.signals.copy()

                noisy_sig, _ = generate_exact_snr_noise(clean_sig, snr_db, seed=base_seed + s_idx)
                out = room_localizer.localize(noisy_sig, fs=fs)
                loc_err = float(np.linalg.norm(np.array([out.pred_x, out.pred_y]) - np.array(s_xy)))

                room_records.append({
                    "room_id": r_id,
                    "condition_id": cond_id,
                    "source_id": src["id"],
                    "localization_error_m": loc_err,
                })

    df_room_evals = pd.DataFrame(room_records)
    df_room_summary = compute_room_generalization_summary(df_room_evals)
    df_room_summary.to_csv(out_path / "unseen_rooms_summary.csv", index=False)
    results_dict["unseen_rooms"] = df_room_summary

    # =========================================================================
    # STUDY N: Blind Black-Box Pipeline Validation (50 Randomized Scenes)
    # =========================================================================
    logger.info("--- Running Study N: Blind Black-Box Pipeline Validation ---")
    blind_records = []
    rng_blind = np.random.default_rng(base_seed + 999)

    for scene_idx in range(blind_scenes_count):
        # Randomized parameters
        rand_x = rng_blind.uniform(0.5, true_dims[0] - 0.5)
        rand_y = rng_blind.uniform(0.5, true_dims[1] - 0.5)
        rand_rt60 = rng_blind.uniform(0.20, 0.35)
        rand_snr = rng_blind.uniform(20.0, 40.0)
        rand_sig_type = rng_blind.choice(signal_types)

        sig_data = generate_diverse_signal(rand_sig_type, fs=fs, duration=0.1, seed=base_seed + scene_idx)
        abs_mat, max_ord = safe_inverse_sabine(rand_rt60, true_dims)

        room = pra.ShoeBox(true_dims, fs=fs, materials=pra.Material(abs_mat), max_order=max_ord)
        room.add_microphone_array(pra.MicrophoneArray(true_mics_3d, fs=fs))
        room.add_source([rand_x, rand_y, src_height], signal=sig_data)
        room.simulate()
        clean_sig = room.mic_array.signals.copy()

        noisy_sig, ach_snr = generate_exact_snr_noise(clean_sig, rand_snr, seed=base_seed + scene_idx)

        # Localize strictly blind
        out = default_localizer.localize(noisy_sig, fs=fs)

        # Post-hoc evaluation
        loc_err = float(np.linalg.norm(np.array([out.pred_x, out.pred_y]) - np.array([rand_x, rand_y])))

        blind_records.append({
            "scene_idx": scene_idx,
            "true_x": rand_x,
            "true_y": rand_y,
            "target_rt60_s": rand_rt60,
            "target_snr_db": rand_snr,
            "signal_type": rand_sig_type,
            "pred_x": out.pred_x,
            "pred_y": out.pred_y,
            "localization_error_m": loc_err,
            "peak_ratio": out.peak_ratio,
        })

    df_blind = pd.DataFrame(blind_records)
    df_blind.to_csv(out_path / "blind_validation.csv", index=False)
    results_dict["blind_validation"] = df_blind

    # Save master summary
    logger.info("============================================================")
    logger.info("Experiment 3.5 Execution Complete! Outputs saved to: %s", out_path)
    logger.info("============================================================")

    return results_dict
