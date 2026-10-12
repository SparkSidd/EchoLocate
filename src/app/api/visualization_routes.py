"""
EchoLocate — Visualization REST API Endpoints
=============================================
Provides structured JSON data payloads for waveforms, spectrograms,
pairwise GCC curves, DeepGCC neural comparisons, SRP spatial heatmaps,
step-by-step SRP explainers, method comparisons, uncertainty, and research timelines.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Body, HTTPException
import numpy as np

from src.app.engine.method_registry import EchoLocateMethodRegistry
from src.app.engine.scenario_manager import PresetScenarioManager
from src.app.state.app_state import ApplicationState
from src.app.visualization import (
    analyze_all_gcc_pairs,
    analyze_deepgcc_comparisons,
    compute_channel_spectrogram,
    compute_latency_profile,
    compute_multi_method_comparison,
    compute_spatial_likelihood_heatmap,
    compute_srp_explainer_step,
    compute_uncertainty_analysis,
    format_trajectory_visualization,
    get_complete_research_timeline,
    process_multichannel_waveform,
)
from src.evaluation.neural_robustness_audit import simulate_room_acoustics
from src.experiments.environment_robustness import generate_exact_snr_noise

logger = logging.getLogger(__name__)


def create_visualization_router(
    app_state: ApplicationState,
    registry: EchoLocateMethodRegistry,
    scenario_mgr: PresetScenarioManager,
) -> APIRouter:
    router = APIRouter(prefix="/api/visualization")

    @router.get("/waveform")
    async def get_waveform_data() -> dict[str, Any]:
        """Fetch latest multichannel waveform display data."""
        # Generate nominal or active frame audio
        audio, _ = scenario_mgr.generate_scenario_audio("demo_1_stable", fs=registry.fs)
        payload = process_multichannel_waveform(audio, fs=registry.fs, frame_id=app_state.total_frames_processed)
        return {
            "frame_id": payload.frame_id,
            "sample_rate": payload.sample_rate,
            "duration_ms": payload.duration_ms,
            "time_axis_ms": payload.time_axis_ms,
            "channels": [vars(c) for c in payload.channels],
            "inter_channel_delays": payload.inter_channel_delays,
            "provenance": payload.provenance.to_dict(),
        }

    @router.get("/spectrogram")
    async def get_spectrogram_data(channel: int = 0) -> dict[str, Any]:
        """Fetch STFT spectrogram for specified microphone channel."""
        audio, _ = scenario_mgr.generate_scenario_audio("demo_3_speech_harmonic_ambiguity", fs=registry.fs)
        ch_idx = max(0, min(audio.shape[0] - 1, channel))
        payload = compute_channel_spectrogram(audio[ch_idx], fs=registry.fs, channel_idx=ch_idx, frame_id=app_state.total_frames_processed)
        return {
            "channel_index": payload.channel_index,
            "time_bins_ms": payload.time_bins_ms,
            "frequency_bins_hz": payload.frequency_bins_hz,
            "spectrogram_db": payload.spectrogram_db,
            "dominant_pitch_hz": payload.dominant_pitch_hz,
            "harmonic_peaks_hz": payload.harmonic_peaks_hz,
            "provenance": payload.provenance.to_dict(),
        }

    @router.post("/gcc")
    async def get_gcc_curves(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        """Fetch pairwise GCC-PHAT cross-correlation curves."""
        scenario_id = payload.get("scenario_id", "demo_2_classical_failure_srp_recovery")
        audio, _ = scenario_mgr.generate_scenario_audio(scenario_id, fs=registry.fs)
        gcc_payload = analyze_all_gcc_pairs(audio, fs=registry.fs, frame_id=app_state.total_frames_processed)
        return {
            "frame_id": gcc_payload.frame_id,
            "pairs": [vars(p) for p in gcc_payload.pairs],
            "provenance": gcc_payload.provenance.to_dict(),
        }

    @router.post("/deepgcc")
    async def get_deepgcc_comparisons(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        """Fetch Calibrated DeepGCC neural likelihood vs. raw GCC comparisons."""
        scenario_id = payload.get("scenario_id", "demo_3_speech_harmonic_ambiguity")
        audio, _ = scenario_mgr.generate_scenario_audio(scenario_id, fs=registry.fs)
        neural_payload = analyze_deepgcc_comparisons(audio, registry.model_calib, fs=registry.fs, frame_id=app_state.total_frames_processed)
        return {
            "frame_id": neural_payload.frame_id,
            "pairs": [vars(p) for p in neural_payload.pairs],
            "provenance": neural_payload.provenance.to_dict(),
        }

    @router.post("/spatial")
    async def get_spatial_heatmap(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        """Fetch 2D SRP spatial likelihood heatmap grid and contours."""
        scenario_id = payload.get("scenario_id", "demo_1_stable")
        audio, cfg = scenario_mgr.generate_scenario_audio(scenario_id, fs=registry.fs)
        registry.update_geometry(cfg.room_dimensions_m, np.array(cfg.mic_positions_2d))
        heatmap_payload = compute_spatial_likelihood_heatmap(registry.localizer_m2, audio, fs=registry.fs, frame_id=app_state.total_frames_processed)
        return {
            "frame_id": heatmap_payload.frame_id,
            "room_dimensions_m": heatmap_payload.room_dimensions_m,
            "grid_extent_xy": heatmap_payload.grid_extent_xy,
            "grid_resolution_m": heatmap_payload.grid_resolution_m,
            "heatmap_matrix": heatmap_payload.heatmap_matrix,
            "primary_peak_xy": heatmap_payload.primary_peak_xy,
            "primary_peak_power": heatmap_payload.primary_peak_power,
            "secondary_peak_xy": heatmap_payload.secondary_peak_xy,
            "secondary_peak_power": heatmap_payload.secondary_peak_power,
            "peak_contrast_ratio": heatmap_payload.peak_contrast_ratio,
            "spatial_entropy": heatmap_payload.spatial_entropy,
            "contour_levels": heatmap_payload.contour_levels,
            "provenance": heatmap_payload.provenance.to_dict(),
        }

    @router.post("/srp_explainer_step")
    async def get_srp_explainer_step(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Fetch educational constructive interference explainer step calculation."""
        step = int(payload.get("step_number", 4))
        cand_x = float(payload.get("candidate_x", 3.2))
        cand_y = float(payload.get("candidate_y", 2.2))
        audio, cfg = scenario_mgr.generate_scenario_audio("demo_1_stable", fs=registry.fs)

        step_res = compute_srp_explainer_step(
            frame_audio=audio,
            candidate_xy=(cand_x, cand_y),
            mic_positions_2d=np.array(cfg.mic_positions_2d),
            room_dimensions_m=cfg.room_dimensions_m,
            speed_of_sound=registry.speed_of_sound,
            fs=registry.fs,
            step_number=step,
        )
        return {
            "step_number": step_res.step_number,
            "step_title": step_res.step_title,
            "explanation": step_res.explanation,
            "candidate_xy": step_res.candidate_xy,
            "accumulated_power": step_res.accumulated_power,
            "is_constructive_peak": step_res.is_constructive_peak,
            "pair_samples": [vars(s) for s in step_res.pair_samples],
            "provenance": step_res.provenance.to_dict(),
        }

    @router.post("/methods")
    async def get_multi_method_comparison(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        """Fetch simultaneous 5-method comparison layout and win-rate matrix."""
        scenario_id = payload.get("scenario_id", "demo_2_classical_failure_srp_recovery")
        audio, cfg = scenario_mgr.generate_scenario_audio(scenario_id, fs=registry.fs)
        registry.update_geometry(cfg.room_dimensions_m, np.array(cfg.mic_positions_2d))
        comp_payload = compute_multi_method_comparison(registry, audio, true_source_xy=cfg.true_source_xy, frame_id=app_state.total_frames_processed)
        return {
            "frame_id": comp_payload.frame_id,
            "true_source_xy": comp_payload.true_source_xy,
            "methods": [vars(m) for m in comp_payload.methods],
            "win_rate_matrix": comp_payload.win_rate_matrix,
            "ranking": comp_payload.ranking,
            "provenance": comp_payload.provenance.to_dict(),
        }

    @router.get("/confidence")
    async def get_confidence_uncertainty() -> dict[str, Any]:
        """Fetch real-time confidence gauge, 2D error ellipse, and risk-coverage curve."""
        latest = app_state.latest_result
        cur_x = latest.smoothed_estimate_xy[0] if latest else 4.5
        cur_y = latest.smoothed_estimate_xy[1] if latest else 3.0
        conf = latest.confidence if latest else 0.85
        contrast = latest.spatial_contrast if latest else 1.85
        hist = [p.get("confidence", 0.85) * 100.0 for p in app_state.trajectory_history] if app_state.trajectory_history else [85.0]

        unc_payload = compute_uncertainty_analysis(cur_x, cur_y, conf, contrast, history_confidences=hist)
        return {
            "current_confidence_pct": unc_payload.current_confidence_pct,
            "confidence_state": unc_payload.confidence_state,
            "ellipse": vars(unc_payload.ellipse),
            "confidence_history": unc_payload.confidence_history,
            "risk_coverage_curve": [vars(rc) for rc in unc_payload.risk_coverage_curve],
            "provenance": unc_payload.provenance.to_dict(),
        }

    @router.get("/latency")
    async def get_pipeline_latency() -> dict[str, Any]:
        """Fetch per-stage pipeline execution latency breakdown and statistics."""
        stats = app_state.get_latency_stats()
        lat_payload = compute_latency_profile(
            total_ms=stats.get("mean_ms", 11.4) or 11.4,
            history_latencies_ms=app_state.latencies_ms,
            queue_delay_ms=0.4,
            dropped_frames=app_state.dropped_frames_total,
        )
        return {
            "total_latency_ms": lat_payload.total_latency_ms,
            "frame_duration_ms": lat_payload.frame_duration_ms,
            "real_time_factor": lat_payload.real_time_factor,
            "queue_delay_ms": lat_payload.queue_delay_ms,
            "stages": [vars(s) for s in lat_payload.stages],
            "mean_latency_ms": lat_payload.mean_latency_ms,
            "median_latency_ms": lat_payload.median_latency_ms,
            "p95_latency_ms": lat_payload.p95_latency_ms,
            "p99_latency_ms": lat_payload.p99_latency_ms,
            "max_latency_ms": lat_payload.max_latency_ms,
            "dropped_frames_total": lat_payload.dropped_frames_total,
            "provenance": lat_payload.provenance.to_dict(),
        }

    @router.get("/history")
    async def get_trajectory_history() -> dict[str, Any]:
        """Fetch temporal tracking history, velocity vectors, and jitter metrics."""
        traj_payload = format_trajectory_visualization(app_state.trajectory_history)
        return {
            "trajectory_points": [vars(pt) for pt in traj_payload.trajectory_points],
            "mean_raw_jitter_m": traj_payload.mean_raw_jitter_m,
            "mean_smoothed_jitter_m": traj_payload.mean_smoothed_jitter_m,
            "jitter_reduction_pct": traj_payload.jitter_reduction_pct,
            "current_velocity_m_s": traj_payload.current_velocity_m_s,
            "provenance": traj_payload.provenance.to_dict(),
        }

    @router.get("/experiments")
    async def get_research_timeline() -> list[dict[str, Any]]:
        """Fetch complete research journey timeline data."""
        timeline = get_complete_research_timeline()
        return [vars(m) for m in timeline]

    @router.post("/simulation_lab")
    async def run_simulation_lab(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Execute full interactive simulation lab with physical failure diagnosis."""
        rt60 = float(payload.get("rt60", 0.35))
        snr = float(payload.get("snr", 20.0))
        src_x = float(payload.get("src_x", 4.0))
        src_y = float(payload.get("src_y", 3.0))
        pert_mm = float(payload.get("mic_perturbation_mm", 0.0))
        sig_type = payload.get("signal_type", "voiced_speech")

        room_dims = [10.0, 6.0, 3.0]
        nominal_mics = np.array([[2.0, 1.5], [8.0, 1.5], [5.0, 4.5], [5.0, 1.5]])
        actual_mics = nominal_mics + (pert_mm * 1e-3 * np.random.normal(0, 1, nominal_mics.shape))

        # Synthesize audio
        t = np.linspace(0, 0.25, int(registry.fs * 0.25))
        sig = np.random.randn(len(t)) if sig_type == "broadband_noise" else np.sin(2 * np.pi * 140.0 * t)

        raw_audio, _ = simulate_room_acoustics(room_dims, actual_mics, [src_x, src_y, 1.2], sig, rt60, fs=registry.fs)
        noisy_audio, _ = generate_exact_snr_noise(raw_audio, target_snr_db=snr, seed=8001)

        registry.update_geometry(room_dims, nominal_mics)
        t0 = time.perf_counter()
        out = registry.execute_method("M12_Confidence_Hybrid", noisy_audio)
        lat = (time.perf_counter() - t0) * 1000.0
        err = float(np.hypot(out.pred_x - src_x, out.pred_y - src_y))

        # Failure diagnosis
        diagnosis = "Nominal High-Accuracy Localization"
        if pert_mm > 5.0:
            diagnosis = f"Physical Array Mismatch ({pert_mm:.1f} mm): Uncalibrated microphone positions degrade spatial coherence."
        elif rt60 > 0.40:
            diagnosis = f"High Reverberation Zone (RT60 = {rt60:.2f}s): Room reflections create multipath spatial ambiguity."
        elif snr < 5.0:
            diagnosis = f"Low SNR Regime ({snr:.0f} dB): High noise floor corrupts cross-correlation phase."

        return {
            "true_source_xy": [src_x, src_y],
            "pred_x": out.pred_x,
            "pred_y": out.pred_y,
            "error_m": err,
            "confidence": out.confidence,
            "latency_ms": lat,
            "spatial_contrast": out.spatial_contrast,
            "status": out.status,
            "failure_diagnosis": diagnosis,
            "provenance": "SIMULATION",
        }

    return router
