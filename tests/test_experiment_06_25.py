"""
EchoLocate — Unit Test Suite for Experiment 6.25
================================================
Tests visualization routines, STFT spectrograms, pairwise GCC/DeepGCC analyses,
SRP heatmap normalizations, SRP explainer steps, uncertainty ellipses, latency profiles,
multi-method comparisons, and visualization REST API routes.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.app.analytics.visualization_metrics import audit_component_synchronization, evaluate_ui_performance
from src.app.engine.method_registry import EchoLocateMethodRegistry
from src.app.main import app
from src.app.visualization import (
    analyze_all_gcc_pairs,
    analyze_deepgcc_comparisons,
    audit_spatial_heatmap_payload,
    audit_waveform_payload,
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
from src.app.visualization.provenance import VisualProvenanceMetadata, VisualProvenanceType


def test_waveform_processing_and_audit():
    """Verify multichannel waveform formatting, RMS calculation, and audit."""
    audio = np.random.randn(4, 3200) * 0.1
    prov = VisualProvenanceMetadata.live(0)
    payload = process_multichannel_waveform(audio, fs=16000, frame_id=0, provenance=prov)

    assert payload.duration_ms == pytest.approx(200.0, abs=1e-1)
    assert len(payload.channels) == 4
    assert payload.channels[0].rms_energy_db < 0.0

    rep = audit_waveform_payload(audio, prov)
    assert rep.is_valid is True


def test_spectrogram_computation():
    """Verify STFT spectrogram computation and harmonic detection."""
    t = np.linspace(0, 0.25, 4000)
    sig = np.sin(2 * np.pi * 140.0 * t) + np.sin(2 * np.pi * 280.0 * t)
    payload = compute_channel_spectrogram(sig, fs=16000, channel_idx=0, frame_id=0)

    assert len(payload.spectrogram_db) > 0
    assert len(payload.spectrogram_db[0]) > 0
    assert payload.dominant_pitch_hz is not None


def test_gcc_and_deepgcc_analyses():
    """Verify pairwise GCC and Calibrated DeepGCC neural likelihood analysis."""
    audio = np.random.randn(4, 3200) * 0.1
    registry = EchoLocateMethodRegistry()

    gcc_pay = analyze_all_gcc_pairs(audio, fs=16000)
    assert len(gcc_pay.pairs) == 6
    assert gcc_pay.pairs[0].peak_to_sidelobe_ratio > 0.0

    deep_pay = analyze_deepgcc_comparisons(audio, registry.model_calib, fs=16000)
    assert len(deep_pay.pairs) == 6
    assert len(deep_pay.pairs[0].neural_gcc_curve) == len(deep_pay.pairs[0].raw_gcc_curve)


def test_spatial_heatmap_and_explainer_step():
    """Verify 2D SRP spatial likelihood heatmap and step-by-step explainer."""
    registry = EchoLocateMethodRegistry()
    audio = np.random.randn(4, 3200) * 0.1

    heat_pay = compute_spatial_likelihood_heatmap(registry.localizer_m2, audio, fs=16000)
    assert len(heat_pay.heatmap_matrix) == registry.localizer_m2.grid.n_y
    assert len(heat_pay.heatmap_matrix[0]) == registry.localizer_m2.grid.n_x
    assert heat_pay.primary_peak_power > 0.0

    rep = audit_spatial_heatmap_payload(np.array(heat_pay.heatmap_matrix), registry.room_dims)
    assert rep.is_valid is True

    # Test Explainer Stepper
    exp_step = compute_srp_explainer_step(
        frame_audio=audio,
        candidate_xy=(3.2, 2.2),
        mic_positions_2d=registry.mics,
        room_dimensions_m=registry.room_dims,
        step_number=4,
    )
    assert exp_step.step_number == 4
    assert len(exp_step.pair_samples) == 6


def test_trajectory_and_jitter_metrics():
    """Verify trajectory history formatting and jitter reduction calculations."""
    records = [
        {"frame": i, "raw_x": 4.0 + (i % 2) * 0.4, "raw_y": 3.0, "x": 4.0 + (i * 0.05), "y": 3.0, "confidence": 0.88}
        for i in range(10)
    ]
    payload = format_trajectory_visualization(records)
    assert len(payload.trajectory_points) == 10
    assert payload.mean_raw_jitter_m > 0.0
    assert payload.mean_smoothed_jitter_m >= 0.0


def test_uncertainty_and_latency_profiles():
    """Verify confidence ellipse calculations and pipeline latency breakdown."""
    unc_pay = compute_uncertainty_analysis(4.5, 3.0, 0.85, 1.85)
    assert unc_pay.current_confidence_pct == 85.0
    assert unc_pay.ellipse.semi_major_axis_m > 0.0
    assert len(unc_pay.risk_coverage_curve) == 6

    lat_pay = compute_latency_profile(11.4)
    assert len(lat_pay.stages) == 5
    assert lat_pay.real_time_factor < 1.0


def test_multi_method_comparison_and_timeline():
    """Verify simultaneous multi-method layout and research journey timeline."""
    registry = EchoLocateMethodRegistry()
    audio = np.random.randn(4, 3200) * 0.1

    comp_pay = compute_multi_method_comparison(registry, audio, true_source_xy=(4.5, 3.2))
    assert len(comp_pay.methods) == 5
    assert "M12" in comp_pay.win_rate_matrix

    timeline = get_complete_research_timeline()
    assert len(timeline) == 10
    assert timeline[0].experiment_id == "Exp 1"
    assert timeline[-1].experiment_id == "Exp 6.25"


def test_analytics_metrics():
    """Verify UI performance and synchronization audits."""
    ui_rep = evaluate_ui_performance()
    assert ui_rep.target_fps == 60.0
    assert ui_rep.is_smooth is True

    timestamps = [0.0, 0.20, 0.40]
    sync_rep = audit_component_synchronization(timestamps, timestamps, timestamps)
    assert sync_rep.is_synchronized is True


def test_fastapi_visualization_endpoints():
    """Verify all 12 visualization REST API routes respond with status 200."""
    client = TestClient(app)

    # 1. Waveform
    r1 = client.get("/api/visualization/waveform")
    assert r1.status_code == 200
    assert "channels" in r1.json()

    # 2. Spectrogram
    r2 = client.get("/api/visualization/spectrogram?channel=0")
    assert r2.status_code == 200
    assert "spectrogram_db" in r2.json()

    # 3. GCC
    r3 = client.post("/api/visualization/gcc", json={})
    assert r3.status_code == 200
    assert len(r3.json()["pairs"]) == 6

    # 4. DeepGCC
    r4 = client.post("/api/visualization/deepgcc", json={})
    assert r4.status_code == 200
    assert len(r4.json()["pairs"]) == 6

    # 5. Spatial
    r5 = client.post("/api/visualization/spatial", json={})
    assert r5.status_code == 200
    assert "heatmap_matrix" in r5.json()

    # 6. SRP Explainer Step
    r6 = client.post("/api/visualization/srp_explainer_step", json={"step_number": 4, "candidate_x": 3.2, "candidate_y": 2.2})
    assert r6.status_code == 200
    assert r6.json()["step_number"] == 4

    # 7. Methods
    r7 = client.post("/api/visualization/methods", json={})
    assert r7.status_code == 200
    assert len(r7.json()["methods"]) == 5

    # 8. Confidence
    r8 = client.get("/api/visualization/confidence")
    assert r8.status_code == 200
    assert "current_confidence_pct" in r8.json()

    # 9. Latency
    r9 = client.get("/api/visualization/latency")
    assert r9.status_code == 200
    assert len(r9.json()["stages"]) == 5

    # 10. History
    r10 = client.get("/api/visualization/history")
    assert r10.status_code == 200

    # 11. Experiments Timeline
    r11 = client.get("/api/visualization/experiments")
    assert r11.status_code == 200
    assert len(r11.json()) == 10

    # 12. Simulation Lab
    r12 = client.post("/api/visualization/simulation_lab", json={"rt60": 0.38, "snr": 18, "src_x": 4.5, "src_y": 3.2, "mic_perturbation_mm": 0})
    assert r12.status_code == 200
    assert "failure_diagnosis" in r12.json()
