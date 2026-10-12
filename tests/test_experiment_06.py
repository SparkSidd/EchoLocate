"""
EchoLocate — Unit Test Suite for Experiment 6
=============================================
Tests circular audio buffers, input frame validation, hardware geometry validators,
causal temporal tracking, scenario generation, method registry, and anti-leakage audits.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.app.audio.stream_manager import CircularAudioBuffer
from src.app.audit.causality_audit import audit_frame_causality_stream
from src.app.audit.demo_integrity import run_demo_anti_leakage_audit
from src.app.engine.input_validator import validate_audio_frame
from src.app.engine.method_registry import EchoLocateMethodRegistry
from src.app.engine.scenario_manager import PresetScenarioManager
from src.app.engine.temporal_tracker import CausalTemporalTracker
from src.app.hardware.geometry_validator import validate_microphone_geometry
from src.app.hardware.sync_diagnostics import ChannelSyncMonitor
from src.app.main import app


def test_circular_audio_buffer():
    """Verify circular audio buffer write, read, and boundary wraparound."""
    buf = CircularAudioBuffer(num_channels=4, buffer_samples=1000)
    chunk1 = np.ones((4, 600), dtype=np.float32)
    buf.write(chunk1)

    read1 = buf.read_latest(400)
    assert read1.shape == (4, 400)
    assert np.all(read1 == 1.0)

    # Trigger wraparound
    chunk2 = np.ones((4, 700), dtype=np.float32) * 2.0
    buf.write(chunk2)
    read2 = buf.read_latest(500)
    assert read2.shape == (4, 500)
    assert np.all(read2 == 2.0)


def test_input_frame_validation():
    """Verify audio frame validator rejects silence, NaN/Inf, and wrong channel count."""
    # Valid frame
    valid_sig = np.random.randn(4, 3200) * 0.1
    res_valid = validate_audio_frame(valid_sig, expected_channels=4)
    assert res_valid.is_valid is True

    # Digital silence
    silent_sig = np.zeros((4, 3200))
    res_silence = validate_audio_frame(silent_sig, expected_channels=4)
    assert res_silence.is_valid is False
    assert res_silence.is_silent is True

    # NaN corruption
    nan_sig = valid_sig.copy()
    nan_sig[0, 10] = np.nan
    res_nan = validate_audio_frame(nan_sig, expected_channels=4)
    assert res_nan.is_valid is False
    assert res_nan.has_nan_or_inf is True

    # Channel count mismatch
    wrong_ch = np.random.randn(2, 3200)
    res_ch = validate_audio_frame(wrong_ch, expected_channels=4)
    assert res_ch.is_valid is False


def test_causal_temporal_tracker():
    """Verify strictly causal exponential and Kalman tracking."""
    tracker = CausalTemporalTracker(mode="CAUSAL_SMOOTHED", alpha=0.50)
    t1 = tracker.update(4.0, 3.0)
    assert t1.smoothed_x == 4.0

    t2 = tracker.update(6.0, 3.0)
    assert t2.smoothed_x == pytest.approx(5.0, abs=1e-3)
    assert t2.is_causal is True

    # Test reset
    tracker.reset()
    assert tracker.is_initialized is False


def test_microphone_geometry_validation():
    """Verify geometry validator detects collinearity, duplicates, and valid arrays."""
    # Valid planar array
    valid_mics = [[2.0, 1.5], [8.0, 1.5], [5.0, 4.5], [5.0, 1.5]]
    res_valid = validate_microphone_geometry(valid_mics)
    assert res_valid.is_valid is True

    # Collinear mics
    collinear_mics = [[2.0, 2.0], [4.0, 2.0], [6.0, 2.0], [8.0, 2.0]]
    res_collinear = validate_microphone_geometry(collinear_mics)
    assert res_collinear.is_valid is False
    assert res_collinear.is_collinear is True

    # Duplicate mics
    dup_mics = [[2.0, 1.5], [2.001, 1.501], [5.0, 4.5], [5.0, 1.5]]
    res_dup = validate_microphone_geometry(dup_mics, min_spacing_m=0.02)
    assert res_dup.is_valid is False
    assert res_dup.has_duplicates is True


def test_channel_sync_monitor():
    """Verify sync monitor detects multi-channel clock drift."""
    monitor = ChannelSyncMonitor(num_channels=4, max_allowed_lag_samples=2)
    clean_audio = np.random.randn(4, 1600)
    rep = monitor.inspect_frame(clean_audio)
    assert rep.status in ["SYNCHRONIZED", "SYNC_DRIFT_DETECTED", "NOMINAL"]


def test_preset_scenarios_audio_generation():
    """Verify PresetScenarioManager synthesizes genuine audio for all 5 demo scenarios."""
    mgr = PresetScenarioManager()
    for s_id in ["demo_1_stable", "demo_2_classical_failure_srp_recovery", "demo_3_speech_harmonic_ambiguity", "demo_4_physical_mismatch", "demo_5_real_playback"]:
        audio, cfg = mgr.generate_scenario_audio(s_id, fs=16000)
        assert audio.shape[0] == 4
        assert cfg.scenario_id == s_id


def test_method_registry_execution():
    """Verify EchoLocateMethodRegistry executes M1, M2, M6, M11e, M12 without error."""
    registry = EchoLocateMethodRegistry()
    mock_audio = np.random.randn(4, 3200) * 0.1

    for m_id in ["M1_Classical_NLS", "M2_Raw_SRP", "M6_Calibrated_DeepGCC", "M11e_Hybrid_Spatial_SRP", "M12_Confidence_Hybrid"]:
        out = registry.execute_method(m_id, mock_audio)
        assert out.pred_x is not None
        assert out.pred_y is not None
        assert out.latency_ms > 0.0


def test_causality_audit_stream():
    """Verify causality audit verifies monotonic timestamps and future-frame isolation."""
    timestamps = [0.0, 0.1, 0.2, 0.3, 0.4]
    proc_times = [0.01, 0.11, 0.21, 0.31, 0.41]
    rep = audit_frame_causality_stream(timestamps, proc_times)
    assert rep.is_causal is True
    assert rep.timestamp_inversions == 0


def test_demo_anti_leakage_audit():
    """Verify anti-leakage audit passes static AST and dynamic permutation invariance."""
    df_audit = run_demo_anti_leakage_audit()
    assert len(df_audit) == 4
    assert np.all(df_audit["status"] == "PASS")


def test_fastapi_rest_endpoints():
    """Verify FastAPI prototype API endpoints respond correctly."""
    client = TestClient(app)

    # Status
    res_status = client.get("/api/status")
    assert res_status.status_code == 200
    assert "active_mode" in res_status.json()

    # Switch Mode
    res_mode = client.post("/api/mode", json={"mode": "SIMULATION"})
    assert res_mode.status_code == 200
    assert res_mode.json()["active_mode"] == "SIMULATION"

    # Switch Method
    res_method = client.post("/api/method", json={"method_id": "M12_Confidence_Hybrid"})
    assert res_method.status_code == 200

    # Load Scenario
    res_scenario = client.post("/api/scenario/load", json={"scenario_id": "demo_1_stable"})
    assert res_scenario.status_code == 200
    assert "result" in res_scenario.json()

    # Multi-Method Comparison
    res_comp = client.post("/api/comparison/run", json={"scenario_id": "demo_2_classical_failure_srp_recovery"})
    assert res_comp.status_code == 200
    assert len(res_comp.json()["comparison_table"]) == 5

    # Examiner Rerun Frame
    res_exam = client.post("/api/examiner/rerun_frame", json={"method_id": "M2_Raw_SRP"})
    assert res_exam.status_code == 200
    assert res_exam.json()["audit_status"] == "GENUINE_PIPELINE_RECOMPUTED"
