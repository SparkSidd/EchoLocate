"""
EchoLocate — Unit Test Suite for Experiment 5
=============================================
Tests real dataset adapters, pre-registered sub-arrays, hybrid gating features,
oracle calculations, sim-to-real gap metrics, and selective prediction.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.datasets.real_dataset_adapter import (
    AV163DatasetAdapter,
    EchoLocateRealRecordingAdapter,
    LOCATADatasetAdapter,
)
from src.datasets.recording_protocol import verify_channel_synchronization
from src.evaluation.failure_taxonomy import classify_real_world_failure_category
from src.evaluation.geometry_calibration import compute_geometry_calibration_summary
from src.evaluation.hybrid_gate_analysis import compute_hybrid_gate_metrics
from src.evaluation.real_world_evaluation import (
    compute_selective_prediction_curve,
    compute_temporal_stability_metrics,
)
from src.evaluation.sim_to_real_analysis import compute_sim_to_real_gap
from src.localization.confidence_gated_hybrid import (
    ConfidenceGatedHybridLocalizer,
    extract_observable_gating_features,
)


def test_locata_adapter_subindices():
    """Verify LOCATA adapter preserves native 15-ch array and pre-registered 4-ch subset."""
    adapter = LOCATADatasetAdapter()
    assert adapter.array_spec.num_channels == 15
    assert adapter.subindices == [0, 4, 10, 14]
    manifest = adapter.get_manifest()
    assert manifest.corpus_name == "LOCATA_Benchmark_Corpus"
    assert manifest.sub_array_rules["dicit_4ch_subset"] == [0, 4, 10, 14]

    audio, meta = adapter.load_scene("Test_Scene_01")
    assert audio.shape[0] == 15
    assert meta.dataset_name == "LOCATA"


def test_av16_3_adapter_subindices():
    """Verify AV16.3 adapter preserves circular array geometry and pre-registered 4-ch cardinal subset."""
    adapter = AV163DatasetAdapter()
    assert adapter.array_spec.num_channels == 8
    assert adapter.subindices == [0, 2, 4, 6]
    manifest = adapter.get_manifest()
    assert manifest.corpus_name == "AV16_3_AudioVisual_Corpus"
    assert manifest.sub_array_rules["circular_4ch_cardinal"] == [0, 2, 4, 6]

    audio, meta = adapter.load_scene("Test_Scene_01")
    assert audio.shape[0] == 8
    assert meta.dataset_name == "AV16_3"


def test_echolocate_real_adapter():
    """Verify EchoLocate real recording adapter returns 4-channel audio and metadata."""
    adapter = EchoLocateRealRecordingAdapter()
    assert adapter.array_spec.num_channels == 4
    audio, meta = adapter.load_scene("EchoLocate_Test_01")
    assert audio.shape[0] == 4
    assert meta.dataset_name == "EchoLocate_Real"


def test_channel_synchronization_check():
    """Verify channel synchronization verification helper."""
    ref = np.random.randn(1600)
    sig = np.stack([ref, ref, ref, ref])
    assert verify_channel_synchronization(sig, threshold_us=10.0, fs=16000) is True


def test_observable_gating_feature_extraction():
    """Verify observable acoustic feature extraction without ground-truth access."""
    class MockSRP:
        primary_peak_power = 1.0
        secondary_peak_power = 0.5
        peak_ratio = 2.0

    class MockNeural:
        primary_peak_power = 1.0
        secondary_peak_power = 0.3
        spatial_entropy = 2.0
        spatial_peak_contrast = 3.33

    features = extract_observable_gating_features(MockSRP(), MockNeural(), (5.0, 5.0), (5.0, 5.0))
    assert features.srp_peak_ratio == pytest.approx(2.0, rel=1e-2)
    assert features.cross_method_disagreement_m == pytest.approx(0.0, abs=1e-6)
    assert features.neural_confidence > 0.0


def test_hybrid_gate_metrics():
    """Verify Oracle Gain, Hybrid Gain, and Gate Efficiency calculations."""
    srp_errs = [1.5, 2.0, 1.0]
    neural_errs = [2.5, 1.0, 1.2]
    hyb_errs = [1.5, 1.0, 1.0]  # Optimal selection on all 3
    decisions = ["SRP", "NEURAL", "SRP"]

    metrics = compute_hybrid_gate_metrics(srp_errs, neural_errs, hyb_errs, decisions)
    assert metrics["mean_oracle_error_m"] == pytest.approx(1.1666, rel=1e-2)
    assert metrics["actual_hybrid_gain_m"] == pytest.approx(metrics["oracle_achievable_gain_m"], rel=1e-2)
    assert metrics["gate_efficiency_pct"] == pytest.approx(100.0, rel=1e-2)
    assert metrics["pct_srp_selected"] == pytest.approx(66.66, rel=1e-1)


def test_sim_to_real_gap():
    """Verify Delta error and relative degradation metrics."""
    sim_errs = [0.5, 0.6, 0.4]
    real_errs = [1.0, 1.2, 0.8]
    gap = compute_sim_to_real_gap(sim_errs, real_errs, "M2_Raw_SRP")
    assert gap["delta_error_m"] == pytest.approx(0.5, rel=1e-3)
    assert gap["relative_degradation_factor"] == pytest.approx(2.0, rel=1e-3)


def test_selective_prediction_curve():
    """Verify risk-coverage curve calculation."""
    confs = [0.9, 0.8, 0.7, 0.4, 0.2]
    errs = [0.1, 0.2, 0.3, 1.5, 2.8]
    df_sel = compute_selective_prediction_curve(confs, errs, coverage_levels=[1.0, 0.6])
    assert len(df_sel) == 2
    assert df_sel[df_sel["target_coverage"] == 0.6]["retained_mean_error_m"].values[0] < df_sel[df_sel["target_coverage"] == 1.0]["retained_mean_error_m"].values[0]


def test_temporal_stability_metrics():
    """Verify trajectory tracking RMSE and jump rate calculations."""
    true_traj = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    pred_traj = [(1.05, 0.95), (2.02, 1.98), (5.0, 5.0)]  # Sudden jump at step 3
    metrics = compute_temporal_stability_metrics(pred_traj, true_traj, jump_threshold_m=1.5)
    assert metrics["trajectory_rmse_m"] > 0.0
    assert metrics["sudden_jump_rate_pct"] > 0.0


def test_real_world_failure_classification():
    """Verify 10-class physical failure categorization."""
    assert classify_real_world_failure_category(0.2, False, 0.1, 1.8, 0.0, 0.0, 20.0, 0.2, 0.9) == "F0_Nominal_Success"
    assert classify_real_world_failure_category(1.8, True, 0.9, 1.8, 0.0, 0.0, 20.0, 0.2, 0.9) == "F2_Speech_Harmonic_Ambiguity"
    assert classify_real_world_failure_category(1.8, False, 0.1, 1.8, 10.0, 0.0, 20.0, 0.2, 0.9) == "F3_Microphone_Geometry_Mismatch"
