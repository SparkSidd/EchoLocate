"""
EchoLocate — Unit Test Suite for Experiment 4.75
=================================================
Tests adversarial perturbation, off-grid generation, speech synthesis,
confidence calibration, geometric equivariance, and statistical validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.adversarial_failure_analysis import (
    attribute_neural_failure_mechanism,
    classify_adversarial_error,
)
from src.evaluation.calibration_generalization import (
    audit_geometric_equivariance,
    compute_confidence_correlation_metrics,
    compute_expected_calibration_error,
    compute_risk_coverage_curve,
)
from src.evaluation.cross_domain_analysis import (
    compute_pitch_stress_metrics,
    generate_adversarial_speech_utterance,
    generate_benchmark_signal,
)
from src.evaluation.neural_robustness_audit import (
    apply_channel_synchronization_jitter,
    apply_microphone_perturbation,
    generate_continuous_offgrid_sources,
)
from src.evaluation.statistical_validation import (
    audit_experiment_04_75_dataset_splits,
    compute_adversarial_paired_statistics,
)


def test_offgrid_source_generation():
    """Verify continuous off-grid sources do not collapse onto integer grid."""
    room_dims = [10.0, 6.0, 3.0]
    sources, d_nearest = generate_continuous_offgrid_sources(n_sources=20, room_dims=room_dims, grid_resolution_m=0.05, seed=42)
    assert len(sources) == 20
    assert len(d_nearest) == 20
    assert np.all(d_nearest >= 0.0)
    assert np.all(d_nearest <= 0.05 * np.sqrt(2))


def test_microphone_perturbation():
    """Verify perturbation shifts mics proportionally to perturbation magnitude."""
    mics = np.array([[2.0, 1.5], [8.0, 1.5], [5.0, 4.5]])
    pert_0 = apply_microphone_perturbation(mics, perturbation_mm=0.0)
    np.testing.assert_array_almost_equal(mics, pert_0)

    pert_10 = apply_microphone_perturbation(mics, perturbation_mm=10.0, seed=42)
    diff = np.linalg.norm(pert_10 - mics, axis=1)
    assert np.all(diff > 0.0)
    assert np.mean(diff) < 0.05  # Within reasonable bound for 10mm std


def test_channel_sync_jitter():
    """Verify synchronization offset applies time delay shift."""
    sig = np.random.randn(3, 1600)
    shifted = apply_channel_synchronization_jitter(sig, fs=16000, offset_us=100.0, mode="bias")
    assert shifted.shape == sig.shape
    np.testing.assert_array_equal(shifted[0], sig[0])  # Reference mic 0 unchanged


def test_adversarial_speech_synthesis():
    """Verify synthetic speech generation produces valid signal and pitch metadata."""
    sig, meta = generate_adversarial_speech_utterance(signal_type="voiced_speech", f0_hz=150.0, duration=0.1, fs=16000)
    assert len(sig) == 1600
    assert np.max(np.abs(sig)) <= 1.0
    assert meta["f0_bin"] == "120-180"


def test_pitch_stress_metrics():
    """Verify PPR, HSR, and CPD calculation."""
    tau_axis = np.linspace(-0.02, 0.02, 129)
    raw_curve = np.zeros(129)
    raw_curve[64] = 1.0   # True peak at 0.0s
    raw_curve[96] = 0.8   # Harmonic peak at +0.01s (100 Hz pitch)

    enh_curve = np.zeros(129)
    enh_curve[64] = 1.0
    enh_curve[96] = 0.2   # Suppressed harmonic

    metrics = compute_pitch_stress_metrics(raw_curve, enh_curve, tau_axis, true_tdoa=0.0, f0_hz=100.0)
    assert metrics["raw_ppr"] == pytest.approx(0.8, rel=1e-3)
    assert metrics["enhanced_ppr"] == pytest.approx(0.2, rel=1e-3)
    assert metrics["harmonic_suppression_ratio"] == pytest.approx(4.0, rel=1e-3)


def test_confidence_and_ece():
    """Verify ECE and Risk-Coverage curve computation."""
    confs = np.array([0.9, 0.85, 0.7, 0.4, 0.2])
    errs = np.array([0.05, 0.12, 0.35, 1.20, 2.50])
    ece, bin_confs, bin_accs = compute_expected_calibration_error(confs, errs, error_threshold_m=0.50, n_bins=5)
    assert 0.0 <= ece <= 1.0

    aurc, covs, risks = compute_risk_coverage_curve(confs, errs, n_points=10)
    assert aurc >= 0.0


def test_geometric_equivariance():
    """Verify geometric equivariance calculation."""
    p_orig = np.array([3.0, 2.0])
    R = np.array([[0.0, -1.0], [1.0, 0.0]])  # 90 deg rotation
    t = np.array([1.0, 1.0])
    p_trans = R @ p_orig + t

    err = audit_geometric_equivariance(p_orig, p_trans, R, t)
    assert err == pytest.approx(0.0, abs=1e-6)


def test_adversarial_failure_classification():
    """Verify failure taxonomy classification."""
    assert classify_adversarial_error(0.05, [3.0, 2.0]) == "excellent"
    assert classify_adversarial_error(0.20, [3.0, 2.0]) == "accurate"
    assert classify_adversarial_error(0.80, [3.0, 2.0]) == "degraded"
    assert classify_adversarial_error(1.50, [3.0, 2.0]) == "large_error"
    assert classify_adversarial_error(2.50, [3.0, 2.0]) == "catastrophic_error"
    assert classify_adversarial_error(float("nan"), [3.0, 2.0]) == "nan_inf"


def test_statistical_validation():
    """Verify paired bootstrap statistics and split integrity."""
    p_errs = np.array([0.2, 0.3, 0.25, 0.4])
    b_errs = np.array([0.5, 0.6, 0.55, 0.7])
    stats = compute_adversarial_paired_statistics(p_errs, b_errs, "Primary", "Baseline", n_bootstraps=50)
    assert stats["wins"] == 4
    assert stats["losses"] == 0
    assert stats["win_rate_pct"] == 100.0

    df_splits = audit_experiment_04_75_dataset_splits()
    assert np.all(df_splits["Isolation Status"] == "STRICT_PASS")
