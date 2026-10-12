"""
Unit tests for Experiment 4: Learned Acoustic Feature Enhancement (DeepGCC + SRP).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.evaluation.acoustic_mechanism_analysis import (
    compute_correlation_entropy,
    extract_correlation_peaks_and_rank,
)
from src.evaluation.confidence_analysis import compute_confidence_calibration_summary
from src.evaluation.neural_leakage_audit import (
    audit_neural_interface_isolation,
    audit_neural_permutation_invariance,
    run_full_neural_leakage_audit,
)
from src.evaluation.speech_mechanism_analysis import analyze_speech_pitch_correlation_peaks
from src.models.acoustic_frontend import (
    extract_pairwise_gcc_features,
    generate_experiment_04_signal,
    generate_gaussian_target_distribution,
)
from src.models.inference import BlindNeuralGCCPredictor
from src.models.neural_gcc import DeepGCCRes1D
from src.models.training import compute_composite_loss


class TestDeepGCCModelArchitecture:
    def test_forward_pass_shape_and_probability_sum(self):
        """Verify DeepGCC outputs a valid probability distribution summing to 1.0."""
        model = DeepGCCRes1D(in_channels=1, hidden_channels=32, num_res_blocks=2)
        model.eval()

        dummy_input = torch.randn(4, 1, 129)
        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (4, 129)
        assert torch.all(output >= 0.0)
        sums = torch.sum(output, dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    def test_composite_loss_calculation(self):
        pred = torch.softmax(torch.randn(2, 129), dim=-1)
        target = torch.softmax(torch.randn(2, 129), dim=-1)
        true_tdoa = torch.tensor([0.005, -0.002])
        tau_axis = torch.linspace(-0.030, 0.030, 129)

        total_loss, l_dict = compute_composite_loss(pred, target, true_tdoa, tau_axis)
        assert torch.isfinite(total_loss)
        assert l_dict["total_loss"] > 0.0
        assert "kl_loss" in l_dict
        assert "tdoa_loss" in l_dict


class TestAcousticFrontendAndSignals:
    def test_speech_signal_generation_with_pitch_meta(self):
        fs = 16000
        sig, meta = generate_experiment_04_signal("speech_male_vowel", fs=fs, duration=0.1, seed=42)
        assert len(sig) == int(0.1 * fs)
        assert meta["f0_hz"] > 0.0
        assert meta["pitch_period_s"] > 0.0

    def test_pairwise_gcc_feature_extraction(self):
        signals = np.random.randn(3, 1600)
        gcc_feat, ipd_feat, tau_axis = extract_pairwise_gcc_features(signals, fs=16000, max_tau_seconds=0.030, num_tau_bins=129)
        assert gcc_feat.shape == (3, 129)
        assert len(tau_axis) == 129
        assert np.isclose(tau_axis[0], -0.030, atol=1e-5)
        assert np.isclose(tau_axis[-1], 0.030, atol=1e-5)


class TestMechanisticAnalysis:
    def test_correlation_entropy(self):
        # Delta spike -> near 0 entropy
        delta_curve = np.zeros(129)
        delta_curve[64] = 1.0
        ent_spike = compute_correlation_entropy(delta_curve)

        # Uniform flat curve -> high entropy
        flat_curve = np.ones(129)
        ent_flat = compute_correlation_entropy(flat_curve)

        assert ent_spike < ent_flat

    def test_peak_extraction_and_rank(self):
        tau_axis = np.linspace(-0.030, 0.030, 129)
        curve = np.exp(-((tau_axis - 0.005)**2) / (2 * 0.001**2))
        res = extract_correlation_peaks_and_rank(curve, tau_axis, true_tdoa=0.005)
        assert res["true_tdoa_peak_rank"] == 1.0
        assert res["peak_competition_ratio"] >= 1.0

    def test_speech_pitch_pseudo_peak_analysis(self):
        tau_axis = np.linspace(-0.030, 0.030, 129)
        f0 = 200.0  # T0 = 5ms = 0.005s
        # True peak at 0.000s, harmonic peak at +0.005s
        curve = np.exp(-(tau_axis**2) / 1e-6) + 0.8 * np.exp(-((tau_axis - 0.005)**2) / 1e-6)
        res = analyze_speech_pitch_correlation_peaks(curve, tau_axis, true_tdoa=0.0, f0_hz=f0)
        assert res["pitch_pseudo_peaks_detected"] >= 1.0


class TestNeuralLeakageAndIsolation:
    def test_neural_interface_isolation(self):
        status, evidence, severity = audit_neural_interface_isolation()
        assert status == "PASS"
        assert severity == "NONE"

    def test_neural_permutation_invariance(self):
        status, evidence, severity = audit_neural_permutation_invariance()
        assert status == "PASS"
        assert severity == "NONE"
