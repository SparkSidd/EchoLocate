"""
EchoLocate — Tests: Experiment 02 Modules
==========================================
Unit tests for:
  - Exact SNR noise generation
  - Geometry metrics (Triangle area, Convexity, TDOA Jacobian, Condition Number, GDOP)
  - Robustness aggregation metrics and confidence intervals
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.geometry_metrics import (
    compute_triangle_area,
    is_point_inside_triangle,
    compute_tdoa_jacobian_and_gdop,
    evaluate_source_geometry_metrics,
)
from src.evaluation.robustness_metrics import compute_condition_summary
from src.experiments.environment_robustness import generate_exact_snr_noise


class TestExactSnrNoiseGeneration:
    """Test AWGN injection scaling according to exact SNR formula."""

    def test_snr_scaling_accuracy(self):
        # Create pure tone test signal
        fs = 16000
        t = np.linspace(0, 1, fs)
        sig = np.sin(2 * np.pi * 440 * t).reshape(1, -1)
        
        target_snrs = [60.0, 40.0, 20.0, 10.0, 0.0]
        for snr_tgt in target_snrs:
            noisy, snr_achieved = generate_exact_snr_noise(sig, target_snr_db=snr_tgt, seed=42)
            # Achieved SNR should match target within 0.5 dB for 16000 samples
            assert abs(snr_achieved - snr_tgt) < 0.5

    def test_reproducibility_with_seed(self):
        sig = np.ones((3, 1000))
        n1, s1 = generate_exact_snr_noise(sig, 20.0, seed=123)
        n2, s2 = generate_exact_snr_noise(sig, 20.0, seed=123)
        n3, s3 = generate_exact_snr_noise(sig, 20.0, seed=999)
        np.testing.assert_allclose(n1, n2)
        assert not np.allclose(n1, n3)


class TestGeometryMetrics:
    """Test triangle area, convexity, and TDOA Jacobian conditioning."""

    def test_triangle_area(self):
        # Right triangle with base 4, height 3 -> Area = 6.0
        mics = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]])
        area = compute_triangle_area(mics)
        assert abs(area - 6.0) < 1e-9

    def test_point_inside_triangle(self):
        tri = np.array([[1.0, 1.0], [9.0, 1.0], [5.0, 5.0]])
        # Inside
        assert is_point_inside_triangle(np.array([5.0, 2.5]), tri) is True
        assert is_point_inside_triangle(np.array([5.0, 4.0]), tri) is True
        # Outside
        assert is_point_inside_triangle(np.array([5.0, 0.5]), tri) is False
        assert is_point_inside_triangle(np.array([0.5, 3.0]), tri) is False
        assert is_point_inside_triangle(np.array([5.0, 5.5]), tri) is False

    def test_tdoa_jacobian_and_gdop_conditioning(self):
        mics = np.array([[1.0, 1.5], [9.0, 1.5], [5.0, 4.5]])
        source_center = np.array([5.0, 3.0])
        cond_num, gdop, H = compute_tdoa_jacobian_and_gdop(source_center, mics, speed_of_sound=343.0)
        
        assert np.isfinite(cond_num)
        assert np.isfinite(gdop)
        assert cond_num > 1.0
        assert gdop > 0.0
        assert H.shape == (3, 2)


class TestRobustnessAggregationMetrics:
    """Test trial aggregation and statistical summarization."""

    def test_compute_condition_summary(self):
        records = []
        for trial in range(5):
            for s in range(15):
                records.append({
                    "trial_idx": trial,
                    "source_id": f"S{s+1:02d}",
                    "true_x": 5.0,
                    "true_y": 3.0,
                    "pred_x": 5.02 + trial * 0.01,
                    "pred_y": 2.99 - trial * 0.01,
                    "localization_error": 0.05 + trial * 0.01,
                    "achieved_snr_db": 30.0,
                    "tdoa_01_estimated_s": 0.0001,
                    "tdoa_01_theoretical_s": 0.0000,
                    "tdoa_02_estimated_s": 0.0081,
                    "tdoa_02_theoretical_s": 0.0080,
                    "tdoa_12_estimated_s": 0.0081,
                    "tdoa_12_theoretical_s": 0.0080,
                })
        df_trials = pd.DataFrame(records)
        
        summary = compute_condition_summary(
            df_trials=df_trials,
            rt60_target=0.2,
            snr_target=30.0,
            rt60_measured_stats={"mean": 0.25, "std": 0.01},
        )
        
        assert summary.num_trials == 5
        assert summary.num_sources == 15
        assert summary.total_evaluations == 75
        assert abs(summary.mean_localization_error_m - 0.07) < 1e-4
        assert summary.pct_within_0_25m == 100.0
        assert summary.pct_within_1_0m == 100.0
        assert summary.rt60_measured_mean_s == 0.25
