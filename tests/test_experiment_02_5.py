"""
EchoLocate — Tests: Experiment 2.5 Modules
===========================================
Unit tests for:
  - Schroeder RIR RT60 measurement and fallback extrapolation
  - Threshold analysis & performance slope computation
  - Failure classification & per-source sensitivity analysis
  - Failure mechanism decoupling
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.failure_analysis import (
    FailureClass,
    classify_evaluation_record,
    compute_per_source_sensitivity,
    analyze_failure_mechanisms,
)
from src.evaluation.threshold_analysis import (
    compute_performance_slopes,
    analyze_reverberation_thresholds,
)
from src.experiments.transition_zone import measure_single_rir_rt60


class TestRirRt60Measurement:
    """Test individual RIR RT60 estimation and extrapolation method labeling."""

    def test_synthetic_exponential_decay(self):
        fs = 16000
        t = np.linspace(0, 1.0, fs)
        # Synthetic decay with RT60 approx 0.3s (decay rate = -60 / (0.3 * 20 * log10(e)) = -6.908 / 0.3 = -23.03)
        decay = np.exp(-23.03 * t)
        noise = (np.random.rand(fs) - 0.5) * 1e-4
        rir = decay + noise
        
        rt60, method = measure_single_rir_rt60(rir, fs)
        assert rt60 > 0.1
        assert rt60 < 0.6
        assert method in ["RT60_direct", "RT30_extrapolated", "RT20_extrapolated"]

    def test_silent_rir_fails_gracefully(self):
        rir = np.zeros(16000)
        rt60, method = measure_single_rir_rt60(rir, 16000)
        assert rt60 == 0.0
        assert method == "measurement_failed"


class TestFailureClassification:
    """Test fine-grained failure taxonomy."""

    def test_classification_categories(self):
        # Accurate
        assert classify_evaluation_record(5.0, 3.0, 0.1, True) == FailureClass.VALID_ACCURATE
        # Moderate
        assert classify_evaluation_record(5.0, 3.0, 0.8, True) == FailureClass.VALID_MODERATE
        # Degraded
        assert classify_evaluation_record(5.0, 3.0, 1.5, True) == FailureClass.VALID_DEGRADED
        # Catastrophic
        assert classify_evaluation_record(5.0, 3.0, 2.5, True) == FailureClass.CATASTROPHIC_ERROR
        # Solver failure
        assert classify_evaluation_record(5.0, 3.0, 0.5, False) == FailureClass.SOLVER_FAILED
        # NaN / Inf
        assert classify_evaluation_record(np.nan, 3.0, 0.5, True) == FailureClass.NAN_INF
        # Out of bounds
        assert classify_evaluation_record(15.0, 3.0, 5.0, True) == FailureClass.OUT_OF_BOUNDS


class TestThresholdAnalysis:
    """Test performance slope and transition region segmentation."""

    def test_threshold_segmentation(self):
        df_summary = pd.DataFrame([
            {"rt60_target_s": 0.20, "rt60_measured_mean_s": 0.24, "mean_localization_error_m": 0.20, "pct_within_1_0m": 100.0, "pct_catastrophic_failure": 0.0},
            {"rt60_target_s": 0.25, "rt60_measured_mean_s": 0.35, "mean_localization_error_m": 0.50, "pct_within_1_0m": 85.0, "pct_catastrophic_failure": 5.0},
            {"rt60_target_s": 0.30, "rt60_measured_mean_s": 0.46, "mean_localization_error_m": 1.45, "pct_within_1_0m": 25.0, "pct_catastrophic_failure": 45.0},
            {"rt60_target_s": 0.35, "rt60_measured_mean_s": 0.58, "mean_localization_error_m": 1.80, "pct_within_1_0m": 15.0, "pct_catastrophic_failure": 50.0},
        ])
        
        res, df_slopes = analyze_reverberation_thresholds(df_summary)
        
        assert res.conservative_operating_upper_rt60_s == 0.35
        assert res.failure_region_lower_rt60_s == 0.46
        assert res.max_error_slope_value > 0.0
        assert len(df_slopes) == 4


class TestPerSourceSensitivity:
    """Test source sensitivity aggregation across RT60 regimes."""

    def test_per_source_summary_schema(self):
        evals = []
        for s in ["S01", "S02"]:
            for t in [0.20, 0.25, 0.30, 0.35]:
                evals.append({
                    "source_id": s,
                    "true_x": 2.0 if s == "S01" else 5.0,
                    "true_y": 1.0 if s == "S01" else 3.0,
                    "rt60_target_s": t,
                    "localization_error": 0.1 if s == "S01" else 0.5,
                    "mean_tdoa_error_ms": 0.05,
                })
        df_evals = pd.DataFrame(evals)
        df_sum = pd.DataFrame([
            {"rt60_target_s": 0.20, "rt60_measured_mean_s": 0.24},
            {"rt60_target_s": 0.25, "rt60_measured_mean_s": 0.35},
            {"rt60_target_s": 0.30, "rt60_measured_mean_s": 0.46},
            {"rt60_target_s": 0.35, "rt60_measured_mean_s": 0.58},
        ])
        
        df_src = compute_per_source_sensitivity(df_evals, df_sum)
        assert len(df_src) == 2
        assert "mean_error_low_rt60_m" in df_src.columns
        assert "relative_sensitivity_index" in df_src.columns
