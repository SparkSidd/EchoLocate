"""
Unit tests for Experiment 3 evaluation and comparison metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.comparison_metrics import (
    compute_paired_comparison_summary,
    compute_per_source_comparison,
)


class TestComparativeMetrics:
    def test_paired_summary_schema_and_deltas(self):
        # Synthetic evaluation DataFrame
        data = {
            "condition_id": ["CondA"] * 10 + ["CondB"] * 10,
            "target_rt60_s": [0.20] * 10 + [0.30] * 10,
            "measured_mean_rt60_s": [0.23] * 10 + [0.45] * 10,
            "snr_target_db": [40.0] * 20,
            "source_id": [f"S{i:02d}" for i in range(1, 11)] * 2,
            "true_x": [5.0] * 20,
            "true_y": [3.0] * 20,
            "nls_localization_error": [0.50] * 10 + [1.50] * 10,
            "srp_raw_error": [0.30] * 10 + [0.80] * 10,
            "srp_refined_error": [0.20] * 10 + [0.60] * 10,
            "srp_peak_ratio": [1.8] * 10 + [1.1] * 10,
            "srp_peak_separation_m": [1.5] * 20,
            "srp_top_k_proximity": [True] * 20,
            "nls_runtime_ms": [15.0] * 20,
            "srp_runtime_ms": [2.5] * 20,
        }
        df_evals = pd.DataFrame(data)

        df_summary = compute_paired_comparison_summary(df_evals)

        assert len(df_summary) == 2
        # Check CondA: NLS=0.50, SRP=0.20 -> delta_mean_error = -0.30
        row_a = df_summary[df_summary["condition_id"] == "CondA"].iloc[0]
        assert np.isclose(row_a["delta_mean_error_m"], -0.30)
        assert np.isclose(row_a["srp_win_rate_pct"], 100.0)

        # Check Per Source Summary
        df_ps = compute_per_source_comparison(df_evals)
        assert len(df_ps) == 10
        assert "delta_mean_error_m" in df_ps.columns
