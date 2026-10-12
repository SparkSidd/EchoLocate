"""
EchoLocate — Hybrid Gating Efficiency & Oracle Headroom Analysis
================================================================
Evaluates Gate A (Fixed SRP), Gate B (Threshold), Gate C (Learned Meta-Gate),
and calculates Oracle headroom, gate efficiency, and decision distributions.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_hybrid_gate_metrics(
    srp_errors_m: list[float] | np.ndarray,
    neural_errors_m: list[float] | np.ndarray,
    hybrid_errors_m: list[float] | np.ndarray,
    hybrid_decisions: list[str],
) -> dict[str, Any]:
    """
    Compute Oracle Gain, Hybrid Gate Gain, Gate Efficiency, and Decision Percentages.
    """
    s_arr = np.asarray(srp_errors_m)
    n_arr = np.asarray(neural_errors_m)
    h_arr = np.asarray(hybrid_errors_m)

    oracle_arr = np.minimum(s_arr, n_arr)

    mean_srp = float(np.mean(s_arr))
    mean_neural = float(np.mean(n_arr))
    mean_hybrid = float(np.mean(h_arr))
    mean_oracle = float(np.mean(oracle_arr))

    best_fixed = min(mean_srp, mean_neural)
    oracle_gain = best_fixed - mean_oracle
    actual_gain = best_fixed - mean_hybrid

    gate_efficiency = float((actual_gain / max(1e-4, oracle_gain)) * 100.0)

    n_total = max(1, len(hybrid_decisions))
    pct_srp = float(np.mean([1 if d == "SRP" else 0 for d in hybrid_decisions]) * 100.0)
    pct_neural = float(np.mean([1 if d == "NEURAL" else 0 for d in hybrid_decisions]) * 100.0)
    pct_abstain = float(np.mean([1 if d == "ABSTAIN" else 0 for d in hybrid_decisions]) * 100.0)

    return {
        "mean_srp_error_m": mean_srp,
        "mean_neural_error_m": mean_neural,
        "mean_hybrid_error_m": mean_hybrid,
        "mean_oracle_error_m": mean_oracle,
        "oracle_achievable_gain_m": oracle_gain,
        "actual_hybrid_gain_m": actual_gain,
        "gate_efficiency_pct": gate_efficiency,
        "pct_srp_selected": pct_srp,
        "pct_neural_selected": pct_neural,
        "pct_abstained": pct_abstain,
    }
