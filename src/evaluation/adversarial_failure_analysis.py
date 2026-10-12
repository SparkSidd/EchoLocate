"""
EchoLocate — Adversarial Failure Analysis & Attribution Taxonomy
================================================================
Implements rigorous multi-class failure categorization and physical root-cause
attribution for Experiment 4.75 stress evaluations.
"""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


def classify_adversarial_error(
    error_m: float | None,
    pred_xy: list[float] | np.ndarray | None,
) -> str:
    """Classify localization outcome into general performance tier."""
    if error_m is None or np.isnan(error_m) or np.isinf(error_m):
        return "nan_inf"

    if pred_xy is None or len(pred_xy) < 2 or np.isnan(pred_xy[0]) or np.isnan(pred_xy[1]):
        return "solver_failed"

    if error_m <= 0.10:
        return "excellent"
    elif error_m <= 0.25:
        return "accurate"
    elif error_m <= 1.00:
        return "degraded"
    elif error_m <= 2.00:
        return "large_error"
    else:
        return "catastrophic_error"


def attribute_neural_failure_mechanism(
    error_m: float,
    peak_contrast: float,
    is_speech: bool,
    harmonic_ppr: float,
    mic_perturbation_mm: float,
    delta_c_mps: float,
    sync_offset_us: float,
    rt60_measured: float,
) -> str:
    """Attribute primary physical/neural failure mechanism if error > 0.50m."""
    if error_m <= 0.50:
        return "nominal_success"

    if is_speech and harmonic_ppr > 0.85:
        return "harmonic_pseudo_peak"

    if peak_contrast < 1.25:
        return "low_peak_contrast"

    if mic_perturbation_mm >= 10.0:
        return "geometry_mismatch"

    if abs(delta_c_mps) >= 5.0:
        return "speed_mismatch"

    if sync_offset_us >= 25.0:
        return "sync_failure"

    if rt60_measured >= 0.40:
        return "diffuse_reverberation"

    return "wrong_spatial_mode"
