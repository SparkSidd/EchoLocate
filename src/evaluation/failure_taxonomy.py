"""
EchoLocate — Real-World 10-Class Failure Taxonomy Module
========================================================
Categorizes catastrophic localization failures into 10 explicit physical
and algorithmic failure mechanisms.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def classify_real_world_failure_category(
    error_m: float,
    is_speech: bool,
    harmonic_ppr: float,
    spatial_contrast: float,
    mic_perturbation_mm: float,
    sync_offset_us: float,
    snr_db: float,
    rt60_s: float,
    confidence: float,
) -> str:
    """Classify physical root cause of real-world localization error."""
    if error_m <= 0.50:
        return "F0_Nominal_Success"

    if is_speech and harmonic_ppr > 0.80:
        return "F2_Speech_Harmonic_Ambiguity"

    if mic_perturbation_mm >= 5.0:
        return "F3_Microphone_Geometry_Mismatch"

    if sync_offset_us >= 25.0:
        return "F4_Synchronization_Mismatch"

    if snr_db < 10.0:
        return "F5_Low_SNR"

    if rt60_s >= 0.50:
        return "F1_Reflection_Peak_Competition"

    if spatial_contrast < 1.25:
        return "F9_SRP_Spatial_Ambiguity"

    if confidence > 0.85 and error_m > 1.50:
        return "F8_Neural_Overconfidence"

    return "F10_Unclassified_Physical_Mismatch"
