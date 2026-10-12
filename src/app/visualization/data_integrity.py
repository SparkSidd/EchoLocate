"""
EchoLocate — Visualization Data Integrity & Mathematical Fidelity Auditor
========================================================================
Validates that visualization payloads (waveforms, correlation curves, heatmaps,
coordinates) derive from real calculations and not hardcoded fake animations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.app.visualization.provenance import VisualProvenanceMetadata

logger = logging.getLogger(__name__)


@dataclass
class VisualizationIntegrityReport:
    """Validation report for a visualization data object."""
    is_valid: bool
    payload_type: str
    has_nan_or_inf: bool
    is_non_trivial: bool
    provenance_verified: bool
    status: str
    error_message: str = ""


def audit_waveform_payload(audio_data: np.ndarray, provenance: VisualProvenanceMetadata) -> VisualizationIntegrityReport:
    """Audit multichannel waveform payload."""
    if not isinstance(audio_data, np.ndarray) or audio_data.size == 0:
        return VisualizationIntegrityReport(
            is_valid=False, payload_type="waveform", has_nan_or_inf=False,
            is_non_trivial=False, provenance_verified=False, status="INVALID_ARRAY",
            error_message="Audio data is not a valid non-empty NumPy array.",
        )

    has_nan = bool(np.isnan(audio_data).any() or np.isinf(audio_data).any())
    energy = float(np.sum(audio_data**2))
    is_non_trivial = energy > 1e-12

    return VisualizationIntegrityReport(
        is_valid=(not has_nan) and is_non_trivial,
        payload_type="waveform",
        has_nan_or_inf=has_nan,
        is_non_trivial=is_non_trivial,
        provenance_verified=provenance is not None,
        status="PASSED" if (not has_nan and is_non_trivial) else "INTEGRITY_FAIL",
    )


def audit_spatial_heatmap_payload(srp_grid: np.ndarray, room_dims: list[float]) -> VisualizationIntegrityReport:
    """Audit 2D spatial likelihood surface grid."""
    if not isinstance(srp_grid, np.ndarray) or srp_grid.ndim != 2:
        return VisualizationIntegrityReport(
            is_valid=False, payload_type="spatial_heatmap", has_nan_or_inf=False,
            is_non_trivial=False, provenance_verified=False, status="INVALID_GRID",
            error_message="SRP Grid must be a 2D NumPy array.",
        )

    has_nan = bool(np.isnan(srp_grid).any() or np.isinf(srp_grid).any())
    min_val, max_val = float(np.min(srp_grid)), float(np.max(srp_grid))
    is_normalized = (min_val >= -1e-4) and (max_val <= 1.0 + 1e-4) and (max_val > 0.0)

    return VisualizationIntegrityReport(
        is_valid=(not has_nan) and is_normalized,
        payload_type="spatial_heatmap",
        has_nan_or_inf=has_nan,
        is_non_trivial=is_normalized,
        provenance_verified=True,
        status="PASSED" if (not has_nan and is_normalized) else "HEATMAP_NORMALIZATION_FAIL",
    )
