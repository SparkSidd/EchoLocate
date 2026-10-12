"""
EchoLocate — Microphone Geometry Validator
===========================================
Validates physical microphone coordinates, array aperture dimensions,
collinearity, and spatial singularity constraints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GeometryValidationResult:
    """Validation report for an acoustic array geometry."""
    is_valid: bool
    num_microphones: int
    aperture_m: float
    is_collinear: bool
    has_duplicates: bool
    status: str
    error_message: str = ""


def validate_microphone_geometry(
    mic_positions_2d: np.ndarray | list[list[float]],
    min_mics: int = 3,
    min_spacing_m: float = 0.02,
) -> GeometryValidationResult:
    """Validate physical microphone array placement against geometric singularity rules."""
    mics = np.asarray(mic_positions_2d, dtype=np.float64)
    n_mics = len(mics)

    if n_mics < min_mics:
        return GeometryValidationResult(
            is_valid=False,
            num_microphones=n_mics,
            aperture_m=0.0,
            is_collinear=False,
            has_duplicates=False,
            status="INVALID_COUNT",
            error_message=f"At least {min_mics} microphones required for 2D localization. Found {n_mics}.",
        )

    # Check for duplicate / overlapping microphones
    dists = []
    has_dups = False
    for i in range(n_mics):
        for j in range(i + 1, n_mics):
            d = float(np.linalg.norm(mics[i] - mics[j]))
            dists.append(d)
            if d < min_spacing_m:
                has_dups = True

    if has_dups:
        return GeometryValidationResult(
            is_valid=False,
            num_microphones=n_mics,
            aperture_m=float(np.max(dists)) if dists else 0.0,
            is_collinear=False,
            has_duplicates=True,
            status="DUPLICATE_MICROPHONES",
            error_message=f"Microphones placed closer than {min_spacing_m*1000:.1f} mm. Possible duplicate.",
        )

    # Check collinearity using rank of centered coordinates
    centered = mics - np.mean(mics, axis=0)
    _, s, _ = np.linalg.svd(centered)
    is_collinear = bool(s[1] < 1e-4) if len(s) > 1 else True

    aperture = float(np.max(dists)) if dists else 0.0

    if is_collinear and n_mics >= 3:
        return GeometryValidationResult(
            is_valid=False,
            num_microphones=n_mics,
            aperture_m=aperture,
            is_collinear=True,
            has_duplicates=False,
            status="COLLINEAR_SINGULARITY",
            error_message="Microphones are strictly collinear. Creates 2D front-back spatial ambiguity.",
        )

    return GeometryValidationResult(
        is_valid=True,
        num_microphones=n_mics,
        aperture_m=aperture,
        is_collinear=False,
        has_duplicates=False,
        status="VALID_PLANAR_ARRAY",
    )
