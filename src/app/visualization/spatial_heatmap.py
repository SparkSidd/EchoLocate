"""
EchoLocate — 2D SRP Spatial Likelihood Heatmap Visualizer
=========================================================
Generates 2D spatial acoustic likelihood surface grids, contour overlays,
primary and secondary peak coordinates, and spatial entropy metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.app.visualization.provenance import VisualProvenanceMetadata
from src.localization.blind_localizer import BlindSRPLocalizer
from src.localization.srp_phat import compute_phat_cross_correlations, evaluate_srp_map

logger = logging.getLogger(__name__)


@dataclass
class SpatialHeatmapPayload:
    """2D spatial acoustic likelihood display payload."""
    frame_id: int
    room_dimensions_m: list[float]
    grid_extent_xy: list[float]
    grid_resolution_m: float
    heatmap_matrix: list[list[float]]
    primary_peak_xy: tuple[float, float]
    primary_peak_power: float
    secondary_peak_xy: tuple[float, float] | None
    secondary_peak_power: float
    peak_contrast_ratio: float
    spatial_entropy: float
    contour_levels: list[float]
    provenance: VisualProvenanceMetadata


def compute_spatial_likelihood_heatmap(
    localizer: BlindSRPLocalizer,
    frame_audio: np.ndarray,
    fs: int = 16000,
    frame_id: int = 0,
    provenance: VisualProvenanceMetadata | None = None,
) -> SpatialHeatmapPayload:
    """Compute normalized 2D SRP likelihood surface grid and spatial metrics."""
    prov = provenance or VisualProvenanceMetadata.live(frame_id)

    # 1. Compute Pairwise GCC-PHAT
    corr_dict, tau_axis = compute_phat_cross_correlations(frame_audio, fs=fs)

    # 2. Accumulate on 2D Grid
    srp_flat, srp_2d = evaluate_srp_map(
        grid=localizer.grid,
        corr_dict=corr_dict,
        tau_axis=tau_axis,
        interpolation_method="linear",
    )

    # Normalize between 0 and 1
    min_val, max_val = float(np.min(srp_2d)), float(np.max(srp_2d))
    norm_grid = (srp_2d - min_val) / max(1e-6, max_val - min_val)

    # Peak detection
    max_idx = int(np.argmax(srp_flat))
    p1_x = float(localizer.grid.points_flat[max_idx, 0])
    p1_y = float(localizer.grid.points_flat[max_idx, 1])
    p1_pow = float(srp_flat[max_idx])

    # Secondary peak
    sorted_indices = np.argsort(srp_flat)[::-1]
    p2_xy = None
    p2_pow = 0.0
    for idx in sorted_indices[1:]:
        gx = float(localizer.grid.points_flat[idx, 0])
        gy = float(localizer.grid.points_flat[idx, 1])
        if np.hypot(gx - p1_x, gy - p1_y) >= 0.50:
            p2_xy = (gx, gy)
            p2_pow = float(srp_flat[idx])
            break

    contrast = float(p1_pow / max(1e-6, p2_pow)) if p2_pow > 0 else 2.5

    # Spatial Entropy
    prob = norm_grid.flatten() / max(1e-12, np.sum(norm_grid))
    entropy = -float(np.sum(prob * np.log(prob + 1e-12)))

    # Contours
    levels = [0.3, 0.5, 0.7, 0.85, 0.95]

    return SpatialHeatmapPayload(
        frame_id=frame_id,
        room_dimensions_m=localizer.assumed_room_dims,
        grid_extent_xy=[0.0, localizer.assumed_room_dims[0], 0.0, localizer.assumed_room_dims[1]],
        grid_resolution_m=localizer.grid_res_m,
        heatmap_matrix=norm_grid.tolist(),
        primary_peak_xy=(p1_x, p1_y),
        primary_peak_power=p1_pow,
        secondary_peak_xy=p2_xy,
        secondary_peak_power=p2_pow,
        peak_contrast_ratio=contrast,
        spatial_entropy=entropy,
        contour_levels=levels,
        provenance=prov,
    )
