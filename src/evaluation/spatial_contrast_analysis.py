"""
EchoLocate — Experiment 4.5 Spatial Contrast & Likelihood Surface Analysis
=========================================================================
Extracts spatial peak contrast (SPC), spatial entropy, competing false maxima,
and spatial curvature from 2D Steered Response Power (SRP) maps.
"""

from __future__ import annotations

import logging
import numpy as np
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


def analyze_spatial_likelihood_surface(
    spatial_map_2d: np.ndarray,  # (Ny, Nx) or (N_grid,)
    grid_points_2d: np.ndarray,  # (N_grid, 2)
    true_xy: np.ndarray,         # (2,)
    grid_resolution_m: float = 0.05,
) -> dict[str, float]:
    """
    Compute fine-grained metrics on 2D spatial response surface S(x, y).
    """
    flat_s = spatial_map_2d.ravel()
    min_val = np.min(flat_s)
    # Non-negative spatial likelihood
    norm_s = flat_s - min_val
    s_sum = np.sum(norm_s)
    if s_sum > 0:
        spatial_probs = norm_s / s_sum
    else:
        spatial_probs = np.ones_like(norm_s) / len(norm_s)

    # 1. Spatial Entropy
    safe_p = np.clip(spatial_probs, 1e-12, 1.0)
    spatial_entropy = -float(np.sum(safe_p * np.log(safe_p)))

    # 2. Peak Sorting & Spatial Peak Contrast (SPC)
    sorted_indices = np.argsort(flat_s)[::-1]
    top1_val = float(flat_s[sorted_indices[0]])
    top2_val = float(flat_s[sorted_indices[1]]) if len(sorted_indices) > 1 else top1_val
    spc = top1_val / max(1e-12, top2_val)

    # 3. Distance from true position to estimated global max
    pred_xy = grid_points_2d[sorted_indices[0]]
    loc_error_m = float(np.linalg.norm(pred_xy - true_xy))

    # 4. True spatial rank
    dists_to_true = np.linalg.norm(grid_points_2d - true_xy, axis=-1)
    true_node_idx = int(np.argmin(dists_to_true))
    true_val = flat_s[true_node_idx]
    true_spatial_rank = int(np.sum(flat_s > true_val)) + 1

    # 5. Spatial PSR (Peak within 0.25m vs outside 0.25m)
    in_mask = dists_to_true <= 0.25
    out_mask = ~in_mask
    in_peak = float(np.max(flat_s[in_mask])) if np.any(in_mask) else 1e-12
    out_peak = float(np.max(flat_s[out_mask])) if np.any(out_mask) else 1e-12
    spatial_psr = in_peak / max(1e-12, out_peak)

    # 6. Competing spatial peaks distance
    competing_dists = []
    # Check top 5 spatial peaks
    for rank_i in range(1, min(6, len(sorted_indices))):
        c_idx = sorted_indices[rank_i]
        c_xy = grid_points_2d[c_idx]
        if np.linalg.norm(c_xy - pred_xy) > 0.20:
            competing_dists.append(float(np.linalg.norm(c_xy - true_xy)))

    dist_nearest_false_max = competing_dists[0] if competing_dists else 5.0

    return {
        "spatial_entropy": spatial_entropy,
        "spatial_peak_contrast": spc,
        "spatial_psr": spatial_psr,
        "true_spatial_rank": float(true_spatial_rank),
        "dist_nearest_false_max_m": dist_nearest_false_max,
        "num_competing_maxima": float(len(competing_dists)),
        "localization_error_m": loc_error_m,
    }
