"""
EchoLocate — Blind Black-Box Localizer
=======================================
A strictly isolated spatial-likelihood localization interface that has zero
access to ground-truth source coordinates, simulation metadata, or RIR parameters.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.localization.srp_phat import (
    SRPGrid,
    SpatialPeak,
    compute_phat_cross_correlations,
    evaluate_srp_map,
    extract_spatial_peaks,
    refine_subgrid_peak,
)

logger = logging.getLogger(__name__)


@dataclass
class BlindLocalizationOutput:
    """Strict output container containing only algorithmically derived estimates."""
    pred_x: float
    pred_y: float
    raw_x: float
    raw_y: float
    primary_peak_power: float
    secondary_peak_power: float
    peak_ratio: float
    peak_separation_m: float
    top_k_peaks: list[SpatialPeak]
    runtime_ms: float
    assumed_c: float
    grid_resolution_m: float


class BlindSRPLocalizer:
    """
    Independent SRP-PHAT Localizer with cached grid support and strict black-box semantics.
    """

    def __init__(
        self,
        assumed_room_dimensions: list[float],
        assumed_mic_positions_2d: np.ndarray,
        grid_resolution_m: float = 0.05,
        boundary_margin_m: float = 0.10,
        assumed_speed_of_sound: float = 343.0,
        subgrid_refinement: bool = True,
        min_peak_separation_m: float = 0.50,
        top_k_peaks: int = 5,
    ):
        self.assumed_room_dims = list(assumed_room_dimensions)
        self.assumed_mics_2d = np.asarray(assumed_mic_positions_2d, dtype=np.float64)
        self.grid_res_m = float(grid_resolution_m)
        self.margin_m = float(boundary_margin_m)
        self.assumed_c = float(assumed_speed_of_sound)
        self.subgrid_refine = bool(subgrid_refinement)
        self.min_peak_sep = float(min_peak_separation_m)
        self.top_k = int(top_k_peaks)

        # Precompute candidate spatial grid and theoretical delays
        self.grid = SRPGrid(
            room_dimensions=self.assumed_room_dims,
            mic_positions_2d=self.assumed_mics_2d,
            grid_resolution_m=self.grid_res_m,
            boundary_margin_m=self.margin_m,
            speed_of_sound=self.assumed_c,
        )

    def localize(
        self,
        mic_signals: np.ndarray,
        fs: int = 16000,
        interpolation_method: Literal["linear", "nearest"] = "linear",
    ) -> BlindLocalizationOutput:
        """
        Estimate 2D source coordinates blindly from raw multi-channel audio signals.
        """
        t_start = time.perf_counter()

        # 1. Gated PHAT cross-correlations
        corr_dict, tau_axis = compute_phat_cross_correlations(mic_signals, fs=fs)

        # 2. Steered Response Power
        srp_flat, srp_2d = evaluate_srp_map(
            grid=self.grid,
            corr_dict=corr_dict,
            tau_axis=tau_axis,
            interpolation_method=interpolation_method,
        )

        # 3. Peak Extraction
        top_1, top_2, ratio, sep, peaks_list, _ = extract_spatial_peaks(
            srp_flat=srp_flat,
            srp_2d=srp_2d,
            grid=self.grid,
            min_peak_separation_m=self.min_peak_sep,
            top_k=self.top_k,
            true_pos_2d=None,  # Intentionally None (no ground truth passed)
        )

        raw_pos = np.array([top_1.grid_x, top_1.grid_y], dtype=np.float64)

        # 4. Optional Sub-grid Refinement
        if self.subgrid_refine:
            peak_y_idx = int(top_1.peak_idx // self.grid.n_x)
            peak_x_idx = int(top_1.peak_idx % self.grid.n_x)
            refined_pos = refine_subgrid_peak(srp_2d, self.grid, peak_y_idx, peak_x_idx)
        else:
            refined_pos = raw_pos.copy()

        t_end = time.perf_counter()
        runtime_ms = (t_end - t_start) * 1000.0

        return BlindLocalizationOutput(
            pred_x=float(refined_pos[0]),
            pred_y=float(refined_pos[1]),
            raw_x=float(raw_pos[0]),
            raw_y=float(raw_pos[1]),
            primary_peak_power=float(top_1.power),
            secondary_peak_power=float(top_2.power),
            peak_ratio=float(ratio),
            peak_separation_m=float(sep),
            top_k_peaks=peaks_list,
            runtime_ms=runtime_ms,
            assumed_c=self.assumed_c,
            grid_resolution_m=self.grid_res_m,
        )
