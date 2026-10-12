"""
EchoLocate — Neural-Enhanced SRP Spatial Likelihood Localizer (Method M4)
========================================================================
Combines DeepGCC enhanced continuous correlation representations L_ij(tau) with
geometry-aware Steered Response Power spatial likelihood maps.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from src.localization.srp_phat import (
    SRPGrid,
    SpatialPeak,
    evaluate_srp_map,
    extract_spatial_peaks,
    refine_subgrid_peak,
)
from src.models.inference import BlindNeuralGCCPredictor

logger = logging.getLogger(__name__)


@dataclass
class NeuralSRPResult:
    """Output container for Neural SRP localization."""
    pred_x: float
    pred_y: float
    raw_x: float
    raw_y: float
    primary_peak_power: float
    secondary_peak_power: float
    spatial_peak_contrast: float    # S_1 / S_2
    spatial_peak_separation_m: float
    spatial_entropy: float
    top_k_peaks: list[SpatialPeak]
    runtime_ms: float
    raw_gcc_dict: dict[tuple[int, int], np.ndarray]
    enhanced_gcc_dict: dict[tuple[int, int], np.ndarray]
    tau_axis: np.ndarray
    srp_2d: np.ndarray


class NeuralSRPLocalizer:
    """
    Method M4: Audio -> DeepGCC Enhanced Correlations -> Geometry-Aware SRP Map -> 2D Position.
    """

    def __init__(
        self,
        assumed_room_dimensions: list[float],
        assumed_mic_positions_2d: np.ndarray,
        predictor: BlindNeuralGCCPredictor,
        grid_resolution_m: float = 0.05,
        boundary_margin_m: float = 0.10,
        assumed_speed_of_sound: float = 343.0,
        subgrid_refinement: bool = True,
        min_peak_separation_m: float = 0.50,
        top_k_peaks: int = 5,
    ):
        self.assumed_room_dims = list(assumed_room_dimensions)
        self.assumed_mics_2d = np.asarray(assumed_mic_positions_2d, dtype=np.float64)
        self.predictor = predictor
        self.grid_res = float(grid_resolution_m)
        self.margin = float(boundary_margin_m)
        self.assumed_c = float(assumed_speed_of_sound)
        self.subgrid_refine = bool(subgrid_refinement)
        self.min_peak_sep = float(min_peak_separation_m)
        self.top_k = int(top_k_peaks)

        # Precompute candidate spatial grid and analytical delay tables
        self.grid = SRPGrid(
            room_dimensions=self.assumed_room_dims,
            mic_positions_2d=self.assumed_mics_2d,
            grid_resolution_m=self.grid_res,
            boundary_margin_m=self.margin,
            speed_of_sound=self.assumed_c,
        )

    def localize(
        self,
        mic_signals: np.ndarray,
        fs: int = 16000,
    ) -> NeuralSRPResult:
        """
        Blindly estimate 2D source position via Neural Enhanced SRP.
        """
        t_start = time.perf_counter()

        # 1. Neural Correlation Enhancement
        enhanced_dict, raw_gcc, tau_axis = self.predictor.enhance_correlations(mic_signals, fs=fs)

        # 2. Geometry-Aware SRP Map Evaluation
        srp_flat, srp_2d = evaluate_srp_map(
            grid=self.grid,
            corr_dict=enhanced_dict,
            tau_axis=tau_axis,
            interpolation_method="linear",
        )

        # 3. Peak Extraction
        top_1, top_2, ratio, sep, peaks_list, _ = extract_spatial_peaks(
            srp_flat=srp_flat,
            srp_2d=srp_2d,
            grid=self.grid,
            min_peak_separation_m=self.min_peak_sep,
            top_k=self.top_k,
            true_pos_2d=None,  # Strict blind evaluation
        )

        raw_pos = np.array([top_1.grid_x, top_1.grid_y], dtype=np.float64)

        # 4. Sub-grid Quadratic Refinement
        if self.subgrid_refine:
            peak_y_idx = int(top_1.peak_idx // self.grid.n_x)
            peak_x_idx = int(top_1.peak_idx % self.grid.n_x)
            refined_pos = refine_subgrid_peak(srp_2d, self.grid, peak_y_idx, peak_x_idx)
        else:
            refined_pos = raw_pos.copy()

        # 5. Spatial Map Entropy
        norm_srp = srp_flat / max(1e-12, np.sum(srp_flat))
        spatial_entropy = -float(np.sum(norm_srp * np.log(norm_srp + 1e-12)))

        t_end = time.perf_counter()
        runtime_ms = (t_end - t_start) * 1000.0

        raw_gcc_dict = {}
        pairs = []
        for i in range(len(self.assumed_mics_2d)):
            for j in range(i + 1, len(self.assumed_mics_2d)):
                pairs.append((i, j))
        for p_idx, pair in enumerate(pairs):
            raw_gcc_dict[pair] = raw_gcc[p_idx]

        return NeuralSRPResult(
            pred_x=float(refined_pos[0]),
            pred_y=float(refined_pos[1]),
            raw_x=float(raw_pos[0]),
            raw_y=float(raw_pos[1]),
            primary_peak_power=float(top_1.power),
            secondary_peak_power=float(top_2.power),
            spatial_peak_contrast=float(ratio),
            spatial_peak_separation_m=float(sep),
            spatial_entropy=spatial_entropy,
            top_k_peaks=peaks_list,
            runtime_ms=runtime_ms,
            raw_gcc_dict=raw_gcc_dict,
            enhanced_gcc_dict=enhanced_dict,
            tau_axis=tau_axis,
            srp_2d=srp_2d,
        )
