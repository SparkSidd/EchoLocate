"""
EchoLocate — Steered Response Power with Phase Transform (SRP-PHAT)
===================================================================
Near-field 2D spatial-likelihood localization engine.

Mathematical Formulation:
-------------------------
For candidate source position p = [x, y] in room domain \u03a9:
  d_i(p) = \u221a((x - x_i)\u00b2 + (y - y_i)\u00b2 + (z_s - z_m)\u00b2)
  \u03c4_ij(p) = (d_i(p) - d_j(p)) / c

Pairwise gated PHAT cross-correlation:
  \u03a8_ij(f) = G_ij(f) / (|G_ij(f)| + \u03b5)
  R_ij(\u03c4) = fftshift(IFFT(\u03a8_ij))

Spatial Likelihood Map:
  S(p) = \u2211_(i,j) R_ij(\u03c4_ij(p))
  p\u0302_grid = argmax_p S(p)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.interpolate import interp1d

from src.localization.gcc_phat import _next_power_of_2

logger = logging.getLogger(__name__)


@dataclass
class SpatialPeak:
    """Represents a local spatial peak in the SRP map."""
    peak_idx: int
    grid_x: float
    grid_y: float
    power: float
    rank: int


@dataclass
class SRPResult:
    """Full result of SRP-PHAT spatial likelihood localization."""
    raw_grid_position: np.ndarray        # [x, y] on discrete grid
    refined_position: np.ndarray         # [x, y] after sub-grid refinement
    raw_grid_error: float | None         # Error of discrete peak
    refined_error: float | None          # Error of refined peak
    srp_map_2d: np.ndarray               # Shape (n_y, n_x) 2D likelihood map
    x_coords: np.ndarray                 # 1D x axis
    y_coords: np.ndarray                 # 1D y axis
    primary_peak_power: float            # S_1
    secondary_peak_power: float          # S_2
    peak_ratio: float                    # S_1 / S_2
    peak_separation_m: float             # ||p_1 - p_2||_2
    top_k_peaks: list[SpatialPeak]       # Top-K spatial local maxima
    top_k_proximity: bool                # True if ground truth is within threshold of any top-K peak
    eval_runtime_ms: float               # Execution time in milliseconds


class SRPGrid:
    """
    Manages 2D candidate spatial grid and precomputed geometric delay tables.
    """

    def __init__(
        self,
        room_dimensions: list[float],
        mic_positions_2d: np.ndarray,
        grid_resolution_m: float = 0.05,
        boundary_margin_m: float = 0.10,
        speed_of_sound: float = 343.0,
        source_height: float = 1.2,
        mic_height: float = 1.2,
    ):
        self.room_dims = room_dimensions
        self.mics_2d = np.asarray(mic_positions_2d, dtype=np.float64)
        self.resolution = float(grid_resolution_m)
        self.margin = float(boundary_margin_m)
        self.c = float(speed_of_sound)
        self.z_s = float(source_height)
        self.z_m = float(mic_height)

        # 1. Build 1D axes respecting boundary margin
        x_min = self.margin
        x_max = self.room_dims[0] - self.margin
        y_min = self.margin
        y_max = self.room_dims[1] - self.margin

        n_x = int(np.floor((x_max - x_min) / self.resolution + 1e-8)) + 1
        n_y = int(np.floor((y_max - y_min) / self.resolution + 1e-8)) + 1
        self.x_coords = np.linspace(x_min, x_min + (n_x - 1) * self.resolution, n_x)
        self.y_coords = np.linspace(y_min, y_min + (n_y - 1) * self.resolution, n_y)
        self.n_x = len(self.x_coords)
        self.n_y = len(self.y_coords)
        self.n_points = self.n_x * self.n_y

        # 2D Meshgrid: indexing='xy' -> shape (n_y, n_x)
        self.X, self.Y = np.meshgrid(self.x_coords, self.y_coords, indexing="xy")
        # Flattened grid points shape (n_points, 2)
        self.points_flat = np.column_stack([self.X.ravel(), self.Y.ravel()])

        # 2. Precompute geometric delays for all microphone pairs
        self.n_mics = len(self.mics_2d)
        self.pairs: list[tuple[int, int]] = []
        for i in range(self.n_mics):
            for j in range(i + 1, self.n_mics):
                self.pairs.append((i, j))

        # Precompute distances from all candidate points to each mic
        # Shape: (n_points, n_mics)
        z_diff_sq = (self.z_s - self.z_m) ** 2
        dists = np.zeros((self.n_points, self.n_mics), dtype=np.float64)
        for m_idx in range(self.n_mics):
            dx = self.points_flat[:, 0] - self.mics_2d[m_idx, 0]
            dy = self.points_flat[:, 1] - self.mics_2d[m_idx, 1]
            dists[:, m_idx] = np.sqrt(dx**2 + dy**2 + z_diff_sq)

        # Precompute theoretical delays: tau_ij(p) = (d_i(p) - d_j(p)) / c
        # Shape: (n_pairs, n_points)
        self.delay_table: dict[tuple[int, int], np.ndarray] = {}
        for pair in self.pairs:
            i, j = pair
            tau_ij = (dists[:, i] - dists[:, j]) / self.c
            self.delay_table[pair] = tau_ij

        logger.debug(
            "SRPGrid initialized: %dx%d (%d points), res=%.3fm, %d pairs",
            self.n_x, self.n_y, self.n_points, self.resolution, len(self.pairs),
        )


def compute_phat_cross_correlations(
    mic_signals: np.ndarray,
    fs: int,
    fft_size: int | None = None,
    gate_threshold: float = 0.01,
) -> tuple[dict[tuple[int, int], np.ndarray], np.ndarray]:
    """
    Compute gated PHAT cross-correlation functions for all distinct microphone pairs.

    Returns
    -------
    corr_dict : dict[tuple[int, int], np.ndarray]
        Centered cross-correlation array R_ij for each pair.
    tau_axis : np.ndarray
        Time delay axis in seconds corresponding to R_ij.
    """
    n_mics, n_samples = mic_signals.shape
    if fft_size is None:
        fft_size = _next_power_of_2(2 * n_samples)
    elif fft_size < 2 * n_samples:
        fft_size = _next_power_of_2(2 * n_samples)

    # Compute RFFT of all microphone channels
    X = np.fft.rfft(mic_signals, n=fft_size, axis=1)  # (n_mics, n_freq_bins)
    power = np.abs(X) ** 2

    # Tau axis
    tau_samples_axis = np.arange(-fft_size // 2, fft_size - fft_size // 2)
    tau_axis = tau_samples_axis / fs
    eps = 1e-12

    corr_dict = {}
    for i in range(n_mics):
        for j in range(i + 1, n_mics):
            X_i = X[i]
            X_j = X[j]
            G_ij = X_i * np.conj(X_j)
            magnitude = np.abs(G_ij)

            # Signal-power gate
            geom_mean_power = np.sqrt(power[i] * power[j])
            max_power = np.max(geom_mean_power)
            if max_power > 0:
                gate = geom_mean_power >= (gate_threshold * max_power)
            else:
                gate = np.ones(len(G_ij), dtype=bool)

            Psi_ij = np.zeros_like(G_ij, dtype=complex)
            Psi_ij[gate] = G_ij[gate] / (magnitude[gate] + eps)

            # Centered IFFT
            R_ij = np.fft.irfft(Psi_ij, n=fft_size)
            R_centred = np.fft.fftshift(R_ij)
            corr_dict[(i, j)] = R_centred

    return corr_dict, tau_axis


def evaluate_srp_map(
    grid: SRPGrid,
    corr_dict: dict[tuple[int, int], np.ndarray],
    tau_axis: np.ndarray,
    interpolation_method: Literal["linear", "nearest"] = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the total Steered Response Power (SRP) across all candidate points.

    Returns
    -------
    srp_flat : np.ndarray
        Shape (n_points,). Likelihood power at each flattened candidate position.
    srp_2d : np.ndarray
        Shape (n_y, n_x). 2D spatial response map over room coordinates.
    """
    srp_flat = np.zeros(grid.n_points, dtype=np.float64)

    for pair, tau_cand in grid.delay_table.items():
        R_ij = corr_dict[pair]

        if interpolation_method == "linear":
            # Fast vectorized 1D linear interpolation
            val = np.interp(tau_cand, tau_axis, R_ij, left=0.0, right=0.0)
        elif interpolation_method == "nearest":
            # Nearest discrete sample index lookup
            fs = 1.0 / (tau_axis[1] - tau_axis[0])
            idx_zero = len(tau_axis) // 2
            idx = np.round(tau_cand * fs).astype(np.int64) + idx_zero
            idx = np.clip(idx, 0, len(R_ij) - 1)
            val = R_ij[idx]
        else:
            raise ValueError(f"Unknown interpolation method: {interpolation_method}")

        srp_flat += val

    srp_2d = srp_flat.reshape(grid.n_y, grid.n_x)
    return srp_flat, srp_2d


def refine_subgrid_peak(
    srp_2d: np.ndarray,
    grid: SRPGrid,
    peak_y_idx: int,
    peak_x_idx: int,
) -> np.ndarray:
    """
    Fit a 2D local quadratic surface around discrete peak to refine continuous coordinates.
    Uses 3x3 local patch: S(dx, dy) = a*dx^2 + b*dy^2 + c*dx*dy + d*dx + e*dy + f.
    """
    ny, nx = srp_2d.shape
    raw_x = grid.x_coords[peak_x_idx]
    raw_y = grid.y_coords[peak_y_idx]

    # If on boundary of grid, cannot fit 3x3 patch; return raw grid coordinate
    if (
        peak_x_idx <= 0
        or peak_x_idx >= nx - 1
        or peak_y_idx <= 0
        or peak_y_idx >= ny - 1
    ):
        return np.array([raw_x, raw_y], dtype=np.float64)

    # Extract 3x3 patch
    patch = srp_2d[peak_y_idx - 1 : peak_y_idx + 2, peak_x_idx - 1 : peak_x_idx + 2]

    # Central difference gradient
    # x is columns (axis 1), y is rows (axis 0)
    gx = (patch[1, 2] - patch[1, 0]) / (2.0 * grid.resolution)
    gy = (patch[2, 1] - patch[0, 1]) / (2.0 * grid.resolution)

    # Central difference Hessian
    hxx = (patch[1, 2] - 2.0 * patch[1, 1] + patch[1, 0]) / (grid.resolution**2)
    hyy = (patch[2, 1] - 2.0 * patch[1, 1] + patch[0, 1]) / (grid.resolution**2)
    hxy = (patch[2, 2] - patch[2, 0] - patch[0, 2] + patch[0, 0]) / (
        4.0 * (grid.resolution**2)
    )

    H = np.array([[hxx, hxy], [hxy, hyy]], dtype=np.float64)
    g = np.array([gx, gy], dtype=np.float64)

    # Check for negative definite Hessian (local maximum)
    det = hxx * hyy - hxy**2
    if det > 1e-12 and hxx < 0:
        # delta = - H^-1 * g
        delta = -np.linalg.solve(H, g)
        # Limit sub-grid shift to within +/- 1 grid cell to avoid extrapolation runaway
        delta = np.clip(delta, -grid.resolution, grid.resolution)
        refined_x = raw_x + delta[0]
        refined_y = raw_y + delta[1]
    else:
        refined_x = raw_x
        refined_y = raw_y

    return np.array([refined_x, refined_y], dtype=np.float64)


def extract_spatial_peaks(
    srp_flat: np.ndarray,
    srp_2d: np.ndarray,
    grid: SRPGrid,
    min_peak_separation_m: float = 0.50,
    top_k: int = 5,
    true_pos_2d: np.ndarray | None = None,
    proximity_thresh_m: float = 0.50,
) -> tuple[SpatialPeak, SpatialPeak, float, float, list[SpatialPeak], bool]:
    """
    Extract primary global maximum and non-overlapping secondary spatial peaks.

    Returns
    -------
    top_1 : SpatialPeak
    top_2 : SpatialPeak
    peak_ratio : float
        S_1 / max(1e-6, S_2)
    peak_separation : float
        ||p_1 - p_2||_2 in metres
    peaks_list : list[SpatialPeak]
        Top-K separated spatial peaks
    top_k_proximity : bool
        True if true_pos is within proximity_thresh_m of any top-K peak
    """
    # Sort all points descending by power
    sorted_indices = np.argsort(srp_flat)[::-1]
    
    extracted_peaks: list[SpatialPeak] = []
    min_sep_sq = min_peak_separation_m ** 2

    for idx in sorted_indices:
        cand_x = float(grid.points_flat[idx, 0])
        cand_y = float(grid.points_flat[idx, 1])
        cand_pwr = float(srp_flat[idx])

        # Check spatial separation against already extracted peaks
        is_separated = True
        for p in extracted_peaks:
            dist_sq = (cand_x - p.grid_x) ** 2 + (cand_y - p.grid_y) ** 2
            if dist_sq < min_sep_sq:
                is_separated = False
                break

        if is_separated:
            extracted_peaks.append(
                SpatialPeak(
                    peak_idx=int(idx),
                    grid_x=cand_x,
                    grid_y=cand_y,
                    power=cand_pwr,
                    rank=len(extracted_peaks) + 1,
                )
            )
            if len(extracted_peaks) >= top_k:
                break

    top_1 = extracted_peaks[0]
    if len(extracted_peaks) > 1:
        top_2 = extracted_peaks[1]
        p1 = np.array([top_1.grid_x, top_1.grid_y])
        p2 = np.array([top_2.grid_x, top_2.grid_y])
        separation = float(np.linalg.norm(p1 - p2))
        # Protect against non-positive power in denominator
        pwr_denom = max(1e-6, abs(top_2.power))
        ratio = float(top_1.power / pwr_denom)
    else:
        top_2 = top_1
        separation = 0.0
        ratio = 1.0

    # Top-K proximity check
    top_k_proximity = False
    if true_pos_2d is not None:
        t_xy = np.asarray(true_pos_2d)
        for p in extracted_peaks:
            d = np.linalg.norm(np.array([p.grid_x, p.grid_y]) - t_xy)
            if d <= proximity_thresh_m:
                top_k_proximity = True
                break

    return top_1, top_2, ratio, separation, extracted_peaks, top_k_proximity


def localize_srp_phat(
    mic_signals: np.ndarray,
    mic_positions_2d: np.ndarray,
    fs: int,
    room_dimensions: list[float],
    grid: SRPGrid | None = None,
    grid_resolution_m: float = 0.05,
    boundary_margin_m: float = 0.10,
    speed_of_sound: float = 343.0,
    interpolation_method: Literal["linear", "nearest"] = "linear",
    subgrid_refinement: bool = True,
    min_peak_separation_m: float = 0.50,
    top_k_peaks: int = 5,
    true_position_2d: np.ndarray | None = None,
    proximity_thresh_m: float = 0.50,
) -> SRPResult:
    """
    High-level entry point for SRP-PHAT spatial likelihood localization.
    """
    import time
    t_start = time.perf_counter()

    if grid is None:
        grid = SRPGrid(
            room_dimensions=room_dimensions,
            mic_positions_2d=mic_positions_2d,
            grid_resolution_m=grid_resolution_m,
            boundary_margin_m=boundary_margin_m,
            speed_of_sound=speed_of_sound,
        )

    # 1. Compute cross-correlations
    corr_dict, tau_axis = compute_phat_cross_correlations(mic_signals, fs=fs)

    # 2. Evaluate SRP Map
    srp_flat, srp_2d = evaluate_srp_map(
        grid=grid,
        corr_dict=corr_dict,
        tau_axis=tau_axis,
        interpolation_method=interpolation_method,
    )

    # 3. Extract Top-K spatial peaks and competition metrics
    top_1, top_2, ratio, sep, peaks_list, prox = extract_spatial_peaks(
        srp_flat=srp_flat,
        srp_2d=srp_2d,
        grid=grid,
        min_peak_separation_m=min_peak_separation_m,
        top_k=top_k_peaks,
        true_pos_2d=true_position_2d,
        proximity_thresh_m=proximity_thresh_m,
    )

    raw_pos = np.array([top_1.grid_x, top_1.grid_y], dtype=np.float64)

    # 4. Optional Sub-grid refinement
    if subgrid_refinement:
        # Find 2D indices of raw peak
        peak_y_idx = int(top_1.peak_idx // grid.n_x)
        peak_x_idx = int(top_1.peak_idx % grid.n_x)
        refined_pos = refine_subgrid_peak(srp_2d, grid, peak_y_idx, peak_x_idx)
    else:
        refined_pos = raw_pos.copy()

    # Errors against ground truth if provided
    raw_err = None
    refined_err = None
    if true_position_2d is not None:
        t_xy = np.asarray(true_position_2d)
        raw_err = float(np.linalg.norm(raw_pos - t_xy))
        refined_err = float(np.linalg.norm(refined_pos - t_xy))

    t_end = time.perf_counter()
    runtime_ms = (t_end - t_start) * 1000.0

    return SRPResult(
        raw_grid_position=raw_pos,
        refined_position=refined_pos,
        raw_grid_error=raw_err,
        refined_error=refined_err,
        srp_map_2d=srp_2d,
        x_coords=grid.x_coords,
        y_coords=grid.y_coords,
        primary_peak_power=top_1.power,
        secondary_peak_power=top_2.power,
        peak_ratio=ratio,
        peak_separation_m=sep,
        top_k_peaks=peaks_list,
        top_k_proximity=prox,
        eval_runtime_ms=runtime_ms,
    )
