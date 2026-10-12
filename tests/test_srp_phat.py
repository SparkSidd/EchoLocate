"""
Unit tests for SRP-PHAT spatial likelihood localization engine.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.localization.srp_phat import (
    SRPGrid,
    compute_phat_cross_correlations,
    evaluate_srp_map,
    extract_spatial_peaks,
    localize_srp_phat,
    refine_subgrid_peak,
)


class TestSRPGrid:
    def test_grid_initialization_bounds(self):
        room_dims = [10.0, 6.0, 3.0]
        mics_2d = np.array([[1.0, 1.5], [9.0, 1.5], [5.0, 4.5]])
        grid = SRPGrid(
            room_dimensions=room_dims,
            mic_positions_2d=mics_2d,
            grid_resolution_m=0.10,
            boundary_margin_m=0.20,
        )

        assert grid.x_coords[0] >= 0.20 - 1e-6
        assert grid.x_coords[-1] <= 9.80 + 1e-6
        assert grid.y_coords[0] >= 0.20 - 1e-6
        assert grid.y_coords[-1] <= 5.80 + 1e-6
        assert grid.n_points == grid.n_x * grid.n_y
        assert len(grid.pairs) == 3

    def test_theoretical_delay_calculation(self):
        room_dims = [10.0, 6.0, 3.0]
        mics_2d = np.array([[1.0, 1.5], [9.0, 1.5], [5.0, 4.5]])
        grid = SRPGrid(
            room_dimensions=room_dims,
            mic_positions_2d=mics_2d,
            grid_resolution_m=0.50,
            speed_of_sound=343.0,
        )

        # Check a specific candidate point p = [5.0, 3.0]
        p = np.array([5.0, 3.0])
        # Find index in points_flat
        dists_to_p = np.linalg.norm(grid.points_flat - p, axis=1)
        idx = int(np.argmin(dists_to_p))
        actual_p = grid.points_flat[idx]

        # Manual delay calculation for pair (0, 1) between M1(1, 1.5) and M2(9, 1.5)
        d0 = np.linalg.norm(actual_p - mics_2d[0])
        d1 = np.linalg.norm(actual_p - mics_2d[1])
        expected_tau = (d0 - d1) / 343.0

        grid_tau = grid.delay_table[(0, 1)][idx]
        assert np.isclose(grid_tau, expected_tau, atol=1e-6)


class TestSRPLocalizationAccuracy:
    def test_synthetic_direct_path_source(self):
        """Synthesize signals with pure theoretical delays and check SRP localization."""
        fs = 16000
        c = 343.0
        room_dims = [10.0, 6.0, 3.0]
        mics_2d = np.array([[1.0, 1.5], [9.0, 1.5], [5.0, 4.5]])
        true_source = np.array([5.0, 3.0])

        # Generate chirp
        from src.simulation.source_generator import generate_chirp
        chirp = generate_chirp(fs=fs, duration=0.05, f_low=300.0, f_high=5000.0)

        # Compute arrival delays
        dists = [np.linalg.norm(true_source - m) for m in mics_2d]
        times = [d / c for d in dists]
        min_time = min(times)
        rel_delays = [t - min_time for t in times]

        # Delay signals by integer samples
        n_samples = len(chirp) + 500
        mic_signals = np.zeros((3, n_samples), dtype=np.float64)
        for i in range(3):
            delay_samples = int(round(rel_delays[i] * fs))
            mic_signals[i, delay_samples : delay_samples + len(chirp)] = chirp

        res = localize_srp_phat(
            mic_signals=mic_signals,
            mic_positions_2d=mics_2d,
            fs=fs,
            room_dimensions=room_dims,
            grid_resolution_m=0.05,
            subgrid_refinement=True,
            true_position_2d=true_source,
        )

        assert res.refined_error is not None
        assert res.refined_error <= 0.05  # Within grid cell resolution
        assert not np.isnan(res.srp_map_2d).any()


class TestPeakExtractionAndSeparation:
    def test_spatial_peak_separation(self):
        room_dims = [10.0, 6.0, 3.0]
        mics_2d = np.array([[1.0, 1.5], [9.0, 1.5], [5.0, 4.5]])
        grid = SRPGrid(
            room_dimensions=room_dims,
            mic_positions_2d=mics_2d,
            grid_resolution_m=0.10,
        )

        # Create synthetic SRP map with two distant peaks
        srp_flat = np.zeros(grid.n_points)
        # Peak 1 at (3.0, 2.0)
        idx1 = int(np.argmin(np.linalg.norm(grid.points_flat - np.array([3.0, 2.0]), axis=1)))
        srp_flat[idx1] = 10.0
        # Peak 2 at (7.0, 4.0)
        idx2 = int(np.argmin(np.linalg.norm(grid.points_flat - np.array([7.0, 4.0]), axis=1)))
        srp_flat[idx2] = 8.0

        srp_2d = srp_flat.reshape(grid.n_y, grid.n_x)

        top_1, top_2, ratio, sep, peaks, prox = extract_spatial_peaks(
            srp_flat=srp_flat,
            srp_2d=srp_2d,
            grid=grid,
            min_peak_separation_m=0.50,
            top_k=3,
        )

        assert np.isclose(top_1.power, 10.0)
        assert np.isclose(top_2.power, 8.0)
        assert np.isclose(ratio, 10.0 / 8.0)
        assert sep >= 0.50
        assert len(peaks) >= 2
