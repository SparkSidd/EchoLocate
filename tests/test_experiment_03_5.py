"""
Unit tests for Experiment 3.5: Reality Check & Adversarial Model-Mismatch Audit.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.leakage_audit import (
    audit_grid_alignment_fraction,
    audit_ground_truth_parameter_access,
    audit_ground_truth_permutation_invariance,
    run_full_leakage_audit,
)
from src.experiments.reality_check import generate_diverse_signal, generate_offgrid_sources
from src.localization.blind_localizer import BlindSRPLocalizer


class TestLeakageAndDecoupling:
    def test_blind_localizer_interface_isolation(self):
        """Verify BlindSRPLocalizer has no ground-truth arguments."""
        status, evidence, severity = audit_ground_truth_parameter_access()
        assert status == "PASS"
        assert severity == "NONE"

    def test_ground_truth_permutation_invariance(self):
        """Verify predicted coordinates are 100% invariant to ground truth labels."""
        status, evidence, severity = audit_ground_truth_permutation_invariance()
        assert status == "PASS"
        assert severity == "NONE"

    def test_leakage_audit_table_generation(self):
        canonical = [{"id": "S01", "xy": [2.0, 1.0]}, {"id": "S02", "xy": [8.0, 1.0]}]
        df_audit = run_full_leakage_audit(canonical)
        assert len(df_audit) == 7
        assert "audit_check" in df_audit.columns
        assert "verdict" in df_audit.columns


class TestOffGridAndSignalGeneration:
    def test_offgrid_source_generator(self):
        room_dims = [10.0, 6.0, 3.0]
        sources = generate_offgrid_sources(count=20, room_dims=room_dims, seed=42)
        assert len(sources) == 20

        # Verify not on multiples of 0.025m
        for src in sources:
            x, y = src["xy"]
            assert not (np.isclose(x % 0.025, 0, atol=0.001) or np.isclose(x % 0.025, 0.025, atol=0.001))
            assert not (np.isclose(y % 0.025, 0, atol=0.001) or np.isclose(y % 0.025, 0.025, atol=0.001))

    def test_diverse_signal_types_generation(self):
        fs = 16000
        dur = 0.05
        for sig_type in ["linear_chirp", "log_chirp", "broadband_noise", "speech_synthetic", "impulsive_transient"]:
            sig = generate_diverse_signal(sig_type, fs=fs, duration=dur, seed=42)
            assert len(sig) == int(fs * dur)
            assert not np.isnan(sig).any()
            assert np.max(np.abs(sig)) > 0.0


class TestModelMismatchCalculations:
    def test_speed_of_sound_localizer_independence(self):
        """Verify speed of sound changes grid delay table without affecting audio."""
        room_dims = [10.0, 6.0, 3.0]
        mics_2d = np.array([[1.0, 1.5], [9.0, 1.5], [5.0, 4.5]])

        loc_343 = BlindSRPLocalizer(room_dims, mics_2d, assumed_speed_of_sound=343.0)
        loc_330 = BlindSRPLocalizer(room_dims, mics_2d, assumed_speed_of_sound=330.0)

        tau_343 = loc_343.grid.delay_table[(0, 1)][0]
        tau_330 = loc_330.grid.delay_table[(0, 1)][0]

        # Delay is inversely proportional to speed: tau = dist_diff / c
        assert np.isclose(tau_343 * 343.0, tau_330 * 330.0, atol=1e-5)
