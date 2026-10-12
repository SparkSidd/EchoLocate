"""
EchoLocate — Unit Tests for Live TDOA & 2D Hyperbola Locus Solver (Exp 6.5)
==========================================================================
"""

from __future__ import annotations

import numpy as np
import pytest

from src.app.localization.hyperbola_solver import solve_hyperbolic_locus
from src.app.localization.live_gcc_phat import compute_live_gcc_phat
from src.app.localization.tdoa_estimator import assess_signal_quality, estimate_tdoa_from_gcc


def test_live_gcc_phat_correlator():
    """Verify live GCC-PHAT cross-correlation calculation and sub-millisecond execution."""
    t = np.linspace(0, 0.2, 3200)
    sig1 = np.sin(2 * np.pi * 500 * t) + np.random.normal(0, 0.05, 3200)
    # Delay by 16 samples (1.0 ms at 16 kHz)
    sig2 = np.roll(sig1, 16)

    res = compute_live_gcc_phat(sig1, sig2, fs=16000, max_tau_ms=10.0)
    assert abs(res.estimated_delay_ms) == pytest.approx(1.0, abs=0.2)
    assert res.peak_power > 0.001
    assert res.latency_ms < 15.0


def test_signal_quality_assessment():
    """Verify signal bandwidth and acoustic quality ranking."""
    t = np.linspace(0, 0.2, 3200)
    chirp = np.sin(2 * np.pi * (300 + 3000 * t) * t)
    speech = np.sin(2 * np.pi * 150 * t) + np.sin(2 * np.pi * 300 * t)
    tone = np.sin(2 * np.pi * 1000 * t)

    q_chirp = assess_signal_quality(chirp, fs=16000)
    q_speech = assess_signal_quality(speech, fs=16000)
    q_tone = assess_signal_quality(tone, fs=16000)

    assert q_chirp.bandwidth_score_pct > q_speech.bandwidth_score_pct
    assert q_speech.bandwidth_score_pct > q_tone.bandwidth_score_pct
    assert q_chirp.is_favorable_for_tdoa is True


def test_2d_hyperbola_locus_solver():
    """Verify 2D hyperbola equation solving and room clipping."""
    m1 = (2.0, 2.0)
    m2 = (6.0, 2.0)  # Baseline = 4.0m
    room = [10.0, 6.0]

    # Zero delay => straight line x = 4.0
    locus_zero = solve_hyperbolic_locus(m1, m2, tdoa_ms=0.0, room_dims=room)
    assert locus_zero.baseline_separation_m == pytest.approx(4.0, abs=1e-3)
    assert locus_zero.is_physically_realizable is True
    assert len(locus_zero.locus_points) > 10

    # Points should lie near x = 4.0
    x_coords = [p[0] for p in locus_zero.locus_points]
    assert np.mean(x_coords) == pytest.approx(4.0, abs=0.2)

    # Positive delay => hyperbola branch bending towards Phone Mic (M2)
    locus_pos = solve_hyperbolic_locus(m1, m2, tdoa_ms=3.0, room_dims=room)
    assert locus_pos.is_physically_realizable is True
    x_pos = [p[0] for p in locus_pos.locus_points]
    assert np.min(x_pos) > 4.0  # Entire curve shifted towards right
