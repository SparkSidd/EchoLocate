"""
EchoLocate — Unit Tests for Distributed Synchronization & Acoustic Calibration (Exp 6.5)
========================================================================================
"""

from __future__ import annotations

import numpy as np
import pytest

from src.app.hardware.acoustic_calibration import generate_reference_calibration_chirp, perform_acoustic_calibration
from src.app.hardware.distributed_sync import DistributedClockSynchronizer


def test_distributed_clock_synchronizer():
    """Verify NTP clock offset estimation, RTT calculation, and jitter filtering."""
    sync = DistributedClockSynchronizer()

    # Feed 10 synthetic probes with 50.0 ms offset and 20.0 ms RTT
    for i in range(10):
        t0 = 1000.0 + i * 100.0
        t1 = t0 + 50.0 + 10.0  # +50ms offset + 10ms one-way delay
        t2 = t1 + 1.0          # 1ms server processing
        t3 = t0 + 20.0         # 20ms RTT

        state = sync.process_probe("phone_1", t0, t1, t2, t3)

    assert state.clock_offset_ms == pytest.approx(50.0, abs=1.0)
    assert state.mean_rtt_ms == pytest.approx(19.0, abs=2.0)
    assert state.sync_confidence_pct > 50.0

    # Translation
    server_time = sync.translate_client_time_to_server("phone_1", 2000.0)
    assert server_time == pytest.approx(2050.0, abs=2.0)


def test_acoustic_calibration_decoupling():
    """Verify decoupling of acoustic propagation delay from hardware clock offset."""
    fs = 16000
    chirp = generate_reference_calibration_chirp(fs=fs, duration_s=0.10)

    # Geometry:
    # Source at (3.0, 3.0)
    # Laptop Mic at (2.0, 2.0) => dist = sqrt(1^2 + 1^2) = 1.414m
    # Phone Mic at (4.0, 2.0) => dist = sqrt(1^2 + 1^2) = 1.414m
    # Symmetric geometry => expected acoustic delay = 0.0 ms
    sig_lap = np.pad(chirp, (0, 1600))
    # Phone audio shifted by 200 samples (12.5 ms hardware clock offset)
    sig_phone = np.pad(chirp, (200, 1400))

    calib = perform_acoustic_calibration(
        laptop_audio=sig_lap,
        phone_audio=sig_phone,
        source_xy=(3.0, 3.0),
        laptop_mic_xy=(2.0, 2.0),
        phone_mic_xy=(4.0, 2.0),
        fs=fs,
    )

    assert calib.expected_acoustic_delay_ms == pytest.approx(0.0, abs=0.1)
    assert calib.estimated_hardware_clock_offset_ms == pytest.approx(-12.5, abs=0.5)
    assert calib.is_valid is True
