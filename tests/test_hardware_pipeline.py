"""
EchoLocate — Unit Tests for Hardware Pipeline & Device Management (Exp 6.5)
===========================================================================
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.app.engine.hardware_pipeline import HardwareMode, LiveHardwarePipeline
from src.app.engine.live_metrics import compute_live_signal_metrics
from src.app.hardware.device_registry import DeviceType, HardwareDevice, MultiDeviceRegistry, SyncStatus
from src.app.main import app


def test_device_registry_operations():
    """Verify registration, coordinate updates, and active geometry extraction."""
    reg = MultiDeviceRegistry()
    assert len(reg.list_devices()) >= 2

    # Update Laptop coordinate
    ok = reg.update_position("laptop_mic_primary", 1.5, 2.5)
    assert ok is True
    dev = reg.get_device("laptop_mic_primary")
    assert dev.position_2d == (1.5, 2.5)

    # Sync metrics update
    reg.update_sync_metrics("smartphone_client_primary", offset_ms=12.5, drift_ppm=4.2, rtt_ms=25.0, status=SyncStatus.CALIBRATED_ACOUSTIC)
    ph = reg.get_device("smartphone_client_primary")
    assert ph.clock_offset_ms == 12.5
    assert ph.sync_status == SyncStatus.CALIBRATED_ACOUSTIC


def test_live_signal_metrics_computation():
    """Verify live audio frame statistics (RMS dB, SNR, dominant frequency, clipping)."""
    t = np.linspace(0, 0.2, 3200)
    sig = 0.5 * np.sin(2 * np.pi * 440.0 * t)

    metrics = compute_live_signal_metrics(sig, fs=16000)
    assert metrics.rms_energy_db > -40.0
    assert metrics.dominant_freq_hz == pytest.approx(440.0, abs=15.0)
    assert metrics.is_clipping is False
    assert metrics.is_silence is False


def test_hardware_pipeline_cycles():
    """Verify Mode H1 and Mode H2 processing cycles and anti-fake hashes."""
    pipeline = LiveHardwarePipeline(sample_rate=16000)

    # Test Mode H1 (Single Laptop)
    pipeline.set_mode(HardwareMode.MODE_H1_SINGLE_LAPTOP)
    out_h1 = pipeline.process_cycle()
    assert out_h1.active_mode == HardwareMode.MODE_H1_SINGLE_LAPTOP
    assert out_h1.computation_hash.startswith("H1-")
    assert out_h1.laptop_metrics.rms_energy_db is not None
    assert out_h1.processing_latency_ms < 50.0

    # Test Mode H2 (Dual Device)
    pipeline.set_mode(HardwareMode.MODE_H2_DUAL_DEVICE)
    out_h2 = pipeline.process_cycle()
    assert out_h2.active_mode == HardwareMode.MODE_H2_DUAL_DEVICE
    assert out_h2.computation_hash.startswith("H2-")
    assert out_h2.gcc_result is not None
    assert out_h2.tdoa_estimate is not None
    assert out_h2.hyperbolic_locus is not None

    # Test Reprocess
    reproc = pipeline.reprocess_current_frame()
    assert reproc.frame_id > out_h2.frame_id


def test_fastapi_hardware_routes():
    """Verify REST endpoints for hardware queries and operations."""
    client = TestClient(app)

    # 1. Devices
    r1 = client.get("/api/hardware/registry_devices")
    assert r1.status_code == 200
    assert len(r1.json()) >= 2

    # 2. Position update
    r2 = client.post("/api/hardware/position", json={"device_id": "laptop_mic_primary", "x": 2.2, "y": 2.2})
    assert r2.status_code == 200
    assert r2.json()["success"] is True

    # 3. Status
    r3 = client.get("/api/hardware/status")
    assert r3.status_code == 200
    assert "computation_hash" in r3.json()

    # 4. Mode switch
    r4 = client.post("/api/hardware/mode", json={"mode": "MODE_H1_SINGLE_LAPTOP"})
    assert r4.status_code == 200
    assert r4.json()["mode"] == "MODE_H1_SINGLE_LAPTOP"

    # 5. Reprocess
    r5 = client.post("/api/hardware/reprocess")
    assert r5.status_code == 200
    assert r5.json()["reprocessed"] is True
