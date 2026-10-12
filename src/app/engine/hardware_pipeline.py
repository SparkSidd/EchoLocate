"""
EchoLocate — Hardware Pipeline Coordinator (Experiment 6.5)
===========================================================
Coordinates physical audio acquisition (sounddevice), smartphone network streaming,
distributed NTP clock synchronization, acoustic calibration, real-time GCC-PHAT,
TDOA estimation, 2D hyperbolic locus rendering, and anti-fake computation hashes.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from src.app.audio.live_capture import LiveMicrophoneCaptureStream
from src.app.audio.network_audio_receiver import NetworkAudioReceiver
from src.app.audio.synchronized_buffer import SynchronizedMultiDeviceBuffer
from src.app.engine.live_metrics import LiveSignalMetrics, compute_live_signal_metrics
from src.app.hardware.acoustic_calibration import AcousticCalibrationResult, perform_acoustic_calibration
from src.app.hardware.device_registry import DeviceType, HardwareDevice, MultiDeviceRegistry, SyncStatus
from src.app.hardware.distributed_sync import ClockSyncState, DistributedClockSynchronizer
from src.app.localization.hyperbola_solver import HyperbolaLocus, solve_hyperbolic_locus
from src.app.localization.live_gcc_phat import LiveGCCResult, compute_live_gcc_phat
from src.app.localization.tdoa_estimator import TDOAEstimate, estimate_tdoa_from_gcc, reset_ema

logger = logging.getLogger(__name__)


class HardwareMode(str, Enum):
    MODE_H1_SINGLE_LAPTOP = "MODE_H1_SINGLE_LAPTOP"
    MODE_H2_DUAL_DEVICE = "MODE_H2_DUAL_DEVICE"
    MODE_H3_EXPANDED_ARRAY = "MODE_H3_EXPANDED_ARRAY"


@dataclass
class HardwarePipelineFrameOutput:
    """Full quantitative output packet of a single live hardware processing cycle."""
    frame_id: int
    timestamp_unix: float
    computation_hash: str
    active_mode: HardwareMode
    laptop_metrics: LiveSignalMetrics
    phone_metrics: LiveSignalMetrics | None
    is_phone_connected: bool
    clock_sync_state: ClockSyncState | None
    latest_calibration: AcousticCalibrationResult | None
    gcc_result: LiveGCCResult | None
    tdoa_estimate: TDOAEstimate | None
    hyperbolic_locus: HyperbolaLocus | None
    processing_latency_ms: float
    mic_gain_offset_db: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp_unix": self.timestamp_unix,
            "computation_hash": self.computation_hash,
            "active_mode": self.active_mode.value,
            "laptop_metrics": self.laptop_metrics.to_dict(),
            "phone_metrics": self.phone_metrics.to_dict() if self.phone_metrics else None,
            "is_phone_connected": self.is_phone_connected,
            "clock_sync_state": self.clock_sync_state.to_dict() if self.clock_sync_state else None,
            "latest_calibration": self.latest_calibration.to_dict() if self.latest_calibration else None,
            "gcc_result": self.gcc_result.to_dict() if self.gcc_result else None,
            "tdoa_estimate": self.tdoa_estimate.to_dict() if self.tdoa_estimate else None,
            "hyperbolic_locus": self.hyperbolic_locus.to_dict() if self.hyperbolic_locus else None,
            "processing_latency_ms": round(self.processing_latency_ms, 2),
            "mic_gain_offset_db": round(self.mic_gain_offset_db, 1),
        }


class LiveHardwarePipeline:
    """Central orchestrator for real hardware audio processing."""

    def __init__(self, sample_rate: int = 16000, room_dims: list[float] = [10.0, 6.0]) -> None:
        self.sample_rate = sample_rate
        self.room_dims = room_dims
        self.active_mode = HardwareMode.MODE_H2_DUAL_DEVICE

        self.registry = MultiDeviceRegistry()
        self.local_capture = LiveMicrophoneCaptureStream(sample_rate=sample_rate)
        self.network_receiver = NetworkAudioReceiver()
        self.clock_sync = DistributedClockSynchronizer()
        self.sync_buffer = SynchronizedMultiDeviceBuffer(
            local_capture=self.local_capture,
            network_receiver=self.network_receiver,
            clock_sync=self.clock_sync,
            sample_rate=sample_rate,
        )

        self.latest_calibration: AcousticCalibrationResult | None = None
        self.mic_gain_offset_db: float = 0.0
        self.total_frames_processed = 0
        self._last_processed_frame: HardwarePipelineFrameOutput | None = None

        # Wire phone audio into the timestamped ring buffer for alignment
        self.network_receiver.sync_buffer_feed = self.sync_buffer.feed_phone_packet

        # Start local capture
        self.local_capture.start()

    def set_mode(self, mode: HardwareMode) -> None:
        """Switch hardware operational mode."""
        self.active_mode = mode
        reset_ema()  # Reset TDOA smoother when mode changes
        logger.info("Hardware operational mode set to: %s", mode.value)

    def calibrate_gain(self) -> float:
        """
        Calibrate hardware sensitivity mismatch when speaker is placed equidistant in the center.
        Calculates: Delta G = RMS_laptop(dB) - RMS_phone(dB)
        """
        dual_audio, is_conn, _ = self.sync_buffer.get_aligned_dual_channel_frame()
        if not is_conn:
            return 0.0
        lap_rms = float(np.sqrt(np.mean(dual_audio[0] ** 2))) + 1e-12
        ph_rms = float(np.sqrt(np.mean(dual_audio[1] ** 2))) + 1e-12
        lap_db = 20.0 * np.log10(lap_rms)
        ph_db = 20.0 * np.log10(ph_rms)
        self.mic_gain_offset_db = float(lap_db - ph_db)
        logger.info("Microphone gain calibration offset set to: %.1f dB", self.mic_gain_offset_db)
        return self.mic_gain_offset_db

    def trigger_acoustic_calibration(
        self,
        source_xy: tuple[float, float] = (3.0, 3.0),
    ) -> AcousticCalibrationResult:
        """Run acoustic calibration using current audio buffers."""
        dual_audio, _, _ = self.sync_buffer.get_aligned_dual_channel_frame()
        dev_lap = self.registry.get_device("laptop_mic_primary")
        dev_ph = self.registry.get_device("smartphone_client_primary")

        m1_xy = dev_lap.position_2d if dev_lap else (2.0, 2.0)
        m2_xy = dev_ph.position_2d if dev_ph else (4.0, 2.0)

        res = perform_acoustic_calibration(
            laptop_audio=dual_audio[0],
            phone_audio=dual_audio[1],
            source_xy=source_xy,
            laptop_mic_xy=m1_xy,
            phone_mic_xy=m2_xy,
            fs=self.sample_rate,
        )
        self.latest_calibration = res

        # Also calibrate mic sensitivity
        self.calibrate_gain()

        # Update registry sync status
        if res.is_valid and dev_ph:
            dev_ph.sync_status = SyncStatus.CALIBRATED_ACOUSTIC
            dev_ph.clock_offset_ms = res.estimated_hardware_clock_offset_ms

        return res

    def process_cycle(self) -> HardwarePipelineFrameOutput:
        """Execute one complete real-time acquisition and localization cycle."""
        t_start = time.perf_counter()
        self.total_frames_processed += 1
        frame_id = self.total_frames_processed
        ts_now = time.time()

        # 1. Fetch aligned dual-channel audio
        dual_audio, is_phone_conn, sync_offset_ms = self.sync_buffer.get_aligned_dual_channel_frame()
        sig_lap = dual_audio[0]
        sig_phone = dual_audio[1]

        # 2. Compute Physical Signal Diagnostics — always compute laptop, phone only when live
        lap_metrics = compute_live_signal_metrics(sig_lap, fs=self.sample_rate)
        phone_metrics = compute_live_signal_metrics(sig_phone, fs=self.sample_rate) if is_phone_conn else None

        # 3. Mode H1: Single Laptop only
        if self.active_mode == HardwareMode.MODE_H1_SINGLE_LAPTOP:
            lat_ms = (time.perf_counter() - t_start) * 1000.0
            h_str = hashlib.sha256(sig_lap.tobytes()).hexdigest()[:12]
            out = HardwarePipelineFrameOutput(
                frame_id=frame_id,
                timestamp_unix=ts_now,
                computation_hash=f"H1-{h_str}",
                active_mode=self.active_mode,
                laptop_metrics=lap_metrics,
                phone_metrics=None,
                is_phone_connected=False,
                clock_sync_state=None,
                latest_calibration=None,
                gcc_result=None,
                tdoa_estimate=None,
                hyperbolic_locus=None,
                processing_latency_ms=lat_ms,
            )
            self._last_processed_frame = out
            return out

        # 4. Mode H2: Dual-Device TDOA and Hyperbolic Locus
        dev_lap = self.registry.get_device("laptop_mic_primary")
        dev_ph = self.registry.get_device("smartphone_client_primary")
        m1_xy = dev_lap.position_2d if dev_lap else (2.0, 2.0)
        m2_xy = dev_ph.position_2d if dev_ph else (4.0, 2.0)

        # Baseline distance & dynamic physical maximum acoustic delay bound (D / c)
        m1 = np.array(m1_xy, dtype=np.float64)
        m2 = np.array(m2_xy, dtype=np.float64)
        baseline_d = float(np.linalg.norm(m2 - m1))
        
        from src.app.localization.tdoa_estimator import speed_of_sound_at_temp
        temp_c = 22.0
        c_sound = speed_of_sound_at_temp(temp_c)
        physical_max_tau_ms = max(1.0, (baseline_d / c_sound) * 1000.0 * 1.1)

        gcc_res: LiveGCCResult | None = None
        tdoa_est: TDOAEstimate | None = None
        hyp_locus: HyperbolaLocus | None = None

        # Only cross-correlate when phone is genuinely streaming real audio
        if is_phone_conn:
            try:
                lap_rms = float(np.sqrt(np.mean(sig_lap ** 2)))
                phone_rms = float(np.sqrt(np.mean(sig_phone ** 2)))

                # Both channels need real signal above noise floor
                if lap_rms > 1e-5 and phone_rms > 1e-5:
                    gcc_res = compute_live_gcc_phat(
                        sig_lap, sig_phone, fs=self.sample_rate, max_tau_ms=physical_max_tau_ms
                    )
                    tdoa_est = estimate_tdoa_from_gcc(
                        tau_axis_ms=np.array(gcc_res.tau_axis_ms),
                        correlation_curve=np.array(gcc_res.correlation_curve),
                        signal_laptop=sig_lap,
                        signal_phone=sig_phone,
                        mic1_xy=m1_xy,
                        mic2_xy=m2_xy,
                        temperature_c=temp_c,
                        gain_offset_db=self.mic_gain_offset_db,
                        fs=self.sample_rate,
                    )
                    hyp_locus = solve_hyperbolic_locus(
                        mic1_xy=m1_xy,
                        mic2_xy=m2_xy,
                        tdoa_ms=tdoa_est.tdoa_ms,
                        room_dims=self.room_dims,
                        temperature_c=temp_c,
                    )
            except Exception as exc:
                logger.error("GCC-PHAT processing error: %s", exc, exc_info=True)
                gcc_res = None
                tdoa_est = None
                hyp_locus = None

        sync_state = self.clock_sync.compute_sync_state("smartphone_client_primary")
        lat_ms = (time.perf_counter() - t_start) * 1000.0
        h_str = hashlib.sha256(dual_audio.tobytes()).hexdigest()[:12]

        out = HardwarePipelineFrameOutput(
            frame_id=frame_id,
            timestamp_unix=ts_now,
            computation_hash=f"H2-{h_str}",
            active_mode=self.active_mode,
            laptop_metrics=lap_metrics,
            phone_metrics=phone_metrics,
            is_phone_connected=is_phone_conn,
            clock_sync_state=sync_state,
            latest_calibration=self.latest_calibration,
            gcc_result=gcc_res,
            tdoa_estimate=tdoa_est,
            hyperbolic_locus=hyp_locus,
            processing_latency_ms=lat_ms,
            mic_gain_offset_db=self.mic_gain_offset_db,
        )
        self._last_processed_frame = out
        return out

    def reprocess_current_frame(self) -> HardwarePipelineFrameOutput:
        """⚡ CAPTURE AND REPROCESS verification action."""
        logger.info("Executing on-demand frame re-processing.")
        return self.process_cycle()
