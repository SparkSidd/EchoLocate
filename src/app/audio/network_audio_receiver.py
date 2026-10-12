"""
EchoLocate — Network Audio WebSocket Receiver (Accuracy Overhaul)
=================================================================
Parses incoming audio frames from browser-based smartphone clients
(Binary ArrayBuffer Float32 or JSON Base64), timestamps them, and queues them.

Key accuracy fix: proper anti-aliased resampling from phone's native rate
(44100 Hz or 48000 Hz on Android Chrome) down to 16000 Hz.

Linear interpolation at ratio 16/44.1 ≈ 0.363 causes severe aliasing —
high-frequency components above 8 kHz fold back into the 0–8 kHz band,
producing wideband noise that raises the GCC-PHAT noise floor and kills PSR.

Fix: low-pass filter at 0.9 × Nyquist_target BEFORE downsampling, then
integer-ratio decimation. Pure numpy — no scipy required.
"""

from __future__ import annotations

import base64
import json
import logging
import struct
import time
from typing import Any

import numpy as np

from src.app.hardware.network_microphone import NetworkAudioPacket, NetworkClientSession

logger = logging.getLogger(__name__)

_TARGET_FS = 16000


# ---------------------------------------------------------------------------
# Anti-aliased resampler
# ---------------------------------------------------------------------------

def _design_lowpass_fir(cutoff_hz: float, fs: int, ntaps: int = 63) -> np.ndarray:
    """Windowed-sinc low-pass FIR filter (Kaiser β=6)."""
    nyq = fs / 2.0
    fc = cutoff_hz / nyq
    beta = 6.0
    win = np.kaiser(ntaps, beta)
    n = np.arange(ntaps) - (ntaps - 1) / 2.0
    h = np.sinc(2 * fc * n) * 2 * fc * win
    h /= h.sum()
    return h.astype(np.float32)


# Pre-compute LP filters for common phone sample rates
_LP_FILTERS: dict[int, np.ndarray] = {}


def _get_lp_filter(src_fs: int) -> np.ndarray:
    if src_fs not in _LP_FILTERS:
        # Cutoff at 90% of target Nyquist to leave a guard band
        cutoff = _TARGET_FS * 0.45
        _LP_FILTERS[src_fs] = _design_lowpass_fir(cutoff, src_fs, ntaps=127)
    return _LP_FILTERS[src_fs]


def _resample_antialias(samples: np.ndarray, src_fs: int, tgt_fs: int = _TARGET_FS) -> np.ndarray:
    """
    Anti-aliased resample from src_fs to tgt_fs.
    Steps:
      1. Low-pass filter at 0.45 × tgt_fs to prevent aliasing
      2. Rational resample via numpy interpolation
    """
    if src_fs == tgt_fs:
        return samples.astype(np.float32)

    # Step 1: Anti-alias LP filter
    fir = _get_lp_filter(src_fs)
    filtered = np.convolve(samples.astype(np.float64), fir.astype(np.float64), mode="same")

    # Step 2: Rational resample
    target_len = int(len(filtered) * tgt_fs / src_fs)
    if target_len <= 0:
        return np.zeros(1, dtype=np.float32)

    resampled = np.interp(
        np.linspace(0, len(filtered) - 1, target_len),
        np.arange(len(filtered)),
        filtered,
    ).astype(np.float32)

    return resampled


# ---------------------------------------------------------------------------
# Receiver
# ---------------------------------------------------------------------------

class NetworkAudioReceiver:
    """Decodes and ingests network audio streams from smartphone clients."""

    def __init__(self) -> None:
        self.active_sessions: dict[str, NetworkClientSession] = {}
        # Reference to the synchronized buffer's feed_phone_packet method
        # Set by hardware_pipeline after both objects are constructed
        self.sync_buffer_feed: Any = None

    def get_or_create_session(
        self,
        device_id: str,
        client_ip: str = "127.0.0.1",
        user_agent: str = "Browser",
    ) -> NetworkClientSession:
        """Get or initialize client session."""
        if device_id not in self.active_sessions:
            session = NetworkClientSession(
                device_id=device_id, client_ip=client_ip, user_agent=user_agent
            )
            self.active_sessions[device_id] = session
            logger.info(
                "Initialized network audio session for device: %s [%s]", device_id, client_ip
            )
        return self.active_sessions[device_id]

    def _ingest_samples(
        self, session: NetworkClientSession, samples: np.ndarray, sample_rate: int, seq: int, client_ts: float
    ) -> NetworkAudioPacket:
        """Resample + push to session queue + feed to timestamped buffer."""
        # Anti-aliased resample to 16 kHz
        if sample_rate != _TARGET_FS:
            samples = _resample_antialias(samples, sample_rate, _TARGET_FS)

        t_server = time.time()
        dur_ms = len(samples) / _TARGET_FS * 1000.0

        packet = NetworkAudioPacket(
            device_id=session.device_id,
            sequence_id=int(seq),
            client_timestamp_ms=float(client_ts),
            server_timestamp_ms=t_server * 1000.0,
            sample_rate=_TARGET_FS,
            samples=samples.copy(),
            duration_ms=dur_ms,
        )
        session.push_packet(packet)

        # Feed into wall-clock-aligned timestamped ring buffer
        if self.sync_buffer_feed is not None:
            try:
                self.sync_buffer_feed(samples, t_server)
            except Exception:
                pass

        return packet

    def decode_json_message(
        self, session: NetworkClientSession, msg_text: str
    ) -> NetworkAudioPacket | None:
        """Parse incoming JSON payload containing Float32 or Base64 PCM data."""
        try:
            data = json.loads(msg_text)
            seq = int(data.get("seq", 0))
            client_ts = float(data.get("client_ts", time.time() * 1000.0))
            sample_rate = int(data.get("sample_rate", _TARGET_FS))

            if "samples" in data and isinstance(data["samples"], list):
                samples = np.array(data["samples"], dtype=np.float32)
            elif "base64" in data:
                raw_bytes = base64.b64decode(data["base64"])
                samples = np.frombuffer(raw_bytes, dtype=np.float32)
            else:
                return None

            return self._ingest_samples(session, samples, sample_rate, seq, client_ts)

        except Exception as e:
            logger.warning("Error decoding JSON audio frame: %s", e)
            return None

    def decode_binary_message(
        self, session: NetworkClientSession, raw_bytes: bytes
    ) -> NetworkAudioPacket | None:
        """
        Parse raw binary Float32Array PCM packet.
        Header format (16 bytes, little-endian): [seq:uint32, client_ts:float64, sr:uint32]
        Followed by: Float32 PCM samples
        """
        try:
            if len(raw_bytes) < 16:
                # Raw audio without header
                samples = np.frombuffer(raw_bytes, dtype=np.float32).copy()
                seq = session.packets_received
                client_ts = time.time() * 1000.0
                sample_rate = _TARGET_FS
            else:
                seq, client_ts, sample_rate = struct.unpack("<IdI", raw_bytes[:16])
                samples = np.frombuffer(raw_bytes[16:], dtype=np.float32).copy()

            if len(samples) == 0:
                return None

            return self._ingest_samples(session, samples, int(sample_rate), int(seq), float(client_ts))

        except Exception as e:
            logger.warning("Error decoding binary audio frame: %s", e)
            return None
