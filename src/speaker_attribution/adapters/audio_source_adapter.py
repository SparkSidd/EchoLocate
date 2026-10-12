"""
EchoLocate — Audio Source Adapter Layer
=======================================
Decouples speaker attribution from underlying hardware, network, or file sources.
Supports WAV files, live sounddevice inputs, WebSocket PCM frames, and simulated arrays.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import soundfile as sf


class AudioSourceAdapter:
    """
    Standardizes heterogeneous input sources into mono 16 kHz numpy audio arrays.
    """

    @staticmethod
    def from_file(file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Load audio from WAV / FLAC / OGG file on disk."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Audio file not found: {p}")
        data, sr = sf.read(str(p), dtype="float32")
        return data, sr

    @staticmethod
    def from_bytes(audio_bytes: bytes, format: str = "WAV") -> Tuple[np.ndarray, int]:
        """Load audio from in-memory encoded audio bytes."""
        bio = io.BytesIO(audio_bytes)
        try:
            data, sr = sf.read(bio, dtype="float32")
            return data, sr
        except Exception as primary_err:
            try:
                bio.seek(0)
                import scipy.io.wavfile as wavfile
                sr, data = wavfile.read(bio)
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32:
                    data = data.astype(np.float32) / 2147483648.0
                elif data.dtype == np.uint8:
                    data = (data.astype(np.float32) - 128.0) / 128.0
                if data.ndim > 1:
                    data = data[:, 0]
                return data.astype(np.float32), int(sr)
            except Exception:
                raise primary_err

    @staticmethod
    def from_raw_pcm_float32(
        raw_bytes: bytes,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> Tuple[np.ndarray, int]:
        """Decode raw uncompressed float32 PCM bytes (e.g. from network WebSocket)."""
        data = np.frombuffer(raw_bytes, dtype=np.float32)
        if channels > 1:
            data = data.reshape(-1, channels).T
        return data, sample_rate

    @staticmethod
    def from_network_packet(
        payload: bytes,
        has_16byte_header: bool = True,
    ) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """
        Decode EchoLocate smartphone WebSocket frame:
        [16-byte header: seq(u32), client_ts(f64), sr(u32)] + Float32Array PCM
        """
        if has_16byte_header and len(payload) > 16:
            header = payload[:16]
            seq = int(np.frombuffer(header[0:4], dtype=np.uint32)[0])
            client_ts = float(np.frombuffer(header[4:12], dtype=np.float64)[0])
            sr = int(np.frombuffer(header[12:16], dtype=np.uint32)[0])
            pcm = np.frombuffer(payload[16:], dtype=np.float32)
            return pcm, sr, {"sequence_id": seq, "client_timestamp_ms": client_ts}
        else:
            pcm = np.frombuffer(payload, dtype=np.float32)
            return pcm, 16000, {}

    @staticmethod
    def from_simulated_room(
        room_signals: np.ndarray,
        sample_rate: int = 16000,
        primary_channel: int = 0,
    ) -> Tuple[np.ndarray, int]:
        """
        Extract primary reference microphone from multi-microphone pyroomacoustics array.
        room_signals: Shape (M, N) or (N, M)
        """
        if room_signals.ndim == 1:
            return room_signals, sample_rate

        if room_signals.shape[0] < room_signals.shape[1]:
            # Shape (M, N)
            selected = room_signals[min(primary_channel, room_signals.shape[0] - 1)]
        else:
            # Shape (N, M)
            selected = room_signals[:, min(primary_channel, room_signals.shape[1] - 1)]

        return selected.astype(np.float32), sample_rate
