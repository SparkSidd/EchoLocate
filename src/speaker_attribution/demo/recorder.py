"""
EchoLocate — Microphone and Audio Input Recorder
================================================
Captures live audio from laptop microphone via sounddevice or loads WAV files
using AudioSourceAdapter.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np

from ..adapters.audio_source_adapter import AudioSourceAdapter
from ..preprocessing.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)


class MicrophoneRecorder:
    """
    Manages live microphone audio acquisition using sounddevice.
    """

    def __init__(self, sample_rate: int = 16000, device: Optional[Union[int, str]] = None):
        self.sample_rate = sample_rate
        self.device = device
        self.processor = AudioProcessor(target_sample_rate=sample_rate)

    @staticmethod
    def is_available() -> bool:
        """Check if sounddevice can access an input device."""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            # Look for at least one input channel
            for d in devices:
                if d.get("max_input_channels", 0) > 0:
                    return True
            return False
        except Exception:
            return False

    def record_live(
        self,
        duration_sec: float = 3.0,
        prompt: Optional[str] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Record `duration_sec` seconds from microphone synchronously.
        Prints a text-based VU meter if `show_progress` is True.
        """
        import sounddevice as sd

        if prompt:
            print(prompt)

        num_samples = int(duration_sec * self.sample_rate)
        buffer = np.zeros(num_samples, dtype=np.float32)

        # Countdown
        for count in range(3, 0, -1):
            sys.stdout.write(f"\rStarting in {count}... ")
            sys.stdout.flush()
            time.sleep(0.4)
        sys.stdout.write("\r[RECORDING NOW] Speak clearly! 🎙️            \n")
        sys.stdout.flush()

        recorded_chunks = []
        chunk_duration = 0.1  # 100 ms chunks
        chunk_samples = int(chunk_duration * self.sample_rate)

        start_time = time.time()
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.device,
            blocksize=chunk_samples,
        ) as stream:
            for _ in range(int(duration_sec / chunk_duration)):
                data, overflowed = stream.read(chunk_samples)
                recorded_chunks.append(data.flatten())

                if show_progress:
                    rms = float(np.sqrt(np.mean(data ** 2)))
                    bar_len = int(min(30, rms * 150))
                    bar = "█" * bar_len + "░" * (30 - bar_len)
                    elapsed = time.time() - start_time
                    sys.stdout.write(f"\r  [{elapsed:.1f}s / {duration_sec:.1f}s] [{bar}] RMS: {rms:.3f}")
                    sys.stdout.flush()

        sys.stdout.write("\nRecording complete.\n")
        sys.stdout.flush()

        raw_audio = np.concatenate(recorded_chunks, axis=0)
        # Apply standard preprocessing: DC removal, peak normalization
        processed = self.processor.process_array(raw_audio, self.sample_rate)
        return processed.audio

    def load_wav_file(self, wav_path: Union[str, Path]) -> np.ndarray:
        """Load audio from WAV file via AudioSourceAdapter and normalize to 16 kHz."""
        path = Path(wav_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        return AudioSourceAdapter.from_wav_file(path, self.processor)
