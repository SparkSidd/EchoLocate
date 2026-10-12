"""
EchoLocate — Synthetic Acoustic Speech & Environment Generator
==============================================================
Generates deterministic multi-speaker acoustic events with known ground truth
for evaluating speaker identification, UNKNOWN rejection, noise, and reverb.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import fftconvolve


class AcousticSpeakerGenerator:
    """
    Generates synthetic speech-like acoustic signals with distinct speaker formant
    signatures, pitch modulations, room impulse responses, and background noise.
    """

    def __init__(self, sample_rate: int = 16000, seed: int = 42):
        self.sample_rate = sample_rate
        self.rng = np.random.default_rng(seed)

        # Baseline speaker acoustic definitions (f0 base in Hz, formant centers)
        self.speaker_specs: Dict[str, Dict[str, Any]] = {
            "Siddharth": {"f0": 115.0, "formants": [550, 1600, 2450], "vibrato_rate": 5.0},
            "Alice": {"f0": 210.0, "formants": [750, 1950, 2800], "vibrato_rate": 5.5},
            "Bob": {"f0": 95.0, "formants": [450, 1400, 2250], "vibrato_rate": 4.5},
            "Charlie": {"f0": 160.0, "formants": [650, 1750, 2600], "vibrato_rate": 5.2},
            "Impostor_Eve": {"f0": 185.0, "formants": [700, 1850, 2700], "vibrato_rate": 5.8},
            "Impostor_Mallory": {"f0": 130.0, "formants": [500, 1500, 2350], "vibrato_rate": 4.8},
        }

    def generate_utterance(
        self,
        speaker_id: str,
        duration_sec: float = 2.0,
        utterance_seed: Optional[int] = None,
        snr_db: Optional[float] = None,
        rt60_sec: float = 0.0,
        distance_m: float = 1.0,
    ) -> np.ndarray:
        """
        Generate a synthetic speech utterance for a specific speaker.
        """
        rng = np.random.default_rng(utterance_seed) if utterance_seed is not None else self.rng
        spec = self.speaker_specs.get(
            speaker_id,
            {"f0": 150.0, "formants": [600, 1700, 2500], "vibrato_rate": 5.0},
        )

        n_samples = int(self.sample_rate * duration_sec)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)

        # 1. Pitch track with natural intonation contour & vibrato
        f0_base = spec["f0"]
        vibrato = 3.0 * np.sin(2 * np.pi * spec["vibrato_rate"] * t)
        # Slow pitch glide
        intonation = 15.0 * np.sin(2 * np.pi * 0.8 * t)
        f0_track = f0_base + vibrato + intonation

        # 2. Glottal source excitation (harmonic series)
        phase = 2 * np.pi * np.cumsum(f0_track) / self.sample_rate
        glottal = np.zeros(n_samples, dtype=np.float32)
        num_harmonics = min(25, int((self.sample_rate / 2) / (f0_base + 30)))

        for h in range(1, num_harmonics + 1):
            # Spectral roll-off: 1 / h^1.2
            amplitude = 1.0 / (h ** 1.2)
            # Add slight jitter
            jitter = 1.0 + 0.05 * rng.standard_normal()
            glottal += (amplitude * jitter * np.sin(h * phase)).astype(np.float32)

        # 3. Vocal tract formant filtering (resonant second-order filters)
        signal = glottal
        for fc in spec["formants"]:
            # Resonator at fc
            q = 10.0
            bw = fc / q
            r = math.exp(-math.pi * bw / self.sample_rate)
            theta = 2 * math.pi * fc / self.sample_rate
            # 2nd order IIR resonator difference eq
            b0 = 1.0 - r
            a1 = -2.0 * r * math.cos(theta)
            a2 = r * r
            filtered = np.zeros_like(signal)
            for i in range(2, len(signal)):
                filtered[i] = b0 * signal[i] - a1 * filtered[i - 1] - a2 * filtered[i - 2]
            signal = signal + 0.6 * filtered

        # 4. Syllabic amplitude envelope (modulation at 3-4 Hz speaking rate)
        syllables = np.clip(np.sin(2 * np.pi * 3.5 * t) + 0.3 * np.sin(2 * np.pi * 7.0 * t), 0, 1)
        envelope = 0.2 + 0.8 * (syllables ** 1.5)
        signal = signal * envelope

        # 5. Distance attenuation (inverse square root of distance)
        dist_factor = 1.0 / max(0.5, math.sqrt(distance_m))
        signal = signal * dist_factor

        # 6. Reverberation (exponential decay impulse response)
        if rt60_sec > 0.02:
            rir_len = int(self.sample_rate * min(rt60_sec, 1.0))
            t_rir = np.linspace(0, rir_len / self.sample_rate, rir_len)
            # RT60 is the time to drop 60 dB (decay = ln(1000) / rt60)
            decay_rate = 6.908 / max(0.05, rt60_sec)
            decay_env = np.exp(-decay_rate * t_rir)
            # Direct path + reflection reflections
            rir = rng.standard_normal(rir_len).astype(np.float32) * decay_env
            rir[0] = 1.0  # Direct arrival
            reverbed = fftconvolve(signal, rir, mode="full")[:n_samples]
            signal = reverbed.astype(np.float32)

        # 7. Additive noise
        if snr_db is not None:
            sig_power = np.mean(signal ** 2) + 1e-12
            noise_power = sig_power / (10.0 ** (snr_db / 10.0))
            noise = np.sqrt(noise_power) * rng.standard_normal(n_samples).astype(np.float32)
            signal = signal + noise

        # Peak normalize to 0.9
        peak = np.max(np.abs(signal))
        if peak > 1e-6:
            signal = (signal / peak) * 0.9

        return signal.astype(np.float32)
