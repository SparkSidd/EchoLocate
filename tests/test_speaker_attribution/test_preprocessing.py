"""
EchoLocate — Tests: Speaker Attribution Preprocessing & Features
===============================================================
Validates audio standardization, polyphase resampling, Fbank feature extraction, and VAD.
"""

import numpy as np
import pytest

from src.speaker_attribution.preprocessing.audio_processor import AudioProcessor
from src.speaker_attribution.preprocessing.features import compute_fbank, create_mel_filterbank
from src.speaker_attribution.preprocessing.vad import EnergyVAD


class TestAudioProcessor:
    def test_mono_conversion(self):
        proc = AudioProcessor(target_sample_rate=16000)
        # Stereo audio (2, 16000)
        stereo = np.ones((2, 16000), dtype=np.float32)
        stereo[0] *= 0.5
        stereo[1] *= 1.0

        avg_mono = proc.convert_to_mono(stereo, policy="average")
        assert avg_mono.ndim == 1
        assert len(avg_mono) == 16000
        assert np.isclose(np.mean(avg_mono), 0.75)

        first_mono = proc.convert_to_mono(stereo, policy="first")
        assert np.isclose(np.mean(first_mono), 0.5)

    def test_resampling(self):
        proc = AudioProcessor(target_sample_rate=16000)
        # 1 second of 48 kHz audio
        audio_48k = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, 48000)).astype(np.float32)
        resampled = proc.resample(audio_48k, orig_sr=48000)
        assert len(resampled) == 16000

    def test_dc_removal_and_normalization(self):
        proc = AudioProcessor(target_sample_rate=16000, remove_dc=True, normalize_amplitude=True, target_peak=0.95)
        # Signal with constant DC offset +0.5
        audio = np.sin(2 * np.pi * 300 * np.linspace(0, 0.5, 8000)) + 0.5
        seg = proc.process(audio, sample_rate=16000)

        assert seg.samples.ndim == 1
        assert np.abs(np.mean(seg.samples)) < 1e-4
        assert np.isclose(np.max(np.abs(seg.samples)), 0.95, atol=1e-2)

    def test_short_duration_validation(self):
        proc = AudioProcessor(min_duration_sec=0.4)
        short_audio = np.zeros(1600, dtype=np.float32)  # 0.1s
        seg = proc.process(short_audio, sample_rate=16000)
        assert not seg.metadata["is_duration_valid"]


class TestFeatureExtraction:
    def test_mel_filterbank_shape(self):
        fb = create_mel_filterbank(num_filters=80, n_fft=512, sample_rate=16000)
        assert fb.shape == (80, 257)
        assert np.all(fb >= 0.0)

    def test_fbank_computation(self):
        # 1 second audio at 16 kHz
        audio = np.sin(2 * np.pi * 500 * np.linspace(0, 1.0, 16000)).astype(np.float32)
        feats = compute_fbank(audio, sample_rate=16000, num_filters=80, apply_cmvn=True)

        assert feats.ndim == 2
        assert feats.shape[1] == 80
        # Expected frames: (16000 - 400) / 160 + 1 = 98 frames
        assert feats.shape[0] >= 90
        # CMVN should make mean ~0 and std ~1 across time
        assert np.abs(np.mean(feats)) < 0.05


class TestEnergyVAD:
    def test_vad_speech_vs_silence(self):
        vad = EnergyVAD()
        # 1. Pure silence
        silence = np.zeros(16000, dtype=np.float32)
        mask_sil = vad.detect(silence, sample_rate=16000)
        assert not np.any(mask_sil)

        # 2. Tone with silence prefix
        signal = np.zeros(32000, dtype=np.float32)
        # Put speech in middle 1 second (16000:32000)
        signal[16000:32000] = 0.5 * np.sin(2 * np.pi * 400 * np.linspace(0, 1.0, 16000))
        segments = vad.get_speech_segments(signal, sample_rate=16000, min_speech_duration_sec=0.3)

        assert len(segments) >= 1
        seg_start, seg_end = segments[0]
        assert seg_start >= 0.8  # Around 1.0s
        assert seg_end <= 2.1
