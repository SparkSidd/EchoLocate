"""
EchoLocate — Speaker Audio Feature Extraction
=============================================
Deterministic 80-dimensional log Mel-filterbank feature extraction,
aligned with standard 3D-Speaker / Kaldi / torchaudio conventions.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import get_window


def hz_to_mel(hz: float | np.ndarray) -> float | np.ndarray:
    """Convert frequency in Hz to Mel scale (HTK formula)."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
    """Convert Mel scale to frequency in Hz (HTK formula)."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def create_mel_filterbank(
    num_filters: int = 80,
    n_fft: int = 512,
    sample_rate: int = 16000,
    low_freq: float = 20.0,
    high_freq: float = 7600.0,
) -> np.ndarray:
    """
    Construct a triangular Mel filterbank matrix.
    Returns:
        np.ndarray of shape (num_filters, n_fft // 2 + 1)
    """
    if high_freq is None or high_freq <= 0:
        high_freq = sample_rate / 2.0

    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = mel_to_hz(mel_points)

    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    num_bins = n_fft // 2 + 1
    filterbank = np.zeros((num_filters, num_bins), dtype=np.float32)

    for i in range(1, num_filters + 1):
        left = bin_points[i - 1]
        center = bin_points[i]
        right = bin_points[i + 1]

        if center > left:
            filterbank[i - 1, left:center] = (
                np.arange(left, center) - left
            ) / (center - left)
        if right > center:
            filterbank[i - 1, center:right] = (
                right - np.arange(center, right)
            ) / (right - center)

    return filterbank


def compute_kaldi_fbank(
    signal: np.ndarray,
    sample_rate: int = 16000,
    num_filters: int = 80,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
    dither: float = 0.0,
    apply_cmn: bool = True,
) -> np.ndarray:
    """
    Official 3D-Speaker Kaldi-compatible 80-dimensional log-Mel filterbank extraction.
    Matches speakerlab.process.processor.FBank with mean_nor=True:
        1. 80-bin FBank at 16kHz, 25ms window, 10ms frame shift, dither=0.
        2. Cepstral Mean Normalization (CMN): feat = feat - feat.mean(dim=0, keepdim=True).
        Does NOT divide by standard deviation.
    """
    if signal is None or len(signal) == 0:
        return np.zeros((0, num_filters), dtype=np.float32)

    signal = signal.astype(np.float32)
    if signal.ndim > 1:
        signal = np.mean(signal, axis=-1)

    try:
        import torch
        import torchaudio.compliance.kaldi as Kaldi
        t_wav = torch.from_numpy(signal).float().unsqueeze(0)
        feat = Kaldi.fbank(
            t_wav,
            num_mel_bins=num_filters,
            sample_frequency=sample_rate,
            frame_length=frame_length_ms,
            frame_shift=frame_shift_ms,
            dither=dither,
            window_type="povey",
            use_energy=False,
            energy_floor=0.0,
            raw_energy=True,
            preemphasis_coefficient=0.97,
            low_freq=20.0,
            high_freq=0.0,
        )
        if apply_cmn and feat.shape[0] > 0:
            feat = feat - feat.mean(dim=0, keepdim=True)
        return feat.cpu().numpy().astype(np.float32)
    except Exception:
        # Pure numpy fallback with CMN (mean normalization only)
        return compute_fbank(
            signal,
            sample_rate=sample_rate,
            num_filters=num_filters,
            frame_length_ms=frame_length_ms,
            frame_shift_ms=frame_shift_ms,
            apply_cmvn=apply_cmn,
            cmn_only=True,
        )


def compute_fbank(
    signal: np.ndarray,
    sample_rate: int = 16000,
    num_filters: int = 80,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
    n_fft: int = 512,
    preemphasis: float = 0.97,
    low_freq: float = 20.0,
    high_freq: float = 7600.0,
    apply_cmvn: bool = True,
    cmn_only: bool = False,
) -> np.ndarray:
    """
    Compute log Mel-filterbank energy features from raw mono PCM audio.
    
    Args:
        signal: 1D numpy array of shape (N,)
        sample_rate: Sample rate in Hz (default: 16000)
        num_filters: Number of Mel bins (default: 80)
        frame_length_ms: Analysis window length in ms (default: 25.0)
        frame_shift_ms: Analysis hop size in ms (default: 10.0)
        n_fft: FFT length (default: 512)
        preemphasis: Pre-emphasis coefficient (default: 0.97)
        low_freq: Lower cut-off frequency in Hz (default: 20.0)
        high_freq: Upper cut-off frequency in Hz (default: 7600.0)
        apply_cmvn: Whether to apply Cepstral Mean & Variance Normalization
        cmn_only: If True, applies only mean subtraction (no std division)

    Returns:
        np.ndarray of shape (num_frames, num_filters), dtype float32
    """
    signal = signal.astype(np.float32)
    if len(signal) == 0:
        return np.zeros((0, num_filters), dtype=np.float32)

    # 1. Pre-emphasis
    if preemphasis > 0:
        signal = np.append(signal[0], signal[1:] - preemphasis * signal[:-1])

    # 2. Framing
    frame_len = int(round(sample_rate * frame_length_ms / 1000.0))
    frame_shift = int(round(sample_rate * frame_shift_ms / 1000.0))

    if len(signal) < frame_len:
        # Pad with zero to at least one frame
        pad_width = frame_len - len(signal)
        signal = np.pad(signal, (0, pad_width), mode="constant")

    num_frames = 1 + int(np.floor((len(signal) - frame_len) / frame_shift))
    shape = (num_frames, frame_len)
    strides = (signal.strides[0] * frame_shift, signal.strides[0])
    frames = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)

    # 3. Windowing (Hamming)
    window = get_window("hamming", frame_len, fftbins=True).astype(np.float32)
    windowed_frames = frames * window

    # 4. Magnitude FFT
    spectrum = np.fft.rfft(windowed_frames, n=n_fft, axis=-1)
    power_spectrum = (np.abs(spectrum) ** 2) / frame_len

    # 5. Filterbank multiplication
    fbank_matrix = create_mel_filterbank(
        num_filters=num_filters,
        n_fft=n_fft,
        sample_rate=sample_rate,
        low_freq=low_freq,
        high_freq=high_freq,
    )
    mel_energy = np.dot(power_spectrum, fbank_matrix.T)
    mel_energy = np.maximum(mel_energy, 1e-6)
    log_mel = np.log(mel_energy).astype(np.float32)

    # 6. Cepstral Normalization
    if apply_cmvn and len(log_mel) > 0:
        mean = np.mean(log_mel, axis=0, keepdims=True)
        if cmn_only:
            log_mel = log_mel - mean
        else:
            std = np.std(log_mel, axis=0, keepdims=True) + 1e-5
            log_mel = (log_mel - mean) / std

    return log_mel

