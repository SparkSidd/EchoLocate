"""
EchoLocate — Source Signal Generator
======================================
Generates acoustic source signals suitable for TDOA estimation experiments.

Design rationale:
    The source signal choice critically affects TDOA estimation quality.

    - WHITE NOISE: broadband but lacks determinism; each realisation is
      different, making experiment reproducibility harder. Still usable.

    - PURE TONE (sine): narrow-band; GCC-PHAT may produce multiple peaks
      (period ambiguity). Bad for TDOA experiments.

    - DIRAC DELTA (ideal impulse): mathematically perfect for measuring
      arrival times, but physically unrealisable and ill-conditioned for
      FFT-based cross-correlation with any noise.

    - CHIRP (linear frequency sweep): sweeps from f_low to f_high across
      a short duration. Produces:
        * broad spectral coverage → GCC-PHAT operates on many frequencies
        * deterministic → reproducible across runs
        * compact time-domain envelope → clear first-arrival peaks
        * high time-bandwidth product → good cross-correlation sharpness
      CHOSEN as the primary signal for Experiment 1.

    - GAUSSIAN PULSE: similar to chirp; very compact, smooth spectrum.
      Provided as alternative.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import chirp as scipy_chirp


def generate_chirp(
    fs: int,
    duration: float,
    f_low: float,
    f_high: float,
    amplitude: float = 1.0,
    pad_before: float = 0.0,
    pad_after: float = 0.0,
    method: str = "linear",
) -> np.ndarray:
    """
    Generate a linear (or logarithmic) chirp signal with optional silence padding.

    A chirp sweeps from f_low to f_high over `duration` seconds.
    The returned array is:
        [silence pad_before] + [chirp] + [silence pad_after]

    Parameters
    ----------
    fs : int
        Sampling frequency in Hz.
    duration : float
        Duration of the chirp itself (seconds, not counting padding).
    f_low : float
        Starting frequency of the sweep (Hz).
    f_high : float
        Ending frequency of the sweep (Hz).
    amplitude : float
        Peak amplitude (dimensionless, default 1.0).
    pad_before : float
        Silence before the chirp in seconds (allows wavefront to propagate
        before hitting microphones with wrap-around artefacts).
    pad_after : float
        Silence after the chirp in seconds (captures reverb tail).
    method : str
        Chirp method passed to scipy.signal.chirp: 'linear', 'logarithmic'.

    Returns
    -------
    np.ndarray
        1-D float64 signal array with shape (n_samples,).
    """
    n_chirp = int(duration * fs)
    t = np.linspace(0.0, duration, n_chirp, endpoint=False)

    # scipy.signal.chirp generates a unit-amplitude frequency sweep
    chirp_signal = amplitude * scipy_chirp(t, f0=f_low, f1=f_high, t1=duration, method=method)

    # Build padding arrays
    n_before = int(pad_before * fs)
    n_after = int(pad_after * fs)
    silence_before = np.zeros(n_before, dtype=np.float64)
    silence_after = np.zeros(n_after, dtype=np.float64)

    return np.concatenate([silence_before, chirp_signal.astype(np.float64), silence_after])


def generate_gaussian_pulse(
    fs: int,
    centre: float = 0.0,
    bw: float = 0.5,
    bwr: float = -6.0,
    tpr: float = -60.0,
    amplitude: float = 1.0,
    total_duration: float = 0.1,
    pad_before: float = 0.01,
) -> np.ndarray:
    """
    Generate a Gaussian-modulated sinusoidal pulse (Ricker wavelet variant).

    Useful as an alternative to the chirp for TDOA sanity checks.

    Parameters
    ----------
    fs : int
        Sampling frequency in Hz.
    centre : float
        Centre frequency in normalised units (fraction of Nyquist, 0–1).
        0.5 means Nyquist/2 = fs/4.
    bw : float
        Fractional bandwidth (default 0.5 = 50%).
    bwr : float
        Reference level for fractional bandwidth (dB, default -6 dB).
    tpr : float
        Truncation level (dB, default -60 dB).
    amplitude : float
        Peak amplitude.
    total_duration : float
        Total signal duration in seconds.
    pad_before : float
        Silence before the pulse.

    Returns
    -------
    np.ndarray
        1-D float64 signal array.
    """
    from scipy.signal import gausspulse

    n_total = int(total_duration * fs)
    n_before = int(pad_before * fs)

    # Centre the pulse at pad_before + half_remaining
    half_pulse_dur = (total_duration - pad_before) / 2.0
    t = np.linspace(-half_pulse_dur, half_pulse_dur, n_total - n_before, endpoint=False)
    pulse, _ = gausspulse(t, fc=centre * fs / 2, bw=bw, bwr=bwr, tpr=tpr, retquad=False, retenv=False)

    silence = np.zeros(n_before, dtype=np.float64)
    return np.concatenate([silence, (amplitude * pulse).astype(np.float64)])


def generate_click(
    fs: int,
    pad_before: float = 0.01,
    total_duration: float = 0.1,
) -> np.ndarray:
    """
    Generate a single-sample impulse (Dirac delta approximation).

    Useful for verifying geometry: the cross-correlation of two click
    signals received at different times should have a single sharp peak.

    Parameters
    ----------
    fs : int
        Sampling frequency in Hz.
    pad_before : float
        Silence before the click (seconds).
    total_duration : float
        Total signal duration (seconds).

    Returns
    -------
    np.ndarray
        1-D float64 signal array with a single 1.0 at sample index n_before.
    """
    n_total = int(total_duration * fs)
    n_before = int(pad_before * fs)
    signal = np.zeros(n_total, dtype=np.float64)
    if n_before < n_total:
        signal[n_before] = 1.0
    return signal


def generate_source_signal(config: dict) -> tuple[np.ndarray, int]:
    """
    Factory function: generate a source signal from the experiment config dict.

    Parameters
    ----------
    config : dict
        The 'source' section of the experiment YAML config, plus 'room.sampling_rate'.

    Returns
    -------
    signal : np.ndarray
        1-D float64 source signal.
    fs : int
        Sampling rate in Hz.
    """
    fs: int = config["sampling_rate"]
    src_cfg = config["source"]
    signal_type: str = src_cfg.get("type", "chirp").lower()

    if signal_type == "chirp":
        signal = generate_chirp(
            fs=fs,
            duration=src_cfg["duration"],
            f_low=src_cfg["f_low"],
            f_high=src_cfg["f_high"],
            amplitude=src_cfg.get("amplitude", 1.0),
            pad_before=src_cfg.get("pad_before", 0.01),
            pad_after=src_cfg.get("pad_after", 0.1),
        )
    elif signal_type == "click":
        signal = generate_click(
            fs=fs,
            pad_before=src_cfg.get("pad_before", 0.01),
            total_duration=src_cfg.get("pad_before", 0.01) + src_cfg.get("pad_after", 0.1),
        )
    elif signal_type == "gaussian_pulse":
        signal = generate_gaussian_pulse(
            fs=fs,
            amplitude=src_cfg.get("amplitude", 1.0),
            total_duration=src_cfg.get("duration", 0.05) + src_cfg.get("pad_after", 0.1),
            pad_before=src_cfg.get("pad_before", 0.01),
        )
    else:
        raise ValueError(f"Unknown source type: '{signal_type}'. Choose from: chirp, click, gaussian_pulse")

    return signal, fs
