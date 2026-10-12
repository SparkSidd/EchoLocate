"""
EchoLocate — GCC-PHAT TDOA Estimator (v2 — fixed peak extraction)
====================================================================
Implements the Generalised Cross-Correlation with Phase Transform (GCC-PHAT)
algorithm for Time Difference of Arrival (TDOA) estimation.

Theory:
-------
Given two microphone signals x_i(t) and x_j(t), the TDOA between them is the
delay τ that maximises the cross-correlation:

    R_ij(τ) = ∫ x_i(t) · x_j(t + τ) dt

In the frequency domain:
    G_ij(f) = X_i(f) · conj(X_j(f))          ← cross-spectrum

GCC-PHAT whitens the cross-spectrum by its magnitude (phase-only):
    Ψ_ij(f) = G_ij(f) / |G_ij(f)|              ← PHAT weighting

The TDOA is then:
    τ̂_ij = argmax_τ { IFFT[ Ψ_ij(f) ] }

Why PHAT?
    The PHAT (Phase Transform) makes GCC robust to coloured noise and
    reverberation by normalising spectral magnitude. Every frequency bin
    contributes equally regardless of signal level. This results in sharper
    correlation peaks than raw cross-correlation.

Sign convention:
    τ̂_ij > 0  means signal arrives at microphone j BEFORE microphone i
               (i.e., mic j is closer to the source).
    τ̂_ij < 0  means signal arrives at microphone i BEFORE microphone j.

    This is consistent with:
        τ_ij = (d_i - d_j) / c
    where d_i = distance from source to mic i.

    Implementation note:
        IFFT of the cross-spectrum gives R_ij[k] for k = 0, 1, ..., N-1.
        This is a circular array where:
            k in [0, N/2)   → positive lags (x_j is delayed w.r.t. x_i)
                               i.e., x_i arrives first, τ_ij > 0
            k in [N/2, N)   → negative lags (x_i is delayed w.r.t. x_j)
                               i.e., x_j arrives first, τ_ij < 0
        The signed lag at index k is:
            signed_k = k if k < N/2 else k - N
        So τ̂_ij = signed_k_peak / fs.

Zero-padding:
    We zero-pad each signal to at least 2 * max(len_i, len_j) before FFT.
    This avoids circular cross-correlation artefacts (aliasing in time domain).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TDOAResult:
    """
    Result of GCC-PHAT TDOA estimation for one microphone pair.

    Attributes
    ----------
    mic_i : int
        Index of the first microphone (0-based).
    mic_j : int
        Index of the second microphone (0-based).
    tau_seconds : float
        Estimated TDOA in seconds (τ̂_ij = τ_i - τ_j).
    tau_samples : float
        Estimated TDOA in samples (may be non-integer due to interpolation).
    gcc_phat_response : np.ndarray
        The full GCC-PHAT correlation function (length = fft_size), centred
        so that index 0 corresponds to τ = 0.
    tau_axis_seconds : np.ndarray
        Time axis corresponding to gcc_phat_response, in seconds.
    fs : int
        Sampling frequency used.
    """

    mic_i: int
    mic_j: int
    tau_seconds: float
    tau_samples: float
    gcc_phat_response: np.ndarray
    tau_axis_seconds: np.ndarray
    fs: int

    @property
    def label(self) -> str:
        """Human-readable label, e.g. 'M1–M2'."""
        return f"M{self.mic_i + 1}–M{self.mic_j + 1}"


def gcc_phat(
    x_i: np.ndarray,
    x_j: np.ndarray,
    fs: int,
    fft_size: int | None = None,
    max_tau_samples: int | None = None,
    mic_i: int = 0,
    mic_j: int = 1,
) -> TDOAResult:
    """
    Estimate TDOA between two signals using GCC-PHAT.

    Parameters
    ----------
    x_i : np.ndarray
        Signal received at microphone i. Shape (n_samples,).
    x_j : np.ndarray
        Signal received at microphone j. Shape (n_samples,).
    fs : int
        Sampling frequency in Hz.
    fft_size : int or None
        FFT size (should be ≥ 2 * max(len(x_i), len(x_j)) to avoid circular
        aliasing). If None, computed automatically as next power-of-2 of
        2 * max(len(x_i), len(x_j)).
    max_tau_samples : int or None
        Restrict TDOA search to ±max_tau_samples. This avoids spurious peaks
        far outside the physically plausible range.
        If None, the full range is searched.
    mic_i : int
        Index of mic i (used for labelling only).
    mic_j : int
        Index of mic j (used for labelling only).

    Returns
    -------
    TDOAResult
    """
    n_i = len(x_i)
    n_j = len(x_j)
    n_max = max(n_i, n_j)

    # --- Choose FFT size --------------------------------------------------
    if fft_size is None:
        # At least 2× length to avoid circular aliasing
        fft_size = _next_power_of_2(2 * n_max)
    else:
        if fft_size < 2 * n_max:
            fft_size = _next_power_of_2(2 * n_max)
            logger.warning(
                "fft_size too small; increased to %d to avoid aliasing.", fft_size
            )

    # --- Compute cross-spectrum --------------------------------------------
    # X_i(f) · conj(X_j(f))
    X_i = np.fft.rfft(x_i, n=fft_size)
    X_j = np.fft.rfft(x_j, n=fft_size)
    G_ij = X_i * np.conj(X_j)

    # --- Gated PHAT weighting ---------------------------------------------
    # Standard PHAT normalises by |G_ij(f)|. However, when the magnitude is
    # uniformly small across frequency (near-identical signals, high noise),
    # the normalised phase becomes dominated by numerical noise.
    #
    # Fix: apply a signal-power gating mask. Only bins where the geometric
    # mean of signal powers exceeds a threshold (5% of peak) contribute to
    # the whitened cross-spectrum. Other bins are zeroed.
    # This is the "selective PHAT" approach described in:
    #   Benesty et al., "Microphone Array Signal Processing", Springer 2008.
    #
    magnitude = np.abs(G_ij)
    # Individual signal power spectra
    power_i = np.abs(X_i) ** 2
    power_j = np.abs(X_j) ** 2
    # Geometric mean power: sqrt(P_i * P_j) — this is proportional to |G_ij|
    # Use a gate: only include bins where power is at least gate_threshold of peak
    gate_threshold = 0.01  # 1% of peak power — low enough not to miss signal
    geom_mean_power = np.sqrt(power_i * power_j)
    max_power = np.max(geom_mean_power)
    if max_power > 0:
        gate = geom_mean_power >= (gate_threshold * max_power)
    else:
        gate = np.ones(len(G_ij), dtype=bool)

    # PHAT: normalise only gated bins; zero out others
    eps = 1e-12
    Psi_ij = np.zeros_like(G_ij, dtype=complex)
    Psi_ij[gate] = G_ij[gate] / (magnitude[gate] + eps)

    logger.debug(
        "GCC-PHAT M%d–M%d: %d/%d bins above gate threshold",
        mic_i + 1, mic_j + 1, int(np.sum(gate)), len(gate),
    )

    # --- Inverse FFT to get correlation -----------------------------------
    # R_ij[k] is the GCC-PHAT value at lag k (in circular/FFT ordering).
    # Interpretation of indices in the output:
    #   k in [0, N/2)    → lag k  (positive delay: x_j delayed w.r.t x_i → τ > 0)
    #   k in [N/2, N)    → lag (k - N)  (negative delay)
    R_ij = np.fft.irfft(Psi_ij, n=fft_size)

    # --- Build centred response and tau axis ------------------------------
    # Use fftshift to move the DC component (τ=0) to the centre.
    R_centred = np.fft.fftshift(R_ij)
    # Tau axis: from -N/2 to N/2-1 (in samples), converted to seconds
    tau_samples_axis = np.arange(-fft_size // 2, fft_size - fft_size // 2)
    tau_axis = tau_samples_axis / fs  # seconds

    # --- Apply search window ----------------------------------------------
    if max_tau_samples is not None:
        # Zero out responses outside the physically plausible range
        mask = np.abs(tau_samples_axis) <= max_tau_samples
        search_response = np.where(mask, R_centred, -np.inf)
    else:
        search_response = R_centred

    # --- Find peak --------------------------------------------------------
    peak_idx = int(np.argmax(search_response))
    tau_samples_est = float(tau_samples_axis[peak_idx])
    tau_seconds_est = tau_samples_est / fs

    logger.debug(
        "GCC-PHAT M%d–M%d: τ = %+.6f s = %+.2f samples",
        mic_i + 1, mic_j + 1, tau_seconds_est, tau_samples_est,
    )

    return TDOAResult(
        mic_i=mic_i,
        mic_j=mic_j,
        tau_seconds=tau_seconds_est,
        tau_samples=tau_samples_est,
        gcc_phat_response=R_centred,
        tau_axis_seconds=tau_axis,
        fs=fs,
    )


def estimate_all_tdoas(
    mic_signals: np.ndarray,
    fs: int,
    pairs: list[tuple[int, int]] | None = None,
    fft_size: int | None = None,
    max_tau_samples: int | None = None,
) -> dict[tuple[int, int], TDOAResult]:
    """
    Estimate TDOAs for all specified microphone pairs.

    Parameters
    ----------
    mic_signals : np.ndarray
        Shape (n_mics, n_samples). Row i = signal at mic i.
    fs : int
        Sampling frequency in Hz.
    pairs : list of (int, int) or None
        Microphone index pairs. Default: all unique pairs (i, j) with i < j.
    fft_size : int or None
        Passed to gcc_phat().
    max_tau_samples : int or None
        Passed to gcc_phat().

    Returns
    -------
    dict
        Keys: (i, j) tuples. Values: TDOAResult objects.
    """
    n_mics = mic_signals.shape[0]

    if pairs is None:
        pairs = [(i, j) for i in range(n_mics) for j in range(i + 1, n_mics)]

    results: dict[tuple[int, int], TDOAResult] = {}
    for i, j in pairs:
        result = gcc_phat(
            x_i=mic_signals[i],
            x_j=mic_signals[j],
            fs=fs,
            fft_size=fft_size,
            max_tau_samples=max_tau_samples,
            mic_i=i,
            mic_j=j,
        )
        results[(i, j)] = result
        logger.info(
            "TDOA(%d,%d): τ̂ = %+.6f s (%+.2f samples)",
            i, j, result.tau_seconds, result.tau_samples,
        )

    return results


def compute_max_tau_from_geometry(
    mic_positions_2d: np.ndarray,
    speed_of_sound: float,
    fs: int,
    safety_factor: float = 1.2,
) -> int:
    """
    Compute the maximum physically plausible TDOA in samples.

    The maximum TDOA between any pair of mics is bounded by:
        τ_max = max_baseline / c
    where max_baseline is the largest distance between any pair of mics.

    Parameters
    ----------
    mic_positions_2d : np.ndarray
        Shape (n_mics, 2).
    speed_of_sound : float
        Speed of sound in m/s.
    fs : int
        Sampling frequency in Hz.
    safety_factor : float
        Multiply computed max by this factor for a margin of safety.

    Returns
    -------
    int
        Maximum TDOA in samples (always positive).
    """
    n_mics = mic_positions_2d.shape[0]
    max_baseline = 0.0
    for i in range(n_mics):
        for j in range(i + 1, n_mics):
            dist = float(np.linalg.norm(mic_positions_2d[i] - mic_positions_2d[j]))
            if dist > max_baseline:
                max_baseline = dist

    tau_max_seconds = max_baseline / speed_of_sound
    tau_max_samples = int(np.ceil(tau_max_seconds * fs * safety_factor))
    logger.info(
        "Max baseline: %.3f m → τ_max = %.6f s = %d samples (×%.1f safety)",
        max_baseline, tau_max_seconds, tau_max_samples, safety_factor,
    )
    return tau_max_samples


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 that is ≥ n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()
