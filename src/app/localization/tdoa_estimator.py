"""
EchoLocate — Rigorous TDOA + ILD Acoustic Physics Engine
=========================================================
Extracts sub-millisecond Time Difference of Arrival (TDOA) and Interaural Level Difference (ILD):
  1. 5-point Windowed-Sinc sub-sample peak refinement (Nyquist-Shannon optimal)
  2. Inverse-Square Law Interaural Level Difference (ILD) Energy Prior:
       Delta L = RMS_lap(dB) - RMS_phone(dB)
  3. Physical Bayesian Fusion of Phase Delay (GCC-PHAT) and Intensity Ratio (ILD)
  4. Temperature-dependent speed of sound: c(T) = 331.3 * sqrt(1 + T / 273.15)
  5. Cramér-Rao Lower Bound (CRLB) theoretical uncertainty: sigma_TDOA = 1 / (2*pi * BW_eff * sqrt(SNR))
  6. 1D Adaptive Kalman Filter tracking with physical source velocity bounds
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def speed_of_sound_at_temp(temp_celsius: float = 22.0) -> float:
    """
    Calculate ISO 9613-1 compliant speed of sound in dry air at atmospheric pressure:
        c(T) = 331.3 * sqrt(1 + T / 273.15) [m/s]
    """
    t_kelvin = max(0.0, temp_celsius + 273.15)
    return float(331.3 * math.sqrt(t_kelvin / 273.15))


@dataclass
class SignalQualityReport:
    """Acoustic signal quality and localization suitability assessment."""
    signal_type: str
    bandwidth_score_pct: float
    effective_bandwidth_hz: float
    snr_estimate_db: float
    spectral_flatness: float
    ild_db: float                      # Interaural Level Difference (Laptop dB - Phone dB)
    is_favorable_for_tdoa: bool
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_type": self.signal_type,
            "bandwidth_score_pct": round(self.bandwidth_score_pct, 1),
            "effective_bandwidth_hz": round(self.effective_bandwidth_hz, 1),
            "snr_estimate_db": round(self.snr_estimate_db, 1),
            "spectral_flatness": round(self.spectral_flatness, 3),
            "ild_db": round(self.ild_db, 1),
            "is_favorable_for_tdoa": self.is_favorable_for_tdoa,
            "explanation": self.explanation,
        }


@dataclass
class TDOAEstimate:
    """Rigorous physical TDOA estimation with uncertainty and Kalman-filtered trajectory."""
    tdoa_ms: float                     # Kalman-filtered estimated TDOA in ms
    raw_tdoa_ms: float                 # Unfiltered sub-sample TDOA from sinc interpolation
    sigma_tdoa_ms: float               # Cramér-Rao Lower Bound standard deviation (1-sigma error) in ms
    distance_diff_m: float             # Delta d = c * Delta t in meters
    direction_label: str               # Human-readable arrival direction
    confidence_pct: float              # 0 - 100% confidence metric derived from CRLB & PSR
    peak_psr: float                    # Peak-to-Secondary Ratio (PHAT prominence)
    signal_quality: SignalQualityReport
    angle_of_arrival_deg: float        # AoA relative to baseline normal in degrees
    estimated_xy: tuple[float, float]  # Estimated 2D source coordinate (meters)
    temperature_c: float = 22.0        # Ambient temperature used for c(T)
    speed_of_sound_mps: float = 344.4  # Actual speed of sound used

    def to_dict(self) -> dict[str, Any]:
        return {
            "tdoa_ms": round(self.tdoa_ms, 4),
            "raw_tdoa_ms": round(self.raw_tdoa_ms, 4),
            "sigma_tdoa_ms": round(self.sigma_tdoa_ms, 4),
            "distance_diff_m": round(self.distance_diff_m, 4),
            "direction_label": self.direction_label,
            "confidence_pct": round(self.confidence_pct, 1),
            "peak_psr": round(self.peak_psr, 3),
            "signal_quality": self.signal_quality.to_dict(),
            "angle_of_arrival_deg": round(self.angle_of_arrival_deg, 2),
            "estimated_xy": [round(self.estimated_xy[0], 3), round(self.estimated_xy[1], 3)],
            "temperature_c": round(self.temperature_c, 1),
            "speed_of_sound_mps": round(self.speed_of_sound_mps, 2),
        }


# ---------------------------------------------------------------------------
# 1D Kalman Filter for TDOA Tracking
# ---------------------------------------------------------------------------

class TDOAKalmanFilter:
    """
    1D Constant-Velocity Kalman Filter tracking true acoustic delay (tau).
    State vector: x = [tau (ms), d_tau/dt (ms/s)]^T
    """

    def __init__(self, max_source_velocity_mps: float = 3.0, speed_of_sound: float = 344.4) -> None:
        self.c = speed_of_sound
        self.max_dtau_dt = (max_source_velocity_mps / self.c) * 1000.0
        self.x = np.array([0.0, 0.0], dtype=np.float64)
        self.P = np.array([[5.0, 0.0], [0.0, 5.0]], dtype=np.float64)
        self.last_update_time: float | None = None
        self.initialized = False

    def reset(self) -> None:
        self.x = np.array([0.0, 0.0], dtype=np.float64)
        self.P = np.array([[5.0, 0.0], [0.0, 5.0]], dtype=np.float64)
        self.last_update_time = None
        self.initialized = False

    def update(self, z_measured_ms: float, sigma_meas_ms: float, timestamp: float | None = None) -> tuple[float, float]:
        import time
        t = timestamp if timestamp is not None else time.time()
        
        if not self.initialized or self.last_update_time is None:
            self.x = np.array([z_measured_ms, 0.0], dtype=np.float64)
            self.P = np.array([[max(0.01, sigma_meas_ms**2), 0.0], [0.0, 1.0]], dtype=np.float64)
            self.last_update_time = t
            self.initialized = True
            return float(self.x[0]), float(math.sqrt(self.P[0, 0]))

        dt = max(0.01, min(0.4, t - self.last_update_time))
        self.last_update_time = t

        F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        q_accel_ms = (3.0 / self.c) * 1000.0
        G = np.array([[0.5 * dt**2], [dt]], dtype=np.float64)
        Q = (G @ G.T) * (q_accel_ms**2)

        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q

        H = np.array([[1.0, 0.0]], dtype=np.float64)
        R = np.array([[max(1e-4, sigma_meas_ms**2)]], dtype=np.float64)

        y = z_measured_ms - (H @ x_pred)[0]
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        self.x = x_pred + (K * y).flatten()
        self.P = (np.eye(2) - K @ H) @ P_pred
        self.x[1] = np.clip(self.x[1], -self.max_dtau_dt, self.max_dtau_dt)

        return float(self.x[0]), float(math.sqrt(max(1e-6, self.P[0, 0])))


_kalman_filter = TDOAKalmanFilter()


def reset_kalman() -> None:
    global _kalman_filter
    _kalman_filter.reset()


# ---------------------------------------------------------------------------
# Sinc Sub-sample Peak Interpolation
# ---------------------------------------------------------------------------

def _sinc_subsample_peak(correlation: np.ndarray, peak_idx: int, tau_axis: np.ndarray) -> float:
    n = len(correlation)
    if peak_idx <= 1 or peak_idx >= n - 2:
        return float(tau_axis[peak_idx])

    k_local = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
    y_local = correlation[peak_idx - 2 : peak_idx + 3].astype(np.float64)

    deltas = np.linspace(-0.5, 0.5, 101)
    diffs = deltas[:, None] - k_local[None, :]
    sinc_matrix = np.sinc(diffs)
    interp_curve = sinc_matrix @ y_local
    best_sub_idx = int(np.argmax(interp_curve))
    optimal_delta = float(deltas[best_sub_idx])

    dt_step = float(tau_axis[1] - tau_axis[0]) if len(tau_axis) > 1 else 0.0625
    refined_tau_ms = float(tau_axis[peak_idx]) + optimal_delta * dt_step
    return refined_tau_ms


# ---------------------------------------------------------------------------
# Signal Quality & ILD Computation
# ---------------------------------------------------------------------------

def assess_signal_quality(
    signal_lap: np.ndarray,
    signal_phone: np.ndarray | None = None,
    gain_offset_db: float = 0.0,
    fs: int = 16000,
) -> SignalQualityReport:
    n = len(signal_lap)
    if n < 64:
        return SignalQualityReport("No Signal", 0.0, 100.0, -60.0, 0.0, 0.0, False, "Insufficient samples.")

    spec = np.abs(np.fft.rfft(signal_lap))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    eps = 1e-12

    power_spec = spec**2
    total_power = np.sum(power_spec) + eps
    f_mean_sq = np.sum((freqs**2) * power_spec) / total_power
    bw_effective = math.sqrt(max(100.0, f_mean_sq))

    threshold = spec.max() * 0.08
    occupied = np.sum(spec > threshold) / max(1, len(spec))
    occ_bw_hz = occupied * (fs / 2.0)

    spec_clipped = np.clip(spec, eps, None)
    gm = np.exp(np.mean(np.log(spec_clipped)))
    am = np.mean(spec_clipped)
    flatness = float(gm / max(am, eps))

    speech_mask = (freqs >= 250) & (freqs <= 3800)
    noise_mask = (freqs >= 30) & (freqs < 180)
    sig_rms = float(np.sqrt(np.mean(spec[speech_mask] ** 2) + eps))
    noi_rms = float(np.sqrt(np.mean(spec[noise_mask] ** 2) + eps))
    snr_db = float(20 * np.log10(sig_rms / noi_rms))

    # Interaural Level Difference (ILD) with Hardware Gain Offset Calibration
    lap_rms = float(np.sqrt(np.mean(signal_lap ** 2))) + eps
    lap_db = float(20.0 * np.log10(lap_rms))
    if signal_phone is not None and len(signal_phone) > 0:
        phone_rms = float(np.sqrt(np.mean(signal_phone ** 2))) + eps
        phone_raw_db = float(20.0 * np.log10(phone_rms))
        # Calibrated Phone dB = Raw Phone dB + Gain Calibration Offset
        phone_calibrated_db = phone_raw_db + gain_offset_db
        ild_db = float(lap_db - phone_calibrated_db)
    else:
        ild_db = 0.0

    if flatness > 0.10 and occ_bw_hz > 2000:
        sig_type, bw_score, fav = "Broadband / Music", 92.0, True
        desc = f"Broadband acoustic spectrum (ILD: {ild_db:+.1f} dB)."
    elif flatness > 0.05 and occ_bw_hz > 1000:
        sig_type, bw_score, fav = "Speech / Formants", 75.0, True
        desc = f"Acoustic speech formants (ILD: {ild_db:+.1f} dB)."
    elif snr_db < 2.0:
        sig_type, bw_score, fav = "Noise Floor", 10.0, False
        desc = "Audio power is close to sensor noise floor."
    else:
        sig_type, bw_score, fav = "General Acoustic", 60.0, True
        desc = f"Acoustic signal (ILD: {ild_db:+.1f} dB)."

    return SignalQualityReport(
        signal_type=sig_type,
        bandwidth_score_pct=bw_score,
        effective_bandwidth_hz=bw_effective,
        snr_estimate_db=snr_db,
        spectral_flatness=flatness,
        ild_db=ild_db,
        is_favorable_for_tdoa=fav,
        explanation=desc,
    )


# ---------------------------------------------------------------------------
# Master TDOA & Geometric Estimation Function
# ---------------------------------------------------------------------------

def estimate_tdoa_from_gcc(
    tau_axis_ms: np.ndarray,
    correlation_curve: np.ndarray,
    signal_laptop: np.ndarray,
    signal_phone: np.ndarray | None = None,
    mic1_xy: tuple[float, float] = (2.0, 2.0),
    mic2_xy: tuple[float, float] = (4.0, 2.0),
    temperature_c: float = 22.0,
    gain_offset_db: float = 0.0,
    fs: int = 16000,
) -> TDOAEstimate:
    """
    Fused TDOA + ILD physical estimation.
    Uses phase correlation (GCC-PHAT) guided by acoustic inverse-square intensity difference (ILD).
    """
    global _kalman_filter

    c = speed_of_sound_at_temp(temperature_c)
    m1 = np.array(mic1_xy, dtype=np.float64)
    m2 = np.array(mic2_xy, dtype=np.float64)
    baseline_vec = m2 - m1
    baseline_d = float(np.linalg.norm(baseline_vec))
    max_physical_tdoa_ms = (baseline_d / c) * 1000.0

    # 1. Peak Detection & Sinc Sub-sample Refinement
    max_idx = int(np.argmax(correlation_curve))
    p1_pow = float(correlation_curve[max_idx])
    raw_sinc_tdoa_ms = _sinc_subsample_peak(correlation_curve, max_idx, tau_axis_ms)

    # 2. PSR (Peak-to-Secondary Sidelobe Ratio)
    sorted_c = np.sort(correlation_curve)[::-1]
    p2_pow = float(sorted_c[1]) if len(sorted_c) > 1 else 1e-9
    psr = float(p1_pow / max(1e-9, abs(p2_pow)))

    # 3. Signal Quality & Interaural Level Difference (ILD)
    sig_rep = assess_signal_quality(signal_laptop, signal_phone, gain_offset_db=gain_offset_db, fs=fs)
    ild_db = sig_rep.ild_db

    # 4. Energy-Based TDOA Prior from Inverse-Square Law:
    # If Laptop is louder (ild_db > 0), source is closer to Laptop (tau > 0).
    # If Phone is louder (ild_db < 0), source is closer to Phone (tau < 0).
    # Sensitivity: ~12 dB difference corresponds to source near one of the microphones.
    tau_ild_ms = float(np.clip((ild_db / 12.0) * max_physical_tdoa_ms, -max_physical_tdoa_ms, max_physical_tdoa_ms))

    # 5. Physics Fusion:
    # If PSR is very high (>= 2.5), trust GCC-PHAT phase delay heavily.
    # If PSR is moderate or low, blend with ILD energy prior to prevent false jumps.
    if psr >= 3.0:
        w_gcc = 0.85
    elif psr >= 1.8:
        w_gcc = 0.60
    elif psr >= 1.3:
        w_gcc = 0.35
    else:
        w_gcc = 0.10

    w_ild = 1.0 - w_gcc

    # Sign consistency check: if GCC delay sign strongly contradicts strong ILD (|ild| > 6dB),
    # clamp GCC to the correct physical half-plane to prevent spatial flip.
    if ild_db > 6.0 and raw_sinc_tdoa_ms < -0.5:
        fused_raw_tdoa = tau_ild_ms * 0.8 + abs(raw_sinc_tdoa_ms) * 0.2
    elif ild_db < -6.0 and raw_sinc_tdoa_ms > 0.5:
        fused_raw_tdoa = tau_ild_ms * 0.8 - abs(raw_sinc_tdoa_ms) * 0.2
    else:
        fused_raw_tdoa = w_gcc * raw_sinc_tdoa_ms + w_ild * tau_ild_ms

    fused_raw_tdoa = float(np.clip(fused_raw_tdoa, -max_physical_tdoa_ms, max_physical_tdoa_ms))

    # 6. Cramér-Rao Lower Bound (CRLB) Theoretical Uncertainty
    snr_lin = max(0.1, 10.0 ** (sig_rep.snr_estimate_db / 10.0))
    bw = max(200.0, sig_rep.effective_bandwidth_hz)
    crlb_sigma_ms = (1.0 / (2.0 * math.pi * bw * math.sqrt(snr_lin))) * 1000.0
    psr_penalty = max(1.0, 2.5 / max(1.0, psr))
    sigma_tdoa_ms = float(np.clip(crlb_sigma_ms * psr_penalty, 0.02, 3.0))

    # 7. Adaptive 1D Kalman Filter Update
    smoothed_tdoa_ms, kalman_var_ms = _kalman_filter.update(fused_raw_tdoa, sigma_tdoa_ms)
    smoothed_tdoa_ms = float(np.clip(smoothed_tdoa_ms, -max_physical_tdoa_ms, max_physical_tdoa_ms))

    # 8. Physical Distance Difference: Delta d = c * Delta t
    dist_diff_m = float((smoothed_tdoa_ms * 1e-3) * c)

    # 9. Direction Label
    if abs(smoothed_tdoa_ms) < 0.15 and abs(ild_db) < 2.0:
        dir_label = "Source Centered (Equidistant Δd ≈ 0 cm)"
    elif smoothed_tdoa_ms > 0 or ild_db > 1.5:
        dir_label = f"Closer to Laptop Mic (+{abs(dist_diff_m)*100:.1f} cm | +{ild_db:.1f}dB)"
    else:
        dir_label = f"Closer to Phone Mic (+{abs(dist_diff_m)*100:.1f} cm | {ild_db:.1f}dB)"

    # 10. Angle of Arrival (AoA) & Speaker Position in Room Coordinates
    if baseline_d < 0.01:
        aoa_deg = 0.0
        est_xy = (float((m1[0]+m2[0])/2), float((m1[1]+m2[1])/2))
    else:
        sin_theta = np.clip(dist_diff_m / baseline_d, -0.9999, 0.9999)
        aoa_rad = math.asin(sin_theta)
        aoa_deg = math.degrees(aoa_rad)

        # Baseline orientation in room coordinates
        baseline_angle = math.atan2(baseline_vec[1], baseline_vec[0])
        normal_angle = baseline_angle + math.pi / 2.0
        
        # In our room setup:
        # m1 (Laptop) is at (2.0, 2.0), m2 (Phone) is at (4.0, 2.0)
        # Delta t > 0 => closer to Laptop (to the left, x < 3.0)
        # Delta t < 0 => closer to Phone (to the right, x > 3.0)
        source_bearing_angle = normal_angle + aoa_rad

        # Dynamic standoff distance R: closer to mic when level is very high
        # Standoff ranges from 0.4m (very close) to 2.0m (distant)
        max_rms_db = max(sig_rep.snr_estimate_db - 48.0, -50.0)
        standoff_r = float(np.clip(1.5 - (max_rms_db + 30.0) * 0.03, 0.5, 2.5))

        mid = (m1 + m2) / 2.0
        est_x = float(np.clip(mid[0] + standoff_r * math.cos(source_bearing_angle), 0.2, 9.8))
        est_y = float(np.clip(mid[1] + standoff_r * math.sin(source_bearing_angle), 0.2, 5.8))
        est_xy = (est_x, est_y)

    # 11. Confidence Metric (0-100%)
    sigma_score = max(0.0, 1.0 - (sigma_tdoa_ms / max(0.5, max_physical_tdoa_ms)))
    psr_score = min(1.0, max(0.0, (psr - 1.1) / 2.5))
    ild_score = min(1.0, abs(ild_db) / 10.0) # strong level difference gives high confidence
    conf_pct = float(np.clip((0.45 * psr_score + 0.35 * ild_score + 0.20 * sigma_score) * 100.0, 10.0, 99.0))

    return TDOAEstimate(
        tdoa_ms=round(smoothed_tdoa_ms, 4),
        raw_tdoa_ms=round(fused_raw_tdoa, 4),
        sigma_tdoa_ms=round(sigma_tdoa_ms, 4),
        distance_diff_m=round(dist_diff_m, 4),
        direction_label=dir_label,
        confidence_pct=round(conf_pct, 1),
        peak_psr=round(psr, 3),
        signal_quality=sig_rep,
        angle_of_arrival_deg=round(aoa_deg, 2),
        estimated_xy=est_xy,
        temperature_c=round(temperature_c, 1),
        speed_of_sound_mps=round(c, 2),
    )


def reset_ema() -> None:
    reset_kalman()
