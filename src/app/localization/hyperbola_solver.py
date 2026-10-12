"""
EchoLocate — 2D Hyperbolic Source Locus Solver & Analytic Geometry
===================================================================
Solves the exact 2D hyperbolic source locus equation for a two-microphone baseline:
    ||x - m1||_2 - ||x - m2||_2 = Delta d = c(T) * Delta t

Mathematical structure:
  - Baseline vector b = m2 - m1, focal distance 2c_f = ||b||_2
  - Hyperbolic parameter 2a_h = |Delta d|  => a_h = |Delta d| / 2
  - Conjugate axis b_h = sqrt(c_f^2 - a_h^2)
  - Asymptotes: y = +/- (b_h / a_h) * x in canonical baseline frame
  - True physical branch (selected based on sign of Delta t)
  - Shadow conjugate branch (dashed reference)
  - Perpendicular bisector (Delta t = 0 baseline normal)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.app.localization.tdoa_estimator import speed_of_sound_at_temp

logger = logging.getLogger(__name__)


@dataclass
class HyperbolaLocus:
    """Sampled 2D points representing the hyperbolic locus of candidate source positions."""
    mic1_xy: tuple[float, float]
    mic2_xy: tuple[float, float]
    tdoa_ms: float
    distance_diff_m: float
    baseline_separation_m: float
    is_physically_realizable: bool
    locus_points: list[tuple[float, float]]         # Physical active branch points [(x, y), ...]
    shadow_branch_points: list[tuple[float, float]] # Conjugate secondary branch points
    asymptote_lines: list[list[tuple[float, float]]] # 2 asymptote segments [[p0, p1], [p2, p3]]
    bisector_line: list[tuple[float, float]]        # Perpendicular bisector segment [p0, p1]
    canonical_a_m: float                           # Semi-major axis a (meters)
    canonical_b_m: float                           # Semi-minor axis b (meters)
    canonical_c_m: float                           # Focal distance c_f (meters)
    eccentricity: float                            # e = c_f / a_h >= 1.0
    temperature_c: float = 22.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mic1_xy": [round(self.mic1_xy[0], 3), round(self.mic1_xy[1], 3)],
            "mic2_xy": [round(self.mic2_xy[0], 3), round(self.mic2_xy[1], 3)],
            "tdoa_ms": round(self.tdoa_ms, 4),
            "distance_diff_m": round(self.distance_diff_m, 4),
            "baseline_separation_m": round(self.baseline_separation_m, 4),
            "is_physically_realizable": self.is_physically_realizable,
            "locus_points": [[round(p[0], 3), round(p[1], 3)] for p in self.locus_points],
            "shadow_branch_points": [[round(p[0], 3), round(p[1], 3)] for p in self.shadow_branch_points],
            "asymptote_lines": [[[round(p[0], 3), round(p[1], 3)] for p in line] for line in self.asymptote_lines],
            "bisector_line": [[round(p[0], 3), round(p[1], 3)] for p in self.bisector_line],
            "canonical_a_m": round(self.canonical_a_m, 4),
            "canonical_b_m": round(self.canonical_b_m, 4),
            "canonical_c_m": round(self.canonical_c_m, 4),
            "eccentricity": round(self.eccentricity, 4),
            "temperature_c": round(self.temperature_c, 1),
        }


def solve_hyperbolic_locus(
    mic1_xy: tuple[float, float],
    mic2_xy: tuple[float, float],
    tdoa_ms: float,
    room_dims: list[float] = [10.0, 6.0],
    temperature_c: float = 22.0,
    num_samples: int = 150,
) -> HyperbolaLocus:
    """
    Solve and sample the exact 2D hyperbola branches, asymptotes, and bisector:
        ||x - m1|| - ||x - m2|| = Delta d = c(T) * Delta t
    """
    c = speed_of_sound_at_temp(temperature_c)
    m1 = np.array(mic1_xy, dtype=np.float64)
    m2 = np.array(mic2_xy, dtype=np.float64)

    baseline_vec = m2 - m1
    baseline_m = float(np.linalg.norm(baseline_vec))
    delta_d = float((tdoa_ms * 1e-3) * c)

    # Physical realizability condition: |Delta d| <= D
    is_realizable = abs(delta_d) <= (baseline_m + 1e-3)
    clamped_delta_d = float(np.clip(delta_d, -0.999 * baseline_m, 0.999 * baseline_m))

    center = (m1 + m2) / 2.0
    baseline_angle = float(np.arctan2(baseline_vec[1], baseline_vec[0]))
    cos_a = math.cos(baseline_angle)
    sin_a = math.sin(baseline_angle)

    # Focal distance: 2 * c_f = baseline_m
    c_f = baseline_m / 2.0
    # Semi-major axis: 2 * a_h = |Delta d|
    a_h = max(1e-4, min(abs(clamped_delta_d) / 2.0, c_f - 1e-4))
    # Semi-minor axis: b_h = sqrt(c_f^2 - a_h^2)
    b_h = math.sqrt(max(1e-6, c_f**2 - a_h**2))
    eccentricity = c_f / a_h

    # Sign convention:
    # If Delta t > 0, closer to M1 => branch opens towards M1 (-X in baseline frame)
    # If Delta t < 0, closer to M2 => branch opens towards M2 (+X in baseline frame)
    phys_branch_sign = -1.0 if delta_d >= 0 else 1.0
    shadow_branch_sign = -phys_branch_sign

    # Parameterize hyperbola: x(v) = sign * a * cosh(v), y(v) = b * sinh(v)
    v_vals = np.linspace(-3.0, 3.0, num_samples)

    def _sample_branch(sign: float) -> list[tuple[float, float]]:
        x_canon = sign * a_h * np.cosh(v_vals)
        y_canon = b_h * np.sinh(v_vals)
        x_rot = x_canon * cos_a - y_canon * sin_a + center[0]
        y_rot = x_canon * sin_a + y_canon * cos_a + center[1]
        pts = []
        rx_max, ry_max = room_dims[0], room_dims[1]
        for x, y in zip(x_rot, y_rot):
            if -0.5 <= x <= rx_max + 0.5 and -0.5 <= y <= ry_max + 0.5:
                pts.append((float(x), float(y)))
        return pts

    active_pts = _sample_branch(phys_branch_sign)
    shadow_pts = _sample_branch(shadow_branch_sign)

    # Asymptotes: y = +/- (b_h / a_h) * x in canonical frame
    slope_asymp = b_h / a_h
    L_asymp = max(room_dims[0], room_dims[1]) * 1.5
    asymp_lines: list[list[tuple[float, float]]] = []
    for s_sign in [1.0, -1.0]:
        x_c = np.array([-L_asymp, L_asymp])
        y_c = s_sign * slope_asymp * x_c
        x_r = x_c * cos_a - y_c * sin_a + center[0]
        y_r = x_c * sin_a + y_c * cos_a + center[1]
        asymp_lines.append([(float(x_r[0]), float(y_r[0])), (float(x_r[1]), float(y_r[1]))])

    # Perpendicular Bisector (Line perpendicular to baseline through center)
    perp_angle = baseline_angle + math.pi / 2.0
    L_bisect = max(room_dims[0], room_dims[1])
    bisect_p0 = (center[0] - L_bisect * math.cos(perp_angle), center[1] - L_bisect * math.sin(perp_angle))
    bisect_p1 = (center[0] + L_bisect * math.cos(perp_angle), center[1] + L_bisect * math.sin(perp_angle))
    bisector_line = [bisect_p0, bisect_p1]

    return HyperbolaLocus(
        mic1_xy=mic1_xy,
        mic2_xy=mic2_xy,
        tdoa_ms=tdoa_ms,
        distance_diff_m=delta_d,
        baseline_separation_m=baseline_m,
        is_physically_realizable=is_realizable,
        locus_points=active_pts,
        shadow_branch_points=shadow_pts,
        asymptote_lines=asymp_lines,
        bisector_line=bisector_line,
        canonical_a_m=a_h,
        canonical_b_m=b_h,
        canonical_c_m=c_f,
        eccentricity=eccentricity,
        temperature_c=temperature_c,
    )
