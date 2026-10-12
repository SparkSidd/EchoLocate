"""
EchoLocate — Evaluation: Geometry & Conditioning Metrics
=========================================================
Metrics and analytical tools for quantifying microphone array geometry,
spatial coverage, convexity, TDOA Jacobian conditioning, and GDOP.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GeometryProperties:
    """Analytical properties of a 3-microphone array configuration."""
    geometry_id: str
    name: str
    mic_positions_2d: np.ndarray  # Shape (3, 2)
    pairwise_distances: dict[tuple[int, int], float]  # (0,1), (0,2), (1,2)
    min_separation_m: float
    max_separation_m: float
    triangle_area_m2: float


def compute_triangle_area(mic_positions_2d: np.ndarray) -> float:
    """
    Compute area of the triangle formed by 3 microphones in the 2D plane.
    
    Parameters
    ----------
    mic_positions_2d : np.ndarray
        Shape (3, 2) array of microphone coordinates.
        
    Returns
    -------
    float
        Area in square metres (always non-negative).
    """
    p1, p2, p3 = mic_positions_2d[0], mic_positions_2d[1], mic_positions_2d[2]
    # Shoelace formula
    area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
    return float(area)


def compute_geometry_properties(geometry_id: str, name: str, mic_positions_2d: np.ndarray) -> GeometryProperties:
    """
    Compute basic structural metrics for a 3-microphone array.
    """
    pairs = [(0, 1), (0, 2), (1, 2)]
    dists = {}
    for i, j in pairs:
        d = float(np.linalg.norm(mic_positions_2d[i] - mic_positions_2d[j]))
        dists[(i, j)] = d
        
    min_sep = min(dists.values())
    max_sep = max(dists.values())
    area = compute_triangle_area(mic_positions_2d)
    
    return GeometryProperties(
        geometry_id=geometry_id,
        name=name,
        mic_positions_2d=mic_positions_2d,
        pairwise_distances=dists,
        min_separation_m=min_sep,
        max_separation_m=max_sep,
        triangle_area_m2=area,
    )


def is_point_inside_triangle(point: np.ndarray, triangle: np.ndarray) -> bool:
    """
    Determine if a 2D point lies strictly inside (or on the boundary of) a triangle.
    Uses barycentric coordinate or cross-product orientation sign.
    """
    p = np.asarray(point, dtype=float)
    a, b, c = triangle[0], triangle[1], triangle[2]

    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)

    has_neg = (d1 < -1e-9) or (d2 < -1e-9) or (d3 < -1e-9)
    has_pos = (d1 > 1e-9) or (d2 > 1e-9) or (d3 > 1e-9)

    return not (has_neg and has_pos)


def compute_tdoa_jacobian_and_gdop(
    source_pos: np.ndarray,
    mic_positions_2d: np.ndarray,
    speed_of_sound: float = 343.0,
    eps: float = 1e-9,
) -> tuple[float, float, np.ndarray]:
    """
    Compute the exact TDOA measurement Jacobian matrix H, its condition number κ(H),
    and the TDOA Geometric Dilution of Precision (GDOP).

    Measurement model:
        τ_ij = (||p - m_i|| - ||p - m_j||) / c

    Jacobian row k corresponding to pair (i, j):
        ∂τ_ij / ∂p = (1 / c) * ( (p - m_i) / ||p - m_i|| - (p - m_j) / ||p - m_j|| )

    Parameters
    ----------
    source_pos : np.ndarray
        Shape (2,) array of source coordinates (x, y).
    mic_positions_2d : np.ndarray
        Shape (3, 2) array of microphone coordinates.
    speed_of_sound : float
        Propagation speed in m/s (default: 343.0).
    eps : float
        Small regularizer to avoid division by zero if source is exactly on a mic.

    Returns
    -------
    condition_number : float
        2D Jacobian condition number κ(H) = σ_max / σ_min.
    gdop : float
        TDOA GDOP = sqrt(trace((H^T H)^(-1))).
    H : np.ndarray
        Shape (3, 2) Jacobian matrix for pairs (0,1), (0,2), (1,2).
    """
    p = np.asarray(source_pos, dtype=float)
    pairs = [(0, 1), (0, 2), (1, 2)]
    
    # Compute unit vectors from source to each microphone
    u = []
    for m in mic_positions_2d:
        diff = p - m
        dist = np.linalg.norm(diff)
        if dist < eps:
            u.append(np.array([0.0, 0.0]))
        else:
            u.append(diff / dist)
            
    H = np.zeros((len(pairs), 2), dtype=float)
    for k, (i, j) in enumerate(pairs):
        H[k, :] = (u[i] - u[j]) / speed_of_sound
        
    # Singular value decomposition & condition number
    s = np.linalg.svd(H, compute_uv=False)
    if len(s) >= 2 and s[1] > 1e-15:
        cond_num = float(s[0] / s[1])
    else:
        cond_num = np.inf
        
    # GDOP calculation: sqrt(trace(inv(H^T H)))
    HtH = H.T @ H
    try:
        inv_HtH = np.linalg.inv(HtH)
        gdop = float(np.sqrt(np.trace(inv_HtH)))
    except np.linalg.LinAlgError:
        gdop = np.inf
        
    return cond_num, gdop, H


def evaluate_source_geometry_metrics(
    source_pos: np.ndarray,
    mic_positions_2d: np.ndarray,
    speed_of_sound: float = 343.0,
) -> dict[str, float | bool]:
    """
    Compute a full dictionary of geometric indicators for a single source position.
    """
    p = np.asarray(source_pos, dtype=float)
    dists = [float(np.linalg.norm(p - m)) for m in mic_positions_2d]
    d_min = min(dists)
    d_max = max(dists)
    d_ratio = d_max / (d_min + 1e-9)
    inside = is_point_inside_triangle(p, mic_positions_2d)
    cond_num, gdop, _ = compute_tdoa_jacobian_and_gdop(p, mic_positions_2d, speed_of_sound)
    
    return {
        "dist_M1_m": dists[0],
        "dist_M2_m": dists[1],
        "dist_M3_m": dists[2],
        "dist_nearest_mic_m": d_min,
        "dist_farthest_mic_m": d_max,
        "distance_ratio": d_ratio,
        "inside_triangle": inside,
        "jacobian_condition_number": cond_num,
        "gdop": gdop,
    }
