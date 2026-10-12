"""
EchoLocate — Neural Robustness & Physical Mismatch Audit
=========================================================
Implements physics, geometry, perturbation, and off-grid validation routines
for Experiment 4.75 adversarial stress-testing.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pyroomacoustics as pra

logger = logging.getLogger(__name__)


def generate_continuous_offgrid_sources(
    n_sources: int,
    room_dims: list[float],
    grid_resolution_m: float = 0.05,
    margin_m: float = 0.50,
    source_height: float = 1.2,
    seed: int = 14501,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate random 2D continuous source coordinates that do NOT lie on discrete grid nodes.

    Returns
    -------
    sources_3d : np.ndarray, shape (n_sources, 3)
    d_nearest_grid : np.ndarray, shape (n_sources,)
        Euclidean distance from each continuous point to the nearest discrete grid node.
    """
    rng = np.random.default_rng(seed)
    x_min, x_max = margin_m, room_dims[0] - margin_m
    y_min, y_max = margin_m, room_dims[1] - margin_m

    sources_3d = []
    d_nearest = []

    # Discrete grid node coordinates
    n_x = int(np.floor((x_max - x_min) / grid_resolution_m + 1e-8)) + 1
    n_y = int(np.floor((y_max - y_min) / grid_resolution_m + 1e-8)) + 1
    x_grid = np.linspace(x_min, x_min + (n_x - 1) * grid_resolution_m, n_x)
    y_grid = np.linspace(y_min, y_min + (n_y - 1) * grid_resolution_m, n_y)
    grid_pts = np.array([[x, y] for x in x_grid for y in y_grid])

    for _ in range(n_sources):
        rx = rng.uniform(x_min, x_max)
        ry = rng.uniform(y_min, y_max)

        # Distance to nearest grid point
        dist_sq = np.sum((grid_pts - np.array([rx, ry])) ** 2, axis=1)
        min_dist = float(np.sqrt(np.min(dist_sq)))

        sources_3d.append([rx, ry, source_height])
        d_nearest.append(min_dist)

    return np.array(sources_3d), np.array(d_nearest)


def apply_microphone_perturbation(
    nominal_mics_2d: np.ndarray,
    perturbation_mm: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Perturb microphone positions independently by a maximum magnitude in mm.
    m_i_actual = m_i_nominal + epsilon_i
    """
    if perturbation_mm <= 0.0:
        return np.array(nominal_mics_2d, dtype=np.float64)

    rng = np.random.default_rng(seed)
    n_mics = len(nominal_mics_2d)
    sigma_m = (perturbation_mm / 1000.0) / np.sqrt(2.0)  # Total 2D displacement norm std

    noise = rng.normal(0.0, sigma_m, size=(n_mics, 2))
    perturbed = np.asarray(nominal_mics_2d, dtype=np.float64) + noise
    return perturbed


def apply_channel_synchronization_jitter(
    multichannel_audio: np.ndarray,
    fs: int = 16000,
    offset_us: float = 0.0,
    mode: str = "bias",  # "bias" or "jitter"
    seed: int = 42,
) -> np.ndarray:
    """
    Introduce inter-channel synchronization offset or clock jitter.
    offset_us: microsecond time shift.
    """
    if offset_us <= 0.0:
        return np.array(multichannel_audio, dtype=np.float64)

    rng = np.random.default_rng(seed)
    n_mics, n_samples = multichannel_audio.shape
    out = np.zeros_like(multichannel_audio)

    for i in range(n_mics):
        if i == 0:
            out[i] = multichannel_audio[i]
            continue

        if mode == "bias":
            shift_s = (i * offset_us) * 1e-6
        else:
            shift_s = rng.uniform(-offset_us, offset_us) * 1e-6

        shift_samples = shift_s * fs
        int_shift = int(np.round(shift_samples))

        if int_shift == 0:
            out[i] = multichannel_audio[i]
        elif int_shift > 0:
            out[i, int_shift:] = multichannel_audio[i, :-int_shift]
        else:
            out[i, :int_shift] = multichannel_audio[i, -int_shift:]

    return out


def simulate_room_acoustics(
    room_dims: list[float],
    mic_positions_2d: np.ndarray,
    source_pos_3d: list[float] | np.ndarray,
    source_signal: np.ndarray,
    rt60_target: float,
    fs: int = 16000,
    mic_height: float = 1.2,
    speed_of_sound: float = 343.0,
) -> tuple[np.ndarray, float]:
    """
    Simulate reverberant multichannel microphone signals in arbitrary room dimensions.
    Returns (mic_signals, measured_rt60).
    """
    pra.constants.set("c", speed_of_sound)
    e_absorption, max_order = pra.inverse_sabine(float(rt60_target), list(room_dims))
    e_absorption = max(0.01, min(0.99, float(e_absorption)))
    max_order = min(int(max_order), 25)

    room = pra.ShoeBox(
        room_dims,
        fs=fs,
        materials=pra.Material(e_absorption),
        max_order=max_order,
        air_absorption=False,
    )

    room.add_source(list(source_pos_3d), signal=source_signal)

    mics_3d = np.zeros((3, len(mic_positions_2d)))
    for idx, (mx, my) in enumerate(mic_positions_2d):
        mics_3d[0, idx] = mx
        mics_3d[1, idx] = my
        mics_3d[2, idx] = mic_height

    room.add_microphone_array(mics_3d)
    room.simulate()

    try:
        measured_rt60 = float(room.measure_rt60()[0][0])
    except Exception:
        measured_rt60 = float(rt60_target)

    # Align signal length
    sig = room.mic_array.signals[:, :len(source_signal)]
    return sig, measured_rt60
