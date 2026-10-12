"""
EchoLocate — Room Simulator
============================
Wraps pyroomacoustics to provide a clean interface for creating rectangular
rooms, placing microphones and sources, and obtaining multichannel audio.

Architecture notes:
    - The room is fundamentally 3D in pyroomacoustics (shoebox model requires
      3 dimensions). We treat the horizontal (x, y) plane as the localization
      plane and fix a constant height for sources and microphones.
    - 2D localization is performed in post-processing (multilateration.py).
    - The wrapper is designed to be replaceable: if pyroomacoustics is
      unavailable, this module can be swapped for another simulator
      (e.g., RoomAcousticsSim, AIRA) without changing the rest of the codebase.

Sign conventions for TDOA:
    TDOA_ij = t_i - t_j = (d_i - d_j) / c
    where d_i is the distance from source to mic i, c is speed of sound.
    TDOA_ij > 0 means sound arrives at mic j BEFORE mic i.
    TDOA_ij < 0 means sound arrives at mic i BEFORE mic j.
    This convention is enforced consistently in this module and gcc_phat.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pyroomacoustics as pra

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Container for the output of one simulation run.

    Attributes
    ----------
    mic_signals : np.ndarray
        Shape (n_mics, n_samples). Row i is the signal received at mic i.
    mic_positions_3d : np.ndarray
        Shape (3, n_mics). Column i is the 3-D (x, y, z) position of mic i.
    source_position_2d : np.ndarray
        Shape (2,). The (x, y) position of the acoustic source.
    source_signal : np.ndarray
        Shape (n_samples_source,). The emitted source signal.
    fs : int
        Sampling frequency in Hz.
    speed_of_sound : float
        Speed of sound used in the simulation (m/s).
    stage : str
        'anechoic' or 'reverberant'.
    room_dimensions : list[float]
        [length_x, width_y, height_z] in metres.
    """

    mic_signals: np.ndarray
    mic_positions_3d: np.ndarray
    source_position_2d: np.ndarray
    source_signal: np.ndarray
    fs: int
    speed_of_sound: float
    stage: str
    room_dimensions: list[float]

    def mic_position_2d(self, mic_index: int) -> np.ndarray:
        """Return (x, y) for mic at index mic_index."""
        return self.mic_positions_3d[:2, mic_index]

    def all_mic_positions_2d(self) -> np.ndarray:
        """Return shape (n_mics, 2) array of mic (x, y) positions."""
        return self.mic_positions_3d[:2, :].T  # (n_mics, 2)

    def theoretical_tdoa(self, i: int, j: int) -> float:
        """
        Compute the theoretical TDOA between mic i and mic j.

        TDOA_ij = (d_i - d_j) / c
        where d_i = Euclidean distance from source to mic i.

        Parameters
        ----------
        i, j : int
            Microphone indices (0-based).

        Returns
        -------
        float
            TDOA in seconds. Positive means sound arrives at j first.
        """
        src = self.source_position_2d
        mic_i = self.mic_position_2d(i)
        mic_j = self.mic_position_2d(j)
        d_i = float(np.linalg.norm(src - mic_i))
        d_j = float(np.linalg.norm(src - mic_j))
        return (d_i - d_j) / self.speed_of_sound


def build_room_anechoic(
    room_dims: list[float],
    mic_positions_3d: np.ndarray,
    source_position_3d: np.ndarray,
    source_signal: np.ndarray,
    fs: int,
    speed_of_sound: float = 343.0,
    noise_level_db: float = -60.0,
) -> SimulationResult:
    """
    Simulate sound propagation in an anechoic (reflection-free) room.

    Uses pyroomacoustics ShoeBox with max_order=0, which models only the
    direct-path wavefront with no reflections from walls, floor, or ceiling.

    Attenuation follows the inverse-square law implicitly through the
    room impulse response computed by pyroomacoustics.

    Parameters
    ----------
    room_dims : list[float]
        [Lx, Ly, Lz] room dimensions in metres.
    mic_positions_3d : np.ndarray
        Shape (3, n_mics). Column per microphone.
    source_position_3d : np.ndarray
        Shape (3,). The source (x, y, z) position.
    source_signal : np.ndarray
        1-D float64 signal emitted by the source.
    fs : int
        Sampling frequency in Hz.
    speed_of_sound : float
        Speed of sound in m/s (default 343.0).
    noise_level_db : float
        SNR of additive white Gaussian noise in dB. Set very low (-60 dB)
        for near-ideal anechoic conditions.

    Returns
    -------
    SimulationResult
    """
    logger.info("Building anechoic room (max_order=0)...")

    # Create shoebox room with absorption = 1 (no reflections; energy absorbed)
    room = pra.ShoeBox(
        room_dims,
        fs=fs,
        materials=pra.Material(1.0),   # absorption = 1.0 → no energy reflected
        max_order=0,
        air_absorption=False,
        ray_tracing=False,
        # NOTE: speed of sound is set via temperature in pyroomacoustics 0.10+.
        # Default temperature (20°C) gives c ≈ 343.0 m/s, which matches our config.
    )

    # Add microphone array (pyroomacoustics expects shape (3, n_mics))
    room.add_microphone(mic_positions_3d)

    # Add source with the emitted signal
    room.add_source(source_position_3d, signal=source_signal)

    # Compute room impulse responses and simulate
    room.simulate()

    mic_signals = room.mic_array.signals.copy()  # (n_mics, n_samples)

    # Add low-level noise for realism (SNR relative to signal power)
    if noise_level_db > -120:
        mic_signals = _add_noise(mic_signals, noise_level_db, seed=42)

    logger.info(
        "Anechoic simulation complete. Signal shape: %s", str(mic_signals.shape)
    )

    return SimulationResult(
        mic_signals=mic_signals,
        mic_positions_3d=mic_positions_3d.copy(),
        source_position_2d=np.array(source_position_3d[:2]),
        source_signal=source_signal.copy(),
        fs=fs,
        speed_of_sound=speed_of_sound,
        stage="anechoic",
        room_dimensions=list(room_dims),
    )


def build_room_reverberant(
    room_dims: list[float],
    mic_positions_3d: np.ndarray,
    source_position_3d: np.ndarray,
    source_signal: np.ndarray,
    fs: int,
    rt60: float = 0.4,
    max_order: int = 17,
    speed_of_sound: float = 343.0,
    noise_level_db: float = -30.0,
) -> SimulationResult:
    """
    Simulate sound propagation in a reverberant room.

    Uses pyroomacoustics ShoeBox with materials derived from the Sabine
    equation so that the room has the target RT60.

    Parameters
    ----------
    room_dims : list[float]
        [Lx, Ly, Lz] room dimensions in metres.
    mic_positions_3d : np.ndarray
        Shape (3, n_mics).
    source_position_3d : np.ndarray
        Shape (3,).
    source_signal : np.ndarray
        1-D float64 signal.
    fs : int
        Sampling frequency in Hz.
    rt60 : float
        Target reverberation time T60 in seconds (time for sound to decay
        by 60 dB). Typical offices: 0.3–0.6 s.
    max_order : int
        Maximum image-source reflection order. Higher = more accurate but
        slower. 17 is a reasonable default for RT60=0.4s in a medium room.
    speed_of_sound : float
        Speed of sound in m/s.
    noise_level_db : float
        SNR of additive white Gaussian noise (dB).

    Returns
    -------
    SimulationResult
    """
    logger.info("Building reverberant room (RT60=%.2f s, max_order=%d)...", rt60, max_order)

    # Derive wall absorption coefficient from Sabine equation:
    #   RT60 = 0.161 * V / (alpha * S)  →  alpha = 0.161 * V / (RT60 * S)
    # pyroomacoustics provides a helper for this.
    e_absorption, max_order_computed = pra.inverse_sabine(rt60, room_dims)

    # Allow config to override max_order if desired
    used_order = max(max_order, max_order_computed)
    logger.info("Computed absorption=%.4f, using max_order=%d", e_absorption, used_order)

    room = pra.ShoeBox(
        room_dims,
        fs=fs,
        materials=pra.Material(e_absorption),
        max_order=used_order,
        air_absorption=True,
        ray_tracing=False,
        # NOTE: speed of sound defaults to 343 m/s at 20°C in pyroomacoustics 0.10+.
    )

    room.add_microphone(mic_positions_3d)
    room.add_source(source_position_3d, signal=source_signal)
    room.simulate()

    mic_signals = room.mic_array.signals.copy()

    if noise_level_db > -120:
        mic_signals = _add_noise(mic_signals, noise_level_db, seed=42)

    logger.info(
        "Reverberant simulation complete. Signal shape: %s", str(mic_signals.shape)
    )

    return SimulationResult(
        mic_signals=mic_signals,
        mic_positions_3d=mic_positions_3d.copy(),
        source_position_2d=np.array(source_position_3d[:2]),
        source_signal=source_signal.copy(),
        fs=fs,
        speed_of_sound=speed_of_sound,
        stage="reverberant",
        room_dimensions=list(room_dims),
    )


def _add_noise(
    signals: np.ndarray, snr_db: float, seed: int = 0
) -> np.ndarray:
    """
    Add additive white Gaussian noise (AWGN) to multichannel signals.

    Parameters
    ----------
    signals : np.ndarray
        Shape (n_channels, n_samples).
    snr_db : float
        Signal-to-noise ratio in dB. Positive = signal stronger than noise.
        E.g. snr_db = +60 means noise is 60 dB below signal power.
        The config uses noise_level_db which is interpreted as SNR.
        A value of -100 means noise is 100 dB below signal (extremely quiet).
        A value of +20 means noise is 20 dB above signal (very noisy).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy signals, same shape.
    """
    rng = np.random.default_rng(seed)
    # Mean signal power across all channels
    signal_power = np.mean(signals ** 2)
    if signal_power < 1e-30:
        # Signal is essentially silent; skip noise injection
        return signals
    # SNR = 10 * log10(S / N)  →  N = S / 10^(SNR/10)
    # Our convention: snr_db > 0 means noise is quiet, snr_db < 0 means noisy
    # noise_level_db in config is treated as: noise is |noise_level_db| dB below signal
    # So noise_power = signal_power * 10^(noise_level_db / 10)
    # For noise_level_db = -60: noise_power = signal_power * 10^(-6) (60 dB below)
    # For noise_level_db = -100: noise_power = signal_power * 10^(-10) (100 dB below)
    noise_power = signal_power * (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=signals.shape)
    return signals + noise


def build_mic_array_3d(
    positions_2d: dict[str, list[float]],
    height: float,
) -> tuple[np.ndarray, list[str]]:
    """
    Convert a dict of 2D mic positions to a (3, n_mics) array for pyroomacoustics.

    Parameters
    ----------
    positions_2d : dict[str, list[float]]
        Mapping from mic label (e.g. 'M1') to [x, y] coordinates.
    height : float
        Common z-coordinate (metres) for all microphones.

    Returns
    -------
    mic_array : np.ndarray
        Shape (3, n_mics) — pyroomacoustics format.
    labels : list[str]
        Ordered list of mic labels corresponding to columns of mic_array.
    """
    labels = sorted(positions_2d.keys())  # Consistent ordering: M1, M2, M3
    coords = []
    for label in labels:
        x, y = positions_2d[label]
        coords.append([x, y, height])
    return np.array(coords, dtype=float).T, labels  # (3, n_mics)


def run_simulation(
    config: dict,
    source_position_2d: list[float],
    source_signal: np.ndarray,
    stage: str = "anechoic",
) -> SimulationResult:
    """
    High-level entry point: run one simulation from config.

    Parameters
    ----------
    config : dict
        Full experiment config (loaded from YAML).
    source_position_2d : list[float]
        [x, y] source position in metres.
    source_signal : np.ndarray
        1-D source signal array.
    stage : str
        'anechoic' or 'reverberant'.

    Returns
    -------
    SimulationResult
    """
    room_cfg = config["room"]
    room_dims = room_cfg["dimensions"]
    fs: int = room_cfg["sampling_rate"]
    c: float = room_cfg["speed_of_sound"]

    mic_cfg = config["microphones"]
    mic_array, labels = build_mic_array_3d(mic_cfg["positions"], mic_cfg["height"])

    src_z = mic_cfg["height"]  # Source at same height as mics (2D problem)
    src_3d = np.array([source_position_2d[0], source_position_2d[1], src_z])

    if stage == "anechoic":
        stage_cfg = config["stage_a"]
        result = build_room_anechoic(
            room_dims=room_dims,
            mic_positions_3d=mic_array,
            source_position_3d=src_3d,
            source_signal=source_signal,
            fs=fs,
            speed_of_sound=c,
            noise_level_db=stage_cfg["noise_level_db"],
        )
    elif stage == "reverberant":
        stage_cfg = config["stage_b"]
        result = build_room_reverberant(
            room_dims=room_dims,
            mic_positions_3d=mic_array,
            source_position_3d=src_3d,
            source_signal=source_signal,
            fs=fs,
            rt60=stage_cfg["rt60"],
            max_order=stage_cfg["max_order"],
            speed_of_sound=c,
            noise_level_db=stage_cfg["noise_level_db"],
        )
    else:
        raise ValueError(f"Unknown stage: '{stage}'. Choose 'anechoic' or 'reverberant'.")

    return result
