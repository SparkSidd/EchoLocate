"""
EchoLocate — Visualization Module
====================================
Generates all required experimental figures and saves them to disk.

Figure inventory:
    Figure 1: Room layout (top-down view) — room, mics, source
    Figure 2: Multichannel waveforms — full received signals
    Figure 3: Zoomed waveforms — arrival time differences visible
    Figure 4: GCC-PHAT cross-correlation responses per pair
    Figure 5: Single-source localization result
    Figure 6: Multi-source localization results (all positions)

All figures are saved to the configured output directory.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no display required)

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def _apply_style(style: str = "seaborn-v0_8-whitegrid") -> None:
    """Apply matplotlib style, with graceful fallback."""
    try:
        plt.style.use(style)
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass  # Default matplotlib style


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 150) -> None:
    """Save figure and log the path."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    logger.info("Figure saved: %s", path)
    plt.close(fig)


# =============================================================================
# Figure 1: Room Layout
# =============================================================================

def plot_room_layout(
    room_dimensions: list[float],
    mic_positions_2d: np.ndarray,
    mic_labels: list[str],
    source_position: np.ndarray | None = None,
    source_label: str = "Source",
    title: str = "Room Layout — Top-Down View",
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Plot a top-down (bird's eye) view of the room.

    Parameters
    ----------
    room_dimensions : list[float]
        [Lx, Ly, ...] room dimensions.
    mic_positions_2d : np.ndarray
        Shape (n_mics, 2). Mic (x, y) positions.
    mic_labels : list[str]
        Labels for each microphone.
    source_position : np.ndarray or None
        Shape (2,). Source (x, y) if known.
    title : str
        Plot title.
    save_path : str or Path or None
        If provided, save figure here.
    dpi : int
        Figure DPI.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    Lx, Ly = room_dimensions[0], room_dimensions[1]

    # Draw room boundary
    room_rect = patches.Rectangle(
        (0, 0), Lx, Ly,
        linewidth=2.5, edgecolor="#2c3e50", facecolor="#ecf0f1", alpha=0.5, zorder=1,
    )
    ax.add_patch(room_rect)

    # Mark microphone positions
    mic_colors = ["#e74c3c", "#3498db", "#2ecc71"]
    for idx, (pos, label) in enumerate(zip(mic_positions_2d, mic_labels)):
        color = mic_colors[idx % len(mic_colors)]
        ax.scatter(pos[0], pos[1], s=200, c=color, zorder=5, marker="^",
                   edgecolors="white", linewidths=1.5, label=label)
        ax.annotate(
            label,
            xy=(pos[0], pos[1]),
            xytext=(pos[0] + 0.15, pos[1] + 0.2),
            fontsize=12, fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.85),
        )
        # Show coordinates
        ax.annotate(
            f"({pos[0]:.1f}, {pos[1]:.1f})",
            xy=(pos[0], pos[1]),
            xytext=(pos[0] + 0.15, pos[1] - 0.35),
            fontsize=8, color="#555",
        )

    # Mark source position if given
    if source_position is not None:
        ax.scatter(source_position[0], source_position[1], s=250, c="#e67e22",
                   marker="*", zorder=6, label=source_label,
                   edgecolors="#d35400", linewidths=1.5)
        ax.annotate(
            f"{source_label}\n({source_position[0]:.1f}, {source_position[1]:.1f})",
            xy=(source_position[0], source_position[1]),
            xytext=(source_position[0] + 0.15, source_position[1] + 0.2),
            fontsize=10, fontweight="bold", color="#e67e22",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#e67e22", alpha=0.85),
        )
        # Draw dashed lines from source to each mic
        for pos in mic_positions_2d:
            ax.plot(
                [source_position[0], pos[0]], [source_position[1], pos[1]],
                "--", color="#aaa", linewidth=0.8, alpha=0.6, zorder=2,
            )

    ax.set_xlim(-0.3, Lx + 0.3)
    ax.set_ylim(-0.3, Ly + 0.3)
    ax.set_xlabel("X (metres)", fontsize=12)
    ax.set_ylabel("Y (metres)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Room dimension annotations
    ax.annotate(
        f"Room: {Lx:.0f} m × {Ly:.0f} m",
        xy=(Lx / 2, -0.2), ha="center", fontsize=10, color="#555",
    )

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, dpi=dpi)
    return fig


# =============================================================================
# Figure 2: Full Multichannel Waveforms
# =============================================================================

def plot_waveforms(
    mic_signals: np.ndarray,
    fs: int,
    mic_labels: list[str],
    title: str = "Received Waveforms — All Microphones",
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Plot the full received waveform for each microphone.

    Parameters
    ----------
    mic_signals : np.ndarray
        Shape (n_mics, n_samples).
    fs : int
        Sampling frequency in Hz.
    mic_labels : list[str]
        Labels for each microphone.
    """
    _apply_style()
    n_mics = mic_signals.shape[0]
    n_samples = mic_signals.shape[1]
    t = np.arange(n_samples) / fs * 1000  # Convert to milliseconds

    fig, axes = plt.subplots(n_mics, 1, figsize=(12, 2.5 * n_mics), sharex=True)
    if n_mics == 1:
        axes = [axes]

    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    for idx, (ax, signal, label) in enumerate(zip(axes, mic_signals, mic_labels)):
        color = colors[idx % len(colors)]
        ax.plot(t, signal, color=color, linewidth=0.7, alpha=0.85)
        ax.set_ylabel(label, fontsize=11, fontweight="bold", rotation=0, labelpad=35)
        ax.set_ylim(signal.min() * 1.15, signal.max() * 1.15 if signal.max() > 0 else 0.1)
        ax.axhline(0, color="#aaa", linewidth=0.5)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

    axes[-1].set_xlabel("Time (ms)", fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, dpi=dpi)
    return fig


# =============================================================================
# Figure 3: Zoomed Waveforms (Arrival Time Comparison)
# =============================================================================

def plot_waveforms_zoomed(
    mic_signals: np.ndarray,
    fs: int,
    mic_labels: list[str],
    zoom_window_ms: tuple[float, float] | None = None,
    title: str = "Zoomed Waveforms — First Arrival Region",
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Plot a zoomed view around the first significant arrival.

    The zoom window is auto-detected if not specified: find the first
    sample where any mic signal exceeds a threshold, then show ±5 ms
    around that point.

    Parameters
    ----------
    zoom_window_ms : tuple (t_start_ms, t_end_ms) or None
        Time window in milliseconds. If None, auto-detected.
    """
    _apply_style()
    n_mics = mic_signals.shape[0]
    n_samples = mic_signals.shape[1]
    t_ms = np.arange(n_samples) / fs * 1000

    if zoom_window_ms is None:
        # Auto-detect first arrival
        threshold = 0.05 * np.max(np.abs(mic_signals))
        first_arrivals = []
        for signal in mic_signals:
            above = np.where(np.abs(signal) > threshold)[0]
            if len(above) > 0:
                first_arrivals.append(above[0])
        if first_arrivals:
            first_sample = max(0, min(first_arrivals) - int(0.003 * fs))  # 3 ms before
            t_start_ms = first_sample / fs * 1000
            t_end_ms = t_start_ms + 30.0  # Show 30 ms window
        else:
            t_start_ms, t_end_ms = 0.0, 50.0
    else:
        t_start_ms, t_end_ms = zoom_window_ms

    fig, axes = plt.subplots(n_mics, 1, figsize=(12, 2.5 * n_mics), sharex=True)
    if n_mics == 1:
        axes = [axes]

    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    sample_start = max(0, int(t_start_ms / 1000 * fs))
    sample_end = min(n_samples, int(t_end_ms / 1000 * fs))

    for idx, (ax, signal, label) in enumerate(zip(axes, mic_signals, mic_labels)):
        color = colors[idx % len(colors)]
        t_slice = t_ms[sample_start:sample_end]
        s_slice = signal[sample_start:sample_end]
        ax.plot(t_slice, s_slice, color=color, linewidth=1.2, alpha=0.9)
        ax.set_ylabel(label, fontsize=11, fontweight="bold", rotation=0, labelpad=35)
        ax.axhline(0, color="#aaa", linewidth=0.5)
        # Mark peak
        if len(s_slice) > 0:
            peak_idx = np.argmax(np.abs(s_slice))
            ax.axvline(t_slice[peak_idx], color=color, linewidth=1.0, linestyle="--", alpha=0.7)

    axes[-1].set_xlabel("Time (ms)", fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, dpi=dpi)
    return fig


# =============================================================================
# Figure 4: GCC-PHAT Responses
# =============================================================================

def plot_gcc_phat(
    tdoa_results: dict,
    speed_of_sound: float,
    room_dimensions: list[float],
    title: str = "GCC-PHAT Cross-Correlation Responses",
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Plot the GCC-PHAT response for each microphone pair.

    A vertical dashed line marks the estimated TDOA peak.

    Parameters
    ----------
    tdoa_results : dict
        Keys: (i, j). Values: TDOAResult objects.
    speed_of_sound : float
        For computing distance axis annotation.
    room_dimensions : list[float]
        Used to determine max plausible tau for axis limits.
    """
    _apply_style()
    n_pairs = len(tdoa_results)
    fig, axes = plt.subplots(n_pairs, 1, figsize=(12, 3.5 * n_pairs), squeeze=False)

    colors = ["#9b59b6", "#e67e22", "#1abc9c"]

    for ax, ((i, j), result), color in zip(axes[:, 0], sorted(tdoa_results.items()), colors):
        response = result.gcc_phat_response
        tau_axis = result.tau_axis_seconds * 1000  # Convert to ms

        # Limit display to ±50 ms for readability
        max_tau_ms = min(50.0, np.max(np.abs(tau_axis)) * 0.8)
        mask = np.abs(tau_axis) <= max_tau_ms

        ax.plot(tau_axis[mask], response[mask], color=color, linewidth=1.2, alpha=0.85)

        # Mark estimated peak
        est_tau_ms = result.tau_seconds * 1000
        ax.axvline(est_tau_ms, color=color, linewidth=1.5, linestyle="--",
                   label=f"τ̂ = {est_tau_ms:+.3f} ms ({result.tau_seconds*speed_of_sound*100:.1f} cm)")
        ax.axvline(0, color="#aaa", linewidth=0.8, linestyle=":")

        ax.set_ylabel("GCC-PHAT", fontsize=10)
        ax.set_title(f"Pair {result.label}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.set_xlabel("τ (ms)", fontsize=10)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, dpi=dpi)
    return fig


# =============================================================================
# Figure 5: Single-Source Localization Result
# =============================================================================

def plot_localization_single(
    room_dimensions: list[float],
    mic_positions_2d: np.ndarray,
    mic_labels: list[str],
    true_position: np.ndarray,
    predicted_position: np.ndarray,
    stage: str = "",
    title: str = "Source Localization Result",
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Plot the actual vs. predicted source position for one source.

    Shows:
    - Room boundary
    - Microphone positions (triangles)
    - True source (star)
    - Predicted source (X marker)
    - Error line connecting true to predicted

    Parameters
    ----------
    true_position : np.ndarray
        Shape (2,). Ground truth.
    predicted_position : np.ndarray
        Shape (2,). NLS estimate.
    stage : str
        'anechoic' or 'reverberant' (shown in title).
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    Lx, Ly = room_dimensions[0], room_dimensions[1]

    # Room
    room_rect = patches.Rectangle(
        (0, 0), Lx, Ly,
        linewidth=2.5, edgecolor="#2c3e50", facecolor="#ecf0f1", alpha=0.4, zorder=1,
    )
    ax.add_patch(room_rect)

    # Microphones
    mic_colors = ["#e74c3c", "#3498db", "#2ecc71"]
    for idx, (pos, label) in enumerate(zip(mic_positions_2d, mic_labels)):
        color = mic_colors[idx % len(mic_colors)]
        ax.scatter(pos[0], pos[1], s=180, c=color, zorder=5, marker="^",
                   edgecolors="white", linewidths=1.5)
        ax.annotate(label, xy=(pos[0], pos[1]), xytext=(pos[0] + 0.15, pos[1] + 0.2),
                    fontsize=11, fontweight="bold", color=color)

    # True position
    ax.scatter(true_position[0], true_position[1], s=280, c="#e67e22",
               marker="*", zorder=7, label="True Position",
               edgecolors="#d35400", linewidths=1.5)

    # Predicted position
    ax.scatter(predicted_position[0], predicted_position[1], s=200, c="#8e44ad",
               marker="X", zorder=7, label="Predicted Position",
               edgecolors="white", linewidths=1.5)

    # Error line
    error_m = float(np.linalg.norm(predicted_position - true_position))
    ax.plot(
        [true_position[0], predicted_position[0]],
        [true_position[1], predicted_position[1]],
        color="#e74c3c", linewidth=2.0, linestyle="-",
        label=f"Error = {error_m:.4f} m",
        zorder=6,
    )

    # Midpoint label
    mid_x = (true_position[0] + predicted_position[0]) / 2
    mid_y = (true_position[1] + predicted_position[1]) / 2
    ax.annotate(
        f"{error_m:.4f} m",
        xy=(mid_x, mid_y), xytext=(mid_x + 0.2, mid_y + 0.2),
        fontsize=10, color="#e74c3c", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5),
    )

    stage_label = f" [{stage.capitalize()}]" if stage else ""
    ax.set_xlim(-0.3, Lx + 0.3)
    ax.set_ylim(-0.3, Ly + 0.3)
    ax.set_xlabel("X (metres)", fontsize=12)
    ax.set_ylabel("Y (metres)", fontsize=12)
    ax.set_title(title + stage_label, fontsize=14, fontweight="bold", pad=12)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, dpi=dpi)
    return fig


# =============================================================================
# Figure 6: Multi-Source Localization Results
# =============================================================================

def plot_localization_multi(
    room_dimensions: list[float],
    mic_positions_2d: np.ndarray,
    mic_labels: list[str],
    true_positions: np.ndarray,
    predicted_positions: np.ndarray,
    errors: np.ndarray,
    stage: str = "",
    title: str = "Multi-Source Localization Results",
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Plot all true vs. predicted positions with error vectors for multi-source experiment.

    Parameters
    ----------
    true_positions : np.ndarray
        Shape (n_sources, 2). Ground truth positions.
    predicted_positions : np.ndarray
        Shape (n_sources, 2). Estimated positions.
    errors : np.ndarray
        Shape (n_sources,). Euclidean errors in metres.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    Lx, Ly = room_dimensions[0], room_dimensions[1]

    # Room
    room_rect = patches.Rectangle(
        (0, 0), Lx, Ly,
        linewidth=2.5, edgecolor="#2c3e50", facecolor="#ecf0f1", alpha=0.3, zorder=1,
    )
    ax.add_patch(room_rect)

    # Microphones
    mic_colors_bg = ["#e74c3c", "#3498db", "#2ecc71"]
    for idx, (pos, label) in enumerate(zip(mic_positions_2d, mic_labels)):
        color = mic_colors_bg[idx % len(mic_colors_bg)]
        ax.scatter(pos[0], pos[1], s=200, c=color, zorder=8, marker="^",
                   edgecolors="white", linewidths=1.5)
        ax.annotate(label, xy=(pos[0], pos[1]), xytext=(pos[0] + 0.15, pos[1] + 0.2),
                    fontsize=11, fontweight="bold", color=color)

    # Colour-map for error magnitude
    norm = plt.Normalize(vmin=0, vmax=max(errors.max(), 0.1))
    cmap = plt.cm.RdYlGn_r  # Green = low error, red = high error

    for idx, (true_pos, pred_pos, err) in enumerate(zip(true_positions, predicted_positions, errors)):
        color = cmap(norm(err))

        # True position
        ax.scatter(true_pos[0], true_pos[1], s=180, c="none", marker="o",
                   edgecolors="#2c3e50", linewidths=2.0, zorder=5)

        # Predicted position
        ax.scatter(pred_pos[0], pred_pos[1], s=120, c=[color], marker="x",
                   linewidths=2.5, zorder=6)

        # Error vector
        ax.annotate(
            "",
            xy=(pred_pos[0], pred_pos[1]),
            xytext=(true_pos[0], true_pos[1]),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=0.85),
            zorder=4,
        )

        # Source ID label near true position
        ax.annotate(
            f"S{idx+1}",
            xy=(true_pos[0], true_pos[1]),
            xytext=(true_pos[0] - 0.35, true_pos[1] + 0.2),
            fontsize=7, color="#333",
        )

    # Colour bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Localization Error (m)", fontsize=10)

    # Legend elements
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="#2c3e50", markersize=10, label="True Position"),
        Line2D([0], [0], marker="x", color="gray", markersize=10, label="Predicted Position"),
        Line2D([0], [0], marker="^", color="#e74c3c", markersize=10, label="Microphone"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    stage_label = f" [{stage.capitalize()}]" if stage else ""
    mean_err = float(np.mean(errors))
    ax.set_xlim(-0.3, Lx + 0.3)
    ax.set_ylim(-0.3, Ly + 0.3)
    ax.set_xlabel("X (metres)", fontsize=12)
    ax.set_ylabel("Y (metres)", fontsize=12)
    ax.set_title(
        f"{title}{stage_label}\n(Mean error: {mean_err:.4f} m, N={len(errors)})",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, dpi=dpi)
    return fig


# =============================================================================
# Bonus: Stage A vs B comparison
# =============================================================================

def plot_error_comparison(
    errors_anechoic: np.ndarray,
    errors_reverberant: np.ndarray,
    title: str = "Localization Error: Anechoic vs. Reverberant",
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Side-by-side box plots comparing localization errors for both stages.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    data = [errors_anechoic, errors_reverberant]
    labels = ["Stage A\n(Anechoic)", "Stage B\n(Reverberant)"]
    colors = ["#3498db", "#e74c3c"]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay individual points
    for i, (d, color) in enumerate(zip(data, colors), start=1):
        jitter = np.random.default_rng(0).uniform(-0.1, 0.1, size=len(d))
        ax.scatter(np.full_like(d, i) + jitter, d, alpha=0.6, s=40, c=color, zorder=5)

    ax.set_ylabel("Localization Error (m)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.4)

    # Annotate means
    for i, (d, label) in enumerate(zip(data, labels), start=1):
        ax.annotate(
            f"Mean: {np.mean(d):.4f} m",
            xy=(i, np.mean(d)), xytext=(i + 0.25, np.mean(d)),
            fontsize=9, color="#333",
            arrowprops=dict(arrowstyle="->", color="#aaa", lw=1.0),
        )

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, dpi=dpi)
    return fig
