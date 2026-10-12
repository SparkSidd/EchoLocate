"""
EchoLocate — Experiment 3 Visualization Suite
==============================================
Generates publication-quality Figures F1 through F13 for SRP-PHAT spatial
likelihood analysis, comparative benchmarks, and spatial response heatmaps.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Standard publication aesthetics
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def generate_all_experiment_03_figures(
    df_summary: pd.DataFrame,
    df_evals: pd.DataFrame,
    df_per_source: pd.DataFrame,
    df_resolution: pd.DataFrame,
    diagnostic_maps: dict[str, Any],
    output_dir: str | Path = "results/experiment_03/figures",
) -> list[Path]:
    """
    Generate all 13 required publication-quality figures for Experiment 3.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_files: list[Path] = []

    logger.info("Generating Experiment 3 Figures F1–F13 into: %s", out_dir)

    # Clean condition labels
    cond_labels = {
        "Condition_A_Stable_Control": "Cond A (0.23s / 40dB)",
        "Condition_B_Transition_Edge": "Cond B (0.34s / 40dB)",
        "Condition_C_Severe_Multipath": "Cond C (0.45s / 20dB)",
        "Condition_D_Extreme_Breakdown": "Cond D (0.56s / 20dB)",
    }
    df_summary_plot = df_summary.copy()
    df_summary_plot["cond_lbl"] = df_summary_plot["condition_id"].map(lambda x: cond_labels.get(x, x))

    df_evals_plot = df_evals.copy()
    df_evals_plot["cond_lbl"] = df_evals_plot["condition_id"].map(lambda x: cond_labels.get(x, x))

    # --- Figure 1: Overall Method Comparison (Grouped Bar Chart) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df_summary_plot))
    width = 0.25
    ax.bar(x - width, df_summary_plot["nls_mean_error_m"], width, label="GCC-PHAT + NLS Baseline", color="#d62728", alpha=0.85)
    ax.bar(x, df_summary_plot["srp_raw_mean_error_m"], width, label="SRP-PHAT (Raw Grid 5cm)", color="#ff7f0e", alpha=0.85)
    ax.bar(x + width, df_summary_plot["srp_refined_mean_error_m"], width, label="SRP-PHAT (Sub-Grid Refined)", color="#2ca02c", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df_summary_plot["cond_lbl"])
    ax.set_ylabel("Mean Localization Error (m)")
    ax.set_title("Figure 1: Mean Localization Error Comparison across Environmental Regimes")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend()
    p1 = out_dir / "F1_overall_method_comparison.png"
    fig.savefig(p1)
    plt.close(fig)
    generated_files.append(p1)

    # --- Figure 2: Mean Error with 95% Confidence Intervals ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df_summary_plot["cond_lbl"], df_summary_plot["nls_mean_error_m"], "o-", label="GCC-PHAT + NLS", color="#d62728", linewidth=2, markersize=8)
    ax.plot(df_summary_plot["cond_lbl"], df_summary_plot["srp_refined_mean_error_m"], "s-", label="SRP-PHAT (Refined)", color="#2ca02c", linewidth=2, markersize=8)
    ax.set_ylabel("Mean Localization Error (m)")
    ax.set_title("Figure 2: Mean Localization Error Trajectory Across Environmental Regimes")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    p2 = out_dir / "F2_mean_error_trajectory.png"
    fig.savefig(p2)
    plt.close(fig)
    generated_files.append(p2)

    # --- Figure 3: Median Error Comparison ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df_summary_plot["cond_lbl"], df_summary_plot["nls_median_error_m"], "o--", label="GCC-PHAT + NLS Median", color="#d62728", linewidth=2, markersize=8)
    ax.plot(df_summary_plot["cond_lbl"], df_summary_plot["srp_refined_median_error_m"], "s--", label="SRP-PHAT Refined Median", color="#2ca02c", linewidth=2, markersize=8)
    ax.set_ylabel("Median Localization Error (m)")
    ax.set_title("Figure 3: Median Localization Error Comparison across Regimes")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    p3 = out_dir / "F3_median_error_comparison.png"
    fig.savefig(p3)
    plt.close(fig)
    generated_files.append(p3)

    # --- Figure 4: Success Rate (<= 1.0 m) Comparison ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df_summary_plot["cond_lbl"], df_summary_plot["nls_pct_within_1_0m"], "o-", label="GCC-PHAT + NLS (<=1.0m)", color="#d62728", linewidth=2, markersize=8)
    ax.plot(df_summary_plot["cond_lbl"], df_summary_plot["srp_refined_pct_within_1_0m"], "s-", label="SRP-PHAT Refined (<=1.0m)", color="#2ca02c", linewidth=2, markersize=8)
    ax.set_ylabel("Success Rate (Error <= 1.0 m) [%]")
    ax.set_ylim(0, 105)
    ax.set_title("Figure 4: Localization Success Rate (<= 1.0 m) across Environmental Regimes")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    p4 = out_dir / "F4_success_rate_comparison.png"
    fig.savefig(p4)
    plt.close(fig)
    generated_files.append(p4)

    # --- Figure 5: Catastrophic Failure Rate (> 2.0 m) ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df_summary_plot["cond_lbl"], df_summary_plot["nls_catastrophic_rate_pct"], "o-", label="GCC-PHAT + NLS Catastrophic", color="#d62728", linewidth=2, markersize=8)
    ax.plot(df_summary_plot["cond_lbl"], df_summary_plot["srp_refined_catastrophic_rate_pct"], "s-", label="SRP-PHAT Refined Catastrophic", color="#2ca02c", linewidth=2, markersize=8)
    ax.set_ylabel("Catastrophic Failure Rate (Error > 2.0 m) [%]")
    ax.set_ylim(-2, max(60, df_summary_plot["nls_catastrophic_rate_pct"].max() + 10))
    ax.set_title("Figure 5: Catastrophic Spatial Failure Rate (> 2.0 m) Comparison")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    p5 = out_dir / "F5_catastrophic_failure_rate.png"
    fig.savefig(p5)
    plt.close(fig)
    generated_files.append(p5)

    # --- Figure 6: Per-Source Error Heatmap ---
    fig, ax = plt.subplots(figsize=(10, 6))
    df_ps_plot = df_per_source.sort_values("source_id")
    x_indices = np.arange(len(df_ps_plot))
    w = 0.35
    ax.bar(x_indices - w/2, df_ps_plot["nls_mean_error_m"], w, label="GCC-PHAT + NLS", color="#d62728", alpha=0.85)
    ax.bar(x_indices + w/2, df_ps_plot["srp_refined_mean_error_m"], w, label="SRP-PHAT Refined", color="#2ca02c", alpha=0.85)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(df_ps_plot["source_id"])
    ax.set_ylabel("Mean Error (m)")
    ax.set_title("Figure 6: Per-Source Mean Localization Error Breakdown (All 15 Canonical Sources)")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend()
    p6 = out_dir / "F6_per_source_breakdown.png"
    fig.savefig(p6)
    plt.close(fig)
    generated_files.append(p6)

    # --- Figures 7, 8, 9: Spatial Heatmap Diagnostic Maps ---
    def plot_single_spatial_map(map_data: dict[str, Any], title: str, filename: str) -> Path:
        fig_m, ax_m = plt.subplots(figsize=(8, 5))
        srp_map = np.array(map_data["srp_map"])
        x_coords = np.array(map_data["x_coords"])
        y_coords = np.array(map_data["y_coords"])
        extent = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]

        im = ax_m.imshow(srp_map, extent=extent, origin="lower", cmap="inferno", aspect="equal")
        plt.colorbar(im, ax=ax_m, label="Steered Response Power")

        # Ground truth
        t_xy = map_data["true_xy"]
        ax_m.plot(t_xy[0], t_xy[1], "g*", markersize=14, label=f"True Source ({t_xy[0]}, {t_xy[1]})")

        # NLS Prediction
        n_xy = map_data["nls_pred_xy"]
        ax_m.plot(n_xy[0], n_xy[1], "rx", markersize=10, markeredgewidth=2, label=f"NLS Pred (Err={map_data['nls_error']:.2f}m)")

        # SRP Prediction
        s_xy = map_data["srp_refined_xy"]
        ax_m.plot(s_xy[0], s_xy[1], "c+", markersize=12, markeredgewidth=2, label=f"SRP Pred (Err={map_data['srp_error']:.2f}m)")

        # Microphones
        ax_m.plot([1.0, 9.0, 5.0], [1.5, 1.5, 4.5], "w^", markersize=9, markeredgecolor="k", label="Mics (M1-M3)")

        ax_m.set_xlim(0, 10)
        ax_m.set_ylim(0, 6)
        ax_m.set_xlabel("Room X (m)")
        ax_m.set_ylabel("Room Y (m)")
        ax_m.set_title(title)
        ax_m.legend(loc="upper right", framealpha=0.9, fontsize=8)

        p_out = out_dir / filename
        fig_m.savefig(p_out)
        plt.close(fig_m)
        return p_out

    # F7: Stable Condition Map
    k7 = "Condition_A_Stable_Control_S06"
    if k7 in diagnostic_maps:
        p7 = plot_single_spatial_map(diagnostic_maps[k7], "Figure 7: SRP Spatial Likelihood Map — Condition A (Stable, S06)", "F7_spatial_map_stable.png")
        generated_files.append(p7)

    # F8: Transition Edge Map (e.g. S10 or S04)
    k8 = "Condition_B_Transition_Edge_S10" if "Condition_B_Transition_Edge_S10" in diagnostic_maps else "Condition_B_Transition_Edge_S04"
    if k8 in diagnostic_maps:
        p8 = plot_single_spatial_map(diagnostic_maps[k8], "Figure 8: SRP Spatial Map — Condition B (Transition Edge, S10)", "F8_spatial_map_transition.png")
        generated_files.append(p8)

    # F9: Severe Multipath Map
    k9 = "Condition_C_Severe_Multipath_S04" if "Condition_C_Severe_Multipath_S04" in diagnostic_maps else "Condition_D_Extreme_Breakdown_S04"
    if k9 in diagnostic_maps:
        p9 = plot_single_spatial_map(diagnostic_maps[k9], "Figure 9: SRP Spatial Map — Severe Multipath Stress", "F9_spatial_map_severe.png")
        generated_files.append(p9)

    # --- Figure 10: Evidence Decoupling Plot ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(df_evals["nls_localization_error"], df_evals["srp_refined_error"], c=df_evals["measured_mean_rt60_s"], cmap="viridis", alpha=0.75, edgecolors="none")
    ax.plot([0, 6], [0, 6], "r--", label="Equality (y = x)")
    ax.set_xlabel("GCC-PHAT + NLS Localization Error (m)")
    ax.set_ylabel("SRP-PHAT Localization Error (m)")
    ax.set_title("Figure 10: Paired Evaluation Scatter (Colored by Measured RT60)")
    ax.grid(True, linestyle="--", alpha=0.6)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Measured Mean RT60 (s)")
    ax.legend(loc="upper left")
    p10 = out_dir / "F10_evidence_decoupling_scatter.png"
    fig.savefig(p10)
    plt.close(fig)
    generated_files.append(p10)

    # --- Figure 11: Resolution Ablation (Accuracy vs Grid Spacing) ---
    if not df_resolution.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df_resolution["grid_resolution_m"] * 100, df_resolution["mean_raw_error_m"], "o-", label="Raw Grid Mean Error", color="#ff7f0e", linewidth=2)
        ax.plot(df_resolution["grid_resolution_m"] * 100, df_resolution["mean_refined_error_m"], "s-", label="Sub-Grid Refined Mean Error", color="#2ca02c", linewidth=2)
        ax.set_xlabel("Grid Resolution Spacing (cm)")
        ax.set_ylabel("Mean Localization Error (m)")
        ax.set_title("Figure 11: Grid Resolution Ablation Study (Localization Accuracy)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        p11 = out_dir / "F11_resolution_ablation_accuracy.png"
        fig.savefig(p11)
        plt.close(fig)
        generated_files.append(p11)

        # --- Figure 12: Computational Efficiency (Runtime vs Grid Spacing) ---
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df_resolution["grid_resolution_m"] * 100, df_resolution["mean_runtime_per_frame_ms"], "d-", color="#1f77b4", linewidth=2, markersize=8)
        ax.set_xlabel("Grid Resolution Spacing (cm)")
        ax.set_ylabel("Mean Evaluation Runtime (ms / frame)")
        ax.set_title("Figure 12: Computational Efficiency vs Grid Spacing")
        ax.grid(True, linestyle="--", alpha=0.6)
        p12 = out_dir / "F12_computational_efficiency_runtime.png"
        fig.savefig(p12)
        plt.close(fig)
        generated_files.append(p12)

    # --- Figure 13: Spatial Peak Competition Distribution ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(df_evals["srp_peak_ratio"], df_evals["srp_refined_error"], c=df_evals["measured_mean_rt60_s"], cmap="plasma", alpha=0.8)
    ax.set_xlabel("Spatial Peak Ratio (Primary Peak S_1 / Secondary Peak S_2)")
    ax.set_ylabel("SRP-PHAT Localization Error (m)")
    ax.set_title("Figure 13: Spatial Peak Dominance Ratio vs Localization Error")
    ax.grid(True, linestyle="--", alpha=0.6)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Measured Mean RT60 (s)")
    p13 = out_dir / "F13_peak_competition_distribution.png"
    fig.savefig(p13)
    plt.close(fig)
    generated_files.append(p13)

    logger.info("Successfully generated all Experiment 3 figures.")
    return generated_files
