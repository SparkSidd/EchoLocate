"""
EchoLocate — Visualizations: Microphone Geometry Study (Figures B1–B8)
=======================================================================
Generates all 8 required figures for Part B of Experiment 2.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

GEOM_COLORS = {
    "geometry_a": "#2563eb",  # Blue
    "geometry_b": "#16a34a",  # Green
    "geometry_c": "#d97706",  # Amber
    "geometry_d": "#dc2626",  # Red
}


def set_plot_theme():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.titlesize": 13,
        "figure.dpi": 300,
    })


def generate_all_geometry_figures(
    cfg: dict,
    df_geom_summary: pd.DataFrame,
    df_pre_analysis: pd.DataFrame,
    df_rankings: pd.DataFrame,
    output_dir: Path,
):
    """
    Generate and save Figures B1 through B8.
    """
    set_plot_theme()
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating Microphone Geometry Figures B1–B8 into: %s", fig_dir)

    room_dims = cfg["room"]["dimensions"]
    geometries_cfg = cfg["geometry_study"]["geometries"]
    
    # Load canonical source positions
    src_file = Path(cfg["source"]["positions_file"])
    if not src_file.is_absolute():
        src_file = Path(__file__).resolve().parent.parent.parent / src_file
    df_sources = pd.read_csv(src_file)

    # --------------------------------------------------------------------------
    # Figure B1: Room Layout showing all 4 Microphone Geometries
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    geom_items = list(geometries_cfg.items())
    
    for ax, (g_key, g_info) in zip(axes.flatten(), geom_items):
        ax.set_xlim(-0.5, room_dims[0] + 0.5)
        ax.set_ylim(-0.5, room_dims[1] + 0.5)
        
        # Room boundary
        ax.add_patch(Rectangle((0, 0), room_dims[0], room_dims[1], fill=False, edgecolor="#333333", linewidth=2))
        
        # Mic coordinates
        m_coords = np.array([g_info["positions"]["M1"], g_info["positions"]["M2"], g_info["positions"]["M3"]])
        tri_color = GEOM_COLORS.get(g_key, "#2563eb")
        
        # Array triangle polygon
        tri = Polygon(m_coords, closed=True, fill=True, facecolor=tri_color, edgecolor=tri_color, linestyle="--", alpha=0.25)
        ax.add_patch(tri)
        
        # Microphones
        for m_lbl, pos in g_info["positions"].items():
            ax.scatter(pos[0], pos[1], marker="^", s=150, color=tri_color, edgecolor="black", zorder=5)
            ax.annotate(f"{m_lbl} ({pos[0]:.1f}, {pos[1]:.1f})", (pos[0], pos[1]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, fontweight="bold")
            
        # Sources
        ax.scatter(df_sources["x"], df_sources["y"], marker="o", s=40, color="#475569", alpha=0.8, zorder=4)
        for _, s_row in df_sources.iterrows():
            ax.annotate(s_row["source_id"], (s_row["x"], s_row["y"]), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=7)
            
        # Triangle area
        p1, p2, p3 = m_coords[0], m_coords[1], m_coords[2]
        area_val = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
        ax.set_title(f"{g_info['name']}\nArray Area: {area_val:.1f} m$^2$", fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, linestyle="--")
        
    axes[1, 0].set_xlabel("X (meters)")
    axes[1, 1].set_xlabel("X (meters)")
    axes[0, 0].set_ylabel("Y (meters)")
    axes[1, 0].set_ylabel("Y (meters)")
    fig.suptitle("Figure B1: Room Layout and Spatial Geometries of the 4 Microphone Arrays", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "figB1_geometry_room_layouts.png")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure B2: Mean Localization Error by Geometry — Anechoic Clean
    # --------------------------------------------------------------------------
    df_c1 = df_geom_summary[df_geom_summary["condition_name"].str.contains("Condition 1")].sort_values("mean_localization_error_m")
    if len(df_c1) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors_c1 = [GEOM_COLORS.get(k, "#2563eb") for k in df_c1["geometry_id"]]
        bars = ax.bar(df_c1["geometry_name"], df_c1["mean_localization_error_m"] * 100.0, color=colors_c1, edgecolor="black", alpha=0.85, width=0.55)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f} cm", (bar.get_x() + bar.get_width() / 2, h), textcoords="offset points", xytext=(0, 4), ha="center", fontweight="bold")
        ax.set_ylabel("Mean Localization Error (cm)")
        ax.set_title("Figure B2: Mean Localization Error by Geometry — Condition 1 (Anechoic Clean / 60 dB SNR)")
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")
        plt.xticks(rotation=15, ha="right")
        fig.tight_layout()
        fig.savefig(fig_dir / "figB2_error_by_geometry_anechoic.png")
        plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure B3: Mean Localization Error by Geometry — Moderate Reverberation
    # --------------------------------------------------------------------------
    df_c3 = df_geom_summary[df_geom_summary["condition_name"].str.contains("Condition 3")].sort_values("mean_localization_error_m")
    if len(df_c3) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors_c3 = [GEOM_COLORS.get(k, "#2563eb") for k in df_c3["geometry_id"]]
        bars = ax.bar(df_c3["geometry_name"], df_c3["mean_localization_error_m"], color=colors_c3, edgecolor="black", alpha=0.85, width=0.55)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f} m", (bar.get_x() + bar.get_width() / 2, h), textcoords="offset points", xytext=(0, 4), ha="center", fontweight="bold")
        ax.set_ylabel("Mean Localization Error (meters)")
        ax.set_title("Figure B3: Mean Localization Error by Geometry — Condition 3 ($RT_{60}=0.3$s / 20 dB SNR)")
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")
        plt.xticks(rotation=15, ha="right")
        fig.tight_layout()
        fig.savefig(fig_dir / "figB3_error_by_geometry_moderate_reverb.png")
        plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure B4: Mean Localization Error by Geometry — Difficult Reverberation
    # --------------------------------------------------------------------------
    df_c5 = df_geom_summary[df_geom_summary["condition_name"].str.contains("Condition 5")].sort_values("mean_localization_error_m")
    if len(df_c5) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors_c5 = [GEOM_COLORS.get(k, "#2563eb") for k in df_c5["geometry_id"]]
        bars = ax.bar(df_c5["geometry_name"], df_c5["mean_localization_error_m"], color=colors_c5, edgecolor="black", alpha=0.85, width=0.55)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f} m", (bar.get_x() + bar.get_width() / 2, h), textcoords="offset points", xytext=(0, 4), ha="center", fontweight="bold")
        ax.set_ylabel("Mean Localization Error (meters)")
        ax.set_title("Figure B4: Mean Localization Error by Geometry — Condition 5 ($RT_{60}=0.6$s / 10 dB SNR)")
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")
        plt.xticks(rotation=15, ha="right")
        fig.tight_layout()
        fig.savefig(fig_dir / "figB4_error_by_geometry_difficult.png")
        plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure B5: Grouped Comparison — Geometry x Environmental Condition
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    cond_names = df_geom_summary["condition_name"].unique()
    x_pos = np.arange(len(cond_names))
    present_geoms = df_geom_summary["geometry_id"].unique()
    n_geom = len(present_geoms)
    width = 0.8 / max(1, n_geom)
    
    for idx, g_key in enumerate(present_geoms):
        g_info = geometries_cfg.get(g_key, {"name": g_key})
        df_sub = df_geom_summary[df_geom_summary["geometry_id"] == g_key]
        vals = []
        for cn in cond_names:
            m_val = df_sub[df_sub["condition_name"] == cn]["mean_localization_error_m"].values
            vals.append(m_val[0] if len(m_val) > 0 else 0.0)
        color = GEOM_COLORS.get(g_key, "#2563eb")
        offset = (idx - (n_geom - 1) / 2.0) * width
        ax.bar(x_pos + offset, vals, width, label=g_info["name"], color=color, edgecolor="black", alpha=0.85)
        
    ax.set_ylabel("Mean Localization Error (meters)")
    ax.set_title("Figure B5: Multi-Condition Performance Comparison across 4 Microphone Geometries")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"C{i+1}\n({cn.split(':')[1].strip()})" for i, cn in enumerate(cond_names)], rotation=0)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax.legend(frameon=True, facecolor="white")
    fig.tight_layout()
    fig.savefig(fig_dir / "figB5_grouped_geometry_condition_comparison.png")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure B6: Source Position x Geometry Error Heatmap
    # --------------------------------------------------------------------------
    # Compute mean error per source per geometry across all conditions
    # Load all evaluation records from per_geometry
    per_geom_dir = output_dir / "per_geometry"
    all_eval_dfs = []
    for g_key in geometries_cfg.keys():
        f_path = per_geom_dir / f"{g_key}_evaluations.csv"
        if f_path.exists():
            all_eval_dfs.append(pd.read_csv(f_path))
            
    if len(all_eval_dfs) > 0:
        df_all_eval = pd.concat(all_eval_dfs, ignore_index=True)
        pivot_src_geom = df_all_eval.groupby(["source_id", "geometry_name"])["localization_error"].mean().unstack()
        
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(pivot_src_geom, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={"label": "Mean Error across Conditions (m)"}, linewidths=0.5, ax=ax)
        ax.set_xlabel("Microphone Array Geometry")
        ax.set_ylabel("Source Position ID")
        ax.set_title("Figure B6: Spatial Robustness Heatmap: Source Position vs. Microphone Geometry")
        plt.xticks(rotation=15, ha="right")
        fig.tight_layout()
        fig.savefig(fig_dir / "figB6_source_vs_geometry_heatmap.png")
        plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure B7: Microphone Triangle Area vs. Overall Localization Error
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.scatter(
        df_rankings["triangle_area_m2"],
        df_rankings["overall_mean_error_m"],
        s=120,
        c=[GEOM_COLORS.get(k, "#2563eb") for k in df_rankings["geometry_id"]],
        edgecolor="black",
        zorder=5,
    )
    
    for _, r in df_rankings.iterrows():
        ax.annotate(f"{r['geometry_name']}\n({r['overall_mean_error_m']:.2f} m)",
                    (r["triangle_area_m2"], r["overall_mean_error_m"]),
                    textcoords="offset points", xytext=(8, -5), fontsize=9, fontweight="bold")
        
    ax.set_xlabel("Microphone Array Triangle Area ($m^2$)")
    ax.set_ylabel("Overall Mean Localization Error (meters)")
    ax.set_title("Figure B7: Array Aperture Area vs. Average Multi-Condition Localization Error")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(fig_dir / "figB7_triangle_area_vs_error.png")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Figure B8: Nearest Microphone Distance vs. Localization Error
    # --------------------------------------------------------------------------
    if len(all_eval_dfs) > 0:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        
        # Merge nearest mic distance from df_pre_analysis
        df_merged = df_all_eval.merge(df_pre_analysis[["geometry_id", "source_id", "dist_nearest_mic_m"]], on=["geometry_id", "source_id"])
        
        for g_key, g_info in geometries_cfg.items():
            df_g = df_merged[df_merged["geometry_id"] == g_key]
            color = GEOM_COLORS.get(g_key, "#2563eb")
            ax.scatter(df_g["dist_nearest_mic_m"], df_g["localization_error"], label=g_info["name"], color=color, alpha=0.5, s=40, edgecolors="none")
            
        ax.set_xlabel("Distance to Nearest Microphone (meters)")
        ax.set_ylabel("Localization Error (meters)")
        ax.set_title("Figure B8: Localization Error vs. Distance to Nearest Microphone by Geometry")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(frameon=True, facecolor="white")
        fig.tight_layout()
        fig.savefig(fig_dir / "figB8_nearest_mic_dist_vs_error.png")
        plt.close(fig)

    logger.info("Successfully generated and saved all 8 Part B figures.")
