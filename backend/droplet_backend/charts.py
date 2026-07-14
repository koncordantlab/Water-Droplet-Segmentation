# droplet_backend/charts.py
"""Headless matplotlib PNG writers for the summary charts and the per-checkpoint
size-distribution histograms."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Plot colors mirror the SummaryPage UI conventions
# (blue = water, red = ice, green = water-ice overlap).
_WATER_PLOT_COLOR = "#1f77b4"
_ICE_PLOT_COLOR = "#d62728"
_WI_OVERLAP_PLOT_COLOR = "#2ca02c"


def _save_chart_pngs(df, charts, overlap_totals, charts_dir):
    """Render the SummaryPage's line plots, donuts, and overlap totals to disk.
    Headless via Agg backend. Output is a matplotlib reconstruction of the
    Plotly figures; styling won't be pixel-identical to the in-browser charts.
    """
    os.makedirs(charts_dir, exist_ok=True)
    x = charts["pct"]["x"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, charts["pct"]["water"], color=_WATER_PLOT_COLOR, marker="o", label="Water (%)")
    ax.plot(x, charts["pct"]["ice"], color=_ICE_PLOT_COLOR, marker="o", label="Ice (%)")
    ax.set_xlabel("Processed Frame (≈ 1 FPS)")
    ax.set_ylabel("%")
    ax.set_title("Water & Ice (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(charts_dir, "water_ice_pct.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, charts["ov"]["ww"], color=_WATER_PLOT_COLOR, marker="o", label="Water–Water")
    ax.plot(x, charts["ov"]["ii"], color=_ICE_PLOT_COLOR, marker="o", label="Ice–Ice")
    ax.plot(x, charts["ov"]["wi"], color=_WI_OVERLAP_PLOT_COLOR, marker="o", label="Water–Ice")
    ax.set_xlabel("Processed Frame (≈ 1 FPS)")
    ax.set_ylabel("Overlap count")
    ax.set_title("Overlap Counts")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(charts_dir, "overlap_counts.png"), dpi=150)
    plt.close(fig)

    donuts = charts["donuts"]

    # Water Count single-slice donut — matches the UI's label-card behavior;
    # skip when zero (the React panel hides itself in that case).
    if donuts.get("water_count", 0) > 0:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie([donuts["water_count"]], labels=[f"Water: {donuts['water_count']}"],
               colors=[_WATER_PLOT_COLOR], wedgeprops=dict(width=0.4))
        ax.set_title("Water Count")
        fig.tight_layout()
        fig.savefig(os.path.join(charts_dir, "donut_water_count.png"), dpi=150)
        plt.close(fig)

    if donuts.get("ice_count", 0) > 0:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie([donuts["ice_count"]], labels=[f"Ice: {donuts['ice_count']}"],
               colors=[_ICE_PLOT_COLOR], wedgeprops=dict(width=0.4))
        ax.set_title("Ice Count")
        fig.tight_layout()
        fig.savefig(os.path.join(charts_dir, "donut_ice_count.png"), dpi=150)
        plt.close(fig)

    void = max(0.0, min(100.0, float(donuts.get("void_pct_avg", 0.0))))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie([void, 100 - void], labels=[f"Void: {void:.1f}%", "Rest"],
           colors=["#7f7f7f", "#d9d9d9"], wedgeprops=dict(width=0.4))
    ax.set_title("Void (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(charts_dir, "donut_void_pct.png"), dpi=150)
    plt.close(fig)

    conf = max(0.0, min(100.0, float(donuts.get("avg_conf", 0.0))))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie([conf, 100 - conf], labels=[f"Avg Conf: {conf:.1f}%", "Rest"],
           colors=["#9467bd", "#d9d9d9"], wedgeprops=dict(width=0.4))
    ax.set_title("Avg Confidence (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(charts_dir, "donut_avg_conf.png"), dpi=150)
    plt.close(fig)


def _save_size_distribution_pngs(size_distribution, charts_dir, video_fname_base):
    """Render per-checkpoint droplet size-distribution histograms (water + ice
    side-by-side) to disk. One PNG per checkpoint, log x-axis, shared bin
    edges and y-range per class across checkpoints so frames are visually
    comparable (matches the frontend's global-binning contract).
    """
    if not size_distribution or not size_distribution.get("checkpoints"):
        return
    os.makedirs(charts_dir, exist_ok=True)

    checkpoints = size_distribution["checkpoints"]
    water_y_max = size_distribution.get("y_max", {}).get("water", 0)
    ice_y_max = size_distribution.get("y_max", {}).get("ice", 0)

    def _global_xlim(class_key):
        for cp in checkpoints:
            edges = cp[class_key]["histogram"]["bin_edges"]
            if len(edges) >= 2:
                return float(edges[0]), float(edges[-1])
        return None

    water_xlim = _global_xlim("water")
    ice_xlim = _global_xlim("ice")

    for cp in checkpoints:
        frame = int(cp["frame"])
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, class_key, color, xlim, y_max in (
            (axes[0], "water", _WATER_PLOT_COLOR, water_xlim, water_y_max),
            (axes[1], "ice", _ICE_PLOT_COLOR, ice_xlim, ice_y_max),
        ):
            block = cp[class_key]
            hist = block["histogram"]
            edges = np.asarray(hist["bin_edges"], dtype=float)
            counts = hist["counts"]
            label = class_key.capitalize()
            if edges.size >= 2 and counts:
                widths = np.diff(edges)
                ax.bar(
                    edges[:-1], counts, width=widths, align="edge",
                    color=color, alpha=0.85, edgecolor="black", linewidth=0.3,
                    label=f"{label} (n={block['count']})",
                )
                if xlim and xlim[0] > 0 and xlim[1] > xlim[0]:
                    ax.set_xscale("log")
                    ax.set_xlim(xlim)
                ax.legend(loc="upper right")
            else:
                ax.text(
                    0.5, 0.5, f"No {label.lower()} droplets",
                    ha="center", va="center", transform=ax.transAxes,
                )
            ax.set_xlabel("Equivalent circular diameter (pixels)")
            ax.set_ylabel("Droplet count")
            ax.set_title(f"{label} — frame {frame}")
            if y_max and y_max > 0:
                ax.set_ylim(0, y_max * 1.05)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Droplet size distribution — frame {frame}")
        fig.tight_layout()
        out_path = os.path.join(
            charts_dir, f"{video_fname_base}_size_dist_frame_{frame:06d}.png"
        )
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
