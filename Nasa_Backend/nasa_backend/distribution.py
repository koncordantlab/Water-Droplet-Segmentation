# nasa_backend/distribution.py
"""Size-distribution math: equivalent-circular diameters, global log-spaced
bin edges, per-checkpoint stats blocks, and long-format histogram frames.
Bin edges are global-per-class across checkpoints and log-spaced to pair with
the frontend's log x-axis — never rebin per checkpoint."""
import numpy as np
import pandas as pd

from nasa_backend.config import SIZE_DIST_BINS


def _eq_diameter(areas):
    """Convert mask pixel areas to equivalent circular diameter in pixels:
    d = √(4·A/π), i.e. the diameter of a circle with the same area as the mask.
    Compresses dynamic range vs raw area so the histogram axis stays readable.
    """
    return np.sqrt(4.0 * np.asarray(areas, dtype=float) / np.pi)


def _shared_bin_edges(all_values):
    """Return common log-spaced histogram edges for all checkpoints of a single
    class, so bars render with uniform visual width on a log x-axis.
    Falls back to a synthetic single-bin range if the data is degenerate, or to
    linear spacing if any value is non-positive (shouldn't happen for
    eq-diameters since area > 0 is enforced upstream).
    Returns None when there are no values at all (caller should treat as empty).
    """
    if not all_values:
        return None
    arr = np.asarray(all_values, dtype=float)
    if arr.size == 1 or arr.min() == arr.max():
        return np.array([float(arr.min()), float(arr.min()) + 1.0])
    lo = float(arr.min())
    hi = float(arr.max())
    if lo <= 0:
        return np.histogram_bin_edges(arr, bins=SIZE_DIST_BINS)
    return np.logspace(np.log10(lo), np.log10(hi), SIZE_DIST_BINS + 1)


def _droplet_stats_block(values, edges=None):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        if edges is not None and len(edges) > 1:
            return {
                "count": 0,
                "stats": {"min": None, "max": None, "mean": None, "median": None, "std": None},
                "histogram": {
                    "bin_edges": [round(float(e), 2) for e in edges],
                    "counts": [0] * (len(edges) - 1),
                },
            }
        return {
            "count": 0,
            "stats": {"min": None, "max": None, "mean": None, "median": None, "std": None},
            "histogram": {"bin_edges": [], "counts": []},
        }
    if edges is not None and len(edges) > 1:
        counts, used_edges = np.histogram(arr, bins=edges)
    elif arr.size == 1 or arr.min() == arr.max():
        used_edges = np.array([float(arr.min()), float(arr.min()) + 1.0])
        counts = np.array([int(arr.size)])
    else:
        counts, used_edges = np.histogram(arr, bins=SIZE_DIST_BINS)
    return {
        "count": int(arr.size),
        "stats": {
            "min": round(float(arr.min()), 2),
            "max": round(float(arr.max()), 2),
            "mean": round(float(arr.mean()), 2),
            "median": round(float(np.median(arr)), 2),
            "std": round(float(arr.std()), 2),
        },
        "histogram": {
            "bin_edges": [round(float(e), 2) for e in used_edges],
            "counts": [int(c) for c in counts],
        },
    }

def _histogram_df(values, edges):
    """Build a long-format histogram DataFrame using pre-computed bin edges.
    `edges` must be the same global edges that drive `size_distribution` so the
    per-frame xlsx matches the on-screen plot bar-for-bar.
    """
    if edges is None or len(edges) < 2:
        return pd.DataFrame(columns=["bin_lo", "bin_hi", "bin_center", "count"])
    edges_arr = np.asarray(edges, dtype=float)
    if values:
        counts, _ = np.histogram(np.asarray(values, dtype=float), bins=edges_arr)
    else:
        counts = np.zeros(len(edges_arr) - 1, dtype=int)
    lo = edges_arr[:-1]
    hi = edges_arr[1:]
    return pd.DataFrame({
        "bin_lo": np.round(lo, 3),
        "bin_hi": np.round(hi, 3),
        "bin_center": np.round((lo + hi) / 2.0, 3),
        "count": counts.astype(int),
    })


def _global_bin_edges_from_size_distribution(size_distribution, class_key):
    """Pull the shared log-spaced bin edges for a class from any non-degenerate
    checkpoint. Returns None if size_distribution is missing or every checkpoint
    has fewer than 2 edges (no data for the class).
    """
    if not size_distribution or not size_distribution.get("checkpoints"):
        return None
    for cp in size_distribution["checkpoints"]:
        edges = cp.get(class_key, {}).get("histogram", {}).get("bin_edges") or []
        if len(edges) >= 2:
            return edges
    return None

