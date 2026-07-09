import time
start = time.time()
import math
import os
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import dash
from dash import html, dcc, Input, Output, State, no_update
from dash.exceptions import PreventUpdate
import dash_daq as daq
from ultralytics import YOLO
from flask import request, jsonify, send_file, url_for
from flask_cors import CORS
import threading
import traceback
import urllib.parse
import uuid
import json
from queue import Queue, Empty
from flask import Response, stream_with_context

# --- Device Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── 1) Model & Helper Functions ─────────────────────────────────────────────
MODEL_PATH = r"app_root/weights_DP(8).pt"
model      = YOLO(MODEL_PATH)
model.to(device)

# COLOR MAPS etc.
COLOR_MAP = {"water": (255, 0, 0), "ice": (0, 128, 0)}
OVERLAP_COLORS = {"ww": (0, 0, 255), "ii": (255, 165, 0), "wi": (255, 255, 0)}
ALPHA_SEG = 0.5
ALPHA_OVERLAP = 0.65

def blend_mask(base_img, mask, color, alpha):
    idx = mask.astype(bool)
    base_img[idx] = base_img[idx] * (1 - alpha) + np.array(color) * alpha
    return base_img

def apply_full_overlay(img, masks_np, class_names, mask_thresh=0.3, full_masks=None):
    h, w = img.shape[:2]
    # full_masks may be supplied pre-computed (already thresholded + resized to
    # (h, w)) so the overlay reuses the GPU-resized masks instead of resizing a
    # third time. When omitted, fall back to the original per-mask cv2 resize.
    if full_masks is None:
        full_masks = [cv2.resize((m > mask_thresh).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) for m in masks_np]
    water_union, ice_union = np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)
    for fm, cls in zip(full_masks, class_names):
        if cls == "water": water_union |= fm
        elif cls == "ice": ice_union |= fm
    ww_mask, ii_mask, wi_mask = np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)
    n = len(full_masks)
    for i in range(n):
        for j in range(i+1, n):
            inter = full_masks[i] & full_masks[j]
            if not np.any(inter): continue
            ni, nj = class_names[i], class_names[j]
            if ni == nj == "water": ww_mask |= inter
            elif ni == nj == "ice": ii_mask |= inter
            else: wi_mask |= inter
    base = img.astype(float)
    base = blend_mask(base, water_union, COLOR_MAP["water"], ALPHA_SEG)
    base = blend_mask(base, ice_union, COLOR_MAP["ice"], ALPHA_SEG)
    base = blend_mask(base, ww_mask, OVERLAP_COLORS["ww"], ALPHA_OVERLAP)
    base = blend_mask(base, ii_mask, OVERLAP_COLORS["ii"], ALPHA_OVERLAP)
    base = blend_mask(base, wi_mask, OVERLAP_COLORS["wi"], ALPHA_OVERLAP)
    return np.clip(base, 0, 255).astype(np.uint8)


# Cache of cv2 INTER_NEAREST source-index maps, keyed by (src_len, dst_len). The
# map is tiny (length dst) and identical for every mask that shares a resize, so
# it is derived once per shape pair and reused.
_NN_INDEX_MAP_CACHE = {}


def _nn_resize_index_map(src_len, dst_len):
    """Source indices cv2.resize(..., INTER_NEAREST) picks along one axis.

    Derived by probing cv2 itself with a labelled row, so the mapping is
    *exactly* OpenCV's nearest-neighbour rule for the installed build rather than
    a re-derived formula that might disagree at boundaries. Returns an int64
    array of length ``dst_len`` with values in ``[0, src_len - 1]``.
    """
    key = (int(src_len), int(dst_len))
    cached = _NN_INDEX_MAP_CACHE.get(key)
    if cached is None:
        probe = np.arange(src_len, dtype=np.float32).reshape(1, src_len)
        picks = cv2.resize(probe, (int(dst_len), 1), interpolation=cv2.INTER_NEAREST)
        cached = picks.reshape(int(dst_len)).astype(np.int64)
        _NN_INDEX_MAP_CACHE[key] = cached
    return cached


def _threshold_masks(prob, thresh=0.3):
    """Binarise a stack of float masks on its own device. uint8, same shape."""
    return (prob > thresh).to(torch.uint8)


def _gather_resize_nn(binm, dst_h, dst_w):
    """Nearest-resize a uint8 mask stack to (dst_h, dst_w) via cv2's index map.

    Pure gather (index_select) using cv2's own nearest index map — NOT torch
    interpolation, whose 'nearest' convention differs from OpenCV's — so the
    result is bit-identical to cv2.resize(INTER_NEAREST) on GPU and CPU alike.
    """
    src_h, src_w = int(binm.shape[-2]), int(binm.shape[-1])
    y_map = torch.as_tensor(_nn_resize_index_map(src_h, dst_h), device=binm.device)
    x_map = torch.as_tensor(_nn_resize_index_map(src_w, dst_w), device=binm.device)
    return binm.index_select(-2, y_map).index_select(-1, x_map)


def _resize_bin_masks_nn(prob, dst_h, dst_w, thresh=0.3):
    """Threshold a stack of float masks and nearest-resize them to full res.

    ``prob`` is a torch tensor of shape (N, src_h, src_w) on any device. Returns
    a uint8 (N, dst_h, dst_w) tensor on the *same* device, bit-identical to
    ``stack(cv2.resize((prob[k] > thresh).astype(uint8), (dst_w, dst_h),
    INTER_NEAREST) for k)``. See test_gpu_mask_equivalence.py.
    """
    return _gather_resize_nn(_threshold_masks(prob, thresh), dst_h, dst_w)


def _mask_areas_from_source(binm, dst_h, dst_w):
    """Per-instance full-resolution pixel areas, straight from the source masks.

    ``binm`` is a (N, src_h, src_w) uint8 tensor of 0/1 values. Returns an (N,)
    int64 numpy array exactly equal to
    ``[int(cv2.resize(binm[k], (dst_w, dst_h), INTER_NEAREST).sum()) for k]`` —
    *without* materialising the (N, dst_h, dst_w) masks. Nearest upsampling
    replicates each source pixel (i, j) exactly ``cy[i] * cx[j]`` times, where
    cy[i] / cx[j] are how many destination rows / cols map onto that source row /
    col (the bincount of cv2's nearest index map). Pure integer arithmetic on the
    masks' device, so it is exact (no float rounding) and only an (N,) vector
    crosses to the CPU — not the multi-GB full-res masks. See
    test_gpu_mask_equivalence.py.
    """
    src_h, src_w = int(binm.shape[-2]), int(binm.shape[-1])
    cy = np.bincount(_nn_resize_index_map(src_h, dst_h), minlength=src_h).astype(np.int32)
    cx = np.bincount(_nn_resize_index_map(src_w, dst_w), minlength=src_w).astype(np.int32)
    cy_t = torch.as_tensor(cy, device=binm.device).view(1, src_h, 1)
    cx_t = torch.as_tensor(cx, device=binm.device).view(1, 1, src_w)
    weight = cy_t * cx_t                                   # (1, src_h, src_w) int32
    areas = (binm.to(torch.int32) * weight).sum(dim=(1, 2), dtype=torch.int64)
    return areas.cpu().numpy()


def _overlap_exists_matrix(masks_2d):
    """(N, N) bool: whether masks k and j share at least one set pixel.

    ``masks_2d`` is an (N, P) torch tensor of 0/1 values. Computed as
    ``(M @ Mᵀ) > 0`` in float32: every product is an exact 0 or 1 and the
    accumulation is non-negative, so a pair that shares no pixel sums to exactly
    0.0 and a pair that shares any pixel sums to ≥ 1.0 — the ``> 0`` test is
    therefore exact regardless of accumulation order or TF32, and reproduces
    ``np.any(mask_k & mask_j)`` bit-for-bit (test_gpu_mask_equivalence.py).
    """
    n = int(masks_2d.shape[0])
    if n == 0:
        return torch.zeros((0, 0), dtype=torch.bool, device=masks_2d.device)
    m = masks_2d.to(torch.float32)
    return (m @ m.t()) > 0


def _classify_overlaps(exists_matrix, class_names):
    """Tally (ww, ii, mixed) over unordered overlapping pairs.

    ``exists_matrix`` is an (N, N) bool array (numpy) whose [k, j] entry marks
    whether instances k and j overlap. Matches the original nested loop exactly:
    a pair counts as ``ww`` only if both classes are "water", ``ii`` only if both
    are "ice", and ``mixed`` for everything else (incl. any non-water/ice class).
    """
    n = len(class_names)
    if n == 0:
        return 0, 0, 0
    cls = np.array([1 if c == "water" else (2 if c == "ice" else 0) for c in class_names])
    iu, ju = np.triu_indices(n, k=1)
    ex = np.asarray(exists_matrix)[iu, ju].astype(bool)
    ci, cj = cls[iu], cls[ju]
    ww = int(np.count_nonzero(ex & (ci == 1) & (cj == 1)))
    ii = int(np.count_nonzero(ex & (ci == 2) & (cj == 2)))
    mixed = int(np.count_nonzero(ex) - ww - ii)
    return ww, ii, mixed


# ── 2) Dash App Setup ───────────────────────────────────────────────────────
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server
# Allow CORS for API endpoints so front-end apps (e.g. React dev on port 3000) can call /api/*
CORS(server, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)


def make_json_serializable(obj):
    """Recursively convert numpy / pandas types to native Python types for JSON."""
    # handle None
    if obj is None:
        return None
    # primitives — non-finite floats become null: json.dumps would emit a bare
    # NaN/Infinity token, which JSON.parse rejects on the SSE frontend
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (str, int, bool)):
        return obj
    # numpy scalar types
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    # numpy arrays
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    # pandas NA
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    # dict
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    # list/tuple
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    # fallback: try to convert using int/float or str
    try:
        return int(obj)
    except Exception:
        try:
            return float(obj)
        except Exception:
            return str(obj)

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="result-store", storage_type="memory"),
    dcc.Store(id="overlap-store", storage_type="memory"),
    html.Div(id="page-content")
])

# ── 3) Page 1 and 2 Layouts, Router, Clientside callback ───────────────────
def page_1_layout():
    return html.Div(style={"height": "100vh", "width": "100vw", "display": "flex", "flexDirection": "row", "background": "linear-gradient(90deg,#000,#737373)", "color": "white", "overflow": "hidden"},
        children=[
            html.Div(style={"flex": "0 0 35%", "display": "flex", "flexDirection": "column", "justifyContent": "space-between", "textAlign": "center", "height": "100%", "padding": "1rem"},
                children=[
                    html.Div([
                        html.H1("NASA", style={"textDecoration": "underline", "fontSize": "45px"}),
                        html.H2("Video Processing Unit", style={"textDecoration": "underline", "fontSize": "45px"})
                    ]),
                    html.Div([
                        html.Button(html.Img(src="/assets/cloud_upload1.png", style={"width": "150px", "cursor": "pointer"}), id="folder-picker-btn", n_clicks=0, style={"border": "none", "background": "transparent"}),
                        dcc.Input(id="folder-input", placeholder="Paste video file path here", type="text", style={"width": "100%", "marginTop": "1rem", "fontSize": "1.25rem"})
                    ]),
                    html.Div([
                        html.Div(style={"display": "flex", "alignItems": "center", "marginTop": "0.5rem"},
                            children=[
                                daq.ToggleSwitch(id="overlay-toggle", value=True, size=55, color="green"),
                                html.Span("Save segmentation overlay video", style={"marginLeft": "10px", "fontSize": "22px"})
                            ]),
                        dcc.Loading(html.Div(id="progress-text", style={"fontSize": "1.25rem"}), type="circle", style={"display": "flex", "justifyContent": "center", "margin": "1rem 0"}),
                        html.Div(style={"display": "flex", "justifyContent": "center"},
                            children=[html.Button("Run Detection", id="run-btn", n_clicks=0, style={"fontSize": "1.25rem", "padding": "0.75rem 1.5rem"})]
                        ),
                        dcc.Download(id="download-summary")
                    ], id="unit3")
                ])
            ,
            html.Div(html.Img(src="/assets/Ice_image.jpg", style={"width": "100%", "height": "100%", "objectFit": "cover", "display": "block"}), style={"flex": "1", "height": "100%", "overflow": "hidden"})
        ])

def page_2_layout():
    # Outer container jo screen size par fix rahega (scrolling fix)
    return html.Div(style={"height": "100vh", "width": "100vw", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
        children=[
            # Inner container jo content ko hold karega aur scroll hoga
            html.Div(style={"flexGrow": 1, "overflowY": "auto", "position": "relative", "background": "linear-gradient(90deg,#000,#737373)", "color": "white", "margin": 0, "padding": 0},
                children=[
                    dcc.Link(html.Button("Back to Home", style={"position": "absolute", "top": "1rem", "left": "1rem", "backgroundColor": "#e6c645", "color": "white", "border": "none", "padding": "0.5rem 1rem", "borderRadius": "12px", "cursor": "pointer", "opacity": "0.8"}), href="/"),
                    html.Div(style={"display": "flex", "justifyContent": "space-between", "width": "95%", "margin": "4rem auto"},
                        children=[
                            html.Div([html.H4("Water (%) & Ice (%)", style={"textAlign": "center"}), dcc.Graph(id="pct-graph"), html.Ul([html.Li("Blue = Water (%)"), html.Li("Red = Ice (%)")], style={"listStyle": "disc", "paddingLeft": "1.5rem", "marginTop": "0.5rem"})], style={"width": "48%"}),
                            html.Div([html.H4("Overlap Counts", style={"textAlign": "center"}), dcc.Graph(id="ov-graph"), html.Ul([html.Li("Blue = Water–Water"), html.Li("Red  = Ice–Ice"), html.Li("Green= Water–Ice")], style={"listStyle": "disc", "paddingLeft": "1.5rem", "marginTop": "0.5rem"})], style={"width": "48%"})
                        ]),
                    html.Div(id="slider-output-container", style={"marginTop": "20px", "textAlign": "center"}),
                    html.Div(dcc.Slider(id="entry-slider", min=1, max=1, value=1, step=1, marks=None, tooltip={"placement": "bottom", "always_visible": True}, updatemode="drag"), style={"width": "80%", "margin": "0 auto", "paddingBottom": "30px"}),
                    
                    # Dynamic paragraph
                    html.Div(id="dynamic-summary-p", style={"width": "80%", "margin": "2rem auto", "textAlign": "center", "fontSize": "18px", "fontStyle": "italic", "color": "#e6c645"}),
                    
                    html.Div(style={"width": "95%", "margin": "2rem auto", "textAlign": "center"},
                        children=[
                            html.Div([dcc.Graph(id="donut-water"), html.Div("Water Count", style={"fontWeight": "bold"})], style={"width": "25%", "display": "inline-block"}),
                            html.Div([dcc.Graph(id="donut-ice"), html.Div("Ice Count", style={"fontWeight": "bold"})], style={"width": "25%", "display": "inline-block"}),
                            html.Div([dcc.Graph(id="donut-void"), html.Div("Void Percentage", style={"fontWeight": "bold"})], style={"width": "25%", "display": "inline-block"}),
                            html.Div([dcc.Graph(id="donut-conf"), html.Div("Avg Confidence", style={"fontWeight": "bold"})], style={"width": "25%", "display": "inline-block"})
                        ]),
                    html.Div(id="overlap-summary", style={"width": "95%", "margin": "2rem auto", "fontSize": "18px"})
                ])
        ])

@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    return page_2_layout() if pathname == "/summary" else page_1_layout()

app.clientside_callback("function(n_clicks) { if (!n_clicks) return ''; return window.prompt('Enter video file path:'); }", Output("folder-input", "value"), Input("folder-picker-btn", "n_clicks"))

# ── 7) run_detection (Updated for BATCH PROCESSING) ────────────────────────
SIZE_DIST_BINS = 30


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


def _per_instance_metrics(full_bin_masks, boxes, class_names, frame_shape, mode="full", areas=None):
    """Compute per-instance shape descriptors for one frame.

    mode: "full" (default) returns the rich descriptor set below. "basic" returns
        only instance_id, class, confidence, pixel_count, eq_diameter_px and skips
        all contour/ellipse/overlap work, so it is also faster.

    full_bin_masks: list of (H, W) uint8 binary masks, already resized to the
        original frame resolution (same objects used for overlap counting). May be
        ``None`` in basic mode when ``areas`` is supplied — basic mode only needs
        the per-instance pixel area, so the full-res masks are never built.
    boxes: ultralytics Boxes object — provides per-instance confidence + bbox.
    class_names: list of lowercased class strings, parallel to the detections.
    frame_shape: (H, W) tuple of the original frame.
    areas: optional precomputed per-instance pixel areas (e.g. from
        _mask_areas_from_source). When omitted, areas are summed from
        full_bin_masks, reproducing the original behaviour exactly.

    Returns a list of dicts (one per non-empty instance). Mask area is the
    full pixel sum; contour-based metrics (perimeter, circularity, feret, etc.)
    are computed on the largest connected component, which is the dominant blob
    for any well-formed YOLO instance mask.
    """
    H, W = frame_shape
    n = len(boxes)
    if areas is None:
        areas = [int(full_bin_masks[k].sum()) for k in range(n)]

    overlap_info = [{"count": 0, "classes": set(), "pixels": 0} for _ in range(n)]
    if mode != "basic":
        for i in range(n):
            for j in range(i + 1, n):
                inter = full_bin_masks[i] & full_bin_masks[j]
                inter_px = int(inter.sum())
                if inter_px == 0:
                    continue
                overlap_info[i]["count"] += 1
                overlap_info[i]["pixels"] += inter_px
                overlap_info[i]["classes"].add(class_names[j])
                overlap_info[j]["count"] += 1
                overlap_info[j]["pixels"] += inter_px
                overlap_info[j]["classes"].add(class_names[i])

    rows = []
    for idx in range(n):
        area = int(areas[idx])
        box = boxes[idx]
        if area == 0:
            continue

        eq_d = float(np.sqrt(4.0 * area / np.pi))

        if mode == "basic":
            rows.append({
                "instance_id": len(rows) + 1,
                "class": class_names[idx],
                "confidence": round(float(box.conf.item()), 4),
                "pixel_count": area,
                "eq_diameter_px": round(eq_d, 3),
            })
            continue

        fm = full_bin_masks[idx]
        ys, xs = np.where(fm > 0)
        cx, cy = float(xs.mean()), float(ys.mean())
        bx1, by1, bx2, by2 = [float(v) for v in box.xyxy[0].cpu().numpy()]
        bw, bh = bx2 - bx1, by2 - by1
        bbox_area = bw * bh

        contours, _ = cv2.findContours(fm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = circularity = solidity = feret_max = None
        ellipse_major = ellipse_minor = ellipse_ecc = ellipse_angle = None
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            perimeter = float(cv2.arcLength(cnt, True))
            if perimeter > 0:
                circularity = float(4.0 * np.pi * area / (perimeter ** 2))
            hull = cv2.convexHull(cnt)
            hull_area = float(cv2.contourArea(hull))
            if hull_area > 0:
                solidity = float(area / hull_area)
            hull_pts = hull.reshape(-1, 2)
            if len(hull_pts) >= 2:
                diffs = hull_pts[:, None, :] - hull_pts[None, :, :]
                feret_max = float(np.sqrt((diffs ** 2).sum(-1).max()))
            if len(cnt) >= 5:
                _, (ax_a, ax_b), e_angle = cv2.fitEllipse(cnt)
                ellipse_major = float(max(ax_a, ax_b))
                ellipse_minor = float(min(ax_a, ax_b))
                ellipse_angle = float(e_angle)
                if ellipse_major > 0:
                    ratio = ellipse_minor / ellipse_major
                    ellipse_ecc = float(np.sqrt(max(0.0, 1.0 - ratio * ratio)))

        extent = float(area / bbox_area) if bbox_area > 0 else None
        touches_border = bool(
            xs.min() == 0 or ys.min() == 0 or xs.max() == W - 1 or ys.max() == H - 1
        )
        oi = overlap_info[idx]
        rows.append({
            "instance_id": len(rows) + 1,
            "class": class_names[idx],
            "confidence": round(float(box.conf.item()), 4),
            "pixel_count": area,
            "eq_diameter_px": round(eq_d, 3),
            "centroid_x": round(cx, 2),
            "centroid_y": round(cy, 2),
            "bbox_x1": round(bx1, 2),
            "bbox_y1": round(by1, 2),
            "bbox_x2": round(bx2, 2),
            "bbox_y2": round(by2, 2),
            "bbox_width": round(bw, 2),
            "bbox_height": round(bh, 2),
            "bbox_area": round(bbox_area, 2),
            "bbox_aspect_ratio": round(bw / bh, 3) if bh > 0 else None,
            "extent": round(extent, 3) if extent is not None else None,
            "perimeter_px": round(perimeter, 2) if perimeter is not None else None,
            "circularity": round(circularity, 3) if circularity is not None else None,
            "solidity": round(solidity, 3) if solidity is not None else None,
            "feret_diameter_max_px": round(feret_max, 2) if feret_max is not None else None,
            "ellipse_major_axis_px": round(ellipse_major, 2) if ellipse_major is not None else None,
            "ellipse_minor_axis_px": round(ellipse_minor, 2) if ellipse_minor is not None else None,
            "ellipse_eccentricity": round(ellipse_ecc, 3) if ellipse_ecc is not None else None,
            "ellipse_angle_deg": round(ellipse_angle, 2) if ellipse_angle is not None else None,
            "touches_border": touches_border,
            "overlap_count": oi["count"],
            "overlap_classes": ",".join(sorted(oi["classes"])) if oi["classes"] else "",
            "overlap_pixels_total": oi["pixels"],
        })
    return rows


def _stats_row(class_name, values):
    """One row of summary stats for a class's eq-diameter list."""
    if not values:
        return {
            "class": class_name, "count": 0,
            "min": None, "max": None, "mean": None, "median": None, "std": None,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "class": class_name,
        "count": int(arr.size),
        "min": round(float(arr.min()), 3),
        "max": round(float(arr.max()), 3),
        "mean": round(float(arr.mean()), 3),
        "median": round(float(np.median(arr)), 3),
        "std": round(float(arr.std()), 3),
    }


def _apply_metric(rows, um_per_px):
    """Add eq_diameter_um and area_um2 to each instance row, in place.

    eq_diameter_um = eq_diameter_px * um_per_px
    area_um2       = pixel_count * um_per_px**2
    Both are np.nan when um_per_px is missing or <= 0. Returns the same list.
    """
    valid = isinstance(um_per_px, (int, float)) and not isinstance(um_per_px, bool) and um_per_px > 0
    for r in rows:
        if valid:
            r["eq_diameter_um"] = round(r["eq_diameter_px"] * um_per_px, 3)
            r["area_um2"] = round(r["pixel_count"] * (um_per_px ** 2), 3)
        else:
            r["eq_diameter_um"] = np.nan
            r["area_um2"] = np.nan
    return rows


def _avg_size_metrics(areas_px, um_per_px):
    """Mean droplet area (µm²) and mean equivalent-circular diameter (µm) over a
    list of per-instance pixel areas. Diameter is computed per droplet
    (sqrt(4*a/pi)) then averaged. Returns (nan, nan) when the scale is
    missing/≤0 or the list is empty."""
    if not areas_px or not um_per_px or um_per_px <= 0:
        return float("nan"), float("nan")
    arr = np.asarray(areas_px, dtype=float)
    avg_area_um2 = float(arr.mean()) * (um_per_px ** 2)
    avg_dia_um = float(np.sqrt(4.0 * arr / np.pi).mean()) * um_per_px
    return avg_area_um2, avg_dia_um


def _resolution_pix_per_um2(um_per_px):
    """Calibration constant: pixels per square micron = 1/um_per_px².
    NaN when the scale is missing/≤0."""
    if not um_per_px or um_per_px <= 0:
        return float("nan")
    return 1.0 / (um_per_px ** 2)


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


def _save_per_frame_instance_xlsx(per_frame_rows, out_dir, video_base, video_meta,
                                  size_distribution=None, mode="full", um_per_px=None):
    """Write one xlsx per checkpoint frame, listing every detected instance.

    per_frame_rows: dict mapping processed_frame_number -> list of instance dicts.
    out_dir: target directory (created if missing).
    video_base: filename stem used to prefix the output files.
    video_meta: dict with fps, stride, width, height, video_name — embedded into
        a small `Frame Info` sheet for downstream joins.
    size_distribution: optional dict returned by process_video. When present,
        adds `Stats`, `Histogram Water`, and `Histogram Ice` sheets so the xlsx
        mirrors the on-screen size-distribution plot exactly (shared global bin
        edges per class).
    mode: "full" (default) writes the rich 5-sheet workbook. "basic" writes a slim
        3-sheet workbook (Instances/Frame Info/Stats) with metric (µm) columns and
        no histogram sheets; size_distribution is ignored in that case.
    um_per_px: microns per pixel for basic-mode metric columns; NaN when missing.
    """
    if not per_frame_rows:
        return
    os.makedirs(out_dir, exist_ok=True)
    fps = video_meta.get("fps") or 0
    stride = video_meta.get("stride") or 1
    water_edges = _global_bin_edges_from_size_distribution(size_distribution, "water")
    ice_edges = _global_bin_edges_from_size_distribution(size_distribution, "ice")
    written = 0
    for frame_number in sorted(per_frame_rows):
        rows = per_frame_rows[frame_number]
        if not rows:
            continue
        original_video_frame = int(frame_number) * int(stride)
        frame_time_seconds = round(original_video_frame / fps, 3) if fps > 0 else None
        out_path = os.path.join(
            out_dir, f"{video_base}_frame_{int(frame_number):06d}_instances.xlsx"
        )

        if mode == "basic":
            _apply_metric(rows, um_per_px)
            cols = ["instance_id", "class", "confidence", "pixel_count",
                    "eq_diameter_px", "eq_diameter_um", "area_um2"]
            instances_df = pd.DataFrame(rows).reindex(columns=cols)
            water_um = [r["eq_diameter_um"] for r in rows if r["class"] == "water"]
            ice_um = [r["eq_diameter_um"] for r in rows if r["class"] == "ice"]
            info_df = pd.DataFrame([{
                "processed_frame_number": int(frame_number),
                "original_video_frame": original_video_frame,
                "frame_time_seconds": frame_time_seconds,
                "video_name": video_meta.get("video_name"),
                "video_fps": fps,
                "video_stride": stride,
                "frame_width": video_meta.get("width"),
                "frame_height": video_meta.get("height"),
                "total_instances": len(rows),
                "water_count": sum(1 for r in rows if r["class"] == "water"),
                "ice_count": sum(1 for r in rows if r["class"] == "ice"),
                "um_per_px": um_per_px if (um_per_px and um_per_px > 0) else None,
            }])
            stats_df = pd.DataFrame([
                _stats_row("water", [v for v in water_um if v == v]),  # v==v drops NaN
                _stats_row("ice", [v for v in ice_um if v == v]),
            ])
            with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
                instances_df.to_excel(writer, sheet_name="Instances", index=False)
                info_df.to_excel(writer, sheet_name="Frame Info", index=False)
                stats_df.to_excel(writer, sheet_name="Stats", index=False)
            written += 1
            continue

        # ---- full mode (rich 5-sheet workbook) ----
        instances_df = pd.DataFrame(rows)
        water_diameters = [r["eq_diameter_px"] for r in rows if r["class"] == "water"]
        ice_diameters = [r["eq_diameter_px"] for r in rows if r["class"] == "ice"]
        water_count = len(water_diameters)
        ice_count = len(ice_diameters)
        info_df = pd.DataFrame([{
            "processed_frame_number": int(frame_number),
            "original_video_frame": original_video_frame,
            "frame_time_seconds": frame_time_seconds,
            "video_name": video_meta.get("video_name"),
            "video_fps": fps,
            "video_stride": stride,
            "frame_width": video_meta.get("width"),
            "frame_height": video_meta.get("height"),
            "total_instances": len(rows),
            "water_count": water_count,
            "ice_count": ice_count,
        }])
        stats_df = pd.DataFrame([
            _stats_row("water", water_diameters),
            _stats_row("ice", ice_diameters),
        ])
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            instances_df.to_excel(writer, sheet_name="Instances", index=False)
            info_df.to_excel(writer, sheet_name="Frame Info", index=False)
            stats_df.to_excel(writer, sheet_name="Stats", index=False)
            _histogram_df(water_diameters, water_edges).to_excel(
                writer, sheet_name="Histogram Water", index=False
            )
            _histogram_df(ice_diameters, ice_edges).to_excel(
                writer, sheet_name="Histogram Ice", index=False
            )
        written += 1
    return written


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}


def _list_videos_in_dir(directory):
    """Return sorted absolute paths of video files directly inside `directory`
    (non-recursive). Extensions matched case-insensitively against VIDEO_EXTENSIONS.
    """
    if not os.path.isdir(directory):
        return []
    out = []
    for name in sorted(os.listdir(directory)):
        full = os.path.join(directory, name)
        if not os.path.isfile(full):
            continue
        if os.path.splitext(name)[1].lower() in VIDEO_EXTENSIONS:
            out.append(full)
    return out


def process_video(video_path: str, save_ovl: bool = True, dist_interval: int = 0, output_dir: str = None, progress_callback = None, output_mode: str = "full", um_per_px = None):
    """Process the video and return (msg, excel_path, rows, overlap_totals, charts, execution_time, size_distribution)

    charts is a dict containing JSON-friendly arrays for plotting:
      - pct: {'x': [...], 'water': [...], 'ice': [...]}
      - ov: {'x': [...], 'ww': [...], 'ii': [...], 'wi': [...]}
      - donuts: {'water_count': int, 'ice_count': int, 'void_pct_avg': float, 'avg_conf': float}

    size_distribution (None when dist_interval <= 0): per-class droplet
    equivalent-circular-diameter distributions (d = √(4·A/π), derived from
    the same mask pixel areas the detection part accumulates into
    `water_pixel_area` / `ice_pixel_area`). Sampled at processed frames
    N, 2N, 3N, ..., plus the final processed frame.
      - {"interval": int, "unit": str, "checkpoints": [{"frame": int, "water": {...}, "ice": {...}}, ...]}
    """
    start_time = time.time()
    if not video_path or not os.path.isfile(video_path):
        if progress_callback:
            progress_callback({"status": "error", "message": f"Invalid video file path: {video_path}"})
        return (f"❌ Invalid video file path: {video_path}", None, None, None, None, None, None)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if progress_callback:
            progress_callback({"status": "error", "message": f"Could not open video file: {video_path}"})
        return (f"❌ Error: Could not open video file {video_path}", None, None, None, None, None, None)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print("Video FPS:", video_fps)
    stride = max(1, int(round(video_fps))) if video_fps > 0 else 1
    BATCH_SIZE = 4
    frame_batch, frame_count_batch = [], []
    video_fname = os.path.basename(video_path)
    video_fname_base = os.path.splitext(video_fname)[0]
    # output_dir overrides the default "next to the input video" location used in
    # file-mode. Batch-mode callers pass a per-video subdirectory here so the
    # Excel, charts/, and overlay all land together.
    if output_dir:
        base_dir = output_dir
        os.makedirs(base_dir, exist_ok=True)
    else:
        base_dir = os.path.dirname(video_path)
    out_video_writer = None
    
    # Manually count frames (more reliable than CAP_PROP_FRAME_COUNT for AVI files)
    print("Counting total frames in video (this may take a moment)...")
    total_frames = 0
    temp_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    while True:
        ret = cap.grab()
        if not ret:
            break
        total_frames += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, temp_pos)  # Reset to beginning
    print(f"Total frames in video: {total_frames}, Processing every {stride} frame(s) for ~{total_frames // stride} total processed frames.")

    try:
        if progress_callback:
            progress_callback({"status": "started", "message": "Video opened successfully. Starting processing..."})
        
        if save_ovl:
            # Batch-mode (output_dir set): overlay goes directly in the per-video
            # folder. File-mode: keep the legacy "segmentation results/" subfolder.
            if output_dir:
                seg_dir = base_dir
            else:
                seg_dir = os.path.join(base_dir, "segmentation results")
                os.makedirs(seg_dir, exist_ok=True)
            h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            output_video_path = os.path.join(seg_dir, f"{video_fname_base}_overlay.mp4")
            out_video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))

        rows, overlap_totals = [], {"ww": 0, "ii": 0, "mixed": 0}
        all_water_areas, all_ice_areas = [], []
        frame_count, processed_frame_count = 0, 0
        size_checkpoints_raw = []  # list of (frame, water_areas, ice_areas)
        last_frame_areas = {"frame": 0, "water": [], "ice": []}
        per_frame_instance_rows = {}  # processed_frame -> list of per-instance dicts
        # Raw data for the most recent non-empty frame; used to write the final
        # frame's per-instance xlsx even when it isn't on a dist_interval boundary.
        last_frame_raw = {
            "frame": 0, "full_bin_masks": None, "areas": None, "boxes": None,
            "class_names": None, "frame_shape": None,
        }

        def process_batch(batch_frames, batch_counts):
            nonlocal rows, overlap_totals, size_checkpoints_raw, last_frame_areas
            nonlocal per_frame_instance_rows, last_frame_raw, all_water_areas, all_ice_areas
            results_list = model(batch_frames, imgsz=640, max_det=2000, verbose=False)

            for i, res in enumerate(results_list):
                original_frame = batch_frames[i]
                current_processed_frame = batch_counts[i]
                h, w = res.orig_shape
                total_px = w * h
                water_cnt, ice_cnt, water_area, ice_area = 0, 0, 0, 0
                confs = []
                ww_count, ii_count, wi_count = 0, 0, 0
                frame_water_areas, frame_ice_areas = [], []

                if res.masks is not None and len(res.boxes):
                    class_names = [res.names[int(b.cls)].lower() for b in res.boxes]
                    binm = _threshold_masks(res.masks.data, 0.3)     # (N, mh, mw) uint8, on device
                    mh, mw = int(binm.shape[-2]), int(binm.shape[-1])

                    # Per-instance full-res pixel areas computed on the masks'
                    # device straight from the source masks (multiplicity trick):
                    # exact (== the old int(cv2.resize(...).sum())) but only an
                    # (N,) vector crosses to the CPU, not the multi-GB full-res
                    # masks. See test_gpu_mask_equivalence.py.
                    areas = _mask_areas_from_source(binm, h, w)

                    # Build the full-res CPU masks ONLY when something needs them:
                    # the overlay (every frame) or full-mode per-instance contour
                    # metrics (checkpoints + the final-frame stash). In basic mode
                    # without overlay they are never built, which removes the
                    # per-frame ~N·h·w GPU→CPU transfer and matching GPU alloc.
                    need_full_masks = save_ovl or (output_mode != "basic" and dist_interval > 0)
                    if need_full_masks:
                        full_np = _gather_resize_nn(binm, h, w).cpu().numpy()
                        full_masks_for_overlap = [full_np[k] for k in range(full_np.shape[0])]
                    else:
                        full_masks_for_overlap = None

                    for idx, box in enumerate(res.boxes):
                        area = int(areas[idx])
                        cls_name = res.names[int(box.cls)].lower()
                        if cls_name == "water":
                            water_cnt += 1
                            water_area += area
                            if area > 0:
                                frame_water_areas.append(area)
                        elif cls_name == "ice":
                            ice_cnt += 1
                            ice_area += area
                            if area > 0:
                                frame_ice_areas.append(area)
                        confs.append(box.conf.item())

                    # Pairwise overlap on the masks' device instead of an O(N²)
                    # Python loop over full-res numpy masks. For nearest
                    # upsampling (both axes scaled up) overlap at source
                    # resolution is identical to overlap at full resolution, so
                    # use the small source masks (cheap, low memory) and fall
                    # back to full-res only when downscaling. The (ww, ii, mixed)
                    # tally matches the original loop bit-for-bit — see
                    # test_gpu_mask_equivalence.py.
                    overlap_src = binm if (h >= mh and w >= mw) else _gather_resize_nn(binm, h, w)
                    exists = _overlap_exists_matrix(
                        overlap_src.reshape(overlap_src.shape[0], -1)
                    ).cpu().numpy()
                    ww_count, ii_count, wi_count = _classify_overlaps(exists, class_names)
                    overlap_totals["ww"] += ww_count
                    overlap_totals["ii"] += ii_count
                    overlap_totals["mixed"] += wi_count

                    if save_ovl and out_video_writer is not None:
                        # Reuse the GPU-resized masks instead of resizing a third
                        # time; identical output (test_gpu_mask_equivalence.py).
                        overlay_frame = apply_full_overlay(original_frame, None, class_names,
                                                           full_masks=full_masks_for_overlap)
                        out_video_writer.write(cv2.cvtColor(overlay_frame, cv2.COLOR_RGB2BGR))

                    # Capture raw segments + per-instance metrics at size-distribution
                    # checkpoints; also stash the latest raw segments so the final
                    # frame can be written even when it isn't on a checkpoint. In
                    # basic mode full_bin_masks is None and the areas alone drive
                    # the per-instance rows.
                    if dist_interval > 0:
                        last_frame_raw = {
                            "frame": current_processed_frame,
                            "full_bin_masks": full_masks_for_overlap,
                            "areas": areas,
                            "boxes": res.boxes,
                            "class_names": list(class_names),
                            "frame_shape": (h, w),
                        }
                        if current_processed_frame % dist_interval == 0:
                            per_frame_instance_rows[current_processed_frame] = _per_instance_metrics(
                                full_masks_for_overlap, res.boxes, class_names, (h, w),
                                mode=output_mode, areas=areas,
                            )

                water_pct = (water_area / total_px * 100) if total_px else 0
                ice_pct = (ice_area / total_px * 100) if total_px else 0
                void_pct = max(0, 100 - water_pct - ice_pct)
                avg_conf = (sum(confs) / len(confs) * 100) if confs else 0

                w_area_um2, w_dia_um = _avg_size_metrics(frame_water_areas, um_per_px)
                i_area_um2, i_dia_um = _avg_size_metrics(frame_ice_areas, um_per_px)
                all_area_um2, all_dia_um = _avg_size_metrics(
                    frame_water_areas + frame_ice_areas, um_per_px
                )
                res_pix_um2 = _resolution_pix_per_um2(um_per_px)

                rows.append({
                    "Frame Number": current_processed_frame,
                    "water_cnt": water_cnt,
                    "ice_cnt": ice_cnt,
                    "void_pct": void_pct,
                    "avg_conf": avg_conf,
                    "Overlap_Water-Water": ww_count,
                    "Overlap_Ice-Ice": ii_count,
                    "Overlap_Water-Ice": wi_count,
                    "Water (%)": round(water_pct, 2),
                    "Ice (%)": round(ice_pct, 2),
                    "Avg Confidence (%)": round(avg_conf, 2),
                    "water_pixel_area": water_area,
                    "ice_pixel_area": ice_area,
                    "Water Avg Area (µm²)": w_area_um2,
                    "Water Avg Diameter (µm)": w_dia_um,
                    "Ice Avg Area (µm²)": i_area_um2,
                    "Ice Avg Diameter (µm)": i_dia_um,
                    "All Avg Area (µm²)": all_area_um2,
                    "All Avg Diameter (µm)": all_dia_um,
                    "Resolution (pix/µm²)": res_pix_um2,
                })

                all_water_areas.extend(frame_water_areas)
                all_ice_areas.extend(frame_ice_areas)

                last_frame_areas = {
                    "frame": current_processed_frame,
                    "water": frame_water_areas,
                    "ice": frame_ice_areas,
                }
                if dist_interval > 0 and current_processed_frame % dist_interval == 0:
                    size_checkpoints_raw.append((
                        current_processed_frame,
                        list(frame_water_areas),
                        list(frame_ice_areas),
                    ))
                if progress_callback:
                    progress_callback({
                        "status": "processing", 
                        "message": f"Processed frame {current_processed_frame}",
                        "processed_frame": current_processed_frame,
                        # eta in seconds (rounded to 2 decimal places) = elapsed_time * (estimated_total_frames / processed_frames - 1)
                        "eta": round((time.time() - start_time) * ( (total_frames // stride) / current_processed_frame - 1), 2) if current_processed_frame > 0 else None,
                        "progress": round((current_processed_frame * stride) / total_frames * 100, 2)
                    })

        print("🚀 Starting video processing...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % stride == 0:
                processed_frame_count += 1
                frame_batch.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_count_batch.append(processed_frame_count)
                if len(frame_batch) == BATCH_SIZE:
                    print(f"➡️  Processing batch... Frames processed so far: {processed_frame_count}")
                    process_batch(frame_batch, frame_count_batch)
                    frame_batch.clear()
                    frame_count_batch.clear()
                    if progress_callback:
                        progress_callback({"status": "batch processed", "processed_frames": processed_frame_count})
            frame_count += 1

        if frame_batch:
            print(f"➡️  Processing final batch... Total frames: {processed_frame_count}")
            process_batch(frame_batch, frame_count_batch)

        if not rows:
            if progress_callback:
                progress_callback({"status": "completed", "message": "Video processed, but no objects were detected."})
            return ("Video processed, but no objects were detected.", None, rows, overlap_totals, None, None, None)

        df = pd.DataFrame(rows)
        excel_path = os.path.join(base_dir, f"{video_fname_base}_detection_summary.xlsx")

        # prepare chart-friendly payload
        x = [r["Frame Number"] for r in rows]
        pct_water = [r["Water (%)"] for r in rows]
        pct_ice = [r["Ice (%)"] for r in rows]
        ov_ww = [r["Overlap_Water-Water"] for r in rows]
        ov_ii = [r["Overlap_Ice-Ice"] for r in rows]
        ov_wi = [r["Overlap_Water-Ice"] for r in rows]
        charts = {
            "pct": {"x": x, "water": pct_water, "ice": pct_ice},
            "ov": {"x": x, "ww": ov_ww, "ii": ov_ii, "wi": ov_wi},
            "donuts": {"water_count": int(df["water_cnt"].sum()), "ice_count": int(df["ice_cnt"].sum()), "void_pct_avg": float(df["void_pct"].mean()), "avg_conf": float(df["avg_conf"].mean())}
        }

        overlap_totals_df = pd.DataFrame([{
            "Water-Water": int(overlap_totals.get("ww", 0)),
            "Ice-Ice": int(overlap_totals.get("ii", 0)),
            "Water-Ice": int(overlap_totals.get("mixed", 0)),
        }])
        sw_area, sw_dia = _avg_size_metrics(all_water_areas, um_per_px)
        si_area, si_dia = _avg_size_metrics(all_ice_areas, um_per_px)
        sa_area, sa_dia = _avg_size_metrics(all_water_areas + all_ice_areas, um_per_px)
        summary_df = pd.DataFrame([{
            "water_count_total": int(charts["donuts"]["water_count"]),
            "ice_count_total": int(charts["donuts"]["ice_count"]),
            "void_pct_avg": float(charts["donuts"]["void_pct_avg"]),
            "avg_conf_mean": float(charts["donuts"]["avg_conf"]),
            "Water Avg Area (µm²)": sw_area,
            "Water Avg Diameter (µm)": sw_dia,
            "Ice Avg Area (µm²)": si_area,
            "Ice Avg Diameter (µm)": si_dia,
            "All Avg Area (µm²)": sa_area,
            "All Avg Diameter (µm)": sa_dia,
            "Resolution (pix/µm²)": _resolution_pix_per_um2(um_per_px),
        }])
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Per-Frame", index=False)
            overlap_totals_df.to_excel(writer, sheet_name="Overlap Totals", index=False)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

        # Best-effort PNG dump — never fail the run on render errors.
        charts_dir = os.path.join(base_dir, f"{video_fname_base}_charts")
        try:
            _save_chart_pngs(df, charts, overlap_totals, charts_dir)
            print(f"📊 Saved chart PNGs to {charts_dir}")
        except Exception as chart_err:
            print(f"⚠️  Failed to save chart PNGs ({chart_err}); continuing.")

        size_distribution = None
        if dist_interval > 0:
            if not size_checkpoints_raw or size_checkpoints_raw[-1][0] != last_frame_areas["frame"]:
                size_checkpoints_raw.append((
                    last_frame_areas["frame"],
                    list(last_frame_areas["water"]),
                    list(last_frame_areas["ice"]),
                ))
            all_water = [d for _, w_a, _ in size_checkpoints_raw for d in _eq_diameter(w_a).tolist()]
            all_ice = [d for _, _, i_a in size_checkpoints_raw for d in _eq_diameter(i_a).tolist()]
            water_edges = _shared_bin_edges(all_water)
            ice_edges = _shared_bin_edges(all_ice)
            checkpoints = [
                {
                    "frame": int(frame),
                    "water": _droplet_stats_block(_eq_diameter(w_a).tolist(), edges=water_edges),
                    "ice": _droplet_stats_block(_eq_diameter(i_a).tolist(), edges=ice_edges),
                }
                for frame, w_a, i_a in size_checkpoints_raw
            ]
            water_y_max = max(
                (max(cp["water"]["histogram"]["counts"], default=0) for cp in checkpoints),
                default=0,
            )
            ice_y_max = max(
                (max(cp["ice"]["histogram"]["counts"], default=0) for cp in checkpoints),
                default=0,
            )
            size_distribution = {
                "interval": int(dist_interval),
                "unit": "pixels (equivalent circular diameter)",
                "bin_count": SIZE_DIST_BINS,
                "y_max": {"water": int(water_y_max), "ice": int(ice_y_max)},
                "checkpoints": checkpoints,
            }

            try:
                _save_size_distribution_pngs(size_distribution, charts_dir, video_fname_base)
                print(f"📊 Saved size distribution PNGs to {charts_dir}")
            except Exception as size_chart_err:
                print(f"⚠️  Failed to save size distribution PNGs ({size_chart_err}); continuing.")

            # Per-instance xlsx dump at the same checkpoints as size_distribution,
            # plus the final non-empty processed frame so users always get the
            # latest snapshot regardless of where it lands relative to dist_interval.
            if (
                last_frame_raw["areas"] is not None
                and last_frame_raw["frame"] not in per_frame_instance_rows
            ):
                per_frame_instance_rows[last_frame_raw["frame"]] = _per_instance_metrics(
                    last_frame_raw["full_bin_masks"],
                    last_frame_raw["boxes"],
                    last_frame_raw["class_names"],
                    last_frame_raw["frame_shape"],
                    mode=output_mode,
                    areas=last_frame_raw["areas"],
                )

            per_frame_xlsx_dir = os.path.join(base_dir, f"{video_fname_base}_per_frame_xlsx")
            try:
                video_meta = {
                    "video_name": video_fname,
                    "fps": float(video_fps) if video_fps else 0,
                    "stride": int(stride),
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                }
                n_written = _save_per_frame_instance_xlsx(
                    per_frame_instance_rows, per_frame_xlsx_dir, video_fname_base, video_meta,
                    size_distribution=size_distribution, mode=output_mode, um_per_px=um_per_px,
                )
                if n_written:
                    print(f"📋 Saved {n_written} per-frame instance xlsx file(s) to {per_frame_xlsx_dir}")
            except Exception as per_frame_err:
                print(f"⚠️  Failed to save per-frame instance xlsx files ({per_frame_err}); continuing.")

        end_time = time.time()
        print(f"✅ Processing complete! Elapsed time: {end_time - start_time:.2f} seconds")
        # execution time in seconds (rounded to 2 decimal places)
        execution_time = round(end_time - start_time, 2)
        if progress_callback:
            progress_callback({"status": "completed", "message": "Processing complete.", "execution_time": execution_time, "excel_path": excel_path, "charts": charts, "rows": rows, "overlap_totals": overlap_totals, "size_distribution": size_distribution})
        return ("✅ Processing complete!", excel_path, rows, overlap_totals, charts, execution_time, size_distribution)

    except Exception as e:
        print(f"An error occurred: {e}")
        if progress_callback:
            progress_callback({"status": "error", "message": f"An error occurred during processing: {e}"})
        return (f"❌ An error occurred during processing: {e}", None, None, None, None, None, None)

    finally:
        if cap:
            cap.release()
        if out_video_writer:
            out_video_writer.release()
        print("Video resources released.")

tasks = {}

# Seconds the SSE stream waits on the queue before deciding whether the worker
# has gone silent. A healthy worker can exceed this during the file-saving tail
# (chart PNGs + per-frame xlsx), so an idle gap is NOT on its own a failure.
SSE_IDLE_TIMEOUT = 300


def _sse_idle_decision(task):
    """Decide what an SSE stream should do after an idle (queue-empty) gap.

    ``task`` is ``tasks.get(task_id)`` — the task record, or ``None`` if it was
    already popped. Returns ``"keep-alive"`` when the worker is still running
    (record present and not yet completed), meaning the stream should emit a
    heartbeat comment and keep waiting; returns ``"timeout"`` when the task is
    gone or finished, meaning the stream should report a real timeout and close.
    """
    if task is not None and not task.get("completed"):
        return "keep-alive"
    return "timeout"


def _mark_task_completed(task_id):
    """Mark a task completed, tolerating the SSE side having already popped it.

    Returns ``True`` if the record still existed and was updated, ``False`` if
    it was already removed (e.g. the client disconnected). This keeps the worker
    thread from raising ``KeyError`` in its ``finally`` after doing all its work.
    """
    if task_id in tasks:
        tasks[task_id]["completed"] = True
        return True
    return False


# REST API endpoints (synchronous) -------------------------------------------
@server.route('/api/process', methods=['POST'])
def api_process():
    """Enqueue processing task and return task_id. Client should connect to /api/events/<task_id> for SSE."""
    print("Received /api/process request")
    
    data = request.get_json(force=True, silent=True) or {}
    video_path = data.get('video_path')
    save_ovl = data.get('save_overlay', True)
    try:
        dist_interval = int(data.get('dist_interval', 0) or 0)
    except (TypeError, ValueError):
        dist_interval = 0
    if dist_interval < 0:
        dist_interval = 0
    output_mode = str(data.get('output_mode', 'full')).strip().lower()
    if output_mode not in ("basic", "full"):
        output_mode = "full"
    try:
        um_per_px = float(data.get('um_per_px'))
        if um_per_px <= 0:
            um_per_px = None
    except (TypeError, ValueError):
        um_per_px = None
    if not video_path:
        return jsonify({"status": "error", "message": "Missing video_path"}), 400
    if not (os.path.isfile(video_path) or os.path.isdir(video_path)):
        return jsonify({"status": "error", "message": f"Path is neither a file nor a directory: {video_path}"}), 400

    task_id = uuid.uuid4().hex
    task_queue = Queue()
    tasks[task_id] = {"queue": task_queue, "completed": False, "status": "queued"}

    # url_for(_external=True) needs a Flask request context, which the worker
    # thread doesn't have. Capture the host here and build URLs by string concat.
    host_url = request.host_url.rstrip('/')

    def _build_download_url(excel_path):
        if not excel_path:
            return None
        return f"{host_url}/api/download_summary?path={urllib.parse.quote(str(excel_path))}"

    def worker():
        try:
            def push(ev):
                try:
                    task_queue.put_nowait(make_json_serializable(ev))
                except Exception:
                    try:
                        task_queue.put_nowait({"message": str(ev)})
                    except Exception:
                        pass

            def _single_video_payload(msg, excel_path, rows, overlaps, charts, execution_time, size_distribution):
                return {
                    "status": "ok" if excel_path else "error",
                    "message": str(msg),
                    "charts": make_json_serializable(charts) if charts else None,
                    "rows": make_json_serializable(rows) if rows else None,
                    "overlaps": make_json_serializable(overlaps) if overlaps else None,
                    "excel_path": str(excel_path) if excel_path else None,
                    "download_url": _build_download_url(excel_path),
                    "execution_time": execution_time,
                    "size_distribution": make_json_serializable(size_distribution) if size_distribution else None,
                }

            if os.path.isdir(video_path):
                # Batch mode: process every video in the directory, sequentially.
                videos = _list_videos_in_dir(video_path)
                print(f"📂 Batch mode: found {len(videos)} video(s) in {video_path}")
                for _v in videos:
                    print(f"   - {_v}")
                if not videos:
                    push({"status": "error", "message": f"No video files found in directory: {video_path}"})
                    task_queue.put_nowait({"status": "error", "message": f"No video files found in directory: {video_path}"})
                    return

                batch_results = []
                last_payload = None
                total = len(videos)
                for idx, vid in enumerate(videos, start=1):
                    print(f"▶️  [{idx}/{total}] Starting {os.path.basename(vid)}")
                    stem = os.path.splitext(os.path.basename(vid))[0]
                    out_dir = os.path.join(video_path, stem)
                    vid_name = os.path.basename(vid)
                    push({
                        "status": "video_started",
                        "message": f"Processing video {idx}/{total}: {vid_name}",
                        "video_index": idx,
                        "video_total": total,
                        "current_video": vid_name,
                    })

                    # Stamp every event with batch position, convert per-video
                    # progress into global progress, and demote intermediate
                    # "completed" events so the frontend doesn't close the SSE
                    # after the first video finishes.
                    def video_push(ev, _idx=idx, _total=total, _name=vid_name):
                        if not isinstance(ev, dict):
                            push(ev)
                            return
                        ev = dict(ev)
                        ev["video_index"] = _idx
                        ev["video_total"] = _total
                        ev["current_video"] = _name
                        if isinstance(ev.get("progress"), (int, float)):
                            per_video_pct = float(ev["progress"])
                            ev["progress"] = round(((_idx - 1) + per_video_pct / 100.0) / _total * 100.0, 2)
                        if ev.get("status") == "completed" and _idx < _total:
                            ev = {
                                "status": "video_completed",
                                "message": ev.get("message") or f"Finished {_name}",
                                "execution_time": ev.get("execution_time"),
                                "excel_path": ev.get("excel_path"),
                                "video_index": _idx,
                                "video_total": _total,
                                "current_video": _name,
                                "progress": round((_idx / _total) * 100.0, 2),
                            }
                        push(ev)

                    # Per-video try/except so one failure can't kill the whole batch.
                    # process_video has its own internal try/except for the main work,
                    # but the cv2 capture-open and frame-counting steps live outside
                    # that block and can still raise.
                    try:
                        msg, excel_path, rows, overlaps, charts, exec_time, size_dist = process_video(
                            vid, save_ovl,
                            dist_interval=dist_interval,
                            output_dir=out_dir,
                            progress_callback=video_push,
                            output_mode=output_mode,
                            um_per_px=um_per_px,
                        )
                    except Exception as per_vid_err:
                        print(f"⚠️  Video {idx}/{total} ({vid_name}) raised: {per_vid_err!r}")
                        push({
                            "status": "error",
                            "message": f"Video {vid_name} failed: {per_vid_err}",
                            "video_index": idx,
                            "video_total": total,
                            "current_video": vid_name,
                        })
                        msg, excel_path = f"❌ {per_vid_err}", None
                        rows = overlaps = charts = size_dist = None
                        exec_time = None

                    batch_results.append({
                        "video": vid_name,
                        "video_path": vid,
                        "output_dir": out_dir,
                        "status": "ok" if excel_path else "error",
                        "message": str(msg),
                        "excel_path": str(excel_path) if excel_path else None,
                        "download_url": _build_download_url(excel_path),
                        "execution_time": exec_time,
                    })
                    if excel_path:
                        last_payload = _single_video_payload(msg, excel_path, rows, overlaps, charts, exec_time, size_dist)

                final_payload = last_payload or {"status": "error", "message": "All videos failed"}
                final_payload["batch_results"] = batch_results
                final_payload["batch_total"] = total
                task_queue.put_nowait({"status": "finished", "data": final_payload})
            else:
                # File mode (unchanged behavior).
                msg, excel_path, rows, overlaps, charts, execution_time, size_distribution = process_video(
                    video_path, save_ovl, dist_interval=dist_interval, progress_callback=push,
                    output_mode=output_mode, um_per_px=um_per_px
                )
                task_queue.put_nowait({"status": "finished", "data": _single_video_payload(
                    msg, excel_path, rows, overlaps, charts, execution_time, size_distribution
                )})
        except Exception as e:
            print(f"❌ Worker fatal: {e!r}")
            traceback.print_exc()
            task_queue.put_nowait({"status": "error", "message": f"An error occurred: {e}"})
        finally:
            # The SSE stream may have already popped this task (e.g. the client
            # disconnected, or an earlier idle-timeout fired). Don't assume the
            # entry still exists — guard the lookup so a healthy worker can't
            # crash here after doing all its real work.
            _mark_task_completed(task_id)
            task_queue.put_nowait({"__done__": True})
            
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return jsonify({"status": "ok", "task_id": task_id}), 202

@server.route('/api/events/<task_id>')
def api_events(task_id):
    """SSE endpoint to stream task progress and final result."""
    if task_id not in tasks:
        return jsonify({"status": "error", "message": "Invalid task_id"}), 404
    
    task = tasks[task_id]
    task_queue = task["queue"]
    
    def event_stream():
        while True:
            try:
                ev = task_queue.get(timeout=SSE_IDLE_TIMEOUT)
            except Empty:
                # A healthy worker can go silent for well over 5 minutes during
                # the tail phases (saving chart PNGs and the per-frame xlsx
                # files), which push no progress events. Don't mistake that
                # silence for a dead worker and tear the task down underneath
                # it: while the task still exists and isn't finished, send an
                # SSE keep-alive comment (EventSource ignores comment lines) and
                # keep waiting. Only report a real timeout if the task is gone.
                if _sse_idle_decision(tasks.get(task_id)) == "keep-alive":
                    yield ": keep-alive\n\n"
                    continue
                yield f"data: {json.dumps({'status': 'error', 'message': 'Timeout: No updates for 5 minutes'})}\n\n"
                break
            if ev is None:
                continue
            if isinstance(ev, dict) and ev.get("__done__"):
                yield f"data: {json.dumps({'status': 'closed'})}\n\n"
                break
            try:
                yield f"data: {json.dumps(ev)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'status': 'error', 'message': f'Error serializing event: {e}'})}\n\n"
            
        try:
            tasks.pop(task_id, None)
        except Exception:
            pass
    return Response(stream_with_context(event_stream()), mimetype='text/event-stream')

@server.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({"status": "ok", "message": "API is running"}), 200

@server.route('/api/download_summary')
def api_download_summary():
    path = request.args.get('path')
    if not path or not os.path.isfile(path):
        return jsonify({"status": "error", "message": "Invalid or missing path"}), 400
    return send_file(path, as_attachment=True)
@app.callback(
    Output("progress-text", "children"),
    Output("download-summary", "data"),
    Output("result-store", "data"),
    Output("overlap-store", "data"),
    Output("url", "pathname"),
    Input("run-btn", "n_clicks"),
    State("folder-input", "value"),
    State("overlay-toggle", "value"),
    prevent_initial_call=True
)
def run_detection(n_clicks, video_path, save_ovl):
    start_time = time.time()
    if not n_clicks:
        raise PreventUpdate
    # call reusable processing helper so UI and API share logic
    msg, excel_path, rows, overlap_totals, charts, execution_time = process_video(video_path, save_ovl)
    if excel_path:
        # return dash-friendly outputs
        return msg, dcc.send_file(excel_path), rows, overlap_totals, "/summary"
    else:
        return msg, None, no_update, no_update, no_update

# --- Callbacks for Page 2 ---
@app.callback(
    Output("pct-graph", "figure"), 
    Output("ov-graph", "figure"), 
    Output("slider-output-container", "children"), 
    Input("entry-slider", "value"), 
    State("result-store", "data")
)
def update_graphs(val, rows):
    if not rows: return {}, {}, "No data."
    df = pd.DataFrame(rows)
    val = min(val, len(df))
    dff = df.iloc[:val]
    fig1 = px.line(dff, x="Frame Number", y=["Water (%)", "Ice (%)"], labels={"Frame Number": "Processed Frame (at 1 FPS)", "value": "%", "variable": "Metric"}, markers=True, line_shape="spline")
    fig1.update_layout(title=f"Water & Ice (%) (first {val} frames)", margin=dict(l=40, r=10, t=30, b=30), transition={'duration': 500, 'easing': 'cubic-in-out'}, uirevision='some-constant-value')
    fig2 = px.line(dff, x="Frame Number", y=["Overlap_Water-Water", "Overlap_Ice-Ice", "Overlap_Water-Ice"], labels={"Frame Number": "Processed Frame (at 1 FPS)", "value": "Count", "variable": "Metric"}, markers=True, line_shape="spline")
    fig2.update_layout(title=f"Overlaps (first {val} frames)", margin=dict(l=40, r=10, t=30, b=30), transition={'duration': 500, 'easing': 'cubic-in-out'}, uirevision='some-constant-value')
    return fig1, fig2, f"Showing first {val} of {len(df)} processed frames"

@app.callback(
    Output("entry-slider", "max"), 
    Output("entry-slider", "value"), 
    Input("result-store", "data")
)
def update_slider(rows):
    if not rows: raise PreventUpdate
    n = len(rows)
    return n, n

@app.callback(
    Output("donut-water", "figure"), 
    Output("donut-ice", "figure"), 
    Output("donut-void", "figure"), 
    Output("donut-conf", "figure"), 
    Input("result-store", "data")
)
def update_donuts(rows):
    if not rows: return {}, {}, {}, {}
    df = pd.DataFrame(rows)
    fig_w = px.pie(names=["Water"], values=[df["water_cnt"].sum()], hole=0.6); fig_w.update_traces(textinfo="value", textposition="inside"); fig_w.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=20), uirevision='donut_revision')
    fig_i = px.pie(names=["Ice"], values=[df["ice_cnt"].sum()], hole=0.6); fig_i.update_traces(textinfo="value", textposition="inside"); fig_i.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=20), uirevision='donut_revision')
    fig_v = px.pie(names=["Void", ""], values=[df["void_pct"].mean(), 100 - df["void_pct"].mean()], hole=0.6); fig_v.update_traces(textinfo="percent", textposition="inside"); fig_v.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=20), uirevision='donut_revision')
    fig_c = px.pie(names=["Conf", ""], values=[df["avg_conf"].mean(), 100 - df["avg_conf"].mean()], hole=0.6); fig_c.update_traces(textinfo="percent", textposition="inside"); fig_c.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=20), uirevision='donut_revision')
    return fig_w, fig_i, fig_v, fig_c

@app.callback(
    Output("overlap-summary", "children"), 
    Input("overlap-store", "data")
)
def display_overlap(o):
    if not o: return html.P("No overlap data.")
    return html.Div([html.H4("Overlap Summary", style={"textAlign": "center"}), html.Ul([html.Li(f"Water–Water total: {o['ww']}"), html.Li(f"Ice–Ice total:   {o['ii']}"), html.Li(f"Water–Ice total: {o['mixed']}")], style={"listStyle": "none", "padding": 0})])

# UPDATED: Callback for dynamic summary paragraph
@app.callback(
    Output("dynamic-summary-p", "children"),
    Input("result-store", "data"),
    Input("overlap-store", "data")
)
def update_dynamic_summary(rows, overlaps):
    if not rows or not overlaps:
        raise PreventUpdate

    df = pd.DataFrame(rows)

    # --- 1. Freezing time calculation ---
    freeze_point_df = df[df['ice_cnt'] >= df['water_cnt']]
    if not freeze_point_df.empty:
        time_taken_to_freeze = freeze_point_df.iloc[0]['Frame Number']
        freeze_text = f"a majority freeze at approximately **{time_taken_to_freeze} seconds**"
    else:
        freeze_text = "no majority freeze point"

    # --- 2. Most common overlap type calculation ---
    if not any(overlaps.values()):
        overlap_type = "N/A"
    else:
        max_overlap_key = max(overlaps, key=overlaps.get)
        overlap_map = {"ww": "Water-Water", "ii": "Ice-Ice", "mixed": "Water-Ice"}
        overlap_type = f"the **{overlap_map.get(max_overlap_key, 'N/A')}** type"

    # --- 3. Growth rate calculation ---
    if len(rows) > 1:
        df['water_area_delta'] = df['water_pixel_area'].diff()
        df['ice_area_delta'] = df['ice_pixel_area'].diff()
        avg_water_growth = df['water_area_delta'].mean()
        avg_ice_growth = df['ice_area_delta'].mean()
        growth_text = f"On average, water area changed by **{avg_water_growth:,.1f} pixels/sec** and ice by **{avg_ice_growth:,.1f} pixels/sec**."
    else:
        growth_text = "Growth rate could not be calculated."

    # --- 4. Final paragraph assembly ---
    final_text_part1 = f"This video achieved {freeze_text}, with most interactions being of {overlap_type}. "
    final_text_part2 = growth_text

    return html.P([final_text_part1, final_text_part2])

if __name__ == "__main__":
    app.run(debug=True, threaded=True)

