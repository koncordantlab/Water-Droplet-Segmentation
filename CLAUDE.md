# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

Two top-level apps:
- `Nasa_Backend/` — Python backend: a Dash + Flask app that wraps an Ultralytics YOLO segmentation model and exposes a REST + SSE API. Also contains the standalone tracking pipeline (`tracking.py` + analysis scripts).
- `nasa-frontend/` — React (Create React App) frontend that calls the backend's `/api/*` endpoints.

The model weights live at `Nasa_Backend/app_root/weights_DP(8).pt` and are loaded at import time by `frontend_nasa13_apiV2.py` and `tracking.py`. Without that file, both will fail on startup.

## Common commands

### Backend (run from `Nasa_Backend/`)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 frontend_nasa13_apiV2.py        # Dash UI + REST API on http://localhost:8050
```

### Tracking pipeline (standalone, separate from the Dash app)
`tracking.py` uses hard-coded relative paths (`../Nasa_Backend/...`) so it expects to be run from a **sibling directory** of `Nasa_Backend/` (e.g. an analysis folder), not from inside `Nasa_Backend/` itself. To run from elsewhere, edit the `MODEL_PATH`, `VIDEO_PATH`, `OUTPUT_PATH`, `DETECTIONS_PATH`, `TRACK_LOG_PATH` constants at the top of the file. Default flow:
```bash
python3 tracking.py                     # writes output_tracked.mp4, detections.json, tracking_log.json
python3 analyze_tracking_json.py        # writes insights_summary.json + suspicious/per-track CSVs
python3 plot_tracking_insights.py       # writes timeline plots into output/insights_visual/
```
Set `EXPORT_DETECTIONS = False` in `tracking.py` to skip the YOLO inference step and re-track from an existing `detections.json`.

### Frontend (run from `nasa-frontend/`)
```bash
npm install
# .env must contain: REACT_APP_BACKEND_API_URL=http://localhost:8050/api
npm start                               # dev server on http://localhost:3000
npm run build                           # production build
npm test                                # react-scripts test (Jest, watch mode)
```

## Architecture notes

### Backend request flow (`frontend_nasa13_apiV2.py`)
The same Python process serves three things off one Flask `server` (created by `dash.Dash`):
1. The legacy Dash UI at `/` (page_1_layout / page_2_layout, with `@app.callback`s).
2. A REST + SSE API under `/api/*` for the React frontend, with CORS enabled for `/api/*` only.
3. Static download of the generated Excel summary.

The API is **asynchronous via thread + SSE**:
- `POST /api/process` enqueues a background `worker()` thread that calls `process_video(...)` and pushes JSON-serializable progress events into a per-task `Queue`. It returns a `task_id` immediately.
- `GET /api/events/<task_id>` is an `text/event-stream` endpoint that drains that queue and emits `data: {...}\n\n` SSE frames. The React `App.js` opens an `EventSource` on this URL and updates progress/results from the stream. The final event includes `charts`, `rows`, `overlap_totals`, `download_url`, `execution_time`, and `size_distribution`.
- `size_distribution` is `null` when the request omits `dist_interval` (or sets it to 0). When present, it carries per-class droplet **equivalent-circular-diameter** distributions (d = √(4·A/π), derived from the same mask pixel areas the detection loop sums into `water_pixel_area` / `ice_pixel_area`), sampled at processed frames N, 2N, …, plus the final frame. Bins and stats are in pixels (length). Bin edges and `y_max` are computed **globally per class across all checkpoints** so the frontend dropdown shows visually comparable charts — don't refactor to per-checkpoint binning. Edges are **log-spaced** (`np.logspace(log10(min), log10(max), SIZE_DIST_BINS+1)`) to pair with the log-scale x-axis in `SizeDistribution.jsx`/`chartsUtils.js`, so outlier bins stay visible without flattening the rest; the frontend relies on each bar's footprint `[center - width/2, center + width/2]` equaling its bin `[lo, hi]`, so don't change the log-spacing in isolation without revisiting the frontend bar geometry. Bin count is the `SIZE_DIST_BINS` constant near the top of the helpers section.
- `tasks` is an in-memory dict keyed by `task_id`. State is lost on restart; there is no persistence layer.
- All payloads are passed through `make_json_serializable()` to coerce numpy/pandas types before JSON encoding — extend this helper rather than adding ad-hoc conversions when adding new fields.

`process_video()` does YOLO inference in batches of 4 frames at `imgsz=640` with `max_det=2000` (raised deliberately for dense droplet fields — changing it changes counts/areas on dense frames), sampling one frame per second (`stride = round(fps)`). It optionally writes an overlay `.mp4` into a `segmentation results/` subfolder of the input video's directory and always writes `<video>_detection_summary.xlsx` next to the input video. Both the `Per-Frame` and `Summary` sheets of that workbook carry seven average droplet-size columns — `Water/Ice/All Avg Area (µm²)`, `Water/Ice/All Avg Diameter (µm)`, and a single `Resolution (pix/µm²)` constant — computed from the same `um_per_px` GUI scale (Resolution = `1/um_per_px²`; equivalent diameter is `√(4·area/π)` per droplet then averaged). They are µm-only and NaN when `um_per_px` is missing or ≤ 0; Per-Frame values average that frame's droplets while Summary pools every droplet across the video. Helpers `_avg_size_metrics()` / `_resolution_pix_per_um2()` drive both sheets — keep the seven header strings byte-identical across them.

When `dist_interval > 0` the same trigger that drives `size_distribution` also writes one `<video>_frame_NNNNNN_instances.xlsx` per checkpoint frame into a `<video>_per_frame_xlsx/` subfolder. Each xlsx has five sheets — `Instances` (one row per detection with class, confidence, pixel_count, eq_diameter_px, bbox, centroid, perimeter, circularity, extent, solidity, feret diameter, ellipse fit, touches_border, and per-instance overlap info), `Frame Info` (frame number, video time, fps, stride, dimensions, totals), `Stats` (count/min/max/mean/median/std per class), and `Histogram Water` + `Histogram Ice` (long-format bin_lo/bin_hi/bin_center/count). The histogram sheets reuse the same global log-spaced bin edges as the size-distribution plot, so they match the on-screen plot bar-for-bar — don't recompute bins per-frame. The final non-empty processed frame is always included, matching the size-distribution checkpoint contract. Empty frames are skipped. Mask math runs on the inference device: per-instance `pixel_count` and the aggregate `water_pixel_area`/`ice_pixel_area` both come from `_mask_areas_from_source()` (exactly equal to summing the cv2-resized full-res mask, so they still match each other exactly), and frame-level overlap totals come from `_overlap_exists_matrix()` on source-resolution masks when both axes upscale (provably identical to full-res for nearest upsampling; full-res fallback otherwise). Full-resolution masks are built by `_gather_resize_nn()` — bit-identical to `cv2.resize(..., INTER_NEAREST)` because it gathers through cv2's own probed index map; never swap in `torch.nn.functional.interpolate`, whose nearest convention differs — and only when actually consumed (overlay, or full-mode checkpoint metrics; basic mode passes `full_bin_masks=None` plus precomputed `areas` into `_per_instance_metrics()`, so don't assume masks are available there). Contour-based fields (perimeter, circularity, feret, ellipse) and the per-instance overlap columns still use the full-resolution masks (largest connected component for contours, pairwise AND for overlaps) in full mode. The cv2 bit-equivalence is pinned by `tests/unit/test_masks.py` — rerun it after any OpenCV/torch upgrade and keep it in sync with the helpers. Full column reference (meaning + derivation of every field across all five sheets) lives in `Nasa_Backend/per_frame_xlsx_schema.md`.

Two `process_video`/`/api/process` parameters control the per-instance output: `output_mode` (`"full"` default, or `"basic"`) and `um_per_px` (optional float). **Full** is the five-sheet, pixel-unit workbook described above. **Basic** emits a slim three-sheet workbook (`Instances` with just `instance_id, class, confidence, pixel_count, eq_diameter_px, eq_diameter_um, area_um2`; `Frame Info` with `um_per_px` added; `Stats` in µm) — no histogram sheets, and the contour/ellipse/overlap work is skipped so Basic is faster. Metric columns use a single scale factor (`eq_diameter_um = eq_diameter_px × um_per_px`, `area_um2 = pixel_count × um_per_px²`) and are NaN when `um_per_px` is missing or ≤ 0. `output_mode` controls **only** these per-instance files — the aggregate `<video>_detection_summary.xlsx`, chart PNGs, and the `size_distribution` payload are produced identically in both modes. Basic-mode columns are documented in the "Basic mode" section of `per_frame_xlsx_schema.md`.

### Tracking pipeline (`tracking.py`)
**Source lives only on the un-merged branch `1-object-tracking-algorithm-v01`** (local + origin) — the working tree on `main`/`stat-distribution-report-v01` has only its outputs (`Nasa_Backend/output/`) and stale `__pycache__` `.pyc` files. Check out or cherry-pick that branch to run it; its `MODEL_PATH` still points at `weights_DP(6).pt`.

Separate from the Dash app. Two-phase design:
1. `export_detections_json()` runs YOLO once and dumps per-frame segments to `detections.json`.
2. `track_from_detections_json()` consumes that JSON and produces `tracking_log.json` plus an annotated `output_tracked.mp4`.

The tracker is custom (not ByteTrack / BoT-SORT). It maintains a `Track` dataclass per object and has three intertwined merge detectors — direct, inferred, and match-growth — each with its own threshold block at the top of the file (≈100 lines of `MERGE_*`, `INFERRED_*`, `INFERRED_SINGLE_PARENT_*` constants). When tweaking tracking behavior, prefer adjusting these constants over editing the matching logic; the constants are the intended tuning surface and the matchers are heavily interdependent.

`analyze_tracking_json.py` and `plot_tracking_insights.py` consume the tracker's outputs to produce diagnostic CSVs and matplotlib plots (Agg backend, headless-safe).

### Frontend (`nasa-frontend/src/App.js`)
- Single-component "router" that switches on `window.location.pathname` between `HomePage` and `SummaryPage` (no react-router).
- `handleSubmit` POSTs to `/api/process`, then a `useEffect` opens an `EventSource` to `/api/events/<task_id>` and folds events into local state.
- Tracks chart-render completion via `markPlotRendered` so `chartRenderElapsed` reflects time-to-paint for the Plotly charts, separate from backend `executionTime`.
- All backend URLs are built from `process.env.REACT_APP_BACKEND_API_URL` — the frontend will silently 404 if `.env` is missing.

## Working in this repo

- **Never commit**: model weights (`*.pt`), input/output videos (`*.mp4`, `*.avi`), generated Excel/CSV summaries, the `segmentation results/`, `Nasa_Backend/output/`, `<video>_charts/`, and `<video>_per_frame_xlsx/` directories, or `nasa-frontend/.env`. The root `.gitignore` covers most of these but not all output artifacts.
- **`docs/` is local-only** (plans, specs, presentation, abstract drafts): never commit or push anything under `docs/`. Enforced via `/docs/` in `.git/info/exclude` (clone-local, not in the tracked `.gitignore`); `git add docs/...` will refuse unless forced — don't force it. The branch history was deliberately rewritten on 2026-06-10 to remove previously committed docs files; don't re-add them.
- **Don't use `git add .`** in this repo (per the README) — large generated artifacts sit alongside source. Stage files explicitly.
- **This box cannot push to GitHub**: https remote with no credential helper, `~/.ssh/id_ed25519` is not authorized for GitHub (it's the Mac→Ubuntu login key), `gh` is not installed, and the koncordantlab org rejects the available MCP token (fine-grained-PAT lifetime policy). Commit locally and tell the user to push from their authenticated environment. Read-only API/fetch works unauthenticated (repo is public).
- **Backend tests**: pytest, run from inside `Nasa_Backend/` in the project's
  Python env (on the lab box that's the conda `droplets` env:
  `~/miniconda3/envs/droplets/bin/python`). Tiers: `python -m pytest -m "not
  local"` is tier 1 (CPU-only; the conftest stubs the import-time YOLO load
  when weights are absent, so it also runs in CI); `python -m pytest -m "local
  and not slow"` is tier 2 (GPU + weights + the two fast basic-mode golden
  masters, minutes) — **required before opening any PR**; `python -m pytest -m
  "local and slow"` is the full-mode golden (`full_um`, ~10 min warm on the
  dense golden clip — the per-instance contour + pairwise-overlap work) — run
  it before merging changes that touch numeric code paths, and inside the
  Docker container when validating an image. Golden masters live in
  `Nasa_Backend/tests/golden/expected/*.json`; re-record with `python
  tests/golden/record_goldens.py` ONLY when a numeric change is intended and
  reviewed. The GPU/cv2 bit-exactness pins live in `tests/unit/test_masks.py`
  — rerun after any OpenCV/torch upgrade.
- Branch off `main`; the README requires PR review before merging.
- GPU is auto-detected via `torch.cuda.is_available()`; both `frontend_nasa13_apiV2.py` and `tracking.py` fall back to CPU silently if CUDA isn't available, which makes runs much slower but still functional.
