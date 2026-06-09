# Per-frame xlsx — column reference

This describes the `<video>_frame_NNNNNN_instances.xlsx` files written into the `<video>_per_frame_xlsx/` folder when `process_video` is called with `dist_interval > 0`.

One file is written per **checkpoint frame**: every Nth processed frame (where N = `dist_interval`) plus the final non-empty processed frame. Empty frames are skipped.

The file contents depend on the run's **output mode**:

- **Full** (`output_mode="full"`, default) — the 5-sheet, pixel-unit workbook documented in this file: `Instances`, `Frame Info`, `Stats`, `Histogram Water`, `Histogram Ice`.
- **Basic** (`output_mode="basic"`) — a slim 3-sheet workbook (`Instances`, `Frame Info`, `Stats`) with metric (µm) columns and no histogram sheets. See the [Basic mode](#basic-mode-output_modebasic) section at the bottom.

The rest of this document describes **Full** mode unless noted otherwise.

> **Units.** In Full mode, all lengths and areas are in **pixel units** of the original video resolution. The model has no physical calibration — in Full mode, if you need µm or mm, apply a separate scale factor downstream, or use Basic mode with a `um_per_px` value. Two videos with different optics produce non-comparable absolute pixel sizes.

---

## Sheet: `Instances`

One row per detected droplet in the frame.

### Identity

| Column | Type | Meaning / derivation |
|---|---|---|
| `instance_id` | int (1-based) | Sequential index within this frame. **Not a track ID** — the same physical droplet may have a different `instance_id` in a different frame. |
| `class` | str | `"water"` or `"ice"` — predicted by the YOLO model (from `res.boxes[i].cls`). |
| `confidence` | float [0,1] | Model's confidence for this detection (`res.boxes[i].conf`). |

### Size (the core measurements)

| Column | Type | Meaning / derivation |
|---|---|---|
| `pixel_count` | int (px) | Number of `1`-pixels in the binary mask. Built from `res.masks.data[i]` by thresholding at `0.3`, then resizing to the original frame resolution with `cv2.INTER_NEAREST`. This is the **fundamental size measurement**; everything else is derived from it. |
| `eq_diameter_px` | float (px) | Equivalent circular diameter: `d = √(4 · pixel_count / π)`. The diameter of a circle with the same area. A monotonic 1-to-1 transform of `pixel_count` — same distribution shape, more intuitive axis. |

### Position

| Column | Type | Meaning / derivation |
|---|---|---|
| `centroid_x`, `centroid_y` | float (px) | Mean (x, y) of the mask's `1`-pixels. Image coords; `(0, 0)` is top-left. |

### Bounding box (from YOLO, not from the mask)

| Column | Type | Meaning / derivation |
|---|---|---|
| `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` | float (px) | Top-left and bottom-right of the axis-aligned bounding box (`res.boxes[i].xyxy`). |
| `bbox_width` | float (px) | `bbox_x2 − bbox_x1`. |
| `bbox_height` | float (px) | `bbox_y2 − bbox_y1`. |
| `bbox_area` | float (px²) | `bbox_width × bbox_height`. Bounding box area, **not** mask area. |
| `bbox_aspect_ratio` | float | `bbox_width / bbox_height`. Hint of elongation along image axes — meaningless for rotated shapes (use `ellipse_eccentricity` instead). |

### Shape descriptors

All from `cv2.findContours(full_bin_mask, RETR_EXTERNAL, CHAIN_APPROX_NONE)` on the **largest connected component** of the mask. If the contour is too small or degenerate, the value is `None`.

| Column | Type | Meaning / derivation |
|---|---|---|
| `extent` | float [0,1] | `pixel_count / bbox_area`. How tightly the mask fills its bbox. 1.0 = bbox fully covered. |
| `perimeter_px` | float (px) | `cv2.arcLength(contour, closed=True)`. Length of the mask outline. |
| `circularity` | float (≤ 1) | `4π · pixel_count / perimeter²`. **1.0 = perfect circle**; lower = irregular or elongated. Use this as a sanity check on `eq_diameter_px` — if circularity is far from 1, the equivalent-circular-diameter is a poor size summary for that shape. |
| `solidity` | float [0,1] | `pixel_count / convex_hull_area`. **1.0 = convex** (no indentations). Lower means concavities. |
| `feret_diameter_max_px` | float (px) | Longest straight-line distance between any two points on the convex hull. The "longest extent" of the shape, orientation-independent. Always ≥ `eq_diameter_px` for non-circular shapes. |

### Ellipse fit

From `cv2.fitEllipse(contour)`. **Requires ≥ 5 contour points** — fields are `None` otherwise.

| Column | Type | Meaning / derivation |
|---|---|---|
| `ellipse_major_axis_px` | float (px) | Longer axis of the fitted ellipse. |
| `ellipse_minor_axis_px` | float (px) | Shorter axis of the fitted ellipse. |
| `ellipse_eccentricity` | float [0,1] | `√(1 − (minor/major)²)`. **0 = circle**, → 1 = highly elongated. |
| `ellipse_angle_deg` | float (deg) | Rotation of the major axis. OpenCV convention (0° = vertical). |

### Boundary flag

| Column | Type | Meaning / derivation |
|---|---|---|
| `touches_border` | bool | `True` if any mask pixel lies on the frame edge. **Important caveat**: when `True`, the droplet is clipped at the edge — its `pixel_count`, `eq_diameter_px`, and shape descriptors all **underestimate** the true size. Filter these rows out for accurate size distributions. |

### Per-instance overlap

Computed pairwise across all instances in this frame using bitwise-AND of their full-resolution binary masks.

| Column | Type | Meaning / derivation |
|---|---|---|
| `overlap_count` | int | Number of other instances whose mask intersects this one. |
| `overlap_classes` | str (CSV) | Distinct classes of overlapping neighbors, e.g. `""`, `"ice"`, `"water,ice"`. |
| `overlap_pixels_total` | int (px) | Sum of intersection pixel counts across all overlapping neighbors. |

---

## Sheet: `Frame Info`

One-row table with metadata for this frame, useful for joining across files.

| Column | Type | Meaning / derivation |
|---|---|---|
| `processed_frame_number` | int (1-based) | Index in the sequence of processed frames (sampled at ~1 FPS). |
| `original_video_frame` | int | `processed_frame_number × video_stride`. Frame index in the original video file. |
| `frame_time_seconds` | float (s) | `original_video_frame / video_fps`. Time in the video. |
| `video_name` | str | Filename of the source video. |
| `video_fps` | float | Source video FPS (`cv2.CAP_PROP_FPS`). |
| `video_stride` | int | Frame-sampling stride used by `process_video` (≈ fps, so ~1 processed frame per second). |
| `frame_width`, `frame_height` | int (px) | Original video resolution. |
| `total_instances` | int | Number of detected droplets. |
| `water_count`, `ice_count` | int | Per-class counts. |

---

## Sheet: `Stats`

Two rows — one per class. Computed on the `eq_diameter_px` column of `Instances` for that class. Matches the stats table shown in the on-screen size-distribution plot.

| Column | Type | Meaning |
|---|---|---|
| `class` | str | `"water"` or `"ice"`. |
| `count` | int | Number of instances of that class. |
| `min`, `max`, `mean`, `median`, `std` | float (px) | Standard summary stats over `eq_diameter_px`. `None` when `count == 0`. |

---

## Sheets: `Histogram Water` and `Histogram Ice`

Long-format histogram for the size distribution. **30 rows per sheet** (one per bin).

| Column | Type | Meaning |
|---|---|---|
| `bin_lo` | float (px) | Lower edge of the bin (inclusive). |
| `bin_hi` | float (px) | Upper edge of the bin (exclusive, except for the last bin). |
| `bin_center` | float (px) | `(bin_lo + bin_hi) / 2`. |
| `count` | int | Number of instances of that class falling into that bin. |

**Bin edges are log-spaced** and **shared across all checkpoint frames in the video** — they're derived globally from the min/max diameter across every checkpoint. This is what lets you compare distributions across frames directly, and it's exactly the binning the on-screen plot uses, bar-for-bar. **Do not assume equal-width bins**: `bin_hi − bin_lo` grows from bin to bin.

---

## End-to-end derivation (per instance)

1. YOLO returns a soft mask `res.masks.data[i]` (float in `[0, 1]`, native resolution ~160×160 for `imgsz=640`).
2. Threshold at `0.3` → `uint8` binary mask.
3. Resize to `(frame_width, frame_height)` with `cv2.INTER_NEAREST` → `full_bin_mask`.
4. `pixel_count = full_bin_mask.sum()`.
5. `eq_diameter_px = √(4 · pixel_count / π)`.
6. `cv2.findContours(full_bin_mask, RETR_EXTERNAL, CHAIN_APPROX_NONE)` on the binary mask → contour list. Pick the largest by area.
7. Shape descriptors (perimeter, circularity, solidity, hull, feret, ellipse) are computed from that contour.
8. Overlap columns come from bitwise-AND between this instance's mask and every other instance's mask in the same frame.

The implementation lives in `_per_instance_metrics()` in `frontend_nasa13_apiV2.py`.

---

## Known caveats

- **No physical calibration** — pixel units only.
- **Equivalent circular diameter assumes circularity** — cross-check with `circularity` and `ellipse_eccentricity`.
- **Boundary droplets** are not excluded — use `touches_border` to filter.
- **Mask resolution is ~160×160** before upscaling. Very small droplets (a few pixels on the native mask) are quantized.
- **Threshold of 0.3** on the soft mask is the project's chosen cutoff; a different cutoff would shift `pixel_count` slightly.
- **Disconnected mask components**: `pixel_count` covers all components, but shape descriptors are computed only on the largest. In practice YOLO instance masks are single-component; the discrepancy is rare and small.

---

## Basic mode (`output_mode="basic"`)

When the run is started in **Basic** mode, each per-frame file is slimmed to three sheets and gains metric (micron) columns derived from the `um_per_px` scale the user supplies. There are **no Histogram sheets** in Basic mode. The `size_distribution` payload/PNGs and the aggregate `<video>_detection_summary.xlsx` are unaffected — they are always produced the same way in both modes; mode controls only these per-instance files.

Basic mode also **skips** the contour/hull/feret/ellipse/overlap computation entirely, so it is faster than Full.

Metric conversion uses a single scale factor:

- `eq_diameter_um = eq_diameter_px × um_per_px`
- `area_um2       = pixel_count × um_per_px²`

If `um_per_px` is omitted or `≤ 0`, both metric columns are written as blank (NaN) and `um_per_px` in `Frame Info` is empty. The run still succeeds.

### Sheet: `Instances` (7 columns, one row per instance)

| Column | Type | Meaning / derivation |
|---|---|---|
| `instance_id` | int (1-based) | Sequential index within this frame. |
| `class` | str | `"water"` or `"ice"`. |
| `confidence` | float [0,1] | Detection confidence, rounded 4dp. |
| `pixel_count` | int (px) | Full mask pixel sum. Identical to Full mode (matches `water_pixel_area`/`ice_pixel_area`). |
| `eq_diameter_px` | float (px) | `√(4 · pixel_count / π)`, rounded 3dp. Identical to Full mode. |
| `eq_diameter_um` | float (µm) | `eq_diameter_px × um_per_px`, rounded 3dp; NaN if scale missing. |
| `area_um2` | float (µm²) | `pixel_count × um_per_px²`, rounded 3dp; NaN if scale missing. |

### Sheet: `Frame Info` (one row)

Same fields as Full mode (`processed_frame_number`, `original_video_frame`, `frame_time_seconds`, `video_name`, `video_fps`, `video_stride`, `frame_width`, `frame_height`, `total_instances`, `water_count`, `ice_count`) **plus** `um_per_px` — the scale used for this run (empty when none was supplied).

### Sheet: `Stats` (one row per class)

`count / min / max / mean / median / std` per class (water, ice), computed over the **`eq_diameter_um`** values (metric units). NaN-valued instances are excluded; a class with no valid metric values reports `count = 0` and `None` stats.

The Basic-mode implementation lives in `_per_instance_metrics(..., mode="basic")`, `_apply_metric()`, and `_save_per_frame_instance_xlsx(..., mode="basic", um_per_px=...)` in `frontend_nasa13_apiV2.py`.
