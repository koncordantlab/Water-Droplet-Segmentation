# nasa_backend/excel.py
"""Per-frame per-instance workbooks (full: five sheets, basic: three).
Column reference: Nasa_Backend/per_frame_xlsx_schema.md. Histogram sheets
reuse the plot's global log-spaced edges — never recompute bins per frame."""
import os

import pandas as pd

from nasa_backend.distribution import _global_bin_edges_from_size_distribution, _histogram_df
from nasa_backend.metrics import _apply_metric, _stats_row


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
