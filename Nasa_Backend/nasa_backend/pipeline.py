# nasa_backend/pipeline.py
"""process_video orchestration: frame sampling (stride = round(fps)), batches
of 4, progress/eta events, summary workbook, chart PNGs, size-distribution
checkpoints, per-frame instance workbooks. Numeric behavior is golden-pinned
(tests/golden). The model is resolved lazily via nasa_backend.model.get_model()
at call time so tests can monkeypatch it."""
import os
import time

import cv2
import numpy as np
import pandas as pd

from nasa_backend import model as model_mod
from nasa_backend.charts import _save_chart_pngs, _save_size_distribution_pngs
from nasa_backend.config import ALPHA_OVERLAP, ALPHA_SEG, COLOR_MAP, OVERLAP_COLORS
from nasa_backend.distribution import (
    _droplet_stats_block,
    _eq_diameter,
    _shared_bin_edges,
    SIZE_DIST_BINS,
)
from nasa_backend.excel import _save_per_frame_instance_xlsx
from nasa_backend.masks import (
    _classify_overlaps,
    _gather_resize_nn,
    _mask_areas_from_source,
    _overlap_exists_matrix,
    _threshold_masks,
)
from nasa_backend.metrics import (
    _avg_size_metrics,
    _per_instance_metrics,
    _resolution_pix_per_um2,
)


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
            results_list = model_mod.get_model().predict(batch_frames)

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
                    # masks. See tests/unit/test_masks.py.
                    areas = _mask_areas_from_source(binm, h, w)

                    # Build the full-res CPU masks ONLY when something needs them:
                    # the overlay (every frame) or full-mode per-instance contour
                    # metrics (checkpoints + the final-frame stash). In basic mode
                    # without overlay they are never built, which removes the
                    # per-frame ~N·h·w GPU→CPU transfer and matching GPU alloc.
                    # Non-checkpoint, non-overlay frames skip the gather entirely —
                    # the small source masks are stashed in last_frame_raw instead
                    # and materialized lazily only if that frame turns out to be
                    # the final one written (see the final-frame block below).
                    is_checkpoint = dist_interval > 0 and current_processed_frame % dist_interval == 0
                    need_full_masks = save_ovl or (output_mode != "basic" and dist_interval > 0 and is_checkpoint)
                    full_t = None
                    if need_full_masks:
                        full_t = _gather_resize_nn(binm, h, w)
                        full_np = full_t.cpu().numpy()
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
                    # tests/unit/test_masks.py.
                    overlap_src = binm if (h >= mh and w >= mw) else (
                        full_t if full_t is not None else _gather_resize_nn(binm, h, w))
                    exists = _overlap_exists_matrix(
                        overlap_src.reshape(overlap_src.shape[0], -1)
                    ).cpu().numpy()
                    ww_count, ii_count, wi_count = _classify_overlaps(exists, class_names)
                    overlap_totals["ww"] += ww_count
                    overlap_totals["ii"] += ii_count
                    overlap_totals["mixed"] += wi_count

                    if save_ovl and out_video_writer is not None:
                        # Reuse the GPU-resized masks instead of resizing a third
                        # time; identical output (tests/unit/test_masks.py).
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
                            "binm": binm,
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
                if (output_mode != "basic" and last_frame_raw.get("full_bin_masks") is None
                        and last_frame_raw.get("binm") is not None):
                    _h, _w = last_frame_raw["frame_shape"]
                    _full = _gather_resize_nn(last_frame_raw["binm"], _h, _w).cpu().numpy()
                    last_frame_raw["full_bin_masks"] = [_full[k] for k in range(_full.shape[0])]
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
