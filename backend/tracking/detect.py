"""YOLO detection stage: load the segmentation model, convert Ultralytics
results into the pipeline's detection lists, and stream per-frame detection
records to detections.json. This is the ONLY module in the tracking package
that imports ultralytics -- the seam where the future droplet_backend.model
wrapper swaps in. Importing this module never loads model weights;
load_model() runs only when the CLI (or a caller) invokes it. Function bodies
verbatim from the monolith."""
import os
from typing import Optional

import numpy as np
from ultralytics import YOLO

from tracking.config import DEFAULT_CONFIG, TrackingConfig
from tracking.detections import FrameDetections
from tracking.geometry import detection_area, is_relevant_circular_segment
from tracking.io import (
    JsonArrayWriter,
    detection_record,
    ensure_parent_dir,
    open_video,
)

def load_model(model_path: str) -> YOLO:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return YOLO(model_path)

def detections_from_results(results, cfg: TrackingConfig) -> FrameDetections:
    dets = FrameDetections()
    dropped_invalid_segment = 0
    dropped_invalid_area = 0
    dropped_low_conf = 0
    dropped_non_circular = 0
    kept_non_circular = 0
    masks_xy = None
    if getattr(results, "masks", None) is not None and getattr(results.masks, "xy", None) is not None:
        masks_xy = results.masks.xy
    if masks_xy is None and cfg.segments.require_segmentation:
        raise RuntimeError("Instance segmentation required, but model returned no masks. Use a segmentation model.")
    if masks_xy is None:
        masks_xy = []
    boxes = getattr(results, "boxes", None)
    if boxes is None and len(masks_xy) > 0:
        raise RuntimeError(
            "Instance segmentation required, but model returned masks without confidence metadata."
        )
    if boxes is not None and len(boxes) < len(masks_xy):
        raise RuntimeError(
            f"Inconsistent segmentation output: masks={len(masks_xy)} exceeds boxes={len(boxes)}."
        )
    if boxes is not None and len(boxes) != len(masks_xy):
        print(
            "Warning: segmentation/box count mismatch; using masks as source of truth "
            f"(masks={len(masks_xy)}, boxes={len(boxes)})."
        )

    for det_idx, mask_points in enumerate(masks_xy):
        box = boxes[det_idx] if boxes is not None else None
        conf = float(box.conf[0]) if box is not None and hasattr(box, "conf") else 0.0
        cls = int(box.cls[0]) if box is not None and hasattr(box, "cls") else 0
        if conf < cfg.segments.min_confidence:
            dropped_low_conf += 1
            continue
        segment = None
        points = np.asarray(mask_points, dtype=np.float32)
        if points.ndim == 2 and points.shape[0] >= 3 and points.shape[1] == 2:
            segment = points
        if cfg.segments.require_segmentation and segment is None:
            dropped_invalid_segment += 1
            continue
        det_area = detection_area(segment)
        if cfg.segments.require_segmentation and det_area <= 0:
            dropped_invalid_area += 1
            continue
        if segment is not None and not is_relevant_circular_segment(segment, cfg):
            if cfg.segments.keep_non_circular_for_association:
                kept_non_circular += 1
            else:
                dropped_non_circular += 1
                continue
        dets.append(segment, det_area, conf, cls, det_idx)
    if (
        dropped_invalid_segment > 0
        or dropped_invalid_area > 0
        or dropped_low_conf > 0
        or dropped_non_circular > 0
        or kept_non_circular > 0
    ):
        print(
            "Warning: dropped segmented instances "
            f"(low_conf={dropped_low_conf}, conf_threshold={cfg.segments.min_confidence}, "
            f"invalid_segment={dropped_invalid_segment}, invalid_area={dropped_invalid_area}, "
            f"non_circular_dropped={dropped_non_circular}, non_circular_kept={kept_non_circular}, "
            f"kept={len(dets)}, total_masks={len(masks_xy)})"
        )
    return dets

def export_detections_json(model, video_path, detections_path, config: Optional[TrackingConfig] = None):
    cfg = DEFAULT_CONFIG if config is None else config
    cap, _, _, _ = open_video(video_path)
    ensure_parent_dir(detections_path)
    writer = JsonArrayWriter(detections_path)
    frame_id = 1
    max_raw_segments = 0
    max_filtered_segments = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, verbose=False)[0]
            masks_xy = getattr(getattr(results, "masks", None), "xy", None)
            raw_segment_count = int(len(masks_xy)) if masks_xy is not None else 0
            max_raw_segments = max(max_raw_segments, raw_segment_count)
            dets = detections_from_results(results, cfg)
            max_filtered_segments = max(max_filtered_segments, len(dets))
            det_records = [detection_record(dets, i) for i in range(len(dets))]
            writer.write({"frame": frame_id, "detections": det_records})
            frame_id += 1
    finally:
        writer.close()
        cap.release()
    print(f"Max raw segmented instances in a frame: {max_raw_segments}")
    print(f"Max kept segmented instances in a frame (after conf/shape filters): {max_filtered_segments}")
    print(f"Segmentation confidence threshold: {cfg.segments.min_confidence}")
    print(f"Detections saved to: {detections_path}")
