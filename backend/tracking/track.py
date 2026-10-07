"""Orchestration of the track stage: read video frames and per-frame
detections, step the Tracker, draw the overlay, and write the annotated mp4,
tracking_log.json, and its flattened CSV. The algorithm lives in
tracking.tracker; nothing here makes a tracking decision."""
import json
import os
from typing import Optional

from tracking.config import DEFAULT_CONFIG, TrackingConfig
from tracking.detections import frame_detections_from_record
from tracking.io import (
    JsonArrayWriter,
    ensure_parent_dir,
    export_tracking_log_csv,
    frame_log_record,
    open_video,
)
from tracking.render import apply_tracking_overlay, create_video_writer
from tracking.tracker import Tracker


def track_from_detections_json(video_path, detections_path, output_path, log_path,
                               config: Optional[TrackingConfig] = None):
    cfg = DEFAULT_CONFIG if config is None else config
    cap, fps, width, height = open_video(video_path)
    ensure_parent_dir(output_path)
    ensure_parent_dir(log_path)
    writer = create_video_writer(output_path, fps, width, height)
    tracker = Tracker(cfg)
    print(f"Tracking candidate prefilter device: {tracker.device}")

    with open(detections_path, "r", encoding="utf-8") as file:
        frames_data = json.load(file)
    if not isinstance(frames_data, list):
        raise RuntimeError("Detections JSON must be a list of frame records.")

    log_writer = JsonArrayWriter(log_path)
    try:
        for frame_index, frame_record in enumerate(frames_data):
            ret, frame = cap.read()
            if not ret:
                break
            frame_id = int(frame_record.get("frame", frame_index + 1))
            dets = frame_detections_from_record(frame_record, cfg)
            result = tracker.step(frame_id, dets)
            if cfg.render.draw_overlay:
                apply_tracking_overlay(frame, tracker.tracks, result.merged_children, cfg)
            writer.write(frame)
            log_writer.write(frame_log_record(frame_id, dets, tracker.tracks, result.events))
            print("Frame complete", frame_id)
    finally:
        log_writer.close()
        cap.release()
        writer.release()

    csv_log_path = os.path.splitext(log_path)[0] + ".csv"
    export_tracking_log_csv(log_path, csv_log_path)
    print(f"Tracking complete. Output saved to: {output_path}")
    print(f"Tracking log saved to: {log_path}")
    print(f"Tracking CSV saved to: {csv_log_path}")
