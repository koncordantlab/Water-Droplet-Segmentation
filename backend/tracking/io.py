"""I/O plumbing: streaming JSON-array writer, video opening, and the
JSON/CSV serialization of tracking logs. Verbatim from the monolith."""
import csv
import json
import os
from typing import Optional

import cv2
import numpy as np

from tracking.geometry import segment_center

class JsonArrayWriter:
    def __init__(self, path: str):
        self._file = open(path, "w", encoding="utf-8")
        self._file.write("[")
        self._first = True

    def write(self, obj):
        if not self._first:
            self._file.write(",\n")
        else:
            self._first = False
        json.dump(obj, self._file, ensure_ascii=True)

    def close(self):
        self._file.write("]\n")
        self._file.close()

def open_video(video_path: str):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return cap, fps, width, height

def serialize_center(segment: Optional[np.ndarray]):
    cx, cy = segment_center(segment)
    return [float(cx), float(cy)]

def serialize_segment(segment: Optional[np.ndarray]):
    if segment is None:
        return None
    if segment.ndim != 2 or segment.shape[0] < 3:
        return None
    return [[float(pt[0]), float(pt[1])] for pt in segment]

def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

def _csv_cell(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    return value

def export_tracking_log_csv(log_json_path: str, csv_path: str) -> None:
    """Flatten tracking_log.json events into a stable CSV view."""
    with open(log_json_path, "r", encoding="utf-8") as file:
        frames_data = json.load(file)
    if not isinstance(frames_data, list):
        raise RuntimeError("Tracking log JSON must be a list of frame records.")

    ensure_parent_dir(csv_path)
    fieldnames = [
        "frame",
        "event",
        "track_id",
        "gen",
        "det_id",
        "parents",
        "detected_parents",
        "unknown_parents",
        "iou",
        "dist_norm",
        "area_ratio",
        "reference_area",
        "growth_threshold",
        "streak",
        "candidate_score",
        "candidate_peak_area_ratio",
        "reason",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for frame_record in frames_data:
            frame_id = int(frame_record.get("frame", 0))
            for event in frame_record.get("events", []):
                writer.writerow(
                    {
                        "frame": frame_id,
                        "event": event.get("type", ""),
                        "track_id": event.get("track_id", ""),
                        "gen": event.get("gen", ""),
                        "det_id": event.get("det_id", ""),
                        "parents": _csv_cell(event.get("parents", [])),
                        "detected_parents": _csv_cell(event.get("detected_parents", [])),
                        "unknown_parents": event.get("unknown_parents", ""),
                        "iou": event.get("iou", ""),
                        "dist_norm": event.get("dist_norm", ""),
                        "area_ratio": event.get("area_ratio", ""),
                        "reference_area": event.get("reference_area", ""),
                        "growth_threshold": event.get("growth_threshold", ""),
                        "streak": event.get("streak", ""),
                        "candidate_score": event.get("candidate_score", ""),
                        "candidate_peak_area_ratio": event.get("candidate_peak_area_ratio", ""),
                        "reason": event.get("reason", ""),
                    }
                )
