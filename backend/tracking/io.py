"""I/O plumbing: streaming JSON-array writer, video opening, and the
JSON/CSV serialization of tracking logs. Verbatim from the monolith."""
import csv
import json
import os
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from tracking.geometry import segment_center

if TYPE_CHECKING:  # typing-only imports
    from tracking.detections import FrameDetections
    from tracking.tracks import Track

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


def detection_record(dets: "FrameDetections", i: int) -> dict:
    """One detection as written to detections.json and to every tracking-log frame."""
    return {
        "det_id": dets.det_ids[i],
        "center": serialize_center(dets.segments[i]),
        "area": float(dets.areas[i]),
        "segment": serialize_segment(dets.segments[i]),
        "conf": dets.confs[i],
        "cls": dets.classes[i],
    }


def track_record(track_id: int, track: "Track") -> dict:
    return {
        "track_id": track_id,
        "gen": track.gen,
        "center": serialize_center(track.segment),
        "area": float(track.area),
        "missed": track.missed,
        "merge_candidate_streak": track.merge_candidate_streak,
        "merge_candidate_score": float(track.merge_candidate_score),
        "last_merge_candidate_frame": int(track.last_merge_candidate_frame),
        "merge_candidate_peak_area_ratio": float(track.merge_candidate_peak_area_ratio),
        "status": "active" if track.missed == 0 else "lost",
        "segment": serialize_segment(track.segment),
    }


def write_config_sidecar(artifact_path: str, cfg) -> str:
    """Write <artifact stem>.config.json next to an output, so the settings that
    produced it travel with it."""
    path = os.path.splitext(artifact_path)[0] + ".config.json"
    with open(path, "w", encoding="utf-8") as file:
        json.dump(cfg.to_dict(), file, indent=2)
        file.write("\n")
    return path


def frame_log_record(frame_id: int, dets: "FrameDetections", tracks: dict, events: list) -> dict:
    """One tracking_log.json element."""
    return {
        "frame": frame_id,
        "detections": [detection_record(dets, i) for i in range(len(dets))],
        "tracks": [track_record(tid, track) for tid, track in tracks.items()],
        "events": events,
    }
