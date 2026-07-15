#!/usr/bin/env python3
"""Analyze tracking outputs for quality and debugging insights. Verbatim
from analyze_tracking_json.py; only this docstring and the DEFAULT_OUTPUT_DIR
anchoring (re-anchored to backend/output/, like config.py) differ.

Usage:
  python -m tracking analyze
  python -m tracking analyze --help
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(_BACKEND_DIR, "output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze detections.json and tracking_log.json.")
    parser.add_argument(
        "--detections",
        default=os.path.join(DEFAULT_OUTPUT_DIR, "detections.json"),
        help="Path to detections.json",
    )
    parser.add_argument(
        "--tracking-log",
        default=os.path.join(DEFAULT_OUTPUT_DIR, "tracking_log.json"),
        help="Path to tracking_log.json",
    )
    parser.add_argument(
        "--summary-out",
        default=os.path.join(DEFAULT_OUTPUT_DIR, "insights_summary.json"),
        help="Output path for aggregated JSON summary",
    )
    parser.add_argument(
        "--suspicious-out",
        default=os.path.join(DEFAULT_OUTPUT_DIR, "suspicious_transitions.csv"),
        help="Output path for suspicious ID carry-over candidates (CSV)",
    )
    parser.add_argument(
        "--per-track-out",
        default=os.path.join(DEFAULT_OUTPUT_DIR, "per_track_insights.csv"),
        help="Output path for per-track insight metrics (CSV, all tracks).",
    )
    parser.add_argument(
        "--jump-dist-norm-thresh",
        type=float,
        default=1.0,
        help="Flag active->active same-track jumps with dist_norm >= this value.",
    )
    parser.add_argument(
        "--jump-iou-thresh",
        type=float,
        default=0.1,
        help="When IoU <= this threshold and area growth is high, flag as suspicious.",
    )
    parser.add_argument(
        "--jump-area-growth-thresh",
        type=float,
        default=1.6,
        help="Flag active->active same-track transitions with large area growth + low IoU.",
    )
    parser.add_argument(
        "--lost-reactivation-dist-norm-thresh",
        type=float,
        default=0.45,
        help="Flag lost->active same-track jumps with dist_norm >= this value.",
    )
    parser.add_argument(
        "--lost-reactivation-area-growth-thresh",
        type=float,
        default=1.8,
        help="Flag lost->active transitions with large area growth as suspicious.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of suspicious transitions to print in console.",
    )
    return parser.parse_args()


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in JSON: {path}")
    return data


def box_area(box: List[float]) -> float:
    if not isinstance(box, list) or len(box) != 4:
        return 0.0
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def box_center(box: List[float]) -> Tuple[float, float]:
    if not isinstance(box, list) or len(box) != 4:
        return 0.0, 0.0
    return (float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0


def box_max_dim(box: List[float]) -> float:
    if not isinstance(box, list) or len(box) != 4:
        return 1.0
    return max(1.0, float(box[2]) - float(box[0]), float(box[3]) - float(box[1]))


def _segment_points(segment: Any) -> List[Tuple[float, float]]:
    if not isinstance(segment, list) or len(segment) < 3:
        return []
    points: List[Tuple[float, float]] = []
    for item in segment:
        if not isinstance(item, list) or len(item) != 2:
            continue
        points.append((float(item[0]), float(item[1])))
    return points


def segment_bounds(segment: Any) -> List[float]:
    points = _segment_points(segment)
    if not points:
        return [0.0, 0.0, 0.0, 0.0]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def segment_center(segment: Any) -> Tuple[float, float]:
    points = _segment_points(segment)
    if not points:
        return 0.0, 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))


def segment_max_dim(segment: Any) -> float:
    b = segment_bounds(segment)
    return max(1.0, float(b[2] - b[0]), float(b[3] - b[1]))


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    if len(box_a) != 4 or len(box_b) != 4:
        return 0.0
    x_a = max(float(box_a[0]), float(box_b[0]))
    y_a = max(float(box_a[1]), float(box_b[1]))
    x_b = min(float(box_a[2]), float(box_b[2]))
    y_b = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x_b - x_a)
    inter_h = max(0.0, y_b - y_a)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = box_area(box_a)
    area_b = box_area(box_b)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def transition_center(record: Dict[str, Any]) -> Tuple[float, float]:
    center = record.get("center")
    if isinstance(center, list) and len(center) == 2:
        return float(center[0]), float(center[1])
    segment = record.get("segment")
    if segment:
        return segment_center(segment)
    return box_center(record.get("bbox", [0.0, 0.0, 0.0, 0.0]))


def transition_max_dim(record: Dict[str, Any]) -> float:
    segment = record.get("segment")
    if segment:
        return segment_max_dim(segment)
    return box_max_dim(record.get("bbox", [0.0, 0.0, 0.0, 0.0]))


def transition_iou(record_a: Dict[str, Any], record_b: Dict[str, Any]) -> float:
    seg_a = record_a.get("segment")
    seg_b = record_b.get("segment")
    if seg_a and seg_b:
        return compute_iou(segment_bounds(seg_a), segment_bounds(seg_b))
    return compute_iou(record_a.get("bbox", [0.0, 0.0, 0.0, 0.0]), record_b.get("bbox", [0.0, 0.0, 0.0, 0.0]))


def transition_distance(record_a: Dict[str, Any], record_b: Dict[str, Any]) -> float:
    ax, ay = transition_center(record_a)
    bx, by = transition_center(record_b)
    return math.hypot(ax - bx, ay - by)


def numeric_summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "min": float(min(values)),
        "median": float(statistics.median(values)),
        "mean": float(statistics.mean(values)),
        "max": float(max(values)),
    }


def build_event_index(tracking_frames: List[Dict[str, Any]]) -> Dict[Tuple[int, int], List[str]]:
    index: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    for frame_record in tracking_frames:
        frame_id = int(frame_record.get("frame", 0))
        for event in frame_record.get("events", []):
            track_id = event.get("track_id")
            if isinstance(track_id, int):
                index[(frame_id, track_id)].append(str(event.get("type", "")))
    return index


def analyze(
    detection_frames: List[Dict[str, Any]],
    tracking_frames: List[Dict[str, Any]],
    jump_dist_norm_thresh: float,
    jump_iou_thresh: float,
    jump_area_growth_thresh: float,
    lost_reactivation_dist_norm_thresh: float,
    lost_reactivation_area_growth_thresh: float,
) -> Dict[str, Any]:
    detection_frame_ids = [int(f.get("frame", i + 1)) for i, f in enumerate(detection_frames)]
    tracking_frame_ids = [int(f.get("frame", i + 1)) for i, f in enumerate(tracking_frames)]

    detection_counts = [len(f.get("detections", [])) for f in detection_frames]
    detection_count_hist = Counter(detection_counts)
    mode_detection_count = int(detection_count_hist.most_common(1)[0][0]) if detection_count_hist else 0
    mode_detection_frames = int(detection_count_hist.most_common(1)[0][1]) if detection_count_hist else 0
    uniform_detection_count_all_frames = (
        bool(detection_counts) and len(set(detection_counts)) == 1
    )
    event_counts = Counter()
    events_per_frame = []
    active_tracks_per_frame = []
    lost_tracks_per_frame = []

    tracks_by_id: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    all_inferred_merges: List[Dict[str, Any]] = []
    all_merge_events: List[Dict[str, Any]] = []
    events_by_track: Dict[int, Counter] = defaultdict(Counter)

    for frame_record in tracking_frames:
        frame_id = int(frame_record.get("frame", 0))
        events = frame_record.get("events", [])
        events_per_frame.append(len(events))
        for e in events:
            e_type = str(e.get("type", ""))
            event_counts[e_type] += 1
            e_track_id = e.get("track_id")
            if isinstance(e_track_id, int):
                events_by_track[e_track_id][e_type] += 1
            if e_type == "inferred_merge":
                all_inferred_merges.append({"frame": frame_id, **e})
            if e_type == "merge":
                all_merge_events.append({"frame": frame_id, **e})

        tracks = frame_record.get("tracks", [])
        active = 0
        lost = 0
        for t in tracks:
            tid = t.get("track_id")
            if not isinstance(tid, int):
                continue
            status = str(t.get("status", ""))
            if status == "active":
                active += 1
            elif status == "lost":
                lost += 1
            bbox = t.get("bbox", [0, 0, 0, 0])
            segment = t.get("segment")
            center = t.get("center")
            area = float(t.get("area", 0.0))
            if area <= 0.0:
                if segment:
                    area = box_area(segment_bounds(segment))
                else:
                    area = box_area(bbox)
            tracks_by_id[tid].append(
                {
                    "frame": frame_id,
                    "status": status,
                    "bbox": bbox,
                    "segment": segment,
                    "center": center,
                    "area": area,
                    "gen": int(t.get("gen", 1)),
                    "missed": int(t.get("missed", 0)),
                }
            )
        active_tracks_per_frame.append(active)
        lost_tracks_per_frame.append(lost)

    event_index = build_event_index(tracking_frames)

    suspicious_transitions: List[Dict[str, Any]] = []
    track_lifetime_active = []
    track_lifetime_total = []
    max_generation_per_track = []
    per_track_insights: List[Dict[str, Any]] = []

    for track_id, history in tracks_by_id.items():
        history.sort(key=lambda x: x["frame"])
        active_count = sum(1 for h in history if h["status"] == "active")
        total_count = len(history)
        max_gen = max((h["gen"] for h in history), default=1)
        track_lifetime_active.append(active_count)
        track_lifetime_total.append(total_count)
        max_generation_per_track.append(max_gen)

        max_lost_streak = 0
        current_lost_streak = 0
        prev_frame = None
        generation_change_count = 0
        prev_gen = None
        for h in history:
            frame = int(h["frame"])
            status = h["status"]
            gen = int(h["gen"])
            if prev_frame is not None and frame != prev_frame + 1:
                current_lost_streak = 0
            if status == "lost":
                current_lost_streak += 1
                max_lost_streak = max(max_lost_streak, current_lost_streak)
            else:
                current_lost_streak = 0
            if prev_gen is not None and gen != prev_gen:
                generation_change_count += 1
            prev_frame = frame
            prev_gen = gen

        transition_iou_values: List[float] = []
        transition_dist_norm_values: List[float] = []
        transition_area_growth_values: List[float] = []
        suspicious_count = 0

        for prev, curr in zip(history, history[1:]):
            if curr["frame"] != prev["frame"] + 1:
                continue
            if curr["status"] != "active":
                continue
            if prev["status"] not in ("active", "lost"):
                continue

            transition_type = f"{prev['status']}_to_active"
            iou = transition_iou(prev, curr)
            dist = transition_distance(prev, curr)
            dist_norm = dist / max(transition_max_dim(prev), transition_max_dim(curr))
            prev_area = max(prev["area"], 1e-6)
            area_growth = curr["area"] / prev_area
            if prev["status"] == "active":
                transition_iou_values.append(float(iou))
                transition_dist_norm_values.append(float(dist_norm))
                transition_area_growth_values.append(float(area_growth))

            reasons = []
            if dist_norm >= jump_dist_norm_thresh:
                reasons.append("far_jump")
            if iou <= jump_iou_thresh and area_growth >= jump_area_growth_thresh:
                reasons.append("low_iou_large_growth")
            if (
                prev["status"] == "lost"
                and (
                    dist_norm >= lost_reactivation_dist_norm_thresh
                    or area_growth >= lost_reactivation_area_growth_thresh
                )
            ):
                reasons.append("lost_reactivation_jump")
            if not reasons:
                continue
            suspicious_count += 1

            frame_events = event_index.get((curr["frame"], track_id), [])
            suspicious_transitions.append(
                {
                    "frame_from": prev["frame"],
                    "frame_to": curr["frame"],
                    "track_id": track_id,
                    "iou": float(iou),
                    "dist_norm": float(dist_norm),
                    "area_prev": float(prev["area"]),
                    "area_curr": float(curr["area"]),
                    "area_growth": float(area_growth),
                    "gen_from": int(prev["gen"]),
                    "gen_to": int(curr["gen"]),
                    "transition_type": transition_type,
                    "events_in_frame_to": frame_events,
                    "reasons": reasons,
                }
            )

        event_counter = events_by_track.get(track_id, Counter())
        merge_like_event_count = (
            int(event_counter.get("merge", 0))
            + int(event_counter.get("inferred_merge", 0))
            + int(event_counter.get("inferred_merge_unknown", 0))
            + int(event_counter.get("large_birth", 0))
        )
        merge_candidate_count = int(event_counter.get("merge_candidate", 0))
        match_rejected_growth_count = int(event_counter.get("match_rejected_growth", 0))
        transition_count = len(transition_iou_values)
        min_transition_iou = min(transition_iou_values) if transition_iou_values else 1.0
        max_transition_dist_norm = max(transition_dist_norm_values) if transition_dist_norm_values else 0.0
        max_transition_area_growth = max(transition_area_growth_values) if transition_area_growth_values else 1.0
        avg_transition_dist_norm = (
            statistics.mean(transition_dist_norm_values) if transition_dist_norm_values else 0.0
        )
        avg_transition_iou = statistics.mean(transition_iou_values) if transition_iou_values else 1.0
        area_values = [float(h["area"]) for h in history]
        lifespan_frames = int(history[-1]["frame"]) - int(history[0]["frame"]) + 1

        anomaly_score = (
            4.0 * suspicious_count
            + 2.0 * match_rejected_growth_count
            + 1.0 * merge_like_event_count
            + 0.5 * merge_candidate_count
            + max(0.0, max_transition_dist_norm - jump_dist_norm_thresh)
            + 0.5 * max(0, max_gen - 1)
            + 0.25 * max_lost_streak
        )
        per_track_insights.append(
            {
                "track_id": int(track_id),
                "first_frame": int(history[0]["frame"]),
                "last_frame": int(history[-1]["frame"]),
                "lifespan_frames": int(lifespan_frames),
                "frames_present": int(total_count),
                "active_frames": int(active_count),
                "lost_frames": int(total_count - active_count),
                "active_ratio": float(active_count / total_count) if total_count else 0.0,
                "max_gen": int(max_gen),
                "generation_change_count": int(generation_change_count),
                "max_lost_streak": int(max_lost_streak),
                "min_area": float(min(area_values)) if area_values else 0.0,
                "mean_area": float(statistics.mean(area_values)) if area_values else 0.0,
                "max_area": float(max(area_values)) if area_values else 0.0,
                "transition_count_active_to_active": int(transition_count),
                "avg_transition_iou": float(avg_transition_iou),
                "min_transition_iou": float(min_transition_iou),
                "avg_transition_dist_norm": float(avg_transition_dist_norm),
                "max_transition_dist_norm": float(max_transition_dist_norm),
                "max_transition_area_growth": float(max_transition_area_growth),
                "suspicious_transition_count": int(suspicious_count),
                "event_birth": int(event_counter.get("birth", 0)),
                "event_match": int(event_counter.get("match", 0)),
                "event_merge": int(event_counter.get("merge", 0)),
                "event_inferred_merge": int(event_counter.get("inferred_merge", 0)),
                "event_inferred_merge_unknown": int(event_counter.get("inferred_merge_unknown", 0)),
                "event_large_birth": int(event_counter.get("large_birth", 0)),
                "event_merge_candidate": int(merge_candidate_count),
                "event_match_rejected_growth": int(match_rejected_growth_count),
                "event_death": int(event_counter.get("death", 0)),
                "merge_like_event_count": int(merge_like_event_count),
                "anomaly_score": float(anomaly_score),
            }
        )

    suspicious_transitions.sort(
        key=lambda x: (
            -x["dist_norm"],
            x["iou"],
            -x["area_growth"],
            x["track_id"],
            x["frame_to"],
        )
    )
    per_track_insights.sort(
        key=lambda row: (
            -float(row["anomaly_score"]),
            -int(row["suspicious_transition_count"]),
            -float(row["max_transition_dist_norm"]),
            row["track_id"],
        )
    )

    inferred_parent_counts = []
    inferred_unknown_parents = []
    inferred_single_parent_count = 0
    inferred_multi_parent_count = 0
    for e in all_inferred_merges:
        detected_parents = e.get("detected_parents", e.get("parents", []))
        if not isinstance(detected_parents, list):
            detected_parents = []
        parent_count = len(detected_parents)
        inferred_parent_counts.append(parent_count)
        if parent_count <= 1:
            inferred_single_parent_count += 1
        else:
            inferred_multi_parent_count += 1
        inferred_unknown_parents.append(int(e.get("unknown_parents", 0)))

    merge_parent_counts = []
    for e in all_merge_events:
        parents = e.get("parents", [])
        if not isinstance(parents, list):
            parents = []
        merge_parent_counts.append(len(parents))

    suspicious_by_event = Counter()
    for row in suspicious_transitions:
        if row["events_in_frame_to"]:
            for t in row["events_in_frame_to"]:
                suspicious_by_event[t] += 1
        else:
            suspicious_by_event["no_event"] += 1

    frame_alignment = {
        "detections_frames": len(detection_frame_ids),
        "tracking_frames": len(tracking_frame_ids),
        "same_count": len(detection_frame_ids) == len(tracking_frame_ids),
        "same_ids": detection_frame_ids == tracking_frame_ids,
    }
    if not frame_alignment["same_ids"]:
        only_in_detections = sorted(set(detection_frame_ids) - set(tracking_frame_ids))
        only_in_tracking = sorted(set(tracking_frame_ids) - set(detection_frame_ids))
        frame_alignment["only_in_detections"] = only_in_detections
        frame_alignment["only_in_tracking"] = only_in_tracking

    result = {
        "frame_alignment": frame_alignment,
        "detections": {
            "per_frame_count": numeric_summary([float(x) for x in detection_counts]),
            "mode_count": mode_detection_count,
            "mode_count_frames": mode_detection_frames,
            "uniform_count_all_frames": uniform_detection_count_all_frames,
        },
        "tracks": {
            "unique_track_ids": len(tracks_by_id),
            "active_tracks_per_frame": numeric_summary([float(x) for x in active_tracks_per_frame]),
            "lost_tracks_per_frame": numeric_summary([float(x) for x in lost_tracks_per_frame]),
            "track_lifetime_active_frames": numeric_summary([float(x) for x in track_lifetime_active]),
            "track_lifetime_total_frames": numeric_summary([float(x) for x in track_lifetime_total]),
            "max_generation_per_track": numeric_summary([float(x) for x in max_generation_per_track]),
        },
        "events": {
            "counts": dict(event_counts),
            "events_per_frame": numeric_summary([float(x) for x in events_per_frame]),
        },
        "merges": {
            "merge_events": len(all_merge_events),
            "inferred_merge_events": len(all_inferred_merges),
            "merge_parent_counts": numeric_summary([float(x) for x in merge_parent_counts]),
            "inferred_detected_parent_counts": numeric_summary([float(x) for x in inferred_parent_counts]),
            "inferred_unknown_parent_counts": numeric_summary([float(x) for x in inferred_unknown_parents]),
            "inferred_single_parent_count": inferred_single_parent_count,
            "inferred_multi_parent_count": inferred_multi_parent_count,
        },
        "suspicious_id_carryover": {
            "criteria": {
                "jump_dist_norm_thresh": jump_dist_norm_thresh,
                "jump_iou_thresh": jump_iou_thresh,
                "jump_area_growth_thresh": jump_area_growth_thresh,
                "lost_reactivation_dist_norm_thresh": lost_reactivation_dist_norm_thresh,
                "lost_reactivation_area_growth_thresh": lost_reactivation_area_growth_thresh,
            },
            "count": len(suspicious_transitions),
            "by_event_type_in_frame_to": dict(suspicious_by_event),
            "items": suspicious_transitions,
        },
        "per_track_insights": per_track_insights,
    }
    return result


def write_suspicious_csv(path: str, items: List[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    fieldnames = [
        "frame_from",
        "frame_to",
        "track_id",
        "transition_type",
        "iou",
        "dist_norm",
        "area_prev",
        "area_curr",
        "area_growth",
        "gen_from",
        "gen_to",
        "events_in_frame_to",
        "reasons",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in items:
            out = dict(row)
            out["events_in_frame_to"] = ",".join(out.get("events_in_frame_to", []))
            out["reasons"] = ",".join(out.get("reasons", []))
            writer.writerow(out)


def write_table_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_report(summary: Dict[str, Any], top_n: int) -> None:
    frame_alignment = summary["frame_alignment"]
    events = summary["events"]["counts"]
    suspicious = summary["suspicious_id_carryover"]["items"]
    by_event = summary["suspicious_id_carryover"]["by_event_type_in_frame_to"]
    merges = summary["merges"]
    per_track = summary.get("per_track_insights", [])

    print("=== Tracking Insights ===")
    print(
        f"Frames: detections={frame_alignment['detections_frames']} "
        f"tracking={frame_alignment['tracking_frames']} "
        f"same_ids={frame_alignment['same_ids']}"
    )
    det_info = summary.get("detections", {})
    print(
        "Detection counts: "
        f"mode={det_info.get('mode_count', 0)} "
        f"mode_frames={det_info.get('mode_count_frames', 0)} "
        f"uniform_all_frames={det_info.get('uniform_count_all_frames', False)}"
    )
    print(f"Unique track IDs: {summary['tracks']['unique_track_ids']}")
    print(f"Event counts: {events}")
    print(
        "Merge events: "
        f"explicit={merges['merge_events']} inferred={merges['inferred_merge_events']} "
        f"(single_parent={merges['inferred_single_parent_count']}, "
        f"multi_parent={merges['inferred_multi_parent_count']})"
    )
    print(
        f"Suspicious ID carry-over candidates: {summary['suspicious_id_carryover']['count']} "
        f"by_event={by_event}"
    )

    if suspicious:
        print(f"\nTop {min(top_n, len(suspicious))} suspicious transitions:")
        for row in suspicious[:top_n]:
            print(
                f"  F{row['frame_from']}->F{row['frame_to']} "
                f"track={row['track_id']} "
                f"type={row.get('transition_type', 'active_to_active')} "
                f"dist_norm={row['dist_norm']:.3f} iou={row['iou']:.3f} "
                f"growth={row['area_growth']:.3f} "
                f"events={row['events_in_frame_to']} reasons={row['reasons']}"
            )

    if per_track:
        print(f"\nTop {min(top_n, len(per_track))} tracks by anomaly score:")
        for row in per_track[:top_n]:
            print(
                f"  track={row['track_id']} score={row['anomaly_score']:.2f} "
                f"suspicious={row['suspicious_transition_count']} "
                f"max_dist={row['max_transition_dist_norm']:.3f} "
                f"max_gen={row['max_gen']} "
                f"merge_like_events={row['merge_like_event_count']}"
            )


def main() -> None:
    args = parse_args()

    detection_frames = load_json_list(args.detections)
    tracking_frames = load_json_list(args.tracking_log)

    summary = analyze(
        detection_frames=detection_frames,
        tracking_frames=tracking_frames,
        jump_dist_norm_thresh=args.jump_dist_norm_thresh,
        jump_iou_thresh=args.jump_iou_thresh,
        jump_area_growth_thresh=args.jump_area_growth_thresh,
        lost_reactivation_dist_norm_thresh=args.lost_reactivation_dist_norm_thresh,
        lost_reactivation_area_growth_thresh=args.lost_reactivation_area_growth_thresh,
    )

    ensure_parent_dir(args.summary_out)
    with open(args.summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    write_suspicious_csv(args.suspicious_out, summary["suspicious_id_carryover"]["items"])
    write_table_csv(args.per_track_out, summary.get("per_track_insights", []))
    print_report(summary, args.top_n)
    print(f"\nWrote summary JSON: {args.summary_out}")
    print(f"Wrote suspicious CSV: {args.suspicious_out}")
    print(f"Wrote per-track CSV: {args.per_track_out}")


if __name__ == "__main__":
    main()
