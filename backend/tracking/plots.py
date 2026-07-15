#!/usr/bin/env python3
"""Generate timeline tables and plots from tracking insights JSON outputs.
Verbatim from plot_tracking_insights.py; only this docstring differs. The
argparse defaults still assume the old sibling-directory cwd, so pass
--tracking-log / --insights-summary / --out-dir explicitly.

Usage:
  python -m tracking plots --tracking-log ... --insights-summary ... --out-dir ...
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

# Headless-safe backend for servers/CLI usage.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create per-track timelines and diagnostic plots.")
    parser.add_argument(
        "--tracking-log",
        default="../Nasa_Backend/output/tracking_log.json",
        help="Path to tracking_log.json",
    )
    parser.add_argument(
        "--insights-summary",
        default="../Nasa_Backend/output/insights_summary.json",
        help="Path to insights_summary.json from analyze_tracking_json.py",
    )
    parser.add_argument(
        "--out-dir",
        default="../Nasa_Backend/output/insights_visual",
        help="Directory for CSV outputs and plots",
    )
    parser.add_argument(
        "--top-suspicious-tracks",
        type=int,
        default=8,
        help="How many suspicious track IDs to include in focus plots.",
    )
    parser.add_argument(
        "--focus-track",
        action="append",
        type=int,
        default=[],
        help="Optional specific track ID(s) to plot. Can be provided multiple times.",
    )
    parser.add_argument(
        "--max-tracks-in-heatmap",
        type=int,
        default=0,
        help="Limit track rows in status heatmap (0 means include all tracks).",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_track_timelines(tracking_frames: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    timelines: List[Dict[str, Any]] = []
    events_by_frame_track: Dict[Tuple[int, int], List[str]] = defaultdict(list)

    for frame_record in tracking_frames:
        frame_id = int(frame_record.get("frame", 0))
        for event in frame_record.get("events", []):
            tid = event.get("track_id")
            if isinstance(tid, int):
                events_by_frame_track[(frame_id, tid)].append(str(event.get("type", "")))

    for frame_record in tracking_frames:
        frame_id = int(frame_record.get("frame", 0))
        for track in frame_record.get("tracks", []):
            tid = track.get("track_id")
            if not isinstance(tid, int):
                continue
            bbox = track.get("bbox", [0, 0, 0, 0])
            if not isinstance(bbox, list) or len(bbox) != 4:
                bbox = [0, 0, 0, 0]
            cx = (float(bbox[0]) + float(bbox[2])) / 2.0
            cy = (float(bbox[1]) + float(bbox[3])) / 2.0
            row = {
                "frame": frame_id,
                "track_id": tid,
                "status": str(track.get("status", "")),
                "gen": int(track.get("gen", 1)),
                "missed": int(track.get("missed", 0)),
                "x1": float(bbox[0]),
                "y1": float(bbox[1]),
                "x2": float(bbox[2]),
                "y2": float(bbox[3]),
                "center_x": float(cx),
                "center_y": float(cy),
                "area": float(track.get("area", 0.0)),
                "event_types": ";".join(events_by_frame_track.get((frame_id, tid), [])),
            }
            timelines.append(row)

    per_track: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in timelines:
        per_track[int(row["track_id"])].append(row)

    track_summary: List[Dict[str, Any]] = []
    for track_id, rows in per_track.items():
        rows.sort(key=lambda r: int(r["frame"]))
        active_frames = sum(1 for r in rows if r["status"] == "active")
        lost_frames = sum(1 for r in rows if r["status"] == "lost")
        event_counter: Counter = Counter()
        for r in rows:
            if r["event_types"]:
                for et in str(r["event_types"]).split(";"):
                    if et:
                        event_counter[et] += 1
        track_summary.append(
            {
                "track_id": track_id,
                "first_frame": int(rows[0]["frame"]),
                "last_frame": int(rows[-1]["frame"]),
                "total_frames_present": len(rows),
                "active_frames": active_frames,
                "lost_frames": lost_frames,
                "max_gen": max(int(r["gen"]) for r in rows),
                "avg_area": sum(float(r["area"]) for r in rows) / max(1, len(rows)),
                "max_area": max(float(r["area"]) for r in rows),
                "event_counts": ";".join(f"{k}:{v}" for k, v in sorted(event_counter.items())),
            }
        )

    track_summary.sort(key=lambda r: (r["first_frame"], r["track_id"]))
    timelines.sort(key=lambda r: (r["frame"], r["track_id"]))
    return timelines, track_summary


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def choose_focus_tracks(
    timeline_rows: List[Dict[str, Any]],
    suspicious_items: List[Dict[str, Any]],
    explicit_focus: List[int],
    top_n: int,
) -> List[int]:
    if explicit_focus:
        return sorted(set(explicit_focus))

    if suspicious_items:
        suspicious_by_tid: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for item in suspicious_items:
            tid = item.get("track_id")
            if isinstance(tid, int):
                suspicious_by_tid[tid].append(item)

        ranked = []
        for tid, items in suspicious_by_tid.items():
            max_dist = max(float(i.get("dist_norm", 0.0)) for i in items)
            max_growth = max(float(i.get("area_growth", 0.0)) for i in items)
            ranked.append((len(items), max_dist, max_growth, tid))
        ranked.sort(key=lambda x: (-x[0], -x[1], -x[2], x[3]))
        return [r[3] for r in ranked[:top_n]]

    by_track = defaultdict(list)
    for r in timeline_rows:
        by_track[int(r["track_id"])].append(r)
    fallback = []
    for tid, rows in by_track.items():
        max_gen = max(int(r["gen"]) for r in rows)
        span = int(rows[-1]["frame"]) - int(rows[0]["frame"]) + 1
        fallback.append((max_gen, span, tid))
    fallback.sort(key=lambda x: (-x[0], -x[1], x[2]))
    return [r[2] for r in fallback[:top_n]]


def plot_per_frame_overview(tracking_frames: List[Dict[str, Any]], out_png: str) -> None:
    frame_ids = []
    active_counts = []
    lost_counts = []
    total_event_counts = []
    merge_like_counts = []

    for fr in tracking_frames:
        frame_id = int(fr.get("frame", 0))
        frame_ids.append(frame_id)
        tracks = fr.get("tracks", [])
        events = fr.get("events", [])
        active_counts.append(sum(1 for t in tracks if t.get("status") == "active"))
        lost_counts.append(sum(1 for t in tracks if t.get("status") == "lost"))
        total_event_counts.append(len(events))
        merge_like_counts.append(
            sum(
                1
                for e in events
                if str(e.get("type", "")) in (
                    "merge",
                    "inferred_merge",
                    "inferred_merge_unknown",
                    "large_birth",
                    "merge_candidate",
                )
            )
        )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(frame_ids, active_counts, marker="o", label="active tracks")
    axes[0].plot(frame_ids, lost_counts, marker="x", label="lost tracks")
    axes[0].set_ylabel("track count")
    axes[0].set_title("Per-frame Track Status")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].bar(frame_ids, total_event_counts, alpha=0.6, label="all events")
    axes[1].plot(frame_ids, merge_like_counts, marker="o", color="tab:red", label="merge-like events")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("event count")
    axes[1].set_title("Per-frame Events")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_focus_tracks(timeline_rows: List[Dict[str, Any]], focus_tracks: List[int], out_png: str) -> None:
    rows_by_track = defaultdict(list)
    for row in timeline_rows:
        tid = int(row["track_id"])
        if tid in focus_tracks:
            rows_by_track[tid].append(row)

    if not rows_by_track:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "No focus tracks found", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    cmap = plt.get_cmap("tab10")

    for idx, track_id in enumerate(focus_tracks):
        rows = sorted(rows_by_track.get(track_id, []), key=lambda r: int(r["frame"]))
        if not rows:
            continue
        color = cmap(idx % 10)
        frames = [int(r["frame"]) for r in rows]
        areas = [float(r["area"]) for r in rows]
        gens = [int(r["gen"]) for r in rows]
        statuses = [str(r["status"]) for r in rows]

        axes[0].plot(frames, areas, marker="o", color=color, label=f"track {track_id}")
        active_idx = [i for i, s in enumerate(statuses) if s == "active"]
        lost_idx = [i for i, s in enumerate(statuses) if s == "lost"]
        if active_idx:
            axes[0].scatter([frames[i] for i in active_idx], [areas[i] for i in active_idx], color=color, marker="o")
        if lost_idx:
            axes[0].scatter([frames[i] for i in lost_idx], [areas[i] for i in lost_idx], color=color, marker="x")

        axes[1].step(frames, gens, where="mid", color=color, label=f"track {track_id}")

    axes[0].set_title("Focus Tracks: Area vs Frame (x marker = lost)")
    axes[0].set_ylabel("area")
    axes[0].grid(alpha=0.3)
    axes[0].legend(ncol=2, fontsize=9)

    axes[1].set_title("Focus Tracks: Generation vs Frame")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("generation")
    axes[1].grid(alpha=0.3)
    axes[1].legend(ncol=2, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_suspicious_scatter(suspicious_items: List[Dict[str, Any]], out_png: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    if not suspicious_items:
        ax.text(0.5, 0.5, "No suspicious transitions", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        return

    def event_bucket(item: Dict[str, Any]) -> str:
        evs = item.get("events_in_frame_to", [])
        if isinstance(evs, list) and evs:
            return str(evs[0])
        return "no_event"

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in suspicious_items:
        buckets[event_bucket(item)].append(item)

    for label, items in sorted(buckets.items()):
        x = [float(it.get("dist_norm", 0.0)) for it in items]
        y = [float(it.get("area_growth", 0.0)) for it in items]
        ax.scatter(x, y, alpha=0.8, label=f"{label} ({len(items)})")

    # Annotate strongest candidates for quick review.
    sorted_items = sorted(
        suspicious_items,
        key=lambda x: (-float(x.get("dist_norm", 0.0)), -float(x.get("area_growth", 0.0))),
    )
    for it in sorted_items[:10]:
        x = float(it.get("dist_norm", 0.0))
        y = float(it.get("area_growth", 0.0))
        tid = it.get("track_id", "?")
        fto = it.get("frame_to", "?")
        ax.annotate(f"T{tid}@F{fto}", (x, y), fontsize=8, alpha=0.8)

    ax.set_title("Suspicious Transitions: Distance vs Area Growth")
    ax.set_xlabel("dist_norm")
    ax.set_ylabel("area_growth")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_all_track_insights(
    insights_summary: Dict[str, Any], track_summary_rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    per_track = insights_summary.get("per_track_insights", [])
    if isinstance(per_track, list) and per_track and isinstance(per_track[0], dict):
        rows = [dict(r) for r in per_track if isinstance(r, dict)]
        return rows

    # Fallback for legacy summary files (without per_track_insights).
    fallback_rows: List[Dict[str, Any]] = []
    for row in track_summary_rows:
        fallback_rows.append(
            {
                "track_id": as_int(row.get("track_id")),
                "first_frame": as_int(row.get("first_frame")),
                "last_frame": as_int(row.get("last_frame")),
                "lifespan_frames": as_int(row.get("total_frames_present")),
                "frames_present": as_int(row.get("total_frames_present")),
                "active_frames": as_int(row.get("active_frames")),
                "lost_frames": as_int(row.get("lost_frames")),
                "max_gen": as_int(row.get("max_gen"), 1),
                "mean_area": as_float(row.get("avg_area")),
                "max_area": as_float(row.get("max_area")),
                "suspicious_transition_count": 0,
                "max_transition_dist_norm": 0.0,
                "anomaly_score": 0.0,
            }
        )
    fallback_rows.sort(key=lambda r: (r["track_id"]))
    return fallback_rows


def plot_all_track_distributions(per_track_rows: List[Dict[str, Any]], out_png: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    if not per_track_rows:
        for ax in axes.ravel():
            ax.axis("off")
        axes[0, 0].text(0.5, 0.5, "No per-track rows", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        return

    lifespan = [as_float(r.get("lifespan_frames", r.get("frames_present", 0))) for r in per_track_rows]
    max_gen = [as_float(r.get("max_gen", 1)) for r in per_track_rows]
    suspicious = [as_float(r.get("suspicious_transition_count", 0)) for r in per_track_rows]
    max_dist = [as_float(r.get("max_transition_dist_norm", 0.0)) for r in per_track_rows]

    axes[0, 0].hist(lifespan, bins=min(30, max(5, len(set(lifespan)))), color="tab:blue", alpha=0.8)
    axes[0, 0].set_title("Track Lifespan (frames)")
    axes[0, 0].set_xlabel("lifespan")
    axes[0, 0].set_ylabel("count")
    axes[0, 0].grid(alpha=0.2)

    axes[0, 1].hist(max_gen, bins=min(20, max(5, len(set(max_gen)))), color="tab:orange", alpha=0.8)
    axes[0, 1].set_title("Max Generation per Track")
    axes[0, 1].set_xlabel("max_gen")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].grid(alpha=0.2)

    axes[1, 0].hist(suspicious, bins=min(20, max(5, len(set(suspicious)))), color="tab:red", alpha=0.8)
    axes[1, 0].set_title("Suspicious Transition Count per Track")
    axes[1, 0].set_xlabel("suspicious transitions")
    axes[1, 0].set_ylabel("count")
    axes[1, 0].grid(alpha=0.2)

    axes[1, 1].hist(max_dist, bins=min(30, max(5, len(set(max_dist)))), color="tab:green", alpha=0.8)
    axes[1, 1].set_title("Max Transition Distance Norm per Track")
    axes[1, 1].set_xlabel("max_transition_dist_norm")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_all_track_scatter(per_track_rows: List[Dict[str, Any]], out_png: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    if not per_track_rows:
        ax.text(0.5, 0.5, "No per-track rows", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        return

    x = np.array([as_float(r.get("lifespan_frames", r.get("frames_present", 0))) for r in per_track_rows], dtype=float)
    y = np.array([as_float(r.get("max_transition_dist_norm", 0.0)) for r in per_track_rows], dtype=float)
    c = np.array([as_float(r.get("suspicious_transition_count", 0.0)) for r in per_track_rows], dtype=float)
    anomaly = np.array([as_float(r.get("anomaly_score", 0.0)) for r in per_track_rows], dtype=float)
    sizes = 25.0 + 20.0 * np.clip(anomaly, 0.0, np.percentile(anomaly, 95) if len(anomaly) else 1.0)

    sc = ax.scatter(x, y, c=c, s=sizes, cmap="viridis", alpha=0.7, edgecolors="none")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("suspicious_transition_count")

    ax.set_title("All Tracks: Lifespan vs Max Transition Distance")
    ax.set_xlabel("lifespan_frames")
    ax.set_ylabel("max_transition_dist_norm")
    ax.grid(alpha=0.3)

    top_rows = sorted(per_track_rows, key=lambda r: -as_float(r.get("anomaly_score", 0.0)))[:15]
    for row in top_rows:
        tx = as_float(row.get("lifespan_frames", row.get("frames_present", 0)))
        ty = as_float(row.get("max_transition_dist_norm", 0.0))
        tid = as_int(row.get("track_id"))
        ax.annotate(f"T{tid}", (tx, ty), fontsize=8, alpha=0.9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_top_anomaly_tracks(per_track_rows: List[Dict[str, Any]], out_png: str, top_n: int = 30) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    if not per_track_rows:
        ax.text(0.5, 0.5, "No per-track rows", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        return

    rows = sorted(per_track_rows, key=lambda r: -as_float(r.get("anomaly_score", 0.0)))[:top_n]
    labels = [str(as_int(r.get("track_id"))) for r in rows]
    scores = [as_float(r.get("anomaly_score", 0.0)) for r in rows]
    suspicious_counts = [as_float(r.get("suspicious_transition_count", 0.0)) for r in rows]

    bars = ax.bar(range(len(rows)), scores, color="tab:purple", alpha=0.75)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=75, fontsize=8)
    ax.set_title(f"Top {len(rows)} Tracks by Anomaly Score")
    ax.set_xlabel("track_id")
    ax.set_ylabel("anomaly_score")
    ax.grid(axis="y", alpha=0.3)

    for bar, suspicious_count in zip(bars, suspicious_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"s{int(suspicious_count)}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_status_heatmap(
    timeline_rows: List[Dict[str, Any]], out_png: str, max_tracks: int = 0
) -> Tuple[int, int]:
    if not timeline_rows:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "No timeline rows", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        return 0, 0

    frame_ids = sorted({as_int(r.get("frame")) for r in timeline_rows})
    track_ids = sorted({as_int(r.get("track_id")) for r in timeline_rows})
    first_frame_by_track: Dict[int, int] = {tid: 10**9 for tid in track_ids}
    for row in timeline_rows:
        tid = as_int(row.get("track_id"))
        frame = as_int(row.get("frame"))
        if frame < first_frame_by_track[tid]:
            first_frame_by_track[tid] = frame
    track_ids.sort(key=lambda tid: (first_frame_by_track.get(tid, 10**9), tid))

    if max_tracks > 0 and len(track_ids) > max_tracks:
        track_ids = track_ids[:max_tracks]

    frame_to_col = {f: idx for idx, f in enumerate(frame_ids)}
    track_to_row = {t: idx for idx, t in enumerate(track_ids)}
    matrix = np.zeros((len(track_ids), len(frame_ids)), dtype=np.int8)

    for row in timeline_rows:
        tid = as_int(row.get("track_id"))
        if tid not in track_to_row:
            continue
        frame = as_int(row.get("frame"))
        if frame not in frame_to_col:
            continue
        status = str(row.get("status", ""))
        value = 1 if status == "active" else 2 if status == "lost" else 0
        matrix[track_to_row[tid], frame_to_col[frame]] = value

    fig_h = max(4, min(18, len(track_ids) * 0.05 + 2))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    cmap = ListedColormap(["#f2f2f2", "#2ca02c", "#d62728"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    _ = im

    ax.set_title("Track Status Heatmap (gray=absent, green=active, red=lost)")
    ax.set_xlabel("frame")
    ax.set_ylabel("track (sorted by first frame)")
    if len(frame_ids) <= 30:
        ax.set_xticks(range(len(frame_ids)))
        ax.set_xticklabels([str(f) for f in frame_ids], rotation=90, fontsize=8)
    else:
        step = max(1, len(frame_ids) // 20)
        idxs = list(range(0, len(frame_ids), step))
        ax.set_xticks(idxs)
        ax.set_xticklabels([str(frame_ids[i]) for i in idxs], rotation=90, fontsize=8)

    if len(track_ids) <= 40:
        ax.set_yticks(range(len(track_ids)))
        ax.set_yticklabels([str(t) for t in track_ids], fontsize=6)
    else:
        step = max(1, len(track_ids) // 30)
        idxs = list(range(0, len(track_ids), step))
        ax.set_yticks(idxs)
        ax.set_yticklabels([str(track_ids[i]) for i in idxs], fontsize=6)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return len(track_ids), len(frame_ids)


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    tracking_frames = load_json(args.tracking_log)
    insights_summary = load_json(args.insights_summary)
    if not isinstance(tracking_frames, list):
        raise ValueError(f"Expected list in tracking log: {args.tracking_log}")
    if not isinstance(insights_summary, dict):
        raise ValueError(f"Expected object in insights summary: {args.insights_summary}")

    timeline_rows, track_summary_rows = build_track_timelines(tracking_frames)
    suspicious_items = insights_summary.get("suspicious_id_carryover", {}).get("items", [])
    if not isinstance(suspicious_items, list):
        suspicious_items = []
    all_track_insights_rows = get_all_track_insights(insights_summary, track_summary_rows)
    focus_tracks = choose_focus_tracks(
        timeline_rows=timeline_rows,
        suspicious_items=suspicious_items,
        explicit_focus=args.focus_track,
        top_n=args.top_suspicious_tracks,
    )

    timeline_csv = os.path.join(args.out_dir, "track_timelines.csv")
    summary_csv = os.path.join(args.out_dir, "track_summary.csv")
    all_track_insights_csv = os.path.join(args.out_dir, "all_track_insights.csv")
    focus_csv = os.path.join(args.out_dir, "focus_tracks.csv")
    plot_overview_png = os.path.join(args.out_dir, "per_frame_overview.png")
    plot_focus_png = os.path.join(args.out_dir, "focus_tracks_timeline.png")
    plot_suspicious_png = os.path.join(args.out_dir, "suspicious_scatter.png")
    plot_distributions_png = os.path.join(args.out_dir, "all_tracks_distributions.png")
    plot_scatter_png = os.path.join(args.out_dir, "all_tracks_scatter.png")
    plot_top_anomaly_png = os.path.join(args.out_dir, "top_anomaly_tracks.png")
    plot_heatmap_png = os.path.join(args.out_dir, "all_tracks_status_heatmap.png")

    write_csv(timeline_csv, timeline_rows)
    write_csv(summary_csv, track_summary_rows)
    write_csv(all_track_insights_csv, all_track_insights_rows)
    write_csv(
        focus_csv,
        [{"track_id": tid} for tid in focus_tracks],
    )

    plot_per_frame_overview(tracking_frames, plot_overview_png)
    plot_focus_tracks(timeline_rows, focus_tracks, plot_focus_png)
    plot_suspicious_scatter(suspicious_items, plot_suspicious_png)
    plot_all_track_distributions(all_track_insights_rows, plot_distributions_png)
    plot_all_track_scatter(all_track_insights_rows, plot_scatter_png)
    plot_top_anomaly_tracks(all_track_insights_rows, plot_top_anomaly_png)
    heatmap_tracks, heatmap_frames = plot_status_heatmap(
        timeline_rows, plot_heatmap_png, max_tracks=args.max_tracks_in_heatmap
    )

    print("=== Tracking Visual Insights ===")
    print(f"Timeline CSV: {timeline_csv}")
    print(f"Track summary CSV: {summary_csv}")
    print(f"All-track insights CSV: {all_track_insights_csv}")
    print(f"Focus tracks CSV: {focus_csv}")
    print(f"Per-frame overview plot: {plot_overview_png}")
    print(f"Focus track timeline plot: {plot_focus_png}")
    print(f"Suspicious transition scatter: {plot_suspicious_png}")
    print(f"All-track distributions plot: {plot_distributions_png}")
    print(f"All-track scatter plot: {plot_scatter_png}")
    print(f"Top anomaly tracks plot: {plot_top_anomaly_png}")
    print(f"All-track status heatmap: {plot_heatmap_png}")
    print(f"Focus tracks: {focus_tracks}")
    print(f"All-track rows used in heatmap: tracks={heatmap_tracks}, frames={heatmap_frames}")


if __name__ == "__main__":
    main()
