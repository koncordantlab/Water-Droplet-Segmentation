"""The tracking main loop: re-hydrate per-frame detections from
detections.json, associate them to Track objects via the matching suite
(mutual-best one-to-one matching plus the direct / inferred / match-growth
merge detectors), and write the annotated mp4, tracking_log.json, and the
flattened CSV. Verbatim from the monolith."""
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from tracking.config import (
    DRAW_TRACK_OVERLAY,
    ENABLE_INFERRED_MERGE,
    ENABLE_INFERRED_UNKNOWN_MERGE,
    ENABLE_MATCH_GROWTH_MERGE_RESCUE,
    ENABLE_MERGE_DETECTION,
    ENABLE_PERSISTENT_GROWTH_MERGE_PROMOTION,
    FILTER_NON_CIRCULAR_SEGMENTS,
    INFERRED_MERGE_MIN_PARENTS,
    INFERRED_MERGE_WARMUP_FRAMES,
    INFERRED_UNKNOWN_MATCH_AREA_GROWTH,
    KEEP_NON_CIRCULAR_FOR_ASSOCIATION,
    MAX_MATCH_AREA_RATIO,
    MAX_MISSED_FRAMES,
    MIN_SEGMENT_CONFIDENCE,
    PERSISTENT_GROWTH_MIN_EVENTS,
    PERSISTENT_GROWTH_MIN_STREAK,
    PERSISTENT_GROWTH_PROMOTION_MIN_AREA_RATIO,
    PERSISTENT_GROWTH_PROMOTION_MIN_SCORE,
    PERSISTENT_GROWTH_SUPPRESS_HIGH_IOU,
    PERSISTENT_GROWTH_SUPPRESS_LOW_MOTION,
    PROMOTE_LARGE_BIRTH_TO_UNKNOWN_MERGE,
    PROMOTE_MATCH_GROWTH_TO_UNKNOWN_MERGE,
    REJECT_MATCH_ON_LARGE_AREA_GROWTH,
    REQUIRE_INSTANCE_SEGMENTATION,
    REQUIRE_MUTUAL_BEST_MATCH,
    TRACKING_PREFILTER_DEVICE,
)
from tracking.geometry import detection_area, is_relevant_circular_segment
from tracking.io import (
    JsonArrayWriter,
    ensure_parent_dir,
    export_tracking_log_csv,
    open_video,
    serialize_center,
    serialize_segment,
)
from tracking.matching import (
    area_growth_ratio,
    area_ratio_values,
    build_match_candidates,
    estimate_adaptive_growth_threshold,
    estimate_unknown_parent_count,
    is_local_growth_continuation,
    is_unknown_merge_from_birth,
    is_unknown_merge_from_match,
    is_valid_match,
    reference_object_area,
    select_inferred_merge_parents,
    select_match_growth_merge_support_parents,
    select_merge_parents,
)
from tracking.render import apply_tracking_overlay, create_video_writer
from tracking.tracks import (
    Track,
    decay_merge_candidate_state,
    register_merge_candidate,
    reset_merge_candidate_state,
)

def detections_from_frame_record(frame_record):
    detections = []
    detection_meta = []
    detection_segments: List[Optional[np.ndarray]] = []
    detection_areas: List[float] = []
    dropped_low_conf = 0
    dropped_non_circular = 0
    kept_non_circular = 0
    total = 0
    for det_idx, det in enumerate(frame_record.get("detections", [])):
        total += 1
        conf = float(det.get("conf", 0.0))
        cls = int(det.get("cls", 0))
        det_id = int(det.get("det_id", det_idx))
        if conf < MIN_SEGMENT_CONFIDENCE:
            dropped_low_conf += 1
            continue
        segment_raw = det.get("segment")
        segment = None
        if isinstance(segment_raw, list):
            points = np.asarray(segment_raw, dtype=np.float32)
            if points.ndim == 2 and points.shape[0] >= 3 and points.shape[1] == 2:
                segment = points
        if REQUIRE_INSTANCE_SEGMENTATION and segment is None:
            frame_id = frame_record.get("frame", "?")
            raise RuntimeError(
                f"Detections JSON is missing valid segment data (frame={frame_id}, det_id={det_id}). "
                "Regenerate detections with segmentation enabled."
            )
        det_area = float(det.get("area", 0.0))
        if det_area <= 0:
            det_area = detection_area(segment)
        if REQUIRE_INSTANCE_SEGMENTATION and det_area <= 0:
            frame_id = frame_record.get("frame", "?")
            raise RuntimeError(
                f"Detections JSON has non-positive segmented area (frame={frame_id}, det_id={det_id})."
            )
        if segment is not None and not is_relevant_circular_segment(segment):
            if KEEP_NON_CIRCULAR_FOR_ASSOCIATION:
                kept_non_circular += 1
            else:
                dropped_non_circular += 1
                continue
        detections.append(segment)
        detection_meta.append((conf, cls, det_id))
        detection_segments.append(segment)
        detection_areas.append(det_area)
    if dropped_low_conf > 0 or dropped_non_circular > 0 or kept_non_circular > 0:
        frame_id = frame_record.get("frame", "?")
        print(
            f"Warning: filtered detections in JSON "
            f"(frame={frame_id}, low_conf={dropped_low_conf}, conf_threshold={MIN_SEGMENT_CONFIDENCE}, "
            f"non_circular_dropped={dropped_non_circular}, non_circular_kept={kept_non_circular}, "
            f"kept={len(detections)}, total={total})"
        )
    return detections, detection_meta, detection_segments, detection_areas

def track_from_detections_json(video_path, detections_path, output_path, log_path):
    cap, fps, width, height = open_video(video_path)
    ensure_parent_dir(output_path)
    ensure_parent_dir(log_path)
    writer = create_video_writer(output_path, fps, width, height)
    print(f"Tracking candidate prefilter device: {TRACKING_PREFILTER_DEVICE}")

    with open(detections_path, "r", encoding="utf-8") as file:
        frames_data = json.load(file)
    if not isinstance(frames_data, list):
        raise RuntimeError("Detections JSON must be a list of frame records.")

    tracked_objects: Dict[int, Track] = {}
    next_id = 1

    log_writer = JsonArrayWriter(log_path)
    try:
        for frame_index, frame_record in enumerate(frames_data):
            ret, frame = cap.read()
            if not ret:
                break

            frame_id = int(frame_record.get("frame", frame_index + 1))
            detections, detection_meta, detection_segments, detection_areas = detections_from_frame_record(frame_record)
            frame_median_area = float(np.median(detection_areas)) if detection_areas else 0.0
            reference_area = reference_object_area(tracked_objects, frame_median_area)
            frame_events = []
            merged_children = set()

            if frame_index == 0:
                for det_idx, _ in enumerate(detections):
                    segment = detection_segments[det_idx] if det_idx < len(detection_segments) else None
                    det_area = detection_areas[det_idx] if det_idx < len(detection_areas) else detection_area(segment)
                    tracked_objects[next_id] = Track(gen=1, missed=0, area=det_area, segment=segment)
                    det_id = detection_meta[det_idx][2] if det_idx < len(detection_meta) else det_idx
                    frame_events.append({
                        "type": "birth",
                        "track_id": next_id,
                        "gen": 1,
                        "det_id": det_id,
                    })
                    next_id += 1
            else:
                active_tracks = {tid: t for tid, t in tracked_objects.items() if t.missed <= MAX_MISSED_FRAMES}
                candidates = build_match_candidates(active_tracks, detection_segments)

                assigned_tracks = set()
                assigned_dets = set()
                merged_parents = set()

                # Merge detection
                if ENABLE_MERGE_DETECTION:
                    for det_idx, cand_list in candidates.items():
                        if det_idx in assigned_dets:
                            continue

                        det_segment = detection_segments[det_idx] if det_idx < len(detection_segments) else None
                        det_area = detection_areas[det_idx] if det_idx < len(detection_areas) else detection_area(det_segment)
                        parent_ids = select_merge_parents(det_segment, det_area, cand_list, active_tracks, assigned_tracks)
                        if len(parent_ids) < 2:
                            continue

                        child_gen = max(active_tracks[pid].gen for pid in parent_ids) + 1
                        child_id = next_id
                        next_id += 1
                        tracked_objects[child_id] = Track(
                            gen=child_gen,
                            missed=0,
                            area=det_area,
                            segment=det_segment,
                        )
                        det_id = detection_meta[det_idx][2] if det_idx < len(detection_meta) else det_idx
                        frame_events.append({
                            "type": "merge",
                            "track_id": child_id,
                            "gen": child_gen,
                            "parents": parent_ids,
                            "det_id": det_id,
                        })
                        merged_children.add(child_id)
                        assigned_tracks.add(child_id)

                        for pid in parent_ids:
                            merged_parents.add(pid)
                            assigned_tracks.add(pid)

                        assigned_dets.add(det_idx)

                # One-to-one matching for remaining detections
                scored_pairs: List[Tuple[float, int, int, float, float]] = []
                for det_idx, cand_list in candidates.items():
                    if det_idx in assigned_dets:
                        continue
                    for tid, iou, dist_norm in cand_list:
                        if tid in assigned_tracks or tid in merged_parents:
                            continue
                        det_area = detection_areas[det_idx] if det_idx < len(detection_areas) else detection_area(detection_segments[det_idx])
                        if not is_valid_match(tracked_objects[tid], det_area, iou, dist_norm):
                            continue
                        score = iou - 0.1 * dist_norm
                        scored_pairs.append((score, tid, det_idx, iou, dist_norm))

                scored_pairs.sort(reverse=True)
                best_track_for_det: Dict[int, Tuple[int, float]] = {}
                best_det_for_track: Dict[int, Tuple[int, float]] = {}
                if REQUIRE_MUTUAL_BEST_MATCH:
                    for score, tid, det_idx, _, _ in scored_pairs:
                        prev_det = best_det_for_track.get(tid)
                        if prev_det is None or score > prev_det[1]:
                            best_det_for_track[tid] = (det_idx, score)
                        prev_track = best_track_for_det.get(det_idx)
                        if prev_track is None or score > prev_track[1]:
                            best_track_for_det[det_idx] = (tid, score)

                frame_growth_merge_threshold = estimate_adaptive_growth_threshold(
                    scored_pairs,
                    tracked_objects,
                    detection_areas,
                )
                frame_growth_reject_threshold = max(
                    INFERRED_UNKNOWN_MATCH_AREA_GROWTH,
                    frame_growth_merge_threshold + 0.35,
                )

                for _, tid, det_idx, iou, dist_norm in scored_pairs:
                    if tid in assigned_tracks or det_idx in assigned_dets:
                        continue
                    if REQUIRE_MUTUAL_BEST_MATCH:
                        best_det = best_det_for_track.get(tid)
                        best_track = best_track_for_det.get(det_idx)
                        if best_det is None or best_track is None:
                            continue
                        if best_det[0] != det_idx or best_track[0] != tid:
                            continue
                    det_area = detection_areas[det_idx] if det_idx < len(detection_areas) else detection_area(detection_segments[det_idx])
                    if not is_valid_match(tracked_objects[tid], det_area, iou, dist_norm):
                        continue
                    track = tracked_objects[tid]
                    decay_merge_candidate_state(track, frame_id)
                    det_id = detection_meta[det_idx][2] if det_idx < len(detection_meta) else det_idx
                    new_segment = detection_segments[det_idx] if det_idx < len(detection_segments) else None
                    prev_area = track.area
                    growth_ratio = area_growth_ratio(prev_area, det_area)
                    is_growth_merge = False
                    is_growth_continuation_match = False
                    area_ratio = 0.0
                    if ENABLE_INFERRED_UNKNOWN_MERGE:
                        is_growth_continuation_match = (
                            det_area > prev_area
                            and area_ratio_values(prev_area, det_area) > MAX_MATCH_AREA_RATIO
                            and is_local_growth_continuation(iou, dist_norm)
                        )
                        is_growth_merge, area_ratio = is_unknown_merge_from_match(
                            prev_area,
                            det_area,
                            frame_growth_reject_threshold,
                        )
                        if (
                            is_growth_merge
                            and REJECT_MATCH_ON_LARGE_AREA_GROWTH
                            and not PROMOTE_MATCH_GROWTH_TO_UNKNOWN_MERGE
                            and not is_growth_continuation_match
                        ):
                            frame_events.append({
                                "type": "match_rejected_growth",
                                "track_id": tid,
                                "det_id": det_id,
                                "iou": float(iou),
                                "dist_norm": float(dist_norm),
                                "area_ratio": float(area_ratio),
                                "reason": "adaptive_growth_reject",
                            })
                            reset_merge_candidate_state(track)
                            continue
                    if ENABLE_MATCH_GROWTH_MERGE_RESCUE:
                        if growth_ratio >= frame_growth_merge_threshold and not is_local_growth_continuation(iou, dist_norm):
                            support_parents = select_match_growth_merge_support_parents(
                                tid,
                                det_idx,
                                new_segment,
                                det_area,
                                tracked_objects,
                                assigned_tracks,
                                merged_parents,
                                candidates.get(det_idx, []),
                                best_det_for_track,
                            )
                            if support_parents:
                                detected_parents = [tid] + support_parents
                                merged_gen = max(tracked_objects[pid].gen for pid in detected_parents) + 1
                                track.missed = 0
                                track.gen = merged_gen
                                track.segment = new_segment
                                track.area = det_area
                                reset_merge_candidate_state(track)
                                assigned_tracks.add(tid)
                                assigned_dets.add(det_idx)
                                merged_children.add(tid)
                                frame_events.append({
                                    "type": "inferred_merge",
                                    "track_id": tid,
                                    "gen": merged_gen,
                                    "parents": support_parents,
                                    "detected_parents": detected_parents,
                                    "unknown_parents": 0,
                                    "det_id": det_id,
                                    "reason": "match_growth_with_competing_parent",
                                    "area_ratio": float(growth_ratio),
                                    "growth_threshold": float(frame_growth_merge_threshold),
                                })
                                for pid in support_parents:
                                    merged_parents.add(pid)
                                    assigned_tracks.add(pid)
                                continue
                    track.missed = 0
                    track.segment = new_segment
                    track.area = det_area
                    assigned_tracks.add(tid)
                    assigned_dets.add(det_idx)
                    if ENABLE_INFERRED_UNKNOWN_MERGE and is_growth_merge and PROMOTE_MATCH_GROWTH_TO_UNKNOWN_MERGE:
                            reset_merge_candidate_state(track)
                            track.gen += 1
                            merged_children.add(tid)
                            frame_events.append({
                                "type": "inferred_merge_unknown",
                                "track_id": tid,
                                "gen": track.gen,
                                "parents": [],
                                "det_id": det_id,
                                "reason": "matched_area_growth",
                                "area_ratio": float(area_ratio),
                            })
                            continue
                    if growth_ratio >= frame_growth_merge_threshold and not is_growth_continuation_match:
                        suppress_promotion_for_wobble = (
                            iou >= PERSISTENT_GROWTH_SUPPRESS_HIGH_IOU
                            and dist_norm <= PERSISTENT_GROWTH_SUPPRESS_LOW_MOTION
                        )
                        if suppress_promotion_for_wobble:
                            # Treat high-overlap low-motion growth as contour wobble, not merge evidence.
                            reset_merge_candidate_state(track)
                            frame_events.append({
                                "type": "match_growth",
                                "track_id": tid,
                                "det_id": det_id,
                                "iou": float(iou),
                                "dist_norm": float(dist_norm),
                                "area_ratio": float(growth_ratio),
                                "reason": "wobble_suppressed_candidate",
                            })
                            continue

                        streak, candidate_score = register_merge_candidate(
                            track,
                            frame_id,
                            growth_ratio,
                            frame_growth_merge_threshold,
                        )
                        if (
                            ENABLE_PERSISTENT_GROWTH_MERGE_PROMOTION
                            and streak >= max(PERSISTENT_GROWTH_MIN_STREAK, PERSISTENT_GROWTH_MIN_EVENTS)
                            and candidate_score >= PERSISTENT_GROWTH_PROMOTION_MIN_SCORE
                            and track.merge_candidate_peak_area_ratio >= PERSISTENT_GROWTH_PROMOTION_MIN_AREA_RATIO
                        ):
                            track.gen += 1
                            merged_children.add(tid)
                            unknown_parent_count = estimate_unknown_parent_count(det_area, [tid], tracked_objects)
                            if unknown_parent_count <= 0:
                                unknown_parent_count = 1
                            frame_events.append({
                                "type": "inferred_merge",
                                "track_id": tid,
                                "gen": track.gen,
                                "parents": [],
                                "detected_parents": [tid],
                                "unknown_parents": int(unknown_parent_count),
                                "det_id": det_id,
                                "reason": "persistent_growth_without_support",
                                "area_ratio": float(growth_ratio),
                                "growth_threshold": float(frame_growth_merge_threshold),
                                "streak": int(streak),
                                "candidate_score": float(candidate_score),
                                "candidate_peak_area_ratio": float(track.merge_candidate_peak_area_ratio),
                            })
                            reset_merge_candidate_state(track)
                            continue
                        frame_events.append({
                            "type": "merge_candidate",
                            "track_id": tid,
                            "det_id": det_id,
                            "iou": float(iou),
                            "dist_norm": float(dist_norm),
                            "area_ratio": float(growth_ratio),
                            "growth_threshold": float(frame_growth_merge_threshold),
                            "streak": int(streak),
                            "candidate_score": float(candidate_score),
                            "candidate_peak_area_ratio": float(track.merge_candidate_peak_area_ratio),
                            "reason": "growth_without_competing_parent",
                        })
                        continue
                    if is_growth_continuation_match:
                        frame_events.append({
                            "type": "match_growth",
                            "track_id": tid,
                            "det_id": det_id,
                            "iou": float(iou),
                            "dist_norm": float(dist_norm),
                            "area_ratio": float(area_ratio_values(prev_area, det_area)),
                        })
                        continue
                    frame_events.append({
                        "type": "match",
                        "track_id": tid,
                        "det_id": det_id,
                        "iou": float(iou),
                        "dist_norm": float(dist_norm),
                    })

                # New detections that weren't assigned
                for det_idx, _ in enumerate(detections):
                    if det_idx in assigned_dets:
                        continue

                    det_id = detection_meta[det_idx][2] if det_idx < len(detection_meta) else det_idx
                    det_segment = detection_segments[det_idx] if det_idx < len(detection_segments) else None
                    det_area = detection_areas[det_idx] if det_idx < len(detection_areas) else detection_area(det_segment)
                    if ENABLE_INFERRED_MERGE:
                        allow_single_parent_inferred = frame_id > INFERRED_MERGE_WARMUP_FRAMES
                        inferred_parents = select_inferred_merge_parents(
                            det_segment,
                            det_area,
                            tracked_objects,
                            assigned_tracks,
                            merged_parents,
                            allow_single_parent=allow_single_parent_inferred,
                        )
                        if len(inferred_parents) >= INFERRED_MERGE_MIN_PARENTS:
                            detected_parents = list(inferred_parents)
                            survivor_id = min(
                                inferred_parents,
                                key=lambda pid: (
                                    tracked_objects[pid].missed,
                                    -tracked_objects[pid].area,
                                ),
                            )
                            survivor_was_lost = tracked_objects[survivor_id].missed > 0
                            survivor_prev_area = tracked_objects[survivor_id].area
                            lineage_parents = [pid for pid in detected_parents if pid != survivor_id]
                            unknown_parent_count = estimate_unknown_parent_count(det_area, detected_parents, tracked_objects)
                            merged_gen = max(tracked_objects[pid].gen for pid in inferred_parents) + 1
                            tracked_objects[survivor_id].missed = 0
                            tracked_objects[survivor_id].gen = merged_gen
                            tracked_objects[survivor_id].segment = det_segment
                            tracked_objects[survivor_id].area = det_area
                            reset_merge_candidate_state(tracked_objects[survivor_id])
                            assigned_tracks.add(survivor_id)
                            assigned_dets.add(det_idx)
                            merged_children.add(survivor_id)

                            inferred_reason = "inferred_multi_parent_unmatched"
                            if len(detected_parents) == 1:
                                inferred_reason = (
                                    "single_parent_inferred_from_lost"
                                    if survivor_was_lost
                                    else "single_parent_inferred_unmatched_active"
                                )
                            frame_events.append({
                                "type": "inferred_merge",
                                "track_id": survivor_id,
                                "gen": merged_gen,
                                "parents": lineage_parents,
                                "detected_parents": detected_parents,
                                "unknown_parents": unknown_parent_count,
                                "det_id": det_id,
                                "reason": inferred_reason,
                                "area_ratio": float(area_growth_ratio(survivor_prev_area, det_area)),
                            })

                            for pid in lineage_parents:
                                merged_parents.add(pid)
                                assigned_tracks.add(pid)
                            continue

                    # Keep non-circular detections for association/merge, but do not spawn new tracks from them.
                    if FILTER_NON_CIRCULAR_SEGMENTS and not is_relevant_circular_segment(det_segment):
                        frame_events.append({
                            "type": "shape_rejected",
                            "det_id": det_id,
                            "reason": "non_circular_unmatched_detection",
                        })
                        continue

                    if ENABLE_INFERRED_UNKNOWN_MERGE:
                        is_unknown_merge, area_ratio = is_unknown_merge_from_birth(det_area, reference_area)
                        if is_unknown_merge:
                            event_type = "inferred_merge_unknown" if PROMOTE_LARGE_BIRTH_TO_UNKNOWN_MERGE else "large_birth"
                            event_gen = 2 if PROMOTE_LARGE_BIRTH_TO_UNKNOWN_MERGE else 1
                            tracked_objects[next_id] = Track(
                                gen=event_gen,
                                missed=0,
                                area=det_area,
                                segment=det_segment,
                            )
                            event_record = {
                                "type": event_type,
                                "track_id": next_id,
                                "gen": event_gen,
                                "det_id": det_id,
                                "reason": "large_unmatched_detection",
                                "area_ratio": float(area_ratio),
                                "reference_area": float(reference_area),
                            }
                            if event_type == "inferred_merge_unknown":
                                event_record["parents"] = []
                            frame_events.append(event_record)
                            if event_type == "inferred_merge_unknown":
                                merged_children.add(next_id)
                            assigned_tracks.add(next_id)
                            next_id += 1
                            continue

                    tracked_objects[next_id] = Track(gen=1, missed=0, area=det_area, segment=det_segment)
                    frame_events.append({
                        "type": "birth",
                        "track_id": next_id,
                        "gen": 1,
                        "det_id": det_id,
                    })
                    assigned_tracks.add(next_id)
                    next_id += 1

                # Update missed counts and remove stale tracks
                to_remove = []
                for tid, track in tracked_objects.items():
                    if tid in merged_parents:
                        to_remove.append(tid)
                        continue
                    if tid not in assigned_tracks:
                        reset_merge_candidate_state(track)
                        track.missed += 1
                        if track.missed > MAX_MISSED_FRAMES:
                            frame_events.append({
                                "type": "death",
                                "track_id": tid,
                                "gen": track.gen,
                            })
                            to_remove.append(tid)

                for tid in to_remove:
                    tracked_objects.pop(tid, None)

            if DRAW_TRACK_OVERLAY:
                apply_tracking_overlay(frame, tracked_objects, merged_children)

            writer.write(frame)
            frame_record_out = {
                "frame": frame_id,
                "detections": [
                    {
                        "det_id": detection_meta[det_idx][2] if det_idx < len(detection_meta) else det_idx,
                        "center": serialize_center(detection_segments[det_idx] if det_idx < len(detection_segments) else None),
                        "area": float(detection_areas[det_idx]) if det_idx < len(detection_areas) else detection_area(detection_segments[det_idx] if det_idx < len(detection_segments) else None),
                        "segment": serialize_segment(detection_segments[det_idx]) if det_idx < len(detection_segments) else None,
                        "conf": detection_meta[det_idx][0] if det_idx < len(detection_meta) else 0.0,
                        "cls": detection_meta[det_idx][1] if det_idx < len(detection_meta) else 0,
                    }
                    for det_idx, _ in enumerate(detections)
                ],
                "tracks": [
                    {
                        "track_id": tid,
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
                    for tid, track in tracked_objects.items()
                ],
                "events": frame_events,
            }
            log_writer.write(frame_record_out)
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
