"""The tracking main loop: re-hydrate per-frame detections from
detections.json, associate them to Track objects via the matching suite
(mutual-best one-to-one matching plus the direct / inferred / match-growth
merge detectors), and write the annotated mp4, tracking_log.json, and the
flattened CSV. Verbatim from the monolith."""
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from tracking import events
from tracking.config import DEFAULT_CONFIG, TrackingConfig
from tracking.detections import frame_detections_from_record
from tracking.geometry import is_relevant_circular_segment
from tracking.io import (
    JsonArrayWriter,
    ensure_parent_dir,
    export_tracking_log_csv,
    frame_log_record,
    open_video,
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
    resolve_prefilter_device,
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

def track_from_detections_json(video_path, detections_path, output_path, log_path,
                               config: Optional[TrackingConfig] = None):
    cfg = DEFAULT_CONFIG if config is None else config
    cap, fps, width, height = open_video(video_path)
    ensure_parent_dir(output_path)
    ensure_parent_dir(log_path)
    writer = create_video_writer(output_path, fps, width, height)
    device = resolve_prefilter_device(cfg)
    print(f"Tracking candidate prefilter device: {device}")

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
            dets = frame_detections_from_record(frame_record, cfg)
            frame_median_area = float(np.median(dets.areas)) if dets.areas else 0.0
            reference_area = reference_object_area(tracked_objects, frame_median_area, cfg)
            frame_events = []
            merged_children = set()

            if frame_index == 0:
                for det_idx in range(len(dets)):
                    segment = dets.segments[det_idx]
                    det_area = dets.areas[det_idx]
                    tracked_objects[next_id] = Track(gen=1, missed=0, area=det_area, segment=segment)
                    det_id = dets.det_ids[det_idx]
                    frame_events.append(events.birth(next_id, 1, det_id))
                    next_id += 1
            else:
                active_tracks = {tid: t for tid, t in tracked_objects.items() if t.missed <= cfg.max_missed_frames}
                candidates = build_match_candidates(active_tracks, dets.segments, cfg, device)

                assigned_tracks = set()
                assigned_dets = set()
                merged_parents = set()

                # Merge detection
                if cfg.direct_merge.enabled:
                    for det_idx, cand_list in candidates.items():
                        if det_idx in assigned_dets:
                            continue

                        det_segment = dets.segments[det_idx]
                        det_area = dets.areas[det_idx]
                        parent_ids = select_merge_parents(det_segment, det_area, cand_list, active_tracks, assigned_tracks, cfg)
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
                        det_id = dets.det_ids[det_idx]
                        frame_events.append(events.merge(child_id, child_gen, parent_ids, det_id))
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
                        det_area = dets.areas[det_idx]
                        if not is_valid_match(tracked_objects[tid], det_area, iou, dist_norm, cfg):
                            continue
                        score = iou - 0.1 * dist_norm
                        scored_pairs.append((score, tid, det_idx, iou, dist_norm))

                scored_pairs.sort(reverse=True)
                best_track_for_det: Dict[int, Tuple[int, float]] = {}
                best_det_for_track: Dict[int, Tuple[int, float]] = {}
                if cfg.match.require_mutual_best:
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
                    dets.areas,
                    cfg,
                )
                frame_growth_reject_threshold = max(
                    cfg.unknown_merge.match_area_growth,
                    frame_growth_merge_threshold + 0.35,
                )

                for _, tid, det_idx, iou, dist_norm in scored_pairs:
                    if tid in assigned_tracks or det_idx in assigned_dets:
                        continue
                    if cfg.match.require_mutual_best:
                        best_det = best_det_for_track.get(tid)
                        best_track = best_track_for_det.get(det_idx)
                        if best_det is None or best_track is None:
                            continue
                        if best_det[0] != det_idx or best_track[0] != tid:
                            continue
                    det_area = dets.areas[det_idx]
                    if not is_valid_match(tracked_objects[tid], det_area, iou, dist_norm, cfg):
                        continue
                    track = tracked_objects[tid]
                    decay_merge_candidate_state(track, frame_id, cfg)
                    det_id = dets.det_ids[det_idx]
                    new_segment = dets.segments[det_idx]
                    prev_area = track.area
                    growth_ratio = area_growth_ratio(prev_area, det_area)
                    is_growth_merge = False
                    is_growth_continuation_match = False
                    area_ratio = 0.0
                    if cfg.unknown_merge.enabled:
                        is_growth_continuation_match = (
                            det_area > prev_area
                            and area_ratio_values(prev_area, det_area) > cfg.match.max_area_ratio
                            and is_local_growth_continuation(iou, dist_norm, cfg)
                        )
                        is_growth_merge, area_ratio = is_unknown_merge_from_match(
                            prev_area,
                            det_area,
                            cfg,
                            frame_growth_reject_threshold,
                        )
                        if (
                            is_growth_merge
                            and cfg.unknown_merge.reject_large_growth_match
                            and not cfg.unknown_merge.promote_match_growth
                            and not is_growth_continuation_match
                        ):
                            frame_events.append(events.growth_rejected(tid, det_id, iou, dist_norm, area_ratio))
                            reset_merge_candidate_state(track)
                            continue
                    if cfg.growth_rescue.enabled:
                        if growth_ratio >= frame_growth_merge_threshold and not is_local_growth_continuation(iou, dist_norm, cfg):
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
                                cfg,
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
                                frame_events.append(events.rescue_merge(tid, merged_gen, support_parents, detected_parents, det_id, growth_ratio, frame_growth_merge_threshold))
                                for pid in support_parents:
                                    merged_parents.add(pid)
                                    assigned_tracks.add(pid)
                                continue
                    track.missed = 0
                    track.segment = new_segment
                    track.area = det_area
                    assigned_tracks.add(tid)
                    assigned_dets.add(det_idx)
                    if cfg.unknown_merge.enabled and is_growth_merge and cfg.unknown_merge.promote_match_growth:
                            reset_merge_candidate_state(track)
                            track.gen += 1
                            merged_children.add(tid)
                            frame_events.append(events.matched_growth_unknown(tid, track.gen, det_id, area_ratio))
                            continue
                    if growth_ratio >= frame_growth_merge_threshold and not is_growth_continuation_match:
                        suppress_promotion_for_wobble = (
                            iou >= cfg.persistent_growth.wobble_min_iou
                            and dist_norm <= cfg.persistent_growth.wobble_max_dist_norm
                        )
                        if suppress_promotion_for_wobble:
                            # Treat high-overlap low-motion growth as contour wobble, not merge evidence.
                            reset_merge_candidate_state(track)
                            frame_events.append(events.wobble_suppressed(tid, det_id, iou, dist_norm, growth_ratio))
                            continue

                        streak, candidate_score = register_merge_candidate(
                            track,
                            frame_id,
                            growth_ratio,
                            frame_growth_merge_threshold,
                            cfg,
                        )
                        if (
                            cfg.persistent_growth.enabled
                            and streak >= max(cfg.persistent_growth.min_streak, cfg.persistent_growth.min_events)
                            and candidate_score >= cfg.persistent_growth.promotion_min_score
                            and track.merge_candidate_peak_area_ratio >= cfg.persistent_growth.promotion_min_area_ratio
                        ):
                            track.gen += 1
                            merged_children.add(tid)
                            unknown_parent_count = estimate_unknown_parent_count(det_area, [tid], tracked_objects, cfg)
                            if unknown_parent_count <= 0:
                                unknown_parent_count = 1
                            frame_events.append(events.persistent_growth_merge(tid, track.gen, unknown_parent_count, det_id, growth_ratio, frame_growth_merge_threshold, streak, candidate_score, track.merge_candidate_peak_area_ratio))
                            reset_merge_candidate_state(track)
                            continue
                        frame_events.append(events.merge_candidate(tid, det_id, iou, dist_norm, growth_ratio, frame_growth_merge_threshold, streak, candidate_score, track.merge_candidate_peak_area_ratio))
                        continue
                    if is_growth_continuation_match:
                        frame_events.append(events.match_growth(tid, det_id, iou, dist_norm, area_ratio_values(prev_area, det_area)))
                        continue
                    frame_events.append(events.match(tid, det_id, iou, dist_norm))

                # New detections that weren't assigned
                for det_idx in range(len(dets)):
                    if det_idx in assigned_dets:
                        continue

                    det_id = dets.det_ids[det_idx]
                    det_segment = dets.segments[det_idx]
                    det_area = dets.areas[det_idx]
                    if cfg.inferred_merge.enabled:
                        allow_single_parent_inferred = frame_id > cfg.inferred_merge.warmup_frames
                        inferred_parents = select_inferred_merge_parents(
                            det_segment,
                            det_area,
                            tracked_objects,
                            assigned_tracks,
                            merged_parents,
                            cfg,
                            allow_single_parent=allow_single_parent_inferred,
                        )
                        if len(inferred_parents) >= cfg.inferred_merge.min_parents:
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
                            unknown_parent_count = estimate_unknown_parent_count(det_area, detected_parents, tracked_objects, cfg)
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
                            frame_events.append(events.unmatched_inferred_merge(survivor_id, merged_gen, lineage_parents, detected_parents, unknown_parent_count, det_id, inferred_reason, area_growth_ratio(survivor_prev_area, det_area)))

                            for pid in lineage_parents:
                                merged_parents.add(pid)
                                assigned_tracks.add(pid)
                            continue

                    # Keep non-circular detections for association/merge, but do not spawn new tracks from them.
                    if cfg.segments.filter_non_circular and not is_relevant_circular_segment(det_segment, cfg):
                        frame_events.append(events.shape_rejected(det_id))
                        continue

                    if cfg.unknown_merge.enabled:
                        is_unknown_merge, area_ratio = is_unknown_merge_from_birth(det_area, reference_area, cfg)
                        if is_unknown_merge:
                            promoted = cfg.unknown_merge.promote_large_birth
                            event_gen = 2 if promoted else 1
                            tracked_objects[next_id] = Track(
                                gen=event_gen,
                                missed=0,
                                area=det_area,
                                segment=det_segment,
                            )
                            frame_events.append(events.large_unmatched(
                                promoted, next_id, event_gen, det_id, area_ratio, reference_area))
                            if promoted:
                                merged_children.add(next_id)
                            assigned_tracks.add(next_id)
                            next_id += 1
                            continue

                    tracked_objects[next_id] = Track(gen=1, missed=0, area=det_area, segment=det_segment)
                    frame_events.append(events.birth(next_id, 1, det_id))
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
                        if track.missed > cfg.max_missed_frames:
                            frame_events.append(events.death(tid, track.gen))
                            to_remove.append(tid)

                for tid in to_remove:
                    tracked_objects.pop(tid, None)

            if cfg.render.draw_overlay:
                apply_tracking_overlay(frame, tracked_objects, merged_children, cfg)

            writer.write(frame)
            frame_record_out = frame_log_record(frame_id, dets, tracked_objects, frame_events)
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
