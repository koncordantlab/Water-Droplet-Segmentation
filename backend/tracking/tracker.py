"""The tracker core: Track state across frames and the per-frame step that
turns one frame's detections into tracking events. No video or file I/O --
orchestration (frames in, overlay and logs out) lives in tracking.track."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from tracking import events
from tracking.config import DEFAULT_CONFIG, TrackingConfig
from tracking.detections import FrameDetections
from tracking.geometry import is_relevant_circular_segment
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
from tracking.tracks import (
    Track,
    decay_merge_candidate_state,
    register_merge_candidate,
    reset_merge_candidate_state,
)

Candidates = Dict[int, List[Tuple[int, float, float]]]  # det_idx -> [(track_id, iou, dist_norm)]


@dataclass
class FrameContext:
    """The working state of one step(); every stage reads and extends it."""
    frame_id: int
    dets: FrameDetections
    reference_area: float
    events: List[dict] = field(default_factory=list)
    merged_children: Set[int] = field(default_factory=set)
    assigned_tracks: Set[int] = field(default_factory=set)
    assigned_dets: Set[int] = field(default_factory=set)
    merged_parents: Set[int] = field(default_factory=set)
    active_tracks: Dict[int, Track] = field(default_factory=dict)
    candidates: Candidates = field(default_factory=dict)


@dataclass
class FrameResult:
    events: List[dict]
    merged_children: Set[int]


class Tracker:
    def __init__(self, cfg: TrackingConfig = DEFAULT_CONFIG, device: Optional[torch.device] = None):
        self.cfg = cfg
        self.device = resolve_prefilter_device(cfg) if device is None else device
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1
        self._frames_seen = 0

    def new_track(self, gen: int, area: float, segment: Optional[np.ndarray]) -> int:
        track_id = self._next_id
        self._next_id += 1
        self.tracks[track_id] = Track(gen=gen, missed=0, area=area, segment=segment)
        return track_id

    def step(self, frame_id: int, dets: FrameDetections) -> FrameResult:
        frame_median_area = float(np.median(dets.areas)) if dets.areas else 0.0
        ctx = FrameContext(
            frame_id=frame_id,
            dets=dets,
            reference_area=reference_object_area(self.tracks, frame_median_area, self.cfg),
        )
        if self._frames_seen == 0:
            self._first_frame(ctx)
        else:
            self._legacy_frame(ctx)  # transitional: carved into stage modules in Tasks 6-7
            self._age_tracks(ctx)
        self._frames_seen += 1
        return FrameResult(ctx.events, ctx.merged_children)

    def _first_frame(self, ctx: FrameContext) -> None:
        for det_idx in range(len(ctx.dets)):
            track_id = self.new_track(1, ctx.dets.areas[det_idx], ctx.dets.segments[det_idx])
            ctx.events.append(events.birth(track_id, 1, ctx.dets.det_ids[det_idx]))

    def _age_tracks(self, ctx: FrameContext) -> None:
        """Merged parents retire; unassigned tracks miss a frame and die past max_missed_frames."""
        to_remove = []
        for tid, track in self.tracks.items():
            if tid in ctx.merged_parents:
                to_remove.append(tid)
                continue
            if tid not in ctx.assigned_tracks:
                reset_merge_candidate_state(track)
                track.missed += 1
                if track.missed > self.cfg.max_missed_frames:
                    ctx.events.append(events.death(tid, track.gen))
                    to_remove.append(tid)
        for tid in to_remove:
            self.tracks.pop(tid, None)

    def _legacy_frame(self, ctx: FrameContext) -> None:
        """Transitional: the lifted association-and-resolution middle of the
        former track.py loop (Tasks 6-7 carve it into stage modules)."""
        ctx.active_tracks = {tid: t for tid, t in self.tracks.items() if t.missed <= self.cfg.max_missed_frames}
        ctx.candidates = build_match_candidates(ctx.active_tracks, ctx.dets.segments, self.cfg, self.device)


        # Merge detection
        if self.cfg.direct_merge.enabled:
            for det_idx, cand_list in ctx.candidates.items():
                if det_idx in ctx.assigned_dets:
                    continue

                det_segment = ctx.dets.segments[det_idx]
                det_area = ctx.dets.areas[det_idx]
                parent_ids = select_merge_parents(det_segment, det_area, cand_list, ctx.active_tracks, ctx.assigned_tracks, self.cfg)
                if len(parent_ids) < 2:
                    continue

                child_gen = max(ctx.active_tracks[pid].gen for pid in parent_ids) + 1
                child_id = self.new_track(child_gen, det_area, det_segment)
                det_id = ctx.dets.det_ids[det_idx]
                ctx.events.append(events.merge(child_id, child_gen, parent_ids, det_id))
                ctx.merged_children.add(child_id)
                ctx.assigned_tracks.add(child_id)

                for pid in parent_ids:
                    ctx.merged_parents.add(pid)
                    ctx.assigned_tracks.add(pid)

                ctx.assigned_dets.add(det_idx)

        # One-to-one matching for remaining detections
        scored_pairs: List[Tuple[float, int, int, float, float]] = []
        for det_idx, cand_list in ctx.candidates.items():
            if det_idx in ctx.assigned_dets:
                continue
            for tid, iou, dist_norm in cand_list:
                if tid in ctx.assigned_tracks or tid in ctx.merged_parents:
                    continue
                det_area = ctx.dets.areas[det_idx]
                if not is_valid_match(self.tracks[tid], det_area, iou, dist_norm, self.cfg):
                    continue
                score = iou - 0.1 * dist_norm
                scored_pairs.append((score, tid, det_idx, iou, dist_norm))

        scored_pairs.sort(reverse=True)
        best_track_for_det: Dict[int, Tuple[int, float]] = {}
        best_det_for_track: Dict[int, Tuple[int, float]] = {}
        if self.cfg.match.require_mutual_best:
            for score, tid, det_idx, _, _ in scored_pairs:
                prev_det = best_det_for_track.get(tid)
                if prev_det is None or score > prev_det[1]:
                    best_det_for_track[tid] = (det_idx, score)
                prev_track = best_track_for_det.get(det_idx)
                if prev_track is None or score > prev_track[1]:
                    best_track_for_det[det_idx] = (tid, score)

        frame_growth_merge_threshold = estimate_adaptive_growth_threshold(
            scored_pairs,
            self.tracks,
            ctx.dets.areas,
            self.cfg,
        )
        frame_growth_reject_threshold = max(
            self.cfg.unknown_merge.match_area_growth,
            frame_growth_merge_threshold + 0.35,
        )

        for _, tid, det_idx, iou, dist_norm in scored_pairs:
            if tid in ctx.assigned_tracks or det_idx in ctx.assigned_dets:
                continue
            if self.cfg.match.require_mutual_best:
                best_det = best_det_for_track.get(tid)
                best_track = best_track_for_det.get(det_idx)
                if best_det is None or best_track is None:
                    continue
                if best_det[0] != det_idx or best_track[0] != tid:
                    continue
            det_area = ctx.dets.areas[det_idx]
            if not is_valid_match(self.tracks[tid], det_area, iou, dist_norm, self.cfg):
                continue
            track = self.tracks[tid]
            decay_merge_candidate_state(track, ctx.frame_id, self.cfg)
            det_id = ctx.dets.det_ids[det_idx]
            new_segment = ctx.dets.segments[det_idx]
            prev_area = track.area
            growth_ratio = area_growth_ratio(prev_area, det_area)
            is_growth_merge = False
            is_growth_continuation_match = False
            area_ratio = 0.0
            if self.cfg.unknown_merge.enabled:
                is_growth_continuation_match = (
                    det_area > prev_area
                    and area_ratio_values(prev_area, det_area) > self.cfg.match.max_area_ratio
                    and is_local_growth_continuation(iou, dist_norm, self.cfg)
                )
                is_growth_merge, area_ratio = is_unknown_merge_from_match(
                    prev_area,
                    det_area,
                    self.cfg,
                    frame_growth_reject_threshold,
                )
                if (
                    is_growth_merge
                    and self.cfg.unknown_merge.reject_large_growth_match
                    and not self.cfg.unknown_merge.promote_match_growth
                    and not is_growth_continuation_match
                ):
                    ctx.events.append(events.growth_rejected(tid, det_id, iou, dist_norm, area_ratio))
                    reset_merge_candidate_state(track)
                    continue
            if self.cfg.growth_rescue.enabled:
                if growth_ratio >= frame_growth_merge_threshold and not is_local_growth_continuation(iou, dist_norm, self.cfg):
                    support_parents = select_match_growth_merge_support_parents(
                        tid,
                        det_idx,
                        new_segment,
                        det_area,
                        self.tracks,
                        ctx.assigned_tracks,
                        ctx.merged_parents,
                        ctx.candidates.get(det_idx, []),
                        best_det_for_track,
                        self.cfg,
                    )
                    if support_parents:
                        detected_parents = [tid] + support_parents
                        merged_gen = max(self.tracks[pid].gen for pid in detected_parents) + 1
                        track.missed = 0
                        track.gen = merged_gen
                        track.segment = new_segment
                        track.area = det_area
                        reset_merge_candidate_state(track)
                        ctx.assigned_tracks.add(tid)
                        ctx.assigned_dets.add(det_idx)
                        ctx.merged_children.add(tid)
                        ctx.events.append(events.rescue_merge(tid, merged_gen, support_parents, detected_parents, det_id, growth_ratio, frame_growth_merge_threshold))
                        for pid in support_parents:
                            ctx.merged_parents.add(pid)
                            ctx.assigned_tracks.add(pid)
                        continue
            track.missed = 0
            track.segment = new_segment
            track.area = det_area
            ctx.assigned_tracks.add(tid)
            ctx.assigned_dets.add(det_idx)
            if self.cfg.unknown_merge.enabled and is_growth_merge and self.cfg.unknown_merge.promote_match_growth:
                    reset_merge_candidate_state(track)
                    track.gen += 1
                    ctx.merged_children.add(tid)
                    ctx.events.append(events.matched_growth_unknown(tid, track.gen, det_id, area_ratio))
                    continue
            if growth_ratio >= frame_growth_merge_threshold and not is_growth_continuation_match:
                suppress_promotion_for_wobble = (
                    iou >= self.cfg.persistent_growth.wobble_min_iou
                    and dist_norm <= self.cfg.persistent_growth.wobble_max_dist_norm
                )
                if suppress_promotion_for_wobble:
                    # Treat high-overlap low-motion growth as contour wobble, not merge evidence.
                    reset_merge_candidate_state(track)
                    ctx.events.append(events.wobble_suppressed(tid, det_id, iou, dist_norm, growth_ratio))
                    continue

                streak, candidate_score = register_merge_candidate(
                    track,
                    ctx.frame_id,
                    growth_ratio,
                    frame_growth_merge_threshold,
                    self.cfg,
                )
                if (
                    self.cfg.persistent_growth.enabled
                    and streak >= max(self.cfg.persistent_growth.min_streak, self.cfg.persistent_growth.min_events)
                    and candidate_score >= self.cfg.persistent_growth.promotion_min_score
                    and track.merge_candidate_peak_area_ratio >= self.cfg.persistent_growth.promotion_min_area_ratio
                ):
                    track.gen += 1
                    ctx.merged_children.add(tid)
                    unknown_parent_count = estimate_unknown_parent_count(det_area, [tid], self.tracks, self.cfg)
                    if unknown_parent_count <= 0:
                        unknown_parent_count = 1
                    ctx.events.append(events.persistent_growth_merge(tid, track.gen, unknown_parent_count, det_id, growth_ratio, frame_growth_merge_threshold, streak, candidate_score, track.merge_candidate_peak_area_ratio))
                    reset_merge_candidate_state(track)
                    continue
                ctx.events.append(events.merge_candidate(tid, det_id, iou, dist_norm, growth_ratio, frame_growth_merge_threshold, streak, candidate_score, track.merge_candidate_peak_area_ratio))
                continue
            if is_growth_continuation_match:
                ctx.events.append(events.match_growth(tid, det_id, iou, dist_norm, area_ratio_values(prev_area, det_area)))
                continue
            ctx.events.append(events.match(tid, det_id, iou, dist_norm))

        # New detections that weren't assigned
        for det_idx in range(len(ctx.dets)):
            if det_idx in ctx.assigned_dets:
                continue

            det_id = ctx.dets.det_ids[det_idx]
            det_segment = ctx.dets.segments[det_idx]
            det_area = ctx.dets.areas[det_idx]
            if self.cfg.inferred_merge.enabled:
                allow_single_parent_inferred = ctx.frame_id > self.cfg.inferred_merge.warmup_frames
                inferred_parents = select_inferred_merge_parents(
                    det_segment,
                    det_area,
                    self.tracks,
                    ctx.assigned_tracks,
                    ctx.merged_parents,
                    self.cfg,
                    allow_single_parent=allow_single_parent_inferred,
                )
                if len(inferred_parents) >= self.cfg.inferred_merge.min_parents:
                    detected_parents = list(inferred_parents)
                    survivor_id = min(
                        inferred_parents,
                        key=lambda pid: (
                            self.tracks[pid].missed,
                            -self.tracks[pid].area,
                        ),
                    )
                    survivor_was_lost = self.tracks[survivor_id].missed > 0
                    survivor_prev_area = self.tracks[survivor_id].area
                    lineage_parents = [pid for pid in detected_parents if pid != survivor_id]
                    unknown_parent_count = estimate_unknown_parent_count(det_area, detected_parents, self.tracks, self.cfg)
                    merged_gen = max(self.tracks[pid].gen for pid in inferred_parents) + 1
                    self.tracks[survivor_id].missed = 0
                    self.tracks[survivor_id].gen = merged_gen
                    self.tracks[survivor_id].segment = det_segment
                    self.tracks[survivor_id].area = det_area
                    reset_merge_candidate_state(self.tracks[survivor_id])
                    ctx.assigned_tracks.add(survivor_id)
                    ctx.assigned_dets.add(det_idx)
                    ctx.merged_children.add(survivor_id)

                    inferred_reason = "inferred_multi_parent_unmatched"
                    if len(detected_parents) == 1:
                        inferred_reason = (
                            "single_parent_inferred_from_lost"
                            if survivor_was_lost
                            else "single_parent_inferred_unmatched_active"
                        )
                    ctx.events.append(events.unmatched_inferred_merge(survivor_id, merged_gen, lineage_parents, detected_parents, unknown_parent_count, det_id, inferred_reason, area_growth_ratio(survivor_prev_area, det_area)))

                    for pid in lineage_parents:
                        ctx.merged_parents.add(pid)
                        ctx.assigned_tracks.add(pid)
                    continue

            # Keep non-circular detections for association/merge, but do not spawn new tracks from them.
            if self.cfg.segments.filter_non_circular and not is_relevant_circular_segment(det_segment, self.cfg):
                ctx.events.append(events.shape_rejected(det_id))
                continue

            if self.cfg.unknown_merge.enabled:
                is_unknown_merge, area_ratio = is_unknown_merge_from_birth(det_area, ctx.reference_area, self.cfg)
                if is_unknown_merge:
                    promoted = self.cfg.unknown_merge.promote_large_birth
                    event_gen = 2 if promoted else 1
                    track_id = self.new_track(event_gen, det_area, det_segment)
                    ctx.events.append(events.large_unmatched(
                        promoted, track_id, event_gen, det_id, area_ratio, ctx.reference_area))
                    if promoted:
                        ctx.merged_children.add(track_id)
                    ctx.assigned_tracks.add(track_id)
                    continue

            track_id = self.new_track(1, det_area, det_segment)
            ctx.events.append(events.birth(track_id, 1, det_id))
            ctx.assigned_tracks.add(track_id)

