"""Unmatched-detection stage of a tracker step: a detection no track
claimed becomes an inferred merge of recently lost tracks, a shape
rejection, a large unknown-parent merge (or large birth), or a plain birth."""
from typing import TYPE_CHECKING, Optional

import numpy as np

from tracking import events
from tracking.geometry import is_relevant_circular_segment
from tracking.matching import (
    area_growth_ratio,
    estimate_unknown_parent_count,
    is_unknown_merge_from_birth,
    select_inferred_merge_parents,
)
from tracking.tracks import reset_merge_candidate_state

if TYPE_CHECKING:
    from tracking.tracker import FrameContext, Tracker


def resolve_unmatched(tracker: "Tracker", ctx: "FrameContext") -> None:
    cfg = tracker.cfg
    for det_idx in range(len(ctx.dets)):
        if det_idx in ctx.assigned_dets:
            continue
        det_id = ctx.dets.det_ids[det_idx]
        det_segment = ctx.dets.segments[det_idx]
        det_area = ctx.dets.areas[det_idx]
        if cfg.inferred_merge.enabled and _infer_merge(tracker, ctx, det_idx, det_id, det_segment, det_area):
            continue
        # Keep non-circular detections for association/merge, but do not spawn new tracks from them.
        if cfg.segments.filter_non_circular and not is_relevant_circular_segment(det_segment, cfg):
            ctx.events.append(events.shape_rejected(det_id))
            continue
        if cfg.unknown_merge.enabled and _large_unmatched(tracker, ctx, det_id, det_segment, det_area):
            continue
        track_id = tracker.new_track(1, det_area, det_segment)
        ctx.events.append(events.birth(track_id, 1, det_id))
        ctx.assigned_tracks.add(track_id)


def _infer_merge(tracker: "Tracker", ctx: "FrameContext", det_idx: int, det_id: int,
                 det_segment: Optional[np.ndarray], det_area: float) -> bool:
    """Recently lost (or unmatched) tracks that explain this detection merged
    into it: the least-missed, largest parent survives as the child."""
    cfg = tracker.cfg
    tracks = tracker.tracks
    inferred_parents = select_inferred_merge_parents(
        det_segment, det_area, tracks, ctx.assigned_tracks, ctx.merged_parents, cfg,
        allow_single_parent=ctx.frame_id > cfg.inferred_merge.warmup_frames,
    )
    if len(inferred_parents) < cfg.inferred_merge.min_parents:
        return False
    detected_parents = list(inferred_parents)
    survivor_id = min(inferred_parents, key=lambda pid: (tracks[pid].missed, -tracks[pid].area))
    survivor = tracks[survivor_id]
    survivor_was_lost = survivor.missed > 0
    survivor_prev_area = survivor.area
    lineage_parents = [pid for pid in detected_parents if pid != survivor_id]
    unknown_parent_count = estimate_unknown_parent_count(det_area, detected_parents, tracks, cfg)
    merged_gen = max(tracks[pid].gen for pid in inferred_parents) + 1
    survivor.missed = 0
    survivor.gen = merged_gen
    survivor.segment = det_segment
    survivor.area = det_area
    reset_merge_candidate_state(survivor)
    ctx.assigned_tracks.add(survivor_id)
    ctx.assigned_dets.add(det_idx)
    ctx.merged_children.add(survivor_id)

    reason = "inferred_multi_parent_unmatched"
    if len(detected_parents) == 1:
        reason = "single_parent_inferred_from_lost" if survivor_was_lost else "single_parent_inferred_unmatched_active"
    ctx.events.append(events.unmatched_inferred_merge(
        survivor_id, merged_gen, lineage_parents, detected_parents, unknown_parent_count, det_id,
        reason, area_growth_ratio(survivor_prev_area, det_area)))
    for pid in lineage_parents:
        ctx.merged_parents.add(pid)
        ctx.assigned_tracks.add(pid)
    return True


def _large_unmatched(tracker: "Tracker", ctx: "FrameContext", det_id: int,
                     det_segment: Optional[np.ndarray], det_area: float) -> bool:
    """A detection much larger than a typical droplet: an unknown-parent merge
    (or a plain large birth when promotion is off)."""
    is_unknown_merge, area_ratio = is_unknown_merge_from_birth(det_area, ctx.reference_area, tracker.cfg)
    if not is_unknown_merge:
        return False
    promoted = tracker.cfg.unknown_merge.promote_large_birth
    event_gen = 2 if promoted else 1
    track_id = tracker.new_track(event_gen, det_area, det_segment)
    ctx.events.append(events.large_unmatched(promoted, track_id, event_gen, det_id, area_ratio, ctx.reference_area))
    if promoted:
        ctx.merged_children.add(track_id)
    ctx.assigned_tracks.add(track_id)
    return True
