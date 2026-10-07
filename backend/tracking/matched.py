"""Matched-pair stage of a tracker step: each mutual-best, valid
(track, detection) pair becomes a plain match, a growth continuation, a
rejected growth, a rescue merge with a competing parent, an unknown-parent
merge, or a step in persistent-growth evidence."""
from typing import TYPE_CHECKING

from tracking import events
from tracking.matching import (
    area_growth_ratio,
    area_ratio_values,
    estimate_unknown_parent_count,
    is_local_growth_continuation,
    is_unknown_merge_from_match,
    is_valid_match,
    select_match_growth_merge_support_parents,
)
from tracking.tracks import (
    Track,
    decay_merge_candidate_state,
    register_merge_candidate,
    reset_merge_candidate_state,
)

if TYPE_CHECKING:
    from tracking.associate import PairPlan
    from tracking.tracker import FrameContext, Tracker


def resolve_matches(tracker: "Tracker", ctx: "FrameContext", plan: "PairPlan") -> None:
    cfg = tracker.cfg
    for _, tid, det_idx, iou, dist_norm in plan.scored_pairs:
        if tid in ctx.assigned_tracks or det_idx in ctx.assigned_dets:
            continue
        if cfg.match.require_mutual_best and not _is_mutual_best(plan, tid, det_idx):
            continue
        det_area = ctx.dets.areas[det_idx]
        if not is_valid_match(tracker.tracks[tid], det_area, iou, dist_norm, cfg):
            continue
        _resolve_pair(tracker, ctx, plan, tid, det_idx, iou, dist_norm, det_area)


def _is_mutual_best(plan: "PairPlan", tid: int, det_idx: int) -> bool:
    best_det = plan.best_det_for_track.get(tid)
    best_track = plan.best_track_for_det.get(det_idx)
    if best_det is None or best_track is None:
        return False
    return best_det[0] == det_idx and best_track[0] == tid


def _resolve_pair(tracker: "Tracker", ctx: "FrameContext", plan: "PairPlan", tid: int, det_idx: int,
                  iou: float, dist_norm: float, det_area: float) -> None:
    cfg = tracker.cfg
    track = tracker.tracks[tid]
    decay_merge_candidate_state(track, ctx.frame_id, cfg)
    det_id = ctx.dets.det_ids[det_idx]
    prev_area = track.area
    growth_ratio = area_growth_ratio(prev_area, det_area)
    is_growth_merge = False
    is_continuation = False
    area_ratio = 0.0
    if cfg.unknown_merge.enabled:
        is_continuation = (
            det_area > prev_area
            and area_ratio_values(prev_area, det_area) > cfg.match.max_area_ratio
            and is_local_growth_continuation(iou, dist_norm, cfg)
        )
        is_growth_merge, area_ratio = is_unknown_merge_from_match(prev_area, det_area, cfg, plan.reject_threshold)
        if (
            is_growth_merge
            and cfg.unknown_merge.reject_large_growth_match
            and not cfg.unknown_merge.promote_match_growth
            and not is_continuation
        ):
            ctx.events.append(events.growth_rejected(tid, det_id, iou, dist_norm, area_ratio))
            reset_merge_candidate_state(track)
            return
    if cfg.growth_rescue.enabled and _rescue_as_merge(
            tracker, ctx, plan, tid, det_idx, iou, dist_norm, det_area, growth_ratio):
        return

    track.missed = 0
    track.segment = ctx.dets.segments[det_idx]
    track.area = det_area
    ctx.assigned_tracks.add(tid)
    ctx.assigned_dets.add(det_idx)
    if cfg.unknown_merge.enabled and is_growth_merge and cfg.unknown_merge.promote_match_growth:
        reset_merge_candidate_state(track)
        track.gen += 1
        ctx.merged_children.add(tid)
        ctx.events.append(events.matched_growth_unknown(tid, track.gen, det_id, area_ratio))
        return
    if growth_ratio >= plan.merge_threshold and not is_continuation:
        _record_growth_evidence(tracker, ctx, plan, track, tid, det_id, iou, dist_norm, det_area, growth_ratio)
        return
    if is_continuation:
        ctx.events.append(events.match_growth(tid, det_id, iou, dist_norm, area_ratio_values(prev_area, det_area)))
        return
    ctx.events.append(events.match(tid, det_id, iou, dist_norm))


def _rescue_as_merge(tracker: "Tracker", ctx: "FrameContext", plan: "PairPlan", tid: int, det_idx: int,
                     iou: float, dist_norm: float, det_area: float, growth_ratio: float) -> bool:
    """Growth that competing live tracks can explain is a merge: the matched
    track absorbs those support parents and its generation advances."""
    cfg = tracker.cfg
    if not (growth_ratio >= plan.merge_threshold and not is_local_growth_continuation(iou, dist_norm, cfg)):
        return False
    det_segment = ctx.dets.segments[det_idx]
    support_parents = select_match_growth_merge_support_parents(
        tid, det_idx, det_segment, det_area, tracker.tracks, ctx.assigned_tracks, ctx.merged_parents,
        ctx.candidates.get(det_idx, []), plan.best_det_for_track, cfg)
    if not support_parents:
        return False
    track = tracker.tracks[tid]
    detected_parents = [tid] + support_parents
    merged_gen = max(tracker.tracks[pid].gen for pid in detected_parents) + 1
    track.missed = 0
    track.gen = merged_gen
    track.segment = det_segment
    track.area = det_area
    reset_merge_candidate_state(track)
    ctx.assigned_tracks.add(tid)
    ctx.assigned_dets.add(det_idx)
    ctx.merged_children.add(tid)
    ctx.events.append(events.rescue_merge(
        tid, merged_gen, support_parents, detected_parents, ctx.dets.det_ids[det_idx],
        growth_ratio, plan.merge_threshold))
    for pid in support_parents:
        ctx.merged_parents.add(pid)
        ctx.assigned_tracks.add(pid)
    return True


def _record_growth_evidence(tracker: "Tracker", ctx: "FrameContext", plan: "PairPlan", track: Track,
                            tid: int, det_id: int, iou: float, dist_norm: float, det_area: float,
                            growth_ratio: float) -> None:
    """Growth no competing track explains: ignore contour wobble; otherwise
    accumulate evidence, promoting to an unknown-parent merge once strong."""
    cfg = tracker.cfg
    pg = cfg.persistent_growth
    if iou >= pg.wobble_min_iou and dist_norm <= pg.wobble_max_dist_norm:
        # Treat high-overlap low-motion growth as contour wobble, not merge evidence.
        reset_merge_candidate_state(track)
        ctx.events.append(events.wobble_suppressed(tid, det_id, iou, dist_norm, growth_ratio))
        return

    streak, candidate_score = register_merge_candidate(track, ctx.frame_id, growth_ratio, plan.merge_threshold, cfg)
    if (
        pg.enabled
        and streak >= max(pg.min_streak, pg.min_events)
        and candidate_score >= pg.promotion_min_score
        and track.merge_candidate_peak_area_ratio >= pg.promotion_min_area_ratio
    ):
        track.gen += 1
        ctx.merged_children.add(tid)
        unknown_parent_count = estimate_unknown_parent_count(det_area, [tid], tracker.tracks, cfg)
        if unknown_parent_count <= 0:
            unknown_parent_count = 1
        ctx.events.append(events.persistent_growth_merge(
            tid, track.gen, unknown_parent_count, det_id, growth_ratio, plan.merge_threshold,
            streak, candidate_score, track.merge_candidate_peak_area_ratio))
        reset_merge_candidate_state(track)
        return
    ctx.events.append(events.merge_candidate(
        tid, det_id, iou, dist_norm, growth_ratio, plan.merge_threshold,
        streak, candidate_score, track.merge_candidate_peak_area_ratio))
