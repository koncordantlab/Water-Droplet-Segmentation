"""Association stage of a tracker step: candidate track/detection pairs,
same-frame direct merges (one detection covering two live tracks), and the
scored one-to-one pair list with the frame's adaptive growth thresholds."""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple

from tracking import events
from tracking.matching import (
    build_match_candidates,
    estimate_adaptive_growth_threshold,
    is_valid_match,
    select_merge_parents,
)

if TYPE_CHECKING:
    from tracking.tracker import FrameContext, Tracker

ScoredPair = Tuple[float, int, int, float, float]  # (score, track_id, det_idx, iou, dist_norm)


@dataclass
class PairPlan:
    """The frame's ranked candidate matches, mutual-best tables, and the two
    growth thresholds: merge_threshold makes a grown match a merge suspect,
    reject_threshold rejects (or promotes) it."""
    scored_pairs: List[ScoredPair]
    best_det_for_track: Dict[int, Tuple[int, float]]
    best_track_for_det: Dict[int, Tuple[int, float]]
    merge_threshold: float
    reject_threshold: float


def collect_candidates(tracker: "Tracker", ctx: "FrameContext") -> None:
    ctx.active_tracks = {
        tid: t for tid, t in tracker.tracks.items() if t.missed <= tracker.cfg.max_missed_frames
    }
    ctx.candidates = build_match_candidates(ctx.active_tracks, ctx.dets.segments, tracker.cfg, tracker.device)


def detect_direct_merges(tracker: "Tracker", ctx: "FrameContext") -> None:
    """A detection strongly overlapping >= 2 live tracks becomes a new child
    track (gen = max parent gen + 1); its parents retire this frame."""
    if not tracker.cfg.direct_merge.enabled:
        return
    for det_idx, cand_list in ctx.candidates.items():
        if det_idx in ctx.assigned_dets:
            continue
        det_segment = ctx.dets.segments[det_idx]
        det_area = ctx.dets.areas[det_idx]
        parent_ids = select_merge_parents(
            det_segment, det_area, cand_list, ctx.active_tracks, ctx.assigned_tracks, tracker.cfg)
        if len(parent_ids) < 2:
            continue
        child_gen = max(ctx.active_tracks[pid].gen for pid in parent_ids) + 1
        child_id = tracker.new_track(child_gen, det_area, det_segment)
        ctx.events.append(events.merge(child_id, child_gen, parent_ids, ctx.dets.det_ids[det_idx]))
        ctx.merged_children.add(child_id)
        ctx.assigned_tracks.add(child_id)
        for pid in parent_ids:
            ctx.merged_parents.add(pid)
            ctx.assigned_tracks.add(pid)
        ctx.assigned_dets.add(det_idx)


def plan_pairs(tracker: "Tracker", ctx: "FrameContext") -> PairPlan:
    cfg = tracker.cfg
    scored_pairs: List[ScoredPair] = []
    for det_idx, cand_list in ctx.candidates.items():
        if det_idx in ctx.assigned_dets:
            continue
        for tid, iou, dist_norm in cand_list:
            if tid in ctx.assigned_tracks or tid in ctx.merged_parents:
                continue
            det_area = ctx.dets.areas[det_idx]
            if not is_valid_match(tracker.tracks[tid], det_area, iou, dist_norm, cfg):
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

    merge_threshold = estimate_adaptive_growth_threshold(scored_pairs, tracker.tracks, ctx.dets.areas, cfg)
    reject_threshold = max(cfg.unknown_merge.match_area_growth, merge_threshold + 0.35)
    return PairPlan(scored_pairs, best_det_for_track, best_track_for_det, merge_threshold, reject_threshold)
