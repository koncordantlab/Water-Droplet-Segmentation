"""The tracker core: Track state across frames and the per-frame step that
turns one frame's detections into tracking events. No video or file I/O --
orchestration (frames in, overlay and logs out) lives in tracking.track."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from tracking import events
from tracking.associate import collect_candidates, detect_direct_merges, plan_pairs
from tracking.config import DEFAULT_CONFIG, TrackingConfig
from tracking.detections import FrameDetections
from tracking.matched import resolve_matches
from tracking.matching import reference_object_area, resolve_prefilter_device
from tracking.tracks import Track, reset_merge_candidate_state
from tracking.unmatched import resolve_unmatched

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
            collect_candidates(self, ctx)
            detect_direct_merges(self, ctx)
            plan = plan_pairs(self, ctx)
            resolve_matches(self, ctx, plan)
            resolve_unmatched(self, ctx)
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
