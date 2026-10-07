"""Per-object track state: the Track dataclass plus the persistent-growth
merge-candidate evidence helpers (reset/decay/register). Verbatim from the
monolith."""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from tracking.config import TrackingConfig

@dataclass
class Track:
    gen: int
    missed: int = 0
    area: float = 0.0
    segment: Optional[np.ndarray] = None
    merge_candidate_streak: int = 0
    merge_candidate_score: float = 0.0
    last_merge_candidate_frame: int = 0
    merge_candidate_peak_area_ratio: float = 0.0

def reset_merge_candidate_state(track: Track) -> None:
    track.merge_candidate_streak = 0
    track.merge_candidate_score = 0.0
    track.last_merge_candidate_frame = 0
    track.merge_candidate_peak_area_ratio = 0.0

def decay_merge_candidate_state(track: Track, frame_id: int, cfg: "TrackingConfig") -> None:
    if track.merge_candidate_score <= 0.0:
        return
    last_frame = track.last_merge_candidate_frame
    if last_frame <= 0:
        reset_merge_candidate_state(track)
        return
    frame_gap = frame_id - last_frame
    if frame_gap <= 0:
        return
    if frame_gap > cfg.persistent_growth.max_frame_gap:
        reset_merge_candidate_state(track)
        return
    track.merge_candidate_score *= cfg.persistent_growth.evidence_decay ** frame_gap
    track.last_merge_candidate_frame = frame_id
    if track.merge_candidate_score < 0.20:
        reset_merge_candidate_state(track)

def register_merge_candidate(
    track: Track,
    frame_id: int,
    growth_ratio: float,
    growth_threshold: float,
    cfg: "TrackingConfig",
) -> Tuple[int, float]:
    """Accumulate growth evidence over short windows (not strictly consecutive frames)."""
    if (
        track.last_merge_candidate_frame > 0
        and frame_id - track.last_merge_candidate_frame > cfg.persistent_growth.max_frame_gap
    ):
        reset_merge_candidate_state(track)

    growth_excess = max(0.0, growth_ratio - growth_threshold)
    score_increment = 1.0 + min(2.0, growth_excess / 0.08)
    track.merge_candidate_score += score_increment
    track.merge_candidate_streak += 1
    track.last_merge_candidate_frame = frame_id
    track.merge_candidate_peak_area_ratio = max(track.merge_candidate_peak_area_ratio, growth_ratio)
    return track.merge_candidate_streak, track.merge_candidate_score
