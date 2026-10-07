"""The matcher suite: tensor-prefiltered candidate generation, the three
interdependent merge-parent selectors (direct/inferred/match-growth), adaptive
growth thresholds, and the match validity gates. Verbatim from the monolith."""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from tracking.config import TrackingConfig
from tracking.geometry import (
    _bbox_iou_matrix_torch,
    segment_bounds,
    segment_center,
    segment_distance,
    segment_distance_norm,
    segment_iou,
    segment_max_dim,
)
from tracking.tracks import Track

def resolve_prefilter_device(cfg: TrackingConfig) -> torch.device:
    """CUDA when cfg.candidates.use_gpu and a GPU is present, else CPU (the
    legacy TRACKING_PREFILTER_DEVICE, now resolved per run instead of at import)."""
    return torch.device("cuda" if cfg.candidates.use_gpu and torch.cuda.is_available() else "cpu")

def build_match_candidates(
    active_tracks: Dict[int, Track],
    detection_segments: List[Optional[np.ndarray]],
    cfg: TrackingConfig,
    device: torch.device,
) -> Dict[int, List[Tuple[int, float, float]]]:
    """Build candidate pairs using GPU/CPU tensor prefilter before exact segment IoU."""
    candidates: Dict[int, List[Tuple[int, float, float]]] = {i: [] for i in range(len(detection_segments))}
    if not active_tracks or not detection_segments:
        return candidates

    track_ids: List[int] = []
    track_segments: List[np.ndarray] = []
    for tid, track in active_tracks.items():
        if track.segment is None:
            continue
        track_ids.append(tid)
        track_segments.append(track.segment)

    det_indices: List[int] = []
    valid_det_segments: List[np.ndarray] = []
    for det_idx, segment in enumerate(detection_segments):
        if segment is None:
            continue
        det_indices.append(det_idx)
        valid_det_segments.append(segment)

    if not track_ids or not det_indices:
        return candidates

    track_centers = np.asarray([segment_center(seg) for seg in track_segments], dtype=np.float32)
    track_dims = np.asarray([segment_max_dim(seg) for seg in track_segments], dtype=np.float32)
    track_bounds = np.asarray([segment_bounds(seg) for seg in track_segments], dtype=np.float32)
    det_centers = np.asarray([segment_center(seg) for seg in valid_det_segments], dtype=np.float32)
    det_dims = np.asarray([segment_max_dim(seg) for seg in valid_det_segments], dtype=np.float32)
    det_bounds = np.asarray([segment_bounds(seg) for seg in valid_det_segments], dtype=np.float32)

    with torch.no_grad():
        t_centers = torch.as_tensor(track_centers, device=device)
        d_centers = torch.as_tensor(det_centers, device=device)
        t_dims = torch.as_tensor(track_dims, device=device)
        d_dims = torch.as_tensor(det_dims, device=device)
        t_bounds = torch.as_tensor(track_bounds, device=device)
        d_bounds = torch.as_tensor(det_bounds, device=device)

        dist = torch.cdist(t_centers, d_centers, p=2.0)
        denom = torch.maximum(t_dims[:, None], d_dims[None, :]).clamp_min(1.0)
        dist_norm = dist / denom
        bbox_iou = _bbox_iou_matrix_torch(t_bounds, d_bounds)
        pre_mask = torch.logical_or(bbox_iou >= cfg.candidates.iou_threshold, dist_norm <= cfg.candidates.dist_factor)

        dist_norm_cpu = dist_norm.cpu().numpy()
        pre_mask_cpu = pre_mask.cpu().numpy()

    for col, det_idx in enumerate(det_indices):
        row_indices = np.flatnonzero(pre_mask_cpu[:, col])
        if row_indices.size == 0:
            continue

        if (
            cfg.candidates.max_per_detection > 0
            and row_indices.size > cfg.candidates.max_per_detection
        ):
            order = np.argsort(dist_norm_cpu[row_indices, col])[:cfg.candidates.max_per_detection]
            row_indices = row_indices[order]

        det_segment = valid_det_segments[col]
        det_candidates: List[Tuple[int, float, float]] = []
        for row in row_indices:
            tid = track_ids[row]
            track = active_tracks[tid]
            iou = effective_match_iou(track, det_segment, cfg)
            dist_norm_val = float(dist_norm_cpu[row, col])
            if iou >= cfg.candidates.iou_threshold or dist_norm_val <= cfg.candidates.dist_factor:
                det_candidates.append((tid, iou, dist_norm_val))

        det_candidates.sort(key=lambda item: (-item[1], item[2]))
        candidates[det_idx] = det_candidates

    return candidates

def select_merge_parents(
    det_segment: Optional[np.ndarray],
    det_area: float,
    cand_list: List[Tuple[int, float, float]],
    active_tracks: Dict[int, Track],
    assigned_tracks: set,
    cfg: TrackingConfig,
) -> List[int]:
    """Return exactly two parent track IDs when a merge hypothesis is strong enough."""
    if len(cand_list) < 2:
        return []
    max_det_dim = segment_max_dim(det_segment)
    strong_candidates: List[Tuple[int, float, float, float]] = []

    for tid, iou, dist_norm in cand_list:
        if tid in assigned_tracks:
            continue
        if iou < cfg.direct_merge.min_iou or dist_norm > cfg.direct_merge.max_dist_norm:
            continue

        parent_area = active_tracks[tid].area
        if det_area < cfg.direct_merge.area_factor * parent_area:
            continue

        parent_det_dist = segment_distance(active_tracks[tid].segment, det_segment)
        if parent_det_dist > cfg.direct_merge.parent_dist_factor * max_det_dim:
            continue

        strong_candidates.append((tid, iou, parent_area, dist_norm))

    if len(strong_candidates) < 2:
        return []

    strong_candidates.sort(key=lambda item: item[1], reverse=True)
    selected: List[Tuple[int, float, float, float]] = []
    combined_parent_area = 0.0
    for candidate in strong_candidates:
        parent_area = candidate[2]
        if not selected:
            selected.append(candidate)
            combined_parent_area = parent_area
            continue
        if det_area >= cfg.direct_merge.combined_area_factor * (combined_parent_area + parent_area):
            selected.append(candidate)
            combined_parent_area += parent_area

    if len(selected) < 2:
        return []

    # Ensure second parent has enough support: overlap or close proximity.
    second_parent_iou = selected[1][1]
    second_parent_dist_norm = selected[1][3]
    if (
        second_parent_iou < cfg.direct_merge.second_parent_min_iou
        and second_parent_dist_norm > cfg.direct_merge.second_parent_max_dist_norm
    ):
        return []
    if (selected[0][1] + selected[1][1]) < cfg.direct_merge.min_combined_parent_iou:
        return []

    # Parents must be reasonably close to each other for a true merge.
    seg_a = active_tracks[selected[0][0]].segment
    seg_b = active_tracks[selected[1][0]].segment
    parent_pair_dist_norm = segment_distance(seg_a, seg_b) / max(
        max_det_dim,
        segment_max_dim(seg_a),
        segment_max_dim(seg_b),
    )
    if parent_pair_dist_norm > cfg.direct_merge.max_parent_pair_dist_norm:
        return []

    return [item[0] for item in selected]

def select_inferred_merge_parents(
    det_segment: Optional[np.ndarray],
    det_area: float,
    tracked_objects: Dict[int, Track],
    assigned_tracks: set,
    merged_parents: set,
    cfg: TrackingConfig,
    allow_single_parent: bool = True,
) -> List[int]:
    """Find recently lost tracks that likely formed a larger merged detection.

    Uses effective missed count (current missed + this frame) so tracks that
    become lost in the current frame can still be considered as parent candidates.
    """
    candidates: List[Tuple[int, int, float, float, float, bool]] = []

    for tid, track in tracked_objects.items():
        if tid in assigned_tracks or tid in merged_parents:
            continue
        was_lost = track.missed > 0
        effective_missed = track.missed + 1
        if effective_missed > cfg.inferred_max_missed:
            continue
        if was_lost and effective_missed > cfg.inferred_merge.lost_parent_max_effective_missed:
            continue

        parent_area = track.area
        if det_area < cfg.inferred_merge.area_factor * parent_area:
            continue

        iou = effective_match_iou(track, det_segment, cfg)
        dist_norm = segment_distance_norm(track.segment, det_segment)
        if iou < cfg.inferred_merge.min_iou and dist_norm > min(cfg.inferred_merge.max_dist_norm, cfg.inferred_merge.parent_max_dist_norm):
            continue

        candidates.append((tid, effective_missed, dist_norm, iou, parent_area, was_lost))

    if len(candidates) < cfg.inferred_merge.min_parents:
        return []

    # Prefer already-lost parents over currently active unmatched tracks.
    candidates.sort(key=lambda item: (-int(item[5]), item[1], item[2], -item[3]))
    chosen: List[Tuple[int, int, float, float, float, bool]] = []
    combined_parent_area = 0.0
    for candidate in candidates:
        parent_area = candidate[4]
        if not chosen:
            chosen.append(candidate)
            combined_parent_area = parent_area
            continue
        if det_area >= cfg.inferred_merge.combined_area_factor * (combined_parent_area + parent_area):
            chosen.append(candidate)
            combined_parent_area += parent_area

    if not chosen:
        return []

    # Multi-parent inferred merge.
    if len(chosen) >= 2:
        if cfg.inferred_merge.require_lost_parent and not any(item[5] for item in chosen):
            return []
        if not any(item[3] >= cfg.inferred_merge.anchor_min_iou or item[2] <= cfg.inferred_merge.anchor_max_dist_norm for item in chosen):
            return []
        return [item[0] for item in chosen]

    if not allow_single_parent:
        return []

    # Single detected parent with significant growth implies hidden/undetected merge contributors.
    single_parent_effective_missed = chosen[0][1]
    single_parent_area = chosen[0][4]
    single_parent_iou = chosen[0][3]
    single_parent_dist_norm = chosen[0][2]
    single_parent_was_lost = chosen[0][5]
    if det_area < cfg.single_parent.min_area_growth * single_parent_area:
        return []
    # Avoid reviving very old lost tracks as self-only inferred merges.
    if single_parent_effective_missed > cfg.single_parent.max_effective_missed:
        return []
    if (
        single_parent_was_lost
        and single_parent_effective_missed > cfg.single_parent.lost_max_effective_missed
    ):
        return []
    if single_parent_was_lost and (
        single_parent_iou < cfg.single_parent.lost_min_iou
        or single_parent_dist_norm > cfg.single_parent.lost_max_dist_norm
    ):
        return []
    if single_parent_iou < cfg.single_parent.min_iou and single_parent_dist_norm > cfg.single_parent.max_dist_norm:
        return []
    if not single_parent_was_lost:
        if det_area < cfg.single_parent.active_min_area_growth * single_parent_area:
            return []
        if (
            single_parent_iou < cfg.single_parent.active_min_iou
            or single_parent_dist_norm > cfg.single_parent.active_max_dist_norm
        ):
            return []
    # High-overlap, tiny-motion growth is usually a single-object size change, not a merge.
    if (
        single_parent_iou >= cfg.single_parent.growth_only_min_iou
        and single_parent_dist_norm <= cfg.single_parent.growth_only_max_dist_norm
    ):
        return []

    return [chosen[0][0]]

def select_match_growth_merge_support_parents(
    primary_tid: int,
    det_idx: int,
    det_segment: Optional[np.ndarray],
    det_area: float,
    tracked_objects: Dict[int, Track],
    assigned_tracks: set,
    merged_parents: set,
    cand_list_for_det: List[Tuple[int, float, float]],
    best_det_for_track: Dict[int, Tuple[int, float]],
    cfg: TrackingConfig,
) -> List[int]:
    """Find additional parent tracks that compete for the same grown detection."""
    candidates: List[Tuple[int, float, float, int, float]] = []
    for sid, iou, dist_norm in cand_list_for_det:
        if sid == primary_tid:
            continue
        if sid in assigned_tracks or sid in merged_parents:
            continue
        track = tracked_objects.get(sid)
        if track is None:
            continue
        effective_missed = track.missed + 1
        # Keep a broader temporal window, but penalize stale candidates by score.
        if effective_missed > cfg.inferred_max_missed:
            continue
        if track.area <= 0 or det_area < (cfg.growth_rescue.parent_area_factor * track.area):
            continue
        best = best_det_for_track.get(sid)
        if best is not None and best[0] != det_idx:
            score_here = iou - 0.1 * dist_norm
            # Relax near-best requirement when growth is strong and support is spatially plausible.
            allow_competing = (
                det_area >= ((cfg.growth_rescue.parent_area_factor + 0.10) * track.area)
                and dist_norm <= cfg.growth_rescue.strong_max_dist_norm
            )
            if not allow_competing and score_here + cfg.growth_rescue.support_score_margin < best[1]:
                continue
        if iou < cfg.match.reliable_min_iou and dist_norm > cfg.growth_rescue.support_max_dist_norm:
            continue
        area_ratio = area_growth_ratio(track.area, det_area)
        score = support_parent_score(iou, dist_norm, area_ratio, effective_missed, cfg)
        if score <= 0.0:
            continue
        candidates.append((sid, iou, dist_norm, effective_missed, score))

    if not candidates:
        return []

    candidates.sort(key=lambda item: (-item[4], -item[1], item[2], item[3]))
    top_iou = candidates[0][1]
    top_dist_norm = candidates[0][2]
    top_effective_missed = candidates[0][3]
    top_score = candidates[0][4]
    if top_iou < cfg.growth_rescue.strong_min_iou and top_dist_norm > cfg.growth_rescue.strong_max_dist_norm:
        return []
    if (
        top_effective_missed >= cfg.growth_rescue.max_effective_missed
        and (
            top_iou < cfg.growth_rescue.stale_strong_min_iou
            or top_dist_norm > cfg.growth_rescue.stale_strong_max_dist_norm
        )
    ):
        return []

    score_floor = max(0.20, top_score * 0.60)
    selected = [item for item in candidates if item[4] >= score_floor]
    if not selected:
        return []

    max_extra = max(1, cfg.growth_rescue.max_extra_parents)
    return [item[0] for item in selected[:max_extra]]


def reference_object_area(tracked_objects: Dict[int, Track], frame_median_area: float, cfg: TrackingConfig) -> float:
    """Estimate typical single-object area from stable active tracks."""
    stable_areas = []
    for track in tracked_objects.values():
        if track.missed != 0 or track.gen != 1:
            continue
        area = track.area
        if area > 0:
            stable_areas.append(area)

    if len(stable_areas) >= cfg.unknown_merge.baseline_min_tracks:
        return float(np.median(stable_areas))

    if frame_median_area > 0:
        return frame_median_area

    if stable_areas:
        return float(np.median(stable_areas))

    return 0.0

def area_growth_ratio(prev_area: float, new_area: float) -> float:
    if prev_area <= 0:
        return float("inf") if new_area > 0 else 1.0
    return new_area / prev_area

def robust_median_mad(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float32)
    median_val = float(np.median(arr))
    mad_val = float(np.median(np.abs(arr - median_val)))
    return median_val, mad_val

def estimate_adaptive_growth_threshold(
    scored_pairs: List[Tuple[float, int, int, float, float]],
    tracked_objects: Dict[int, Track],
    detection_areas: List[float],
    cfg: TrackingConfig,
) -> float:
    """Estimate growth threshold from likely continuation matches in the current frame."""
    best_per_track: Dict[int, Tuple[int, float, float]] = {}
    for _, tid, det_idx, iou, dist_norm in scored_pairs:
        if tid not in best_per_track:
            best_per_track[tid] = (det_idx, iou, dist_norm)

    baseline_growth: List[float] = []
    for tid, (det_idx, iou, dist_norm) in best_per_track.items():
        if iou < cfg.match.active_min_iou or dist_norm > cfg.match.active_max_dist_norm:
            continue
        if det_idx >= len(detection_areas):
            continue
        track = tracked_objects.get(tid)
        if track is None:
            continue
        if track.missed != 0:
            # Use only active-continuation behavior for adaptive baseline.
            continue
        ratio = area_growth_ratio(track.area, detection_areas[det_idx])
        if np.isfinite(ratio) and ratio > 0:
            baseline_growth.append(float(ratio))

    if len(baseline_growth) < cfg.adaptive_growth.min_samples:
        return cfg.adaptive_growth.fallback_threshold

    arr = np.asarray(baseline_growth, dtype=np.float32)
    median_growth, mad_growth = robust_median_mad(baseline_growth)
    robust_tail = median_growth + (cfg.adaptive_growth.mad_multiplier * max(mad_growth, 0.01))
    p97_tail = float(np.quantile(arr, 0.97))
    p99_tail = float(np.quantile(arr, 0.99))
    adaptive = max(
        robust_tail,
        p97_tail + 0.015,
        p99_tail + 0.008,
    )
    adaptive = max(cfg.adaptive_growth.min_threshold, adaptive)
    adaptive = min(adaptive, cfg.unknown_merge.match_area_growth)
    return float(adaptive)

def support_parent_score(iou: float, dist_norm: float, area_ratio: float, effective_missed: int,
                         cfg: TrackingConfig) -> float:
    """Score competing parent support with soft penalties for stale tracks."""
    overlap_score = max(0.0, min(1.0, iou))
    dist_score = max(0.0, 1.0 - (dist_norm / max(cfg.growth_rescue.support_max_dist_norm, 1e-6)))
    area_score = max(0.0, min(1.0, (area_ratio - cfg.growth_rescue.parent_area_factor) / 0.35))
    stale_penalty = min(0.55, 0.10 * max(0, effective_missed - 1))
    return (0.60 * overlap_score) + (0.25 * dist_score) + (0.15 * area_score) - stale_penalty

def is_unknown_merge_from_match(
    prev_area: float,
    new_area: float,
    cfg: TrackingConfig,
    growth_threshold: Optional[float] = None,
) -> Tuple[bool, float]:
    ratio = area_growth_ratio(prev_area, new_area)
    threshold = cfg.unknown_merge.match_area_growth if growth_threshold is None else growth_threshold
    return ratio >= threshold, ratio

def is_unknown_merge_from_birth(det_area: float, reference_area: float, cfg: TrackingConfig) -> Tuple[bool, float]:
    if reference_area <= 0:
        return False, 0.0
    ratio = det_area / reference_area
    return ratio >= cfg.unknown_merge.birth_median_factor, ratio

def area_ratio_values(area_a: float, area_b: float) -> float:
    if area_a <= 0 or area_b <= 0:
        return float("inf")
    return max(area_a, area_b) / min(area_a, area_b)

def is_local_growth_continuation(iou: float, dist_norm: float, cfg: TrackingConfig) -> bool:
    return iou >= cfg.match.continuation_min_iou and dist_norm <= cfg.match.continuation_max_dist_norm

def estimate_unknown_parent_count(
    det_area: float,
    known_parent_ids: List[int],
    tracked_objects: Dict[int, Track],
    cfg: TrackingConfig,
) -> int:
    if len(known_parent_ids) != 1:
        return 0
    parent_track = tracked_objects.get(known_parent_ids[0])
    if parent_track is None:
        return 0

    parent_area = parent_track.area
    if parent_area <= 0:
        return 0

    ratio = det_area / parent_area
    if ratio < cfg.single_parent.min_area_growth:
        return 0

    estimated = max(1, int(np.floor(ratio)) - 1)
    return min(cfg.inferred_merge.max_unknown_parent_estimate, estimated)

def is_reliable_match(iou: float, dist_norm: float, cfg: TrackingConfig) -> bool:
    return iou >= cfg.match.reliable_min_iou or dist_norm <= cfg.match.reliable_max_dist_norm

def is_valid_match(track: Track, det_area: float, iou: float, dist_norm: float, cfg: TrackingConfig) -> bool:
    """Conservative match gate to reduce ID carry-over between nearby objects."""
    if not is_reliable_match(iou, dist_norm, cfg):
        return False

    # For currently visible tracks, require at least moderate overlap.
    if track.missed == 0:
        if iou < cfg.match.active_min_iou:
            return False
        if dist_norm > cfg.match.active_max_dist_norm and iou < (cfg.match.active_min_iou + 0.10):
            return False
    else:
        if iou < cfg.match.lost_min_iou:
            return False
        if dist_norm > cfg.match.lost_max_dist_norm:
            return False
        if track.missed >= 2 and (iou < cfg.match.recovery_min_iou or dist_norm > cfg.match.recovery_max_dist_norm):
            return False

    # Reject drastic size changes as regular matches.
    track_area = track.area
    if area_ratio_values(track_area, det_area) > cfg.match.max_area_ratio:
        if det_area > track_area and is_local_growth_continuation(iou, dist_norm, cfg):
            return True
        return False

    return True

def effective_match_iou(
    track: Track,
    det_segment: Optional[np.ndarray],
    cfg: TrackingConfig,
) -> float:
    seg_iou = segment_iou(track.segment, det_segment, cfg.segments.iou_max_raster_pixels)
    if seg_iou is None:
        raise RuntimeError("Instance segmentation required for matching, but missing track/detection segment.")
    return float(seg_iou)
