"""Tracking-log event records, one builder per event variant. This module IS
the tracking_log.json event schema: analyze.py and plots.py parse it, so key
order and the float()/int() conversions are part of the format (the refactor
reference logs pin them byte for byte)."""
from typing import List


def birth(track_id: int, gen: int, det_id: int) -> dict:
    return {"type": "birth", "track_id": track_id, "gen": gen, "det_id": det_id}


def death(track_id: int, gen: int) -> dict:
    return {"type": "death", "track_id": track_id, "gen": gen}


def merge(track_id: int, gen: int, parents: List[int], det_id: int) -> dict:
    """Direct merge: one detection covers >= 2 live tracks; a new child track."""
    return {"type": "merge", "track_id": track_id, "gen": gen, "parents": parents, "det_id": det_id}


def match(track_id: int, det_id: int, iou: float, dist_norm: float) -> dict:
    return {"type": "match", "track_id": track_id, "det_id": det_id,
            "iou": float(iou), "dist_norm": float(dist_norm)}


def match_growth(track_id: int, det_id: int, iou: float, dist_norm: float, area_ratio: float) -> dict:
    """A match accepted through the near-stationary growth-continuation exception."""
    return {"type": "match_growth", "track_id": track_id, "det_id": det_id,
            "iou": float(iou), "dist_norm": float(dist_norm), "area_ratio": float(area_ratio)}


def wobble_suppressed(track_id: int, det_id: int, iou: float, dist_norm: float, area_ratio: float) -> dict:
    """Growth with high overlap and almost no motion: contour wobble, not merge evidence."""
    return {"type": "match_growth", "track_id": track_id, "det_id": det_id,
            "iou": float(iou), "dist_norm": float(dist_norm), "area_ratio": float(area_ratio),
            "reason": "wobble_suppressed_candidate"}


def growth_rejected(track_id: int, det_id: int, iou: float, dist_norm: float, area_ratio: float) -> dict:
    return {"type": "match_rejected_growth", "track_id": track_id, "det_id": det_id,
            "iou": float(iou), "dist_norm": float(dist_norm), "area_ratio": float(area_ratio),
            "reason": "adaptive_growth_reject"}


def merge_candidate(track_id: int, det_id: int, iou: float, dist_norm: float, area_ratio: float,
                    growth_threshold: float, streak: int, candidate_score: float,
                    peak_area_ratio: float) -> dict:
    return {"type": "merge_candidate", "track_id": track_id, "det_id": det_id,
            "iou": float(iou), "dist_norm": float(dist_norm), "area_ratio": float(area_ratio),
            "growth_threshold": float(growth_threshold), "streak": int(streak),
            "candidate_score": float(candidate_score),
            "candidate_peak_area_ratio": float(peak_area_ratio),
            "reason": "growth_without_competing_parent"}


def rescue_merge(track_id: int, gen: int, parents: List[int], detected_parents: List[int],
                 det_id: int, area_ratio: float, growth_threshold: float) -> dict:
    """A grown match that a competing live track explains: the matched track absorbs it."""
    return {"type": "inferred_merge", "track_id": track_id, "gen": gen, "parents": parents,
            "detected_parents": detected_parents, "unknown_parents": 0, "det_id": det_id,
            "reason": "match_growth_with_competing_parent", "area_ratio": float(area_ratio),
            "growth_threshold": float(growth_threshold)}


def persistent_growth_merge(track_id: int, gen: int, unknown_parents: int, det_id: int,
                            area_ratio: float, growth_threshold: float, streak: int,
                            candidate_score: float, peak_area_ratio: float) -> dict:
    return {"type": "inferred_merge", "track_id": track_id, "gen": gen, "parents": [],
            "detected_parents": [track_id], "unknown_parents": int(unknown_parents),
            "det_id": det_id, "reason": "persistent_growth_without_support",
            "area_ratio": float(area_ratio), "growth_threshold": float(growth_threshold),
            "streak": int(streak), "candidate_score": float(candidate_score),
            "candidate_peak_area_ratio": float(peak_area_ratio)}


def matched_growth_unknown(track_id: int, gen: int, det_id: int, area_ratio: float) -> dict:
    return {"type": "inferred_merge_unknown", "track_id": track_id, "gen": gen, "parents": [],
            "det_id": det_id, "reason": "matched_area_growth", "area_ratio": float(area_ratio)}


def unmatched_inferred_merge(track_id: int, gen: int, parents: List[int], detected_parents: List[int],
                             unknown_parents: int, det_id: int, reason: str, area_ratio: float) -> dict:
    """An unmatched detection explained by recently lost tracks; track_id is the surviving parent."""
    return {"type": "inferred_merge", "track_id": track_id, "gen": gen, "parents": parents,
            "detected_parents": detected_parents, "unknown_parents": unknown_parents,
            "det_id": det_id, "reason": reason, "area_ratio": float(area_ratio)}


def large_unmatched(promoted: bool, track_id: int, gen: int, det_id: int, area_ratio: float,
                    reference_area: float) -> dict:
    """An unmatched detection much larger than a typical droplet."""
    record = {"type": "inferred_merge_unknown" if promoted else "large_birth",
              "track_id": track_id, "gen": gen, "det_id": det_id,
              "reason": "large_unmatched_detection", "area_ratio": float(area_ratio),
              "reference_area": float(reference_area)}
    if promoted:
        record["parents"] = []
    return record


def shape_rejected(det_id: int) -> dict:
    return {"type": "shape_rejected", "det_id": det_id, "reason": "non_circular_unmatched_detection"}
