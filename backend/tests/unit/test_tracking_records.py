"""Tier-1 tests for the tracker's data shapes: FrameDetections parsing and
the tracking-log event schema (key order is part of the log format that
analyze.py/plots.py parse)."""
import json

import numpy as np
import pytest

from tracking import events
from tracking.config import DEFAULT_CONFIG
from tracking.detections import FrameDetections, frame_detections_from_record


def _det(det_id, conf, x0=0, size=10, area=None):
    det = {"det_id": det_id, "conf": conf, "cls": 1,
           "segment": [[x0, 0], [x0 + size, 0], [x0 + size, size], [x0, size]]}
    if area is not None:
        det["area"] = area
    return det


def test_low_confidence_dropped_ids_and_order_kept():
    record = {"frame": 7, "detections": [_det(4, 0.9), _det(5, 0.3), _det(6, 0.8, x0=50, area=123.0)]}
    dets = frame_detections_from_record(record, DEFAULT_CONFIG)
    assert dets.det_ids == [4, 6]
    assert dets.areas == [100.0, 123.0]  # polygon area when "area" is absent
    assert dets.confs == [0.9, 0.8]
    assert dets.classes == [1, 1]
    assert len(dets) == 2
    assert {len(c) for c in (dets.segments, dets.areas, dets.confs, dets.classes, dets.det_ids)} == {2}


def test_missing_segment_is_an_error_when_segmentation_required():
    record = {"frame": 7, "detections": [{"det_id": 9, "conf": 0.9, "cls": 0}]}
    with pytest.raises(RuntimeError, match=r"frame=7, det_id=9"):
        frame_detections_from_record(record, DEFAULT_CONFIG)


def test_append_keeps_columns_aligned():
    dets = FrameDetections()
    dets.append(None, 1.0, 0.5, 0, 3)
    assert (len(dets), dets.det_ids, dets.areas) == (1, [3], [1.0])


@pytest.mark.parametrize("event, keys", [
    (events.birth(1, 1, 0), ["type", "track_id", "gen", "det_id"]),
    (events.death(1, 2), ["type", "track_id", "gen"]),
    (events.merge(3, 2, [1, 2], 0), ["type", "track_id", "gen", "parents", "det_id"]),
    (events.match(1, 0, 0.5, 0.1), ["type", "track_id", "det_id", "iou", "dist_norm"]),
    (events.match_growth(1, 0, 0.9, 0.05, 1.4),
     ["type", "track_id", "det_id", "iou", "dist_norm", "area_ratio"]),
    (events.wobble_suppressed(1, 0, 0.9, 0.05, 1.2),
     ["type", "track_id", "det_id", "iou", "dist_norm", "area_ratio", "reason"]),
    (events.growth_rejected(1, 0, 0.5, 0.1, 1.9),
     ["type", "track_id", "det_id", "iou", "dist_norm", "area_ratio", "reason"]),
    (events.merge_candidate(1, 0, 0.5, 0.1, 1.2, 1.1, 1, 1.5, 1.2),
     ["type", "track_id", "det_id", "iou", "dist_norm", "area_ratio", "growth_threshold",
      "streak", "candidate_score", "candidate_peak_area_ratio", "reason"]),
    (events.rescue_merge(1, 2, [2], [1, 2], 0, 1.3, 1.1),
     ["type", "track_id", "gen", "parents", "detected_parents", "unknown_parents", "det_id",
      "reason", "area_ratio", "growth_threshold"]),
    (events.persistent_growth_merge(1, 2, 1, 0, 1.4, 1.1, 3, 4.5, 1.4),
     ["type", "track_id", "gen", "parents", "detected_parents", "unknown_parents", "det_id",
      "reason", "area_ratio", "growth_threshold", "streak", "candidate_score",
      "candidate_peak_area_ratio"]),
    (events.matched_growth_unknown(1, 2, 0, 1.7),
     ["type", "track_id", "gen", "parents", "det_id", "reason", "area_ratio"]),
    (events.unmatched_inferred_merge(1, 2, [3], [1, 3], 0, 0, "inferred_multi_parent_unmatched", 1.5),
     ["type", "track_id", "gen", "parents", "detected_parents", "unknown_parents", "det_id",
      "reason", "area_ratio"]),
    (events.large_unmatched(True, 1, 2, 0, 2.0, 100.0),
     ["type", "track_id", "gen", "det_id", "reason", "area_ratio", "reference_area", "parents"]),
    (events.large_unmatched(False, 1, 1, 0, 2.0, 100.0),
     ["type", "track_id", "gen", "det_id", "reason", "area_ratio", "reference_area"]),
    (events.shape_rejected(0), ["type", "det_id", "reason"]),
])
def test_event_key_order(event, keys):
    assert list(event) == keys


def test_event_numbers_are_plain_python():
    event = events.merge_candidate(1, 0, np.float32(0.5), np.float64(0.1), np.float32(1.2), 1.1,
                                   np.int64(2), np.float32(1.5), np.float32(1.2))
    json.dumps(event)
    assert type(event["streak"]) is int and type(event["iou"]) is float
    assert events.large_unmatched(True, 1, 2, 0, 2.0, 1.0)["type"] == "inferred_merge_unknown"
    assert events.large_unmatched(False, 1, 1, 0, 2.0, 1.0)["type"] == "large_birth"
