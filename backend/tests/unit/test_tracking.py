"""Tier-1 tests for the backend/tracking package (port of the
1-object-tracking-algorithm-v01 scripts; behavior frozen at the JSON
boundaries — see the tracking_freeze goldens)."""
import subprocess
import sys

import numpy as np


def test_config_imports_without_side_effects():
    # Importing config must not load weights, open videos, or write files.
    # (torch.device probing is the one allowed exception, verbatim behavior.)
    code = "from tracking import config; print(config.MERGE_MIN_IOU)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        timeout=120, cwd=None,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "0.2"


def test_tuning_constants_pinned():
    from tracking import config
    # Spot-pin values across each constants family (the tuning surface).
    assert config.MAX_MISSED_FRAMES == 10
    assert config.IOU_MATCH_THRESHOLD == 0.05
    assert config.MERGE_COMBINED_AREA_FACTOR == 0.85
    assert config.INFERRED_SINGLE_PARENT_ACTIVE_MIN_IOU == 0.60
    assert config.MATCH_GROWTH_MERGE_MIN_AREA_GROWTH == 1.12
    assert config.MIN_SEGMENT_CONFIDENCE == 0.50
    assert config.SEGMENT_IOU_MAX_RASTER_PIXELS == 250_000


def _square_segment(x0, y0, size):
    return np.array([[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size]],
                    dtype=np.float32)


def test_geometry_known_values():
    from tracking import geometry
    sq = _square_segment(0, 0, 10)
    assert geometry.segment_area(sq) == 100.0
    assert geometry.segment_center(sq) == (5.0, 5.0)
    assert geometry.segment_max_dim(sq) == 10.0
    # identical squares -> IoU 1.0; disjoint -> 0.0
    assert geometry.segment_iou(sq, sq) == 1.0
    far = _square_segment(100, 100, 10)
    assert geometry.segment_iou(sq, far) == 0.0
    # circularity of a square is 4*pi*A/P^2 = pi/4
    assert abs(geometry.segment_circularity(sq) - np.pi / 4) < 1e-6


def test_json_array_writer_roundtrip(tmp_path):
    import json
    from tracking.io import JsonArrayWriter
    p = tmp_path / "arr.json"
    w = JsonArrayWriter(str(p))
    w.write({"a": 1})
    w.write({"b": [1, 2]})
    w.close()
    assert json.loads(p.read_text()) == [{"a": 1}, {"b": [1, 2]}]


def test_build_match_candidates_smoke():
    """One live track + one overlapping detection -> exactly one candidate
    pair (0, track_id); a far detection yields no pair for that track."""
    from tracking.matching import build_match_candidates
    from tracking.tracks import Track
    near = _square_segment(0, 0, 10)
    far = _square_segment(500, 500, 10)
    track = Track(gen=1, area=100.0, segment=near)
    tracked = {7: track}
    dets = [near.copy(), far]
    # Signature is (active_tracks, detection_segments) and the return is
    # {det_idx: [(tid, iou, dist_norm), ...]} — not a flat pair list.
    candidates = build_match_candidates(tracked, dets)
    pairs = {(d, t) for d, cand_list in candidates.items() for t, *_ in cand_list}
    assert (0, 7) in pairs and (1, 7) not in pairs
