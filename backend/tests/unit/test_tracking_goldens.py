"""Local refactor gate: the tracker must reproduce, byte for byte, the logs
the PRE-refactor code recorded under 11 settings variants x 2 real inputs
(backend/output/tracking_refactor_goldens; recorded by its record_goldens.py,
provenance in manifest.json). CUDA box only: on these inputs the CPU and
CUDA prefilters order equal-decision events differently. Point
TRACKING_GOLDENS_DIR at the set when running from a worktree."""
import collections
import filecmp
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.local

_DEFAULT_DIR = Path(__file__).resolve().parents[2] / "output" / "tracking_refactor_goldens"
GOLDENS = Path(os.environ.get("TRACKING_GOLDENS_DIR", _DEFAULT_DIR))
INPUTS = ("freeze10", "march20")

# The variants record_goldens.py recorded, in TrackingConfig override form.
VARIANT_OVERRIDES = {
    "default": {},
    "short_memory": {"max_missed_frames": 2},
    "growth_open": {"match": {"max_area_ratio": 3.0, "continuation_min_iou": 0.40,
                              "continuation_max_dist_norm": 0.30}},
    "growth_reject": {"match": {"max_area_ratio": 3.0, "continuation_min_iou": 0.40,
                                "continuation_max_dist_norm": 0.30},
                      "unknown_merge": {"promote_match_growth": False}},
    "continuation_open": {"match": {"continuation_min_iou": 0.40, "continuation_max_dist_norm": 0.30}},
    "persistent_easy": {"persistent_growth": {"min_streak": 1, "min_events": 1,
                                              "promotion_min_score": 1.0,
                                              "promotion_min_area_ratio": 1.0}},
    "shapes": {"segments": {"filter_non_circular": True},
               "unknown_merge": {"promote_large_birth": False}},
    "shapes_drop": {"segments": {"filter_non_circular": True,
                                 "keep_non_circular_for_association": False}},
    "detectors_off": {"direct_merge": {"enabled": False}, "inferred_merge": {"enabled": False},
                      "unknown_merge": {"enabled": False}, "growth_rescue": {"enabled": False},
                      "persistent_growth": {"enabled": False},
                      "match": {"require_mutual_best": False}},
    "cpu_prefilter": {"candidates": {"use_gpu": False}},
    "topk_lostparent": {"candidates": {"max_per_detection": 3},
                        "inferred_merge": {"require_lost_parent": True}},
}
EXPECTED_EVENTS = {
    "birth/", "death/", "match/", "match_growth/", "match_growth/wobble_suppressed_candidate",
    "match_rejected_growth/adaptive_growth_reject", "merge/",
    "merge_candidate/growth_without_competing_parent",
    "inferred_merge/match_growth_with_competing_parent",
    "inferred_merge/persistent_growth_without_support",
    "inferred_merge/inferred_multi_parent_unmatched",
    "inferred_merge/single_parent_inferred_from_lost",
    "inferred_merge/single_parent_inferred_unmatched_active",
    "inferred_merge_unknown/matched_area_growth",
    "inferred_merge_unknown/large_unmatched_detection",
    "large_birth/large_unmatched_detection",
    "shape_rejected/non_circular_unmatched_detection",
}


def _manifest():
    path = GOLDENS / "manifest.json"
    if not path.exists():
        if "TRACKING_GOLDENS_DIR" in os.environ:
            pytest.fail(f"TRACKING_GOLDENS_DIR={GOLDENS} holds no manifest.json")
        pytest.skip(f"no recorded reference set at {GOLDENS}")
    return json.loads(path.read_text())


def _first_difference(got_path, want_path):
    got = json.loads(Path(got_path).read_text())
    want = json.loads(Path(want_path).read_text())
    if len(got) != len(want):
        return f"frame count {len(got)} vs golden {len(want)}"
    for g, w in zip(got, want):
        for key in ("frame", "detections", "tracks", "events"):
            if g.get(key) != w.get(key):
                if key == "events":
                    for i, (ge, we) in enumerate(zip(g["events"], w["events"])):
                        if ge != we:
                            return f"frame {w['frame']} event #{i}: got {ge} want {we}"
                    return f"frame {w['frame']}: {len(g['events'])} events vs golden {len(w['events'])}"
                return f"frame {w.get('frame')}: '{key}' differs"
    return "logs parse equal; difference is in serialization bytes (key order or float formatting)"


def test_manifest_matches_the_variant_table():
    manifest = _manifest()
    assert set(manifest["variants"]) == set(VARIANT_OVERRIDES)
    assert set(manifest["inputs"]) == set(INPUTS)


def test_reference_logs_cover_every_event_variant():
    _manifest()
    seen = set()
    for inp in INPUTS:
        for variant in VARIANT_OVERRIDES:
            frames = json.loads((GOLDENS / inp / variant / "tracking_log.json").read_text())
            seen.update(collections.Counter(
                f"{e['type']}/{e.get('reason', '')}" for f in frames for e in f["events"]))
    assert seen == EXPECTED_EVENTS


@pytest.mark.parametrize("variant", sorted(VARIANT_OVERRIDES))
@pytest.mark.parametrize("inp", INPUTS)
def test_tracker_reproduces_reference_log(inp, variant, tmp_path):
    _manifest()
    from tracking.config import DEFAULT_CONFIG
    from tracking.track import track_from_detections_json

    cfg = DEFAULT_CONFIG.with_overrides(VARIANT_OVERRIDES[variant])
    golden = GOLDENS / inp / variant
    log = tmp_path / "tracking_log.json"
    track_from_detections_json(str(GOLDENS / "inputs" / f"{inp}.mp4"),
                               str(GOLDENS / "inputs" / f"{inp}_detections.json"),
                               str(tmp_path / "out.mp4"), str(log), config=cfg)
    assert filecmp.cmp(log, golden / "tracking_log.json", shallow=False), \
        _first_difference(log, golden / "tracking_log.json")
    assert filecmp.cmp(tmp_path / "tracking_log.csv", golden / "tracking_log.csv", shallow=False)
