"""Tier-1 tests for TrackingConfig: the 1:1 migration of the legacy
UPPER_CASE constants (frozen in fixtures/tracking/legacy_constants.json),
strict overrides, and the JSON round trip."""
import dataclasses
import json
import re
from pathlib import Path

import pytest

from tracking.config import DEFAULT_CONFIG, TrackingConfig, load_config

FIXTURE = Path(__file__).parent.parent / "fixtures" / "tracking" / "legacy_constants.json"

# Legacy constant -> TrackingConfig attribute path. This table is also the
# substitution table for threading cfg through the modules (legacy NAME
# becomes cfg.<path>).
LEGACY_FIELD_MAP = {
    "MAX_MISSED_FRAMES": "max_missed_frames",
    "IOU_MATCH_THRESHOLD": "candidates.iou_threshold",
    "DIST_MATCH_FACTOR": "candidates.dist_factor",
    "TRACKING_USE_GPU_PREFILTER": "candidates.use_gpu",
    "TRACKING_PREFILTER_MAX_CANDIDATES_PER_DET": "candidates.max_per_detection",
    "MIN_RELIABLE_MATCH_IOU": "match.reliable_min_iou",
    "MAX_RELIABLE_MATCH_DIST_NORM": "match.reliable_max_dist_norm",
    "ACTIVE_TRACK_MIN_IOU": "match.active_min_iou",
    "ACTIVE_TRACK_MAX_DIST_NORM": "match.active_max_dist_norm",
    "LOST_TRACK_MIN_IOU": "match.lost_min_iou",
    "LOST_TRACK_MAX_DIST_NORM": "match.lost_max_dist_norm",
    "LOST_TRACK_RECOVERY_MIN_IOU": "match.recovery_min_iou",
    "LOST_TRACK_RECOVERY_MAX_DIST_NORM": "match.recovery_max_dist_norm",
    "MAX_MATCH_AREA_RATIO": "match.max_area_ratio",
    "GROWTH_CONTINUATION_MIN_IOU": "match.continuation_min_iou",
    "GROWTH_CONTINUATION_MAX_DIST_NORM": "match.continuation_max_dist_norm",
    "REQUIRE_MUTUAL_BEST_MATCH": "match.require_mutual_best",
    "ENABLE_MERGE_DETECTION": "direct_merge.enabled",
    "MERGE_AREA_FACTOR": "direct_merge.area_factor",
    "MERGE_COMBINED_AREA_FACTOR": "direct_merge.combined_area_factor",
    "MERGE_MIN_IOU": "direct_merge.min_iou",
    "MERGE_MAX_DIST_NORM": "direct_merge.max_dist_norm",
    "MERGE_PARENT_DIST_FACTOR": "direct_merge.parent_dist_factor",
    "MERGE_SECOND_PARENT_MIN_IOU": "direct_merge.second_parent_min_iou",
    "MERGE_SECOND_PARENT_MAX_DIST_NORM": "direct_merge.second_parent_max_dist_norm",
    "MERGE_MIN_COMBINED_PARENT_IOU": "direct_merge.min_combined_parent_iou",
    "MERGE_MAX_PARENT_PAIR_DIST_NORM": "direct_merge.max_parent_pair_dist_norm",
    "ENABLE_INFERRED_MERGE": "inferred_merge.enabled",
    "INFERRED_MERGE_MIN_PARENTS": "inferred_merge.min_parents",
    "INFERRED_MERGE_MAX_MISSED": "inferred_max_missed",
    "INFERRED_MERGE_AREA_FACTOR": "inferred_merge.area_factor",
    "INFERRED_MERGE_COMBINED_AREA_FACTOR": "inferred_merge.combined_area_factor",
    "INFERRED_MERGE_MIN_IOU": "inferred_merge.min_iou",
    "INFERRED_MERGE_MAX_DIST_NORM": "inferred_merge.max_dist_norm",
    "INFERRED_MERGE_PARENT_MAX_DIST_NORM": "inferred_merge.parent_max_dist_norm",
    "INFERRED_MERGE_ANCHOR_MIN_IOU": "inferred_merge.anchor_min_iou",
    "INFERRED_MERGE_ANCHOR_MAX_DIST_NORM": "inferred_merge.anchor_max_dist_norm",
    "INFERRED_MERGE_REQUIRE_LOST_PARENT": "inferred_merge.require_lost_parent",
    "INFERRED_MERGE_WARMUP_FRAMES": "inferred_merge.warmup_frames",
    "INFERRED_MERGE_LOST_PARENT_MAX_EFFECTIVE_MISSED": "inferred_merge.lost_parent_max_effective_missed",
    "MAX_UNKNOWN_PARENT_COUNT_ESTIMATE": "inferred_merge.max_unknown_parent_estimate",
    "INFERRED_SINGLE_PARENT_MIN_AREA_GROWTH": "single_parent.min_area_growth",
    "INFERRED_SINGLE_PARENT_MIN_IOU": "single_parent.min_iou",
    "INFERRED_SINGLE_PARENT_MAX_DIST_NORM": "single_parent.max_dist_norm",
    "INFERRED_SINGLE_PARENT_ACTIVE_MIN_AREA_GROWTH": "single_parent.active_min_area_growth",
    "INFERRED_SINGLE_PARENT_ACTIVE_MIN_IOU": "single_parent.active_min_iou",
    "INFERRED_SINGLE_PARENT_ACTIVE_MAX_DIST_NORM": "single_parent.active_max_dist_norm",
    "INFERRED_SINGLE_PARENT_GROWTH_ONLY_IOU": "single_parent.growth_only_min_iou",
    "INFERRED_SINGLE_PARENT_GROWTH_ONLY_MAX_DIST_NORM": "single_parent.growth_only_max_dist_norm",
    "INFERRED_SINGLE_PARENT_MAX_EFFECTIVE_MISSED": "single_parent.max_effective_missed",
    "INFERRED_SINGLE_PARENT_LOST_MAX_EFFECTIVE_MISSED": "single_parent.lost_max_effective_missed",
    "INFERRED_SINGLE_PARENT_LOST_MIN_IOU": "single_parent.lost_min_iou",
    "INFERRED_SINGLE_PARENT_LOST_MAX_DIST_NORM": "single_parent.lost_max_dist_norm",
    "ENABLE_INFERRED_UNKNOWN_MERGE": "unknown_merge.enabled",
    "INFERRED_UNKNOWN_MATCH_AREA_GROWTH": "unknown_merge.match_area_growth",
    "INFERRED_UNKNOWN_BIRTH_MEDIAN_FACTOR": "unknown_merge.birth_median_factor",
    "UNKNOWN_MERGE_BASELINE_MIN_TRACKS": "unknown_merge.baseline_min_tracks",
    "PROMOTE_LARGE_BIRTH_TO_UNKNOWN_MERGE": "unknown_merge.promote_large_birth",
    "PROMOTE_MATCH_GROWTH_TO_UNKNOWN_MERGE": "unknown_merge.promote_match_growth",
    "REJECT_MATCH_ON_LARGE_AREA_GROWTH": "unknown_merge.reject_large_growth_match",
    "ENABLE_MATCH_GROWTH_MERGE_RESCUE": "growth_rescue.enabled",
    "MATCH_GROWTH_MERGE_PARENT_AREA_FACTOR": "growth_rescue.parent_area_factor",
    "MATCH_GROWTH_MERGE_SUPPORT_MAX_DIST_NORM": "growth_rescue.support_max_dist_norm",
    "MATCH_GROWTH_MERGE_STRONG_MIN_IOU": "growth_rescue.strong_min_iou",
    "MATCH_GROWTH_MERGE_STRONG_MAX_DIST_NORM": "growth_rescue.strong_max_dist_norm",
    "MATCH_GROWTH_MERGE_SUPPORT_SCORE_MARGIN": "growth_rescue.support_score_margin",
    "MATCH_GROWTH_MERGE_MAX_EFFECTIVE_MISSED": "growth_rescue.max_effective_missed",
    "MATCH_GROWTH_MERGE_STALE_STRONG_MIN_IOU": "growth_rescue.stale_strong_min_iou",
    "MATCH_GROWTH_MERGE_STALE_STRONG_MAX_DIST_NORM": "growth_rescue.stale_strong_max_dist_norm",
    "MATCH_GROWTH_MERGE_MAX_EXTRA_PARENTS": "growth_rescue.max_extra_parents",
    "ENABLE_PERSISTENT_GROWTH_MERGE_PROMOTION": "persistent_growth.enabled",
    "PERSISTENT_GROWTH_MIN_STREAK": "persistent_growth.min_streak",
    "PERSISTENT_GROWTH_MIN_EVENTS": "persistent_growth.min_events",
    "PERSISTENT_GROWTH_MAX_FRAME_GAP": "persistent_growth.max_frame_gap",
    "PERSISTENT_GROWTH_EVIDENCE_DECAY": "persistent_growth.evidence_decay",
    "PERSISTENT_GROWTH_PROMOTION_MIN_SCORE": "persistent_growth.promotion_min_score",
    "PERSISTENT_GROWTH_PROMOTION_MIN_AREA_RATIO": "persistent_growth.promotion_min_area_ratio",
    "PERSISTENT_GROWTH_SUPPRESS_HIGH_IOU": "persistent_growth.wobble_min_iou",
    "PERSISTENT_GROWTH_SUPPRESS_LOW_MOTION": "persistent_growth.wobble_max_dist_norm",
    "MATCH_GROWTH_MERGE_MIN_AREA_GROWTH": "adaptive_growth.fallback_threshold",
    "ADAPTIVE_GROWTH_BASELINE_MIN_SAMPLES": "adaptive_growth.min_samples",
    "ADAPTIVE_GROWTH_MAD_MULTIPLIER": "adaptive_growth.mad_multiplier",
    "ADAPTIVE_GROWTH_MIN_THRESHOLD": "adaptive_growth.min_threshold",
    "MIN_SEGMENT_CONFIDENCE": "segments.min_confidence",
    "REQUIRE_INSTANCE_SEGMENTATION": "segments.require_segmentation",
    "FILTER_NON_CIRCULAR_SEGMENTS": "segments.filter_non_circular",
    "MIN_SEGMENT_CIRCULARITY": "segments.min_circularity",
    "MIN_SEGMENT_AXIS_RATIO": "segments.min_axis_ratio",
    "KEEP_NON_CIRCULAR_FOR_ASSOCIATION": "segments.keep_non_circular_for_association",
    "SEGMENT_IOU_MAX_RASTER_PIXELS": "segments.iou_max_raster_pixels",
    "DRAW_TRACK_OVERLAY": "render.draw_overlay",
    "SHOW_ONLY_MERGED_INSTANCES": "render.show_only_merged",
    "SHOW_ONLY_MERGED_INCLUDE_LOST": "render.show_only_merged_include_lost",
    "SEGMENT_FILL_ALPHA": "render.fill_alpha",
    "DRAW_SEGMENT_BORDER": "render.draw_border",
    "GREEN_BOX_COLOR": "render.active_color",
    "RED_BOX_COLOR": "render.lost_color",
    "YELLOW_BOX_COLOR": "render.merged_color",
    "THICKNESS": "render.label_thickness",
    "FONT": "render.font",
    "FONT_SCALE": "render.font_scale",
}
DROPPED = {
    "EXPORT_DETECTIONS": "superseded by the CLI subcommands",
    "MATCH_GROWTH_MERGE_SUPPORT_MIN_IOU": "never read by any code path",
}


def _resolve(cfg, dotted):
    value = cfg
    for part in dotted.split("."):
        value = getattr(value, part)
    return value


def test_every_legacy_constant_is_mapped_or_dropped():
    legacy = json.loads(FIXTURE.read_text())
    assert len(LEGACY_FIELD_MAP) == 101
    assert set(legacy) == set(LEGACY_FIELD_MAP) | set(DROPPED)
    assert not set(LEGACY_FIELD_MAP) & set(DROPPED)


def test_defaults_equal_legacy_values_and_types():
    legacy = json.loads(FIXTURE.read_text())
    mismatches = []
    for name, path in sorted(LEGACY_FIELD_MAP.items()):
        value = _resolve(DEFAULT_CONFIG, path)
        if isinstance(value, tuple):
            value = list(value)
        if value != legacy[name] or type(value) is not type(legacy[name]):
            mismatches.append(f"{name} -> {path}: {value!r} vs legacy {legacy[name]!r}")
    assert not mismatches, "\n".join(mismatches)


def test_inferred_max_missed_follows_max_missed_frames():
    assert DEFAULT_CONFIG.inferred_max_missed == 10
    assert DEFAULT_CONFIG.with_overrides({"max_missed_frames": 2}).inferred_max_missed == 2
    pinned = DEFAULT_CONFIG.with_overrides({"max_missed_frames": 2, "inferred_merge": {"max_missed": 5}})
    assert pinned.inferred_max_missed == 5


def test_override_unknown_key_names_it():
    with pytest.raises(ValueError, match=r"'direct_merge\.min_iuo'.*'min_iou'"):
        DEFAULT_CONFIG.with_overrides({"direct_merge": {"min_iuo": 0.3}})
    with pytest.raises(ValueError, match=r"'segmnets'"):
        DEFAULT_CONFIG.with_overrides({"segmnets": {}})


@pytest.mark.parametrize("overrides, message", [
    ({"match": {"require_mutual_best": 1}}, "expected true or false"),
    ({"max_missed_frames": 2.5}, "expected an integer"),
    ({"max_missed_frames": True}, "expected an integer"),
    ({"direct_merge": {"min_iou": "0.3"}}, "expected a number"),
    ({"direct_merge": 0.3}, "expected an object of settings"),
    ({"render": {"active_color": [0, 255]}}, "expected a list of 3 integers"),
    ({"direct_merge": {"min_iou": None}}, "null is not allowed"),
])
def test_override_wrong_type_is_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        DEFAULT_CONFIG.with_overrides(overrides)


@pytest.mark.parametrize("overrides, key", [
    # 0 crashed mid-run: min() over an empty parent list / quantiles of an empty sample.
    ({"inferred_merge": {"min_parents": 0}}, "inferred_merge.min_parents"),
    ({"adaptive_growth": {"min_samples": 0}}, "adaptive_growth.min_samples"),
])
def test_override_below_minimum_is_rejected(overrides, key):
    with pytest.raises(ValueError, match=rf"{re.escape(key)}: must be >= 1"):
        DEFAULT_CONFIG.with_overrides(overrides)


def test_int_into_float_setting_becomes_float():
    cfg = DEFAULT_CONFIG.with_overrides({"direct_merge": {"min_iou": 1}})
    assert cfg.direct_merge.min_iou == 1.0 and isinstance(cfg.direct_merge.min_iou, float)


def test_overrides_return_a_new_config_and_leave_the_default_untouched():
    cfg = DEFAULT_CONFIG.with_overrides({"max_missed_frames": 3})
    assert cfg.max_missed_frames == 3
    assert DEFAULT_CONFIG.max_missed_frames == 10
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.max_missed_frames = 4


def test_full_dump_round_trips(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(DEFAULT_CONFIG.to_dict()))
    assert load_config(str(path)) == DEFAULT_CONFIG
    assert load_config(None) is DEFAULT_CONFIG
    assert isinstance(DEFAULT_CONFIG, TrackingConfig)


def test_load_config_error_names_the_file(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"match": {"max_area_ration": 2.0}}))
    with pytest.raises(ValueError, match=r"bad\.json: unknown setting 'match\.max_area_ration'"):
        load_config(str(path))
