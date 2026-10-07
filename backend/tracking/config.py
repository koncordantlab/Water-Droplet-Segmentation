"""All tracking constants, verbatim from the monolith -- values are part of
the behavior freeze (backend/output/tracking_freeze). Paths are re-anchored
to the post-rename repo layout; everything else is byte-identical."""
import dataclasses
import difflib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

import cv2
import torch

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_DIR = os.path.dirname(_BACKEND_DIR)

MODEL_PATH = os.path.join(_BACKEND_DIR, "app_root", "weights_DP(6).pt")      # Path to YOLO model
VIDEO_PATH = os.path.join(_REPO_DIR, "Test_videos", "20 seconds.mp4")        # Input video
OUTPUT_PATH = os.path.join(_BACKEND_DIR, "output", "output_tracked.mp4")     # Output video
DETECTIONS_PATH = os.path.join(_BACKEND_DIR, "output", "detections.json")    # JSON detections log
TRACK_LOG_PATH = os.path.join(_BACKEND_DIR, "output", "tracking_log.json")   # JSON per-frame tracking log

# **Run mode**
EXPORT_DETECTIONS = True

# **Visualization settings**
GREEN_BOX_COLOR = (0, 255, 0)
RED_BOX_COLOR = (0, 0, 255)
YELLOW_BOX_COLOR = (0, 255, 255)
THICKNESS = 1
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
# If True, output video is written with segmentation overlays (polygons), not boxes.
DRAW_TRACK_OVERLAY = True
# If True, only merged tracks are drawn in output frames.
# A track is considered merged when its generation is >1 or it was merged in the current frame.
SHOW_ONLY_MERGED_INSTANCES = True
SHOW_ONLY_MERGED_INCLUDE_LOST = False
SEGMENT_FILL_ALPHA = 0.35
DRAW_SEGMENT_BORDER = False

# **Tracking settings**
# Track lifecycle + candidate prefilter gates.
MAX_MISSED_FRAMES = 10  # Keep unmatched tracks alive for this many frames.
IOU_MATCH_THRESHOLD = 0.05  # Initial candidate gate on segment IoU.
DIST_MATCH_FACTOR = 1.2  # Initial candidate gate on normalized center distance.

# Direct merge detector (same-frame multi-parent detection).
ENABLE_MERGE_DETECTION = True
MERGE_AREA_FACTOR = 1.05  # Child area must be at least this factor of each parent area.
MERGE_COMBINED_AREA_FACTOR = 0.85  # Child area vs. combined selected parent areas.
MERGE_MIN_IOU = 0.2  # Minimum IoU for a parent candidate in direct merge mode.
MERGE_MAX_DIST_NORM = 1.5  # Maximum normalized distance for a parent candidate.
MERGE_PARENT_DIST_FACTOR = 2.0  # Parent center must be within this * child max-dim.
MERGE_SECOND_PARENT_MIN_IOU = 0.24  # Require non-trivial support from second parent.
MERGE_SECOND_PARENT_MAX_DIST_NORM = 0.55  # Allow low-IoU second parent when still close.
MERGE_MIN_COMBINED_PARENT_IOU = 0.58  # Sum IoU check to avoid weak dual-parent claims.
MERGE_MAX_PARENT_PAIR_DIST_NORM = 1.25  # Parents must be reasonably close to each other.

# Inferred merge detector (handles hidden/missed parents).
ENABLE_INFERRED_MERGE = True
INFERRED_MERGE_MIN_PARENTS = 1  # Allow single-parent inferred merge with unknown contributors.
INFERRED_MERGE_MAX_MISSED = MAX_MISSED_FRAMES  # Max effective missed window for inferred parents.
INFERRED_MERGE_AREA_FACTOR = 1.05  # Child area must exceed parent area by this factor.
INFERRED_MERGE_COMBINED_AREA_FACTOR = 0.75  # Child area vs combined inferred parent areas.
INFERRED_MERGE_MIN_IOU = 0.05  # Loose IoU gate to keep inferred mode tolerant.
INFERRED_MERGE_MAX_DIST_NORM = 2.5  # Loose distance gate for inferred mode.
INFERRED_MERGE_PARENT_MAX_DIST_NORM = 1.2  # Tighter parent distance cap.
INFERRED_MERGE_ANCHOR_MIN_IOU = 0.08  # At least one anchor parent should overlap reasonably.
INFERRED_MERGE_ANCHOR_MAX_DIST_NORM = 0.6  # Or be spatially close as an anchor.
INFERRED_MERGE_REQUIRE_LOST_PARENT = False  # If True, inferred merge needs >=1 already-lost parent.
INFERRED_MERGE_WARMUP_FRAMES = 4  # During early frames, suppress single-parent inferred merge promotions.
INFERRED_MERGE_LOST_PARENT_MAX_EFFECTIVE_MISSED = 4  # Guard against reviving very old lost tracks.
INFERRED_SINGLE_PARENT_MIN_AREA_GROWTH = 1.28  # Allow moderate growth single-parent inferred merges.
INFERRED_SINGLE_PARENT_MIN_IOU = 0.10  # Base IoU gate for single-parent inferred merge.
INFERRED_SINGLE_PARENT_MAX_DIST_NORM = 0.70  # Base distance gate for single-parent inferred merge.
INFERRED_SINGLE_PARENT_ACTIVE_MIN_AREA_GROWTH = 1.45  # Stricter growth when parent is still active.
INFERRED_SINGLE_PARENT_ACTIVE_MIN_IOU = 0.60  # Require clear overlap for active-parent single inferred merges.
INFERRED_SINGLE_PARENT_ACTIVE_MAX_DIST_NORM = 0.20  # Keep active-parent single inferred merges very local.
INFERRED_SINGLE_PARENT_GROWTH_ONLY_IOU = 0.85  # High IoU + tiny motion likely means plain growth.
INFERRED_SINGLE_PARENT_GROWTH_ONLY_MAX_DIST_NORM = 0.08
INFERRED_SINGLE_PARENT_MAX_EFFECTIVE_MISSED = 3  # Do not use stale tracks for single-parent mode.
INFERRED_SINGLE_PARENT_LOST_MAX_EFFECTIVE_MISSED = 2  # Even stricter when parent was already lost.
INFERRED_SINGLE_PARENT_LOST_MIN_IOU = 0.12
INFERRED_SINGLE_PARENT_LOST_MAX_DIST_NORM = 0.35
MAX_UNKNOWN_PARENT_COUNT_ESTIMATE = 5  # Cap estimated hidden parent count in event logs.

# Unknown-merge promotion controls for large growth/birth events.
ENABLE_INFERRED_UNKNOWN_MERGE = True
INFERRED_UNKNOWN_MATCH_AREA_GROWTH = 1.6  # Matched track area growth threshold.
INFERRED_UNKNOWN_BIRTH_MEDIAN_FACTOR = 1.8  # Birth area vs frame baseline threshold.
UNKNOWN_MERGE_BASELINE_MIN_TRACKS = 8  # Need enough stable tracks before baseline is trusted.
PROMOTE_LARGE_BIRTH_TO_UNKNOWN_MERGE = True  # False keeps large births as "large_birth".
PROMOTE_MATCH_GROWTH_TO_UNKNOWN_MERGE = True  # False avoids gen bumps on growth-only matches.
REJECT_MATCH_ON_LARGE_AREA_GROWTH = True  # Reject suspicious growth matches unless continuation.

# Rescue: convert some growth matches to inferred merges when competing parent evidence exists.
ENABLE_MATCH_GROWTH_MERGE_RESCUE = True
MATCH_GROWTH_MERGE_MIN_AREA_GROWTH = 1.12  # Trigger rescue for smaller-but-real growth merges.
MATCH_GROWTH_MERGE_PARENT_AREA_FACTOR = 1.02  # Detection must also be large enough for support parent.
MATCH_GROWTH_MERGE_SUPPORT_MIN_IOU = 0.10  # Loose support gate.
MATCH_GROWTH_MERGE_SUPPORT_MAX_DIST_NORM = 0.70
MATCH_GROWTH_MERGE_STRONG_MIN_IOU = 0.16  # Need one stronger support parent.
MATCH_GROWTH_MERGE_STRONG_MAX_DIST_NORM = 0.70
MATCH_GROWTH_MERGE_SUPPORT_SCORE_MARGIN = 0.04  # Allow near-best competing supports.
MATCH_GROWTH_MERGE_MAX_EFFECTIVE_MISSED = 3  # Allow one extra missed frame for support parents.
MATCH_GROWTH_MERGE_STALE_STRONG_MIN_IOU = 0.22  # If support is older, require stronger overlap.
MATCH_GROWTH_MERGE_STALE_STRONG_MAX_DIST_NORM = 0.55  # If support is older, require closer distance.
MATCH_GROWTH_MERGE_MAX_EXTRA_PARENTS = 3  # Limit number of attached support parents.
ENABLE_PERSISTENT_GROWTH_MERGE_PROMOTION = True  # Promote repeated growth-only merge candidates.
PERSISTENT_GROWTH_MIN_STREAK = 2  # Require repeated growth evidence before unknown-parent merge promotion.
PERSISTENT_GROWTH_MIN_EVENTS = 3  # Require at least this many candidate events before promotion.
PERSISTENT_GROWTH_MAX_FRAME_GAP = 3  # Allow accumulation across short non-consecutive gaps.
PERSISTENT_GROWTH_EVIDENCE_DECAY = 0.70  # Decay candidate evidence across frame gaps.
PERSISTENT_GROWTH_PROMOTION_MIN_SCORE = 4.2  # Minimum evidence score for promotion.
PERSISTENT_GROWTH_PROMOTION_MIN_AREA_RATIO = 1.34  # Avoid promoting moderate growth oscillations.
PERSISTENT_GROWTH_SUPPRESS_HIGH_IOU = 0.82  # Suppress promotion for near-static contour wobble.
PERSISTENT_GROWTH_SUPPRESS_LOW_MOTION = 0.06
ADAPTIVE_GROWTH_BASELINE_MIN_SAMPLES = 12  # Minimum local continuation samples before adapting growth threshold.
ADAPTIVE_GROWTH_MAD_MULTIPLIER = 4.0  # Robust spread scaling for adaptive growth threshold.
ADAPTIVE_GROWTH_MIN_THRESHOLD = 1.08  # Absolute lower bound for adaptive growth trigger.

# Match validity gates (active vs lost track matching behavior).
MIN_RELIABLE_MATCH_IOU = 0.05  # Global reliability floor.
MAX_RELIABLE_MATCH_DIST_NORM = 0.6
ACTIVE_TRACK_MIN_IOU = 0.30  # Stricter overlap requirement for currently visible tracks.
ACTIVE_TRACK_MAX_DIST_NORM = 0.45
LOST_TRACK_MIN_IOU = 0.12  # Slightly looser for re-acquiring lost tracks.
LOST_TRACK_MAX_DIST_NORM = 0.55
LOST_TRACK_RECOVERY_MIN_IOU = 0.18  # Tighten after longer lost periods.
LOST_TRACK_RECOVERY_MAX_DIST_NORM = 0.40
MAX_MATCH_AREA_RATIO = 1.3  # Reject abrupt size changes as plain matches.
GROWTH_CONTINUATION_MIN_IOU = 0.82  # Exception: near-stationary high-overlap growth continuation.
GROWTH_CONTINUATION_MAX_DIST_NORM = 0.10
REQUIRE_MUTUAL_BEST_MATCH = True

# Segmentation confidence gate.
MIN_SEGMENT_CONFIDENCE = 0.50  # Drop segmented instances below this confidence.

# Segmentation-only assumptions and geometric filtering.
REQUIRE_INSTANCE_SEGMENTATION = True
FILTER_NON_CIRCULAR_SEGMENTS = False
MIN_SEGMENT_CIRCULARITY = 0.58  # 4*pi*A/P^2 floor.
MIN_SEGMENT_AXIS_RATIO = 0.55  # min(width,height)/max(width,height) floor.
KEEP_NON_CIRCULAR_FOR_ASSOCIATION = True  # Allow non-circular shapes for matching/merge evidence only.
SEGMENT_IOU_MAX_RASTER_PIXELS = 250_000  # Raster downscale cap for segment IoU speed.

# GPU/CPU tensor prefilter before exact segment IoU.
TRACKING_USE_GPU_PREFILTER = True
TRACKING_PREFILTER_MAX_CANDIDATES_PER_DET = 0  # Top-k candidates per detection after prefilter.
TRACKING_PREFILTER_DEVICE = torch.device(
    "cuda" if TRACKING_USE_GPU_PREFILTER and torch.cuda.is_available() else "cpu"
)


# ---------------------------------------------------------------------------
# TrackingConfig: the tuning surface as frozen dataclasses. Each field's
# trailing comment names the legacy constant it replaces; the 1:1 migration
# is pinned by tests/unit/test_tracking_config.py.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateConfig:
    """Candidate prefilter: which track/detection pairs get an exact segment IoU."""
    iou_threshold: float = 0.05  # IOU_MATCH_THRESHOLD
    dist_factor: float = 1.2  # DIST_MATCH_FACTOR: normalized center-distance gate
    use_gpu: bool = True  # TRACKING_USE_GPU_PREFILTER: CUDA when available, else CPU
    max_per_detection: int = 0  # TRACKING_PREFILTER_MAX_CANDIDATES_PER_DET: top-k, 0 = no cap


@dataclass(frozen=True)
class MatchConfig:
    """Match validity gates (active vs lost tracks) and one-to-one assignment."""
    reliable_min_iou: float = 0.05  # MIN_RELIABLE_MATCH_IOU: global reliability floor
    reliable_max_dist_norm: float = 0.6  # MAX_RELIABLE_MATCH_DIST_NORM
    active_min_iou: float = 0.30  # ACTIVE_TRACK_MIN_IOU: stricter for visible tracks
    active_max_dist_norm: float = 0.45  # ACTIVE_TRACK_MAX_DIST_NORM
    lost_min_iou: float = 0.12  # LOST_TRACK_MIN_IOU: looser to re-acquire lost tracks
    lost_max_dist_norm: float = 0.55  # LOST_TRACK_MAX_DIST_NORM
    recovery_min_iou: float = 0.18  # LOST_TRACK_RECOVERY_MIN_IOU: tighter after longer losses
    recovery_max_dist_norm: float = 0.40  # LOST_TRACK_RECOVERY_MAX_DIST_NORM
    max_area_ratio: float = 1.3  # MAX_MATCH_AREA_RATIO: abrupt size change is not a plain match
    continuation_min_iou: float = 0.82  # GROWTH_CONTINUATION_MIN_IOU: near-stationary growth exception
    continuation_max_dist_norm: float = 0.10  # GROWTH_CONTINUATION_MAX_DIST_NORM
    require_mutual_best: bool = True  # REQUIRE_MUTUAL_BEST_MATCH


@dataclass(frozen=True)
class DirectMergeConfig:
    """Same-frame direct merge: one detection covering two live tracks."""
    enabled: bool = True  # ENABLE_MERGE_DETECTION
    area_factor: float = 1.05  # MERGE_AREA_FACTOR: child >= this x each parent
    combined_area_factor: float = 0.85  # MERGE_COMBINED_AREA_FACTOR: child vs summed parents
    min_iou: float = 0.2  # MERGE_MIN_IOU
    max_dist_norm: float = 1.5  # MERGE_MAX_DIST_NORM
    parent_dist_factor: float = 2.0  # MERGE_PARENT_DIST_FACTOR: parent center within this x child max-dim
    second_parent_min_iou: float = 0.24  # MERGE_SECOND_PARENT_MIN_IOU
    second_parent_max_dist_norm: float = 0.55  # MERGE_SECOND_PARENT_MAX_DIST_NORM
    min_combined_parent_iou: float = 0.58  # MERGE_MIN_COMBINED_PARENT_IOU
    max_parent_pair_dist_norm: float = 1.25  # MERGE_MAX_PARENT_PAIR_DIST_NORM


@dataclass(frozen=True)
class InferredMergeConfig:
    """Inferred merge: an unmatched detection explained by recently lost tracks."""
    enabled: bool = True  # ENABLE_INFERRED_MERGE
    min_parents: int = 1  # INFERRED_MERGE_MIN_PARENTS
    max_missed: Optional[int] = None  # INFERRED_MERGE_MAX_MISSED; None = follow max_missed_frames
    area_factor: float = 1.05  # INFERRED_MERGE_AREA_FACTOR
    combined_area_factor: float = 0.75  # INFERRED_MERGE_COMBINED_AREA_FACTOR
    min_iou: float = 0.05  # INFERRED_MERGE_MIN_IOU
    max_dist_norm: float = 2.5  # INFERRED_MERGE_MAX_DIST_NORM
    parent_max_dist_norm: float = 1.2  # INFERRED_MERGE_PARENT_MAX_DIST_NORM
    anchor_min_iou: float = 0.08  # INFERRED_MERGE_ANCHOR_MIN_IOU
    anchor_max_dist_norm: float = 0.6  # INFERRED_MERGE_ANCHOR_MAX_DIST_NORM
    require_lost_parent: bool = False  # INFERRED_MERGE_REQUIRE_LOST_PARENT
    warmup_frames: int = 4  # INFERRED_MERGE_WARMUP_FRAMES
    lost_parent_max_effective_missed: int = 4  # INFERRED_MERGE_LOST_PARENT_MAX_EFFECTIVE_MISSED
    max_unknown_parent_estimate: int = 5  # MAX_UNKNOWN_PARENT_COUNT_ESTIMATE


@dataclass(frozen=True)
class SingleParentConfig:
    """Single-parent inferred merge: one visible parent plus hidden contributors."""
    min_area_growth: float = 1.28  # INFERRED_SINGLE_PARENT_MIN_AREA_GROWTH
    min_iou: float = 0.10  # INFERRED_SINGLE_PARENT_MIN_IOU
    max_dist_norm: float = 0.70  # INFERRED_SINGLE_PARENT_MAX_DIST_NORM
    active_min_area_growth: float = 1.45  # INFERRED_SINGLE_PARENT_ACTIVE_MIN_AREA_GROWTH
    active_min_iou: float = 0.60  # INFERRED_SINGLE_PARENT_ACTIVE_MIN_IOU
    active_max_dist_norm: float = 0.20  # INFERRED_SINGLE_PARENT_ACTIVE_MAX_DIST_NORM
    growth_only_min_iou: float = 0.85  # INFERRED_SINGLE_PARENT_GROWTH_ONLY_IOU
    growth_only_max_dist_norm: float = 0.08  # INFERRED_SINGLE_PARENT_GROWTH_ONLY_MAX_DIST_NORM
    max_effective_missed: int = 3  # INFERRED_SINGLE_PARENT_MAX_EFFECTIVE_MISSED
    lost_max_effective_missed: int = 2  # INFERRED_SINGLE_PARENT_LOST_MAX_EFFECTIVE_MISSED
    lost_min_iou: float = 0.12  # INFERRED_SINGLE_PARENT_LOST_MIN_IOU
    lost_max_dist_norm: float = 0.35  # INFERRED_SINGLE_PARENT_LOST_MAX_DIST_NORM


@dataclass(frozen=True)
class UnknownMergeConfig:
    """Unknown-parent merges promoted from large growth or large births."""
    enabled: bool = True  # ENABLE_INFERRED_UNKNOWN_MERGE
    match_area_growth: float = 1.6  # INFERRED_UNKNOWN_MATCH_AREA_GROWTH
    birth_median_factor: float = 1.8  # INFERRED_UNKNOWN_BIRTH_MEDIAN_FACTOR
    baseline_min_tracks: int = 8  # UNKNOWN_MERGE_BASELINE_MIN_TRACKS
    promote_large_birth: bool = True  # PROMOTE_LARGE_BIRTH_TO_UNKNOWN_MERGE
    promote_match_growth: bool = True  # PROMOTE_MATCH_GROWTH_TO_UNKNOWN_MERGE
    reject_large_growth_match: bool = True  # REJECT_MATCH_ON_LARGE_AREA_GROWTH


@dataclass(frozen=True)
class GrowthRescueConfig:
    """Match-growth rescue: growth a competing live track explains becomes a merge."""
    enabled: bool = True  # ENABLE_MATCH_GROWTH_MERGE_RESCUE
    parent_area_factor: float = 1.02  # MATCH_GROWTH_MERGE_PARENT_AREA_FACTOR
    support_max_dist_norm: float = 0.70  # MATCH_GROWTH_MERGE_SUPPORT_MAX_DIST_NORM
    strong_min_iou: float = 0.16  # MATCH_GROWTH_MERGE_STRONG_MIN_IOU
    strong_max_dist_norm: float = 0.70  # MATCH_GROWTH_MERGE_STRONG_MAX_DIST_NORM
    support_score_margin: float = 0.04  # MATCH_GROWTH_MERGE_SUPPORT_SCORE_MARGIN
    max_effective_missed: int = 3  # MATCH_GROWTH_MERGE_MAX_EFFECTIVE_MISSED
    stale_strong_min_iou: float = 0.22  # MATCH_GROWTH_MERGE_STALE_STRONG_MIN_IOU
    stale_strong_max_dist_norm: float = 0.55  # MATCH_GROWTH_MERGE_STALE_STRONG_MAX_DIST_NORM
    max_extra_parents: int = 3  # MATCH_GROWTH_MERGE_MAX_EXTRA_PARENTS


@dataclass(frozen=True)
class PersistentGrowthConfig:
    """Repeated unexplained growth, promoted to an unknown-parent merge."""
    enabled: bool = True  # ENABLE_PERSISTENT_GROWTH_MERGE_PROMOTION
    min_streak: int = 2  # PERSISTENT_GROWTH_MIN_STREAK
    min_events: int = 3  # PERSISTENT_GROWTH_MIN_EVENTS
    max_frame_gap: int = 3  # PERSISTENT_GROWTH_MAX_FRAME_GAP
    evidence_decay: float = 0.70  # PERSISTENT_GROWTH_EVIDENCE_DECAY
    promotion_min_score: float = 4.2  # PERSISTENT_GROWTH_PROMOTION_MIN_SCORE
    promotion_min_area_ratio: float = 1.34  # PERSISTENT_GROWTH_PROMOTION_MIN_AREA_RATIO
    wobble_min_iou: float = 0.82  # PERSISTENT_GROWTH_SUPPRESS_HIGH_IOU: contour wobble, not growth
    wobble_max_dist_norm: float = 0.06  # PERSISTENT_GROWTH_SUPPRESS_LOW_MOTION


@dataclass(frozen=True)
class AdaptiveGrowthConfig:
    """The per-frame growth threshold estimated from continuation matches."""
    fallback_threshold: float = 1.12  # MATCH_GROWTH_MERGE_MIN_AREA_GROWTH: used below min_samples
    min_samples: int = 12  # ADAPTIVE_GROWTH_BASELINE_MIN_SAMPLES
    mad_multiplier: float = 4.0  # ADAPTIVE_GROWTH_MAD_MULTIPLIER
    min_threshold: float = 1.08  # ADAPTIVE_GROWTH_MIN_THRESHOLD


@dataclass(frozen=True)
class SegmentConfig:
    """Which segmented detections the tracker keeps, and segment-IoU cost."""
    min_confidence: float = 0.50  # MIN_SEGMENT_CONFIDENCE
    require_segmentation: bool = True  # REQUIRE_INSTANCE_SEGMENTATION
    filter_non_circular: bool = False  # FILTER_NON_CIRCULAR_SEGMENTS
    min_circularity: float = 0.58  # MIN_SEGMENT_CIRCULARITY: 4*pi*A/P^2 floor
    min_axis_ratio: float = 0.55  # MIN_SEGMENT_AXIS_RATIO
    keep_non_circular_for_association: bool = True  # KEEP_NON_CIRCULAR_FOR_ASSOCIATION
    iou_max_raster_pixels: int = 250_000  # SEGMENT_IOU_MAX_RASTER_PIXELS


@dataclass(frozen=True)
class RenderConfig:
    """Annotated-video overlay. Never affects tracking decisions or logs."""
    draw_overlay: bool = True  # DRAW_TRACK_OVERLAY
    show_only_merged: bool = True  # SHOW_ONLY_MERGED_INSTANCES
    show_only_merged_include_lost: bool = False  # SHOW_ONLY_MERGED_INCLUDE_LOST
    fill_alpha: float = 0.35  # SEGMENT_FILL_ALPHA
    draw_border: bool = False  # DRAW_SEGMENT_BORDER
    active_color: Tuple[int, int, int] = (0, 255, 0)  # GREEN_BOX_COLOR (BGR)
    lost_color: Tuple[int, int, int] = (0, 0, 255)  # RED_BOX_COLOR
    merged_color: Tuple[int, int, int] = (0, 255, 255)  # YELLOW_BOX_COLOR
    label_thickness: int = 1  # THICKNESS
    font: int = cv2.FONT_HERSHEY_SIMPLEX  # FONT
    font_scale: float = 0.5  # FONT_SCALE


@dataclass(frozen=True)
class TrackingConfig:
    """Every tracking setting for one run. Frozen: build variants with
    with_overrides(); the result of a run is a function of (detections, config)."""
    max_missed_frames: int = 10  # MAX_MISSED_FRAMES: keep unmatched tracks alive this many frames
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
    direct_merge: DirectMergeConfig = field(default_factory=DirectMergeConfig)
    inferred_merge: InferredMergeConfig = field(default_factory=InferredMergeConfig)
    single_parent: SingleParentConfig = field(default_factory=SingleParentConfig)
    unknown_merge: UnknownMergeConfig = field(default_factory=UnknownMergeConfig)
    growth_rescue: GrowthRescueConfig = field(default_factory=GrowthRescueConfig)
    persistent_growth: PersistentGrowthConfig = field(default_factory=PersistentGrowthConfig)
    adaptive_growth: AdaptiveGrowthConfig = field(default_factory=AdaptiveGrowthConfig)
    segments: SegmentConfig = field(default_factory=SegmentConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    @property
    def inferred_max_missed(self) -> int:
        """Max effective missed window for inferred-merge parents: the explicit
        inferred_merge.max_missed, else max_missed_frames (the legacy constant
        was defined as INFERRED_MERGE_MAX_MISSED = MAX_MISSED_FRAMES)."""
        if self.inferred_merge.max_missed is None:
            return self.max_missed_frames
        return self.inferred_merge.max_missed

    def with_overrides(self, overrides: Mapping[str, Any]) -> "TrackingConfig":
        """A copy with nested overrides applied, e.g. {"direct_merge": {"min_iou": 0.3}}.
        Unknown keys and wrong types raise ValueError naming the setting."""
        return _apply(self, overrides, "")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


DEFAULT_CONFIG = TrackingConfig()


def load_config(path: Optional[str] = None) -> TrackingConfig:
    """DEFAULT_CONFIG, or DEFAULT_CONFIG with the JSON overrides file at path applied."""
    if path is None:
        return DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as file:
        overrides = json.load(file)
    try:
        return DEFAULT_CONFIG.with_overrides(overrides)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from None


def _apply(obj: Any, overrides: Any, path: str) -> Any:
    if not isinstance(overrides, Mapping):
        raise ValueError(f"{path or 'settings'}: expected an object of settings, got {overrides!r}")
    fields = {f.name: f for f in dataclasses.fields(obj)}
    changes = {}
    for key, value in overrides.items():
        where = f"{path}.{key}" if path else key
        if key not in fields:
            hint = difflib.get_close_matches(key, list(fields), n=1)
            suffix = f" (did you mean {hint[0]!r}?)" if hint else ""
            raise ValueError(f"unknown setting {where!r}{suffix}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current):
            changes[key] = _apply(current, value, where)
        else:
            changes[key] = _coerce(value, current, fields[key].default is None, where)
    return dataclasses.replace(obj, **changes)


def _coerce(value: Any, current: Any, nullable: bool, where: str) -> Any:
    """Type-check one override against the setting's current value (nullable
    settings are integers whose default is None)."""
    if value is None:
        if nullable:
            return None
        raise ValueError(f"{where}: null is not allowed here")
    if isinstance(current, bool):
        if isinstance(value, bool):
            return value
        raise ValueError(f"{where}: expected true or false, got {value!r}")
    if isinstance(current, int) or (current is None and nullable):
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raise ValueError(f"{where}: expected an integer, got {value!r}")
    if isinstance(current, float):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        raise ValueError(f"{where}: expected a number, got {value!r}")
    if isinstance(current, tuple):
        if (isinstance(value, (list, tuple)) and len(value) == len(current)
                and all(isinstance(v, int) and not isinstance(v, bool) for v in value)):
            return tuple(value)
        raise ValueError(f"{where}: expected a list of {len(current)} integers, got {value!r}")
    raise TypeError(f"{where}: unsupported setting type {type(current).__name__}")
