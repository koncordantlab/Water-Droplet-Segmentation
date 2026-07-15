"""All tracking constants, verbatim from the monolith -- values are part of
the behavior freeze (backend/output/tracking_freeze). Paths are re-anchored
to the post-rename repo layout; everything else is byte-identical."""
import os

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
