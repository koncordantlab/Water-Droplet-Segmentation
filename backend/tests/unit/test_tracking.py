"""Tier-1 tests for the backend/tracking package (port of the
1-object-tracking-algorithm-v01 scripts; behavior frozen at the JSON
boundaries — see the tracking_freeze goldens)."""
import subprocess
import sys


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
