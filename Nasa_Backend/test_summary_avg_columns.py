"""Standalone checks for the summary average-size columns.

Run: ~/miniconda3/envs/droplets/bin/python Nasa_Backend/test_summary_avg_columns.py
Exits non-zero on failure; prints ALL CHECKS PASSED on success. No pytest.
"""
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# The module loads the YOLO model at import time using a relative weights path,
# so we must import from within Nasa_Backend/.
os.chdir(_HERE)
import frontend_nasa13_apiV2 as app


def check_avg_size_metrics():
    # areas 16 px and 9 px, um_per_px = 2.0
    a, d = app._avg_size_metrics([16, 9], 2.0)
    # mean area = 12.5 px * 2^2 = 50.0 µm²
    assert abs(a - 50.0) < 1e-9, a
    # per-droplet eq dia px: sqrt(4*16/pi), sqrt(4*9/pi); mean * 2.0
    exp_d = ((math.sqrt(4 * 16 / math.pi) + math.sqrt(4 * 9 / math.pi)) / 2) * 2.0
    assert abs(d - exp_d) < 1e-9, d
    # NaN when scale missing/≤0
    for bad in (None, 0, -1):
        a2, d2 = app._avg_size_metrics([16, 9], bad)
        assert a2 != a2 and d2 != d2, bad  # NaN != NaN
    # NaN when no droplets
    a3, d3 = app._avg_size_metrics([], 2.0)
    assert a3 != a3 and d3 != d3
    print("OK avg_size_metrics")


def check_resolution():
    assert app._resolution_pix_per_um2(2.0) == 0.25
    for bad in (None, 0, -1):
        r = app._resolution_pix_per_um2(bad)
        assert r != r, bad  # NaN
    print("OK resolution")


def check_json_serializable_non_finite():
    """Non-finite floats must serialize to JSON null: json.dumps emits a bare
    NaN token otherwise, which JSON.parse in the SSE frontend rejects."""
    import json
    import numpy as np

    assert app.make_json_serializable(float("nan")) is None
    assert app.make_json_serializable(float("inf")) is None
    assert app.make_json_serializable(float("-inf")) is None
    assert app.make_json_serializable(np.float64("nan")) is None
    row = {
        "Water Avg Area (µm²)": float("nan"),
        "Resolution (pix/µm²)": np.float64("inf"),
        "Water (%)": 12.5,
    }
    clean = app.make_json_serializable([row])[0]
    assert clean["Water Avg Area (µm²)"] is None, clean
    assert clean["Resolution (pix/µm²)"] is None, clean
    assert clean["Water (%)"] == 12.5, clean
    # strict round-trip — same strictness as browser JSON.parse
    json.loads(json.dumps(app.make_json_serializable([row]), allow_nan=False))
    print("OK json_serializable non-finite")


if __name__ == "__main__":
    check_avg_size_metrics()
    check_resolution()
    check_json_serializable_non_finite()
    print("\nALL CHECKS PASSED")
