"""Standalone checks for Basic/Full output modes.

Run: python3 Nasa_Backend/test_output_modes.py
Exits non-zero on failure; prints ALL CHECKS PASSED on success.
No pytest dependency.
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# The module loads the YOLO model at import time using a relative weights path
# (app_root/weights_DP(8).pt), so we must import from within Nasa_Backend/.
os.chdir(_HERE)
import frontend_nasa13_apiV2 as app


class _Arr:
    def __init__(self, a):
        self._a = a

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _Conf:
    def __init__(self, v):
        self._v = v

    def item(self):
        return self._v


class _XY:
    def __init__(self, a):
        self._a = a

    def __getitem__(self, i):
        return _Arr(self._a[i])


class _FakeBox:
    """Minimal stand-in for an ultralytics Boxes element."""

    def __init__(self, conf, xyxy):
        self._conf = conf
        self._xyxy = np.array([xyxy], dtype=float)

    @property
    def conf(self):
        return _Conf(self._conf)

    @property
    def xyxy(self):
        return _XY(self._xyxy)


def _synthetic():
    H, W = 20, 40
    m1 = np.zeros((H, W), np.uint8)
    m1[2:6, 2:6] = 1   # 16 px
    m2 = np.zeros((H, W), np.uint8)
    m2[10:13, 20:23] = 1   # 9 px
    masks = [m1, m2]
    boxes = [_FakeBox(0.9, [2, 2, 6, 6]), _FakeBox(0.8, [20, 10, 23, 13])]
    return masks, boxes, ["water", "ice"], (H, W)


def check_apply_metric():
    rows = [{"eq_diameter_px": 10.0, "pixel_count": 4}]
    app._apply_metric(rows, 2.0)
    assert rows[0]["eq_diameter_um"] == 20.0, rows
    assert rows[0]["area_um2"] == 16.0, rows
    for bad in (None, 0, -1):
        r = [{"eq_diameter_px": 10.0, "pixel_count": 4}]
        app._apply_metric(r, bad)
        assert r[0]["eq_diameter_um"] != r[0]["eq_diameter_um"], bad  # NaN != NaN
        assert r[0]["area_um2"] != r[0]["area_um2"], bad
    assert app._apply_metric([], 2.0) == []
    print("OK apply_metric")


def check_metrics_modes():
    masks, boxes, names, shape = _synthetic()
    basic = app._per_instance_metrics(masks, boxes, names, shape, mode="basic")
    full = app._per_instance_metrics(masks, boxes, names, shape, mode="full")
    assert set(basic[0].keys()) == {
        "instance_id", "class", "confidence", "pixel_count", "eq_diameter_px"
    }, basic[0].keys()
    assert [r["pixel_count"] for r in basic] == [r["pixel_count"] for r in full]
    assert [r["eq_diameter_px"] for r in basic] == [r["eq_diameter_px"] for r in full]
    print("OK metrics modes")


def check_instance_id_sequential_with_empty_mask():
    """Empty masks are skipped, but instance_id must stay sequential 1..N
    (per per_frame_xlsx_schema.md) — no gaps from the skipped indices."""
    H, W = 20, 40
    m1 = np.zeros((H, W), np.uint8)
    m1[2:6, 2:6] = 1
    m_empty = np.zeros((H, W), np.uint8)  # zero area -> skipped
    m3 = np.zeros((H, W), np.uint8)
    m3[10:13, 20:23] = 1
    masks = [m1, m_empty, m3]
    boxes = [
        _FakeBox(0.9, [2, 2, 6, 6]),
        _FakeBox(0.5, [0, 0, 1, 1]),
        _FakeBox(0.8, [20, 10, 23, 13]),
    ]
    names = ["water", "ice", "ice"]
    for mode in ("basic", "full"):
        rows = app._per_instance_metrics(masks, boxes, names, (H, W), mode=mode)
        ids = [r["instance_id"] for r in rows]
        assert ids == [1, 2], (mode, ids)
        assert [r["class"] for r in rows] == ["water", "ice"], rows
    print("OK instance_id sequential with empty mask")


def check_save_modes():
    masks, boxes, names, shape = _synthetic()
    meta = {"video_name": "t.mp4", "fps": 30, "stride": 30, "width": 40, "height": 20}
    written_files = []
    with tempfile.TemporaryDirectory() as d:
        rows_b = app._per_instance_metrics(masks, boxes, names, shape, mode="basic")
        app._save_per_frame_instance_xlsx({1: rows_b}, d, "t", meta,
                                          mode="basic", um_per_px=2.0)
        f = os.path.join(d, "t_frame_000001_instances.xlsx")
        assert os.path.isfile(f), f
        written_files.append(f)
        xl = pd.ExcelFile(f)
        assert xl.sheet_names == ["Instances", "Frame Info", "Stats"], xl.sheet_names
        cols = list(pd.read_excel(f, sheet_name="Instances").columns)
        assert cols == ["instance_id", "class", "confidence", "pixel_count",
                        "eq_diameter_px", "eq_diameter_um", "area_um2"], cols
        info = pd.read_excel(f, sheet_name="Frame Info")
        assert "um_per_px" in info.columns, info.columns
        # 16 px -> eq_d_px = sqrt(4*16/pi); um = px*2; spot-check water row metric present
        inst = pd.read_excel(f, sheet_name="Instances")
        assert inst["eq_diameter_um"].notna().all(), inst
        print("OK basic save ->", f)

        rows_f = app._per_instance_metrics(masks, boxes, names, shape, mode="full")
        app._save_per_frame_instance_xlsx({1: rows_f}, d, "tf", meta,
                                          size_distribution=None, mode="full")
        ff = os.path.join(d, "tf_frame_000001_instances.xlsx")
        assert os.path.isfile(ff), ff
        written_files.append(ff)
        assert pd.ExcelFile(ff).sheet_names == [
            "Instances", "Frame Info", "Stats", "Histogram Water", "Histogram Ice"
        ], pd.ExcelFile(ff).sheet_names
        print("OK full save ->", ff)

        # NaN metric path: basic with no scale
        rows_b2 = app._per_instance_metrics(masks, boxes, names, shape, mode="basic")
        app._save_per_frame_instance_xlsx({2: rows_b2}, d, "tn", meta,
                                          mode="basic", um_per_px=None)
        fn = os.path.join(d, "tn_frame_000002_instances.xlsx")
        assert os.path.isfile(fn), fn
        written_files.append(fn)
        inst_n = pd.read_excel(fn, sheet_name="Instances")
        assert inst_n["eq_diameter_um"].isna().all(), inst_n
        print("OK basic save (no scale -> NaN) ->", fn)

        print("\nInventory of files written this run:")
        for p in written_files:
            print(f"  - {p} ({os.path.getsize(p)} bytes)")


if __name__ == "__main__":
    check_apply_metric()
    check_metrics_modes()
    check_instance_id_sequential_with_empty_mask()
    check_save_modes()
    print("\nALL CHECKS PASSED")
