"""Basic/Full per-instance output modes. Migrated from test_output_modes.py."""
import os

import numpy as np
import pandas as pd


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


_META = {"video_name": "t.mp4", "fps": 30, "stride": 30, "width": 40, "height": 20}


def test_basic_mode_columns_and_shared_pixel_math(app_module):
    masks, boxes, names, shape = _synthetic()
    basic = app_module._per_instance_metrics(masks, boxes, names, shape, mode="basic")
    full = app_module._per_instance_metrics(masks, boxes, names, shape, mode="full")
    assert set(basic[0].keys()) == {
        "instance_id", "class", "confidence", "pixel_count", "eq_diameter_px"
    }
    assert [r["pixel_count"] for r in basic] == [r["pixel_count"] for r in full]
    assert [r["eq_diameter_px"] for r in basic] == [r["eq_diameter_px"] for r in full]


def test_instance_ids_sequential_when_empty_mask_skipped(app_module):
    H, W = 20, 40
    m1 = np.zeros((H, W), np.uint8)
    m1[2:6, 2:6] = 1
    m_empty = np.zeros((H, W), np.uint8)
    m3 = np.zeros((H, W), np.uint8)
    m3[10:13, 20:23] = 1
    boxes = [_FakeBox(0.9, [2, 2, 6, 6]), _FakeBox(0.5, [0, 0, 1, 1]), _FakeBox(0.8, [20, 10, 23, 13])]
    names = ["water", "ice", "ice"]
    for mode in ("basic", "full"):
        rows = app_module._per_instance_metrics([m1, m_empty, m3], boxes, names, (H, W), mode=mode)
        assert [r["instance_id"] for r in rows] == [1, 2], mode
        assert [r["class"] for r in rows] == ["water", "ice"]


def test_basic_workbook_sheets_and_columns(app_module, tmp_path):
    masks, boxes, names, shape = _synthetic()
    rows = app_module._per_instance_metrics(masks, boxes, names, shape, mode="basic")
    app_module._save_per_frame_instance_xlsx({1: rows}, str(tmp_path), "t", _META,
                                             mode="basic", um_per_px=2.0)
    f = tmp_path / "t_frame_000001_instances.xlsx"
    assert f.is_file()
    assert pd.ExcelFile(f).sheet_names == ["Instances", "Frame Info", "Stats"]
    inst = pd.read_excel(f, sheet_name="Instances")
    assert list(inst.columns) == ["instance_id", "class", "confidence", "pixel_count",
                                  "eq_diameter_px", "eq_diameter_um", "area_um2"]
    assert inst["eq_diameter_um"].notna().all()
    info = pd.read_excel(f, sheet_name="Frame Info")
    assert "um_per_px" in info.columns


def test_full_workbook_sheets(app_module, tmp_path):
    masks, boxes, names, shape = _synthetic()
    rows = app_module._per_instance_metrics(masks, boxes, names, shape, mode="full")
    app_module._save_per_frame_instance_xlsx({1: rows}, str(tmp_path), "tf", _META,
                                             size_distribution=None, mode="full")
    ff = tmp_path / "tf_frame_000001_instances.xlsx"
    assert ff.is_file()
    assert pd.ExcelFile(ff).sheet_names == [
        "Instances", "Frame Info", "Stats", "Histogram Water", "Histogram Ice"
    ]


def test_basic_workbook_nan_metrics_without_scale(app_module, tmp_path):
    masks, boxes, names, shape = _synthetic()
    rows = app_module._per_instance_metrics(masks, boxes, names, shape, mode="basic")
    app_module._save_per_frame_instance_xlsx({2: rows}, str(tmp_path), "tn", _META,
                                             mode="basic", um_per_px=None)
    fn = tmp_path / "tn_frame_000002_instances.xlsx"
    assert fn.is_file()
    inst = pd.read_excel(fn, sheet_name="Instances")
    assert inst["eq_diameter_um"].isna().all()


def test_empty_rows_writes_nothing(app_module, tmp_path):
    app_module._save_per_frame_instance_xlsx({}, str(tmp_path / "sub"), "x", _META, mode="full")
    assert not (tmp_path / "sub").exists()
