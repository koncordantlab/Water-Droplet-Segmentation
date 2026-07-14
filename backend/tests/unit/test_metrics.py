"""Numeric helpers: avg-size metrics, resolution constant, per-row µm scaling,
eq-diameter, and stats rows. Migrated from test_summary_avg_columns.py and
test_output_modes.py (check_apply_metric); values preserved exactly."""
import math

import numpy as np
import pytest

from droplet_backend import metrics


def test_avg_size_metrics_known_values():
    a, d = metrics._avg_size_metrics([16, 9], 2.0)
    assert a == pytest.approx(50.0, abs=1e-9)  # mean area 12.5 px² × 2²
    exp_d = ((math.sqrt(4 * 16 / math.pi) + math.sqrt(4 * 9 / math.pi)) / 2) * 2.0
    assert d == pytest.approx(exp_d, abs=1e-9)


@pytest.mark.parametrize("bad_scale", [None, 0, -1])
def test_avg_size_metrics_nan_on_bad_scale(bad_scale):
    a, d = metrics._avg_size_metrics([16, 9], bad_scale)
    assert math.isnan(a) and math.isnan(d)


def test_avg_size_metrics_nan_on_empty():
    a, d = metrics._avg_size_metrics([], 2.0)
    assert math.isnan(a) and math.isnan(d)


def test_resolution_constant():
    assert metrics._resolution_pix_per_um2(2.0) == 0.25


@pytest.mark.parametrize("bad_scale", [None, 0, -1])
def test_resolution_nan_on_bad_scale(bad_scale):
    assert math.isnan(metrics._resolution_pix_per_um2(bad_scale))


def test_apply_metric_scales_rows_in_place():
    rows = [{"eq_diameter_px": 10.0, "pixel_count": 4}]
    metrics._apply_metric(rows, 2.0)
    assert rows[0]["eq_diameter_um"] == 20.0
    assert rows[0]["area_um2"] == 16.0


@pytest.mark.parametrize("bad_scale", [None, 0, -1])
def test_apply_metric_nan_on_bad_scale(bad_scale):
    rows = [{"eq_diameter_px": 10.0, "pixel_count": 4}]
    metrics._apply_metric(rows, bad_scale)
    assert math.isnan(rows[0]["eq_diameter_um"])
    assert math.isnan(rows[0]["area_um2"])


def test_apply_metric_empty_rows():
    assert metrics._apply_metric([], 2.0) == []


def test_stats_row_values_and_empty():
    row = metrics._stats_row("water", [1.0, 2.0, 3.0])
    assert row == {
        "class": "water", "count": 3,
        "min": 1.0, "max": 3.0, "mean": 2.0, "median": 2.0,
        "std": round(float(np.std([1.0, 2.0, 3.0])), 3),
    }
    empty = metrics._stats_row("ice", [])
    assert empty["count"] == 0
    assert all(empty[k] is None for k in ("min", "max", "mean", "median", "std"))


def test_full_mode_circle_metrics_match_geometry():
    """Spec §7 tier-1 row: synthetic circle -> circularity ~= 1, exact pixel count."""
    import numpy as np
    import torch

    H, W, r = 128, 128, 20
    yy, xx = np.mgrid[0:H, 0:W]
    circle = ((yy - 64) ** 2 + (xx - 64) ** 2 <= r * r).astype(np.uint8)

    class _Box:
        cls = torch.tensor([0.0])
        conf = torch.tensor([0.9])
        xyxy = torch.tensor([[44.0, 44.0, 84.0, 84.0]])

    rows = metrics._per_instance_metrics([circle], [_Box()], ["water"], (H, W), mode="full")
    assert len(rows) == 1
    row = rows[0]
    assert row["pixel_count"] == int(circle.sum())
    assert abs(row["eq_diameter_px"] - 2 * r) < 1.5
    assert 0.85 < row["circularity"] <= 1.1
    assert row["touches_border"] in (False, 0)
