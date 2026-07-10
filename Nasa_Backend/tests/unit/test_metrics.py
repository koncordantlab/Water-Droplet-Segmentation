"""Numeric helpers: avg-size metrics, resolution constant, per-row µm scaling,
eq-diameter, and stats rows. Migrated from test_summary_avg_columns.py and
test_output_modes.py (check_apply_metric); values preserved exactly."""
import math

import numpy as np
import pytest


def test_avg_size_metrics_known_values(app_module):
    a, d = app_module._avg_size_metrics([16, 9], 2.0)
    assert a == pytest.approx(50.0, abs=1e-9)  # mean area 12.5 px² × 2²
    exp_d = ((math.sqrt(4 * 16 / math.pi) + math.sqrt(4 * 9 / math.pi)) / 2) * 2.0
    assert d == pytest.approx(exp_d, abs=1e-9)


@pytest.mark.parametrize("bad_scale", [None, 0, -1])
def test_avg_size_metrics_nan_on_bad_scale(app_module, bad_scale):
    a, d = app_module._avg_size_metrics([16, 9], bad_scale)
    assert math.isnan(a) and math.isnan(d)


def test_avg_size_metrics_nan_on_empty(app_module):
    a, d = app_module._avg_size_metrics([], 2.0)
    assert math.isnan(a) and math.isnan(d)


def test_resolution_constant(app_module):
    assert app_module._resolution_pix_per_um2(2.0) == 0.25


@pytest.mark.parametrize("bad_scale", [None, 0, -1])
def test_resolution_nan_on_bad_scale(app_module, bad_scale):
    assert math.isnan(app_module._resolution_pix_per_um2(bad_scale))


def test_apply_metric_scales_rows_in_place(app_module):
    rows = [{"eq_diameter_px": 10.0, "pixel_count": 4}]
    app_module._apply_metric(rows, 2.0)
    assert rows[0]["eq_diameter_um"] == 20.0
    assert rows[0]["area_um2"] == 16.0


@pytest.mark.parametrize("bad_scale", [None, 0, -1])
def test_apply_metric_nan_on_bad_scale(app_module, bad_scale):
    rows = [{"eq_diameter_px": 10.0, "pixel_count": 4}]
    app_module._apply_metric(rows, bad_scale)
    assert math.isnan(rows[0]["eq_diameter_um"])
    assert math.isnan(rows[0]["area_um2"])


def test_apply_metric_empty_rows(app_module):
    assert app_module._apply_metric([], 2.0) == []


def test_eq_diameter_formula(app_module):
    # d = sqrt(4A/pi): area pi/4 -> d = 1
    out = app_module._eq_diameter([math.pi / 4.0, 16.0])
    assert out[0] == pytest.approx(1.0, abs=1e-12)
    assert out[1] == pytest.approx(math.sqrt(64.0 / math.pi), abs=1e-12)


def test_stats_row_values_and_empty(app_module):
    row = app_module._stats_row("water", [1.0, 2.0, 3.0])
    assert row == {
        "class": "water", "count": 3,
        "min": 1.0, "max": 3.0, "mean": 2.0, "median": 2.0,
        "std": round(float(np.std([1.0, 2.0, 3.0])), 3),
    }
    empty = app_module._stats_row("ice", [])
    assert empty["count"] == 0
    assert all(empty[k] is None for k in ("min", "max", "mean", "median", "std"))
