"""Size-distribution binning invariants: log-spaced global edges, stats blocks,
long-format histogram frames. New coverage (no prior standalone script)."""
import math

import numpy as np
import pytest

from droplet_backend import distribution


def test_shared_bin_edges_are_log_spaced():
    values = [1.0, 10.0, 100.0]
    edges = distribution._shared_bin_edges(values)
    expected = np.logspace(np.log10(1.0), np.log10(100.0), distribution.SIZE_DIST_BINS + 1)
    assert np.allclose(edges, expected)
    assert len(edges) == distribution.SIZE_DIST_BINS + 1


def test_shared_bin_edges_degenerate_and_empty():
    assert distribution._shared_bin_edges([]) is None
    single = distribution._shared_bin_edges([5.0])
    assert list(single) == [5.0, 6.0]
    same = distribution._shared_bin_edges([3.0, 3.0, 3.0])
    assert list(same) == [3.0, 4.0]


def test_shared_bin_edges_nonpositive_falls_back_linear():
    edges = distribution._shared_bin_edges([0.0, 5.0, 10.0])
    expected = np.histogram_bin_edges(np.array([0.0, 5.0, 10.0]), bins=distribution.SIZE_DIST_BINS)
    assert np.allclose(edges, expected)


def test_droplet_stats_block_with_shared_edges():
    edges = np.array([1.0, 2.0, 4.0, 8.0])
    block = distribution._droplet_stats_block([1.5, 3.0, 3.5, 7.0], edges=edges)
    assert block["count"] == 4
    assert block["stats"]["min"] == 1.5 and block["stats"]["max"] == 7.0
    assert block["histogram"]["bin_edges"] == [1.0, 2.0, 4.0, 8.0]
    assert block["histogram"]["counts"] == [1, 2, 1]


def test_droplet_stats_block_empty_with_and_without_edges():
    edges = np.array([1.0, 2.0, 4.0])
    with_edges = distribution._droplet_stats_block([], edges=edges)
    assert with_edges["count"] == 0
    assert with_edges["histogram"]["counts"] == [0, 0]
    assert all(v is None for v in with_edges["stats"].values())
    bare = distribution._droplet_stats_block([])
    assert bare["histogram"] == {"bin_edges": [], "counts": []}


def test_histogram_df_long_format_and_empty():
    edges = [1.0, 2.0, 4.0]
    df = distribution._histogram_df([1.5, 2.5, 3.0], edges)
    assert list(df.columns) == ["bin_lo", "bin_hi", "bin_center", "count"]
    assert df["count"].tolist() == [1, 2]
    assert df["bin_center"].tolist() == [1.5, 3.0]
    empty = distribution._histogram_df([], edges)
    assert empty["count"].tolist() == [0, 0]
    none_edges = distribution._histogram_df([1.0], None)
    assert none_edges.empty


def test_global_bin_edges_lookup():
    sd = {"checkpoints": [
        {"frame": 5, "water": {"histogram": {"bin_edges": []}}, "ice": {"histogram": {"bin_edges": [1.0, 2.0]}}},
        {"frame": 10, "water": {"histogram": {"bin_edges": [3.0, 4.0, 5.0]}}, "ice": {"histogram": {"bin_edges": [1.0, 2.0]}}},
    ]}
    assert distribution._global_bin_edges_from_size_distribution(sd, "water") == [3.0, 4.0, 5.0]
    assert distribution._global_bin_edges_from_size_distribution(sd, "ice") == [1.0, 2.0]
    assert distribution._global_bin_edges_from_size_distribution(None, "water") is None
    assert distribution._global_bin_edges_from_size_distribution({"checkpoints": []}, "water") is None


def test_stats_and_histogram_rounding_precision():
    # Values with >3 decimals distinguish the stats blocks' 2dp rounding from
    # _histogram_df's 3dp rounding, and pin mean/median/std numerically.
    vals = [1.23456, 2.34567, 3.45678]
    arr = np.asarray(vals, dtype=float)
    block = distribution._droplet_stats_block(vals)
    for key, ref in (
        ("min", arr.min()), ("max", arr.max()), ("mean", arr.mean()),
        ("median", np.median(arr)), ("std", arr.std()),
    ):
        assert block["stats"][key] == round(float(ref), 2), key
    assert block["stats"]["mean"] != round(float(arr.mean()), 3)  # proves 2dp, not 3dp

    edges = [1.11111, 2.22222, 4.44444]
    df = distribution._histogram_df(vals, edges)
    assert df["bin_lo"].tolist() == [1.111, 2.222]    # proves 3dp
    assert df["bin_hi"].tolist() == [2.222, 4.444]
    assert df["bin_lo"].tolist() != [1.11, 2.22]      # proves not 2dp
    assert df["bin_center"].tolist() == [
        float(np.round((1.11111 + 2.22222) / 2.0, 3)),
        float(np.round((2.22222 + 4.44444) / 2.0, 3)),
    ]


def test_eq_diameter_formula():
    # d = sqrt(4A/pi): area pi/4 -> d = 1
    out = distribution._eq_diameter([math.pi / 4.0, 16.0])
    assert out[0] == pytest.approx(1.0, abs=1e-12)
    assert out[1] == pytest.approx(math.sqrt(64.0 / math.pi), abs=1e-12)
