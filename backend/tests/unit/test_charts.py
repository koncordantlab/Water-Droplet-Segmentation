"""Chart PNG writers produce files headlessly (Agg). Pixel content is not
golden-pinned — only that every expected artifact lands non-empty."""
import os

import pandas as pd

from droplet_backend import charts


def _sample_inputs():
    rows = [
        {"Frame Number": i, "Water (%)": 50.0 - i, "Ice (%)": 10.0 + i,
         "Overlap_Water-Water": i, "Overlap_Ice-Ice": 0, "Overlap_Water-Ice": 1,
         "water_cnt": 5, "ice_cnt": 2, "void_pct": 40.0, "avg_conf": 88.0}
        for i in range(1, 4)
    ]
    df = pd.DataFrame(rows)
    charts_payload = {
        "pct": {"x": [1, 2, 3], "water": [49, 48, 47], "ice": [11, 12, 13]},
        "ov": {"x": [1, 2, 3], "ww": [1, 2, 3], "ii": [0, 0, 0], "wi": [1, 1, 1]},
        "donuts": {"water_count": 15, "ice_count": 6, "void_pct_avg": 40.0, "avg_conf": 88.0},
    }
    return df, charts_payload, {"ww": 6, "ii": 0, "mixed": 3}


def test_save_chart_pngs_writes_files(tmp_path):
    df, charts_payload, overlaps = _sample_inputs()
    charts._save_chart_pngs(df, charts_payload, overlaps, str(tmp_path))
    pngs = [f for f in os.listdir(tmp_path) if f.endswith(".png")]
    assert pngs, "no chart PNGs written"
    for f in pngs:
        assert os.path.getsize(tmp_path / f) > 1024, f"{f} suspiciously small"


def test_save_size_distribution_pngs_writes_per_checkpoint(tmp_path):
    sd = {"interval": 2, "unit": "processed_frames", "bin_count": 3,
          "y_max": {"water": 4, "ice": 2},
          "checkpoints": [{
              "frame": 2,
              "water": {"count": 3, "histogram": {
                  "bin_edges": [1.0, 2.0, 4.0, 8.0],
                  "counts": [1, 1, 1]},
                  "stats": {"min": 1.1, "max": 6.0,
                           "mean": 3.0, "median": 2.5, "std": 1.9}},
              "ice": {"count": 1, "histogram": {
                  "bin_edges": [2.0, 4.0, 8.0],
                  "counts": [1, 0]},
                  "stats": {"min": 2.5, "max": 4.0,
                           "mean": 3.0, "median": 3.0, "std": 0.7}},
          }]}
    charts._save_size_distribution_pngs(sd, str(tmp_path), "vid")
    pngs = [f for f in os.listdir(tmp_path) if f.endswith(".png")]
    assert pngs, "no size-distribution PNGs written"
    for f in pngs:
        assert os.path.getsize(tmp_path / f) > 1024, f"{f} suspiciously small"
