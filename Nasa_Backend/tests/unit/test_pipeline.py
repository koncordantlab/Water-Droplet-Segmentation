# tests/unit/test_pipeline.py
"""Tier-1 end-to-end: process_video on a tiny synthetic clip with a fake
model — no weights, no GPU. Pins checkpoint selection (N, 2N, ..., final),
the seven avg-size headers byte-identical across Per-Frame and Summary
sheets, and the size_distribution payload shape."""
import math
import os

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from nasa_backend import pipeline
import nasa_backend.model as model_mod

SEVEN_HEADERS = [
    "Water Avg Area (µm²)", "Water Avg Diameter (µm)",
    "Ice Avg Area (µm²)", "Ice Avg Diameter (µm)",
    "All Avg Area (µm²)", "All Avg Diameter (µm)",
    "Resolution (pix/µm²)",
]


class _Box:
    def __init__(self, cls_id, conf, xyxy):
        self.cls = torch.tensor([float(cls_id)])
        self.conf = torch.tensor([conf])
        self.xyxy = torch.tensor([xyxy], dtype=torch.float32)


class _FakeResult:
    def __init__(self, h, w):
        from types import SimpleNamespace

        self.orig_shape = (h, w)
        self.names = {0: "water", 1: "ice"}
        m = torch.zeros((2, h, w), dtype=torch.float32)
        m[0, 8:20, 8:20] = 1.0   # water square
        m[1, 30:40, 30:44] = 1.0  # ice rectangle
        self.masks = SimpleNamespace(data=m)
        self.boxes = [_Box(0, 0.9, [8, 8, 20, 20]), _Box(1, 0.8, [30, 30, 44, 40])]


class _FakeModel:
    def predict(self, frames):
        return [_FakeResult(64, 64) for _ in frames]


@pytest.fixture
def synthetic_video(tmp_path):
    """40 frames at 10 fps -> stride 10 -> processed frames 1..4."""
    path = str(tmp_path / "synth.mp4")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    for i in range(40):
        frame = np.full((64, 64, 3), i % 255, dtype=np.uint8)
        w.write(frame)
    w.release()
    assert os.path.getsize(path) > 0
    return path


@pytest.fixture
def synthetic_video_60(tmp_path):
    """60 frames at 10 fps -> stride 10 -> processed frames 1..6."""
    path = str(tmp_path / "synth60.mp4")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    for i in range(60):
        frame = np.full((64, 64, 3), i % 255, dtype=np.uint8)
        w.write(frame)
    w.release()
    assert os.path.getsize(path) > 0
    return path


def test_fake_model_end_to_end(synthetic_video, synthetic_video_60, tmp_path, monkeypatch):
    monkeypatch.setattr(model_mod, "get_model", lambda: _FakeModel())
    events = []
    msg, excel_path, rows, overlaps, charts, exec_time, size_dist = pipeline.process_video(
        synthetic_video, save_ovl=False, dist_interval=3,
        output_dir=str(tmp_path / "out"), progress_callback=events.append,
        output_mode="full", um_per_px=2.0,
    )
    assert excel_path and os.path.isfile(excel_path)
    assert len(rows) == 4

    # checkpoint selection: N, 2N, ... plus the final processed frame
    per_frame_dir = os.path.join(str(tmp_path / "out"), "synth_per_frame_xlsx")
    written = sorted(os.listdir(per_frame_dir))
    assert [f for f in written if f.endswith(".xlsx")] == [
        "synth_frame_000003_instances.xlsx",
        "synth_frame_000004_instances.xlsx",
    ]

    # seven avg-size headers byte-identical across both summary sheets
    sheets = pd.read_excel(excel_path, sheet_name=None)
    for h in SEVEN_HEADERS:
        assert h in sheets["Per-Frame"].columns, f"{h!r} missing from Per-Frame"
        assert h in sheets["Summary"].columns, f"{h!r} missing from Summary"

    # size_distribution shape
    assert size_dist and size_dist["interval"] == 3
    assert [c["frame"] for c in size_dist["checkpoints"]] == [3, 4]

    # progress events carried eta + progress
    processing = [e for e in events if e.get("status") == "processing"]
    assert processing and all("eta" in e and "progress" in e for e in processing)

    # Deferred full-res gather: 60 frames -> processed 1..6, dist_interval=4 ->
    # checkpoint at frame 4 only (6 % 4 != 0), so the final processed frame (6)
    # is written via the last_frame_raw stash of the small SOURCE masks
    # ("binm"), materialized lazily at the final-frame write rather than
    # gathered eagerly during process_batch like the checkpoint frame was.
    *_, size_dist_60 = pipeline.process_video(
        synthetic_video_60, save_ovl=False, dist_interval=4,
        output_dir=str(tmp_path / "out60"), output_mode="full", um_per_px=2.0,
    )
    per_frame_dir_60 = os.path.join(str(tmp_path / "out60"), "synth60_per_frame_xlsx")
    written_60 = sorted(os.listdir(per_frame_dir_60))
    assert [f for f in written_60 if f.endswith(".xlsx")] == [
        "synth60_frame_000004_instances.xlsx",
        "synth60_frame_000006_instances.xlsx",
    ]
    assert [c["frame"] for c in size_dist_60["checkpoints"]] == [4, 6]


def test_dist_interval_zero_yields_no_size_distribution(synthetic_video, tmp_path, monkeypatch):
    monkeypatch.setattr(model_mod, "get_model", lambda: _FakeModel())
    *_, size_dist = pipeline.process_video(
        synthetic_video, save_ovl=False, dist_interval=0,
        output_dir=str(tmp_path / "out2"), output_mode="basic", um_per_px=None,
    )
    assert size_dist is None
