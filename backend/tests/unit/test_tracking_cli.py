"""Tier-1 tests for the tracking CLI's settings handling (subprocess, CPU)."""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from tracking.config import DEFAULT_CONFIG

BACKEND = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent.parent / "fixtures" / "tracking"


def _cli(*args):
    return subprocess.run([sys.executable, "-m", "tracking", *args], cwd=BACKEND,
                          capture_output=True, text=True, timeout=300)


def _blank_video(path, n_frames, size=640):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (size, size))
    for _ in range(n_frames):
        writer.write(np.zeros((size, size, 3), dtype=np.uint8))
    writer.release()


def test_cli_config_prints_the_defaults():
    out = _cli("config")
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == json.loads(json.dumps(DEFAULT_CONFIG.to_dict()))


def test_cli_rejects_typo_in_config_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"direct_merge": {"min_iuo": 0.3}}))
    out = _cli("config", "--config", str(bad))
    assert out.returncode == 2
    assert "min_iuo" in out.stderr and "min_iou" in out.stderr


def test_cli_track_applies_overrides_and_writes_sidecar(tmp_path):
    _blank_video(tmp_path / "v.mp4", 4)
    overrides = tmp_path / "o.json"
    overrides.write_text(json.dumps({"direct_merge": {"enabled": False}}))
    for name, extra in (("default", []), ("override", ["--config", str(overrides)])):
        out = _cli("track", "--video", str(tmp_path / "v.mp4"),
                   "--detections", str(FIXTURES / "synthetic_detections.json"),
                   "--output", str(tmp_path / f"{name}.mp4"), "--log", str(tmp_path / f"{name}.json"), *extra)
        assert out.returncode == 0, out.stderr
    frame4 = {name: json.loads((tmp_path / f"{name}.json").read_text())[3]["events"]
              for name in ("default", "override")}
    assert [e["type"] for e in frame4["default"]] == ["merge"]
    assert "merge" not in [e["type"] for e in frame4["override"]]
    sidecar = json.loads((tmp_path / "override.config.json").read_text())
    assert sidecar["direct_merge"]["enabled"] is False
    assert json.loads((tmp_path / "default.config.json").read_text()) == \
        json.loads(json.dumps(DEFAULT_CONFIG.to_dict()))
