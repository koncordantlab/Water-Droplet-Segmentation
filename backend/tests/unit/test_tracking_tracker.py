"""Tier-1 tests for the Tracker core (no video, no files) and the thin
orchestrator's frame handling."""
import json
from pathlib import Path

import cv2
import numpy as np

from tracking import events
from tracking.config import DEFAULT_CONFIG
from tracking.detections import FrameDetections, frame_detections_from_record

FIXTURES = Path(__file__).parent.parent / "fixtures" / "tracking"


def _square(x0, y0, size):
    return np.array([[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size]],
                    dtype=np.float32)


def _dets(*squares):
    dets = FrameDetections()
    for i, (x0, y0, size) in enumerate(squares):
        dets.append(_square(x0, y0, size), float(size * size), 0.9, 0, i)
    return dets


def _blank_video(path, n_frames, size=640):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (size, size))
    for _ in range(n_frames):
        writer.write(np.zeros((size, size, 3), dtype=np.uint8))
    writer.release()


def test_first_frame_births_get_sequential_ids():
    from tracking.tracker import Tracker
    tracker = Tracker(DEFAULT_CONFIG)
    result = tracker.step(1, _dets((0, 0, 20), (100, 0, 20)))
    assert result.events == [events.birth(1, 1, 0), events.birth(2, 1, 1)]
    assert list(tracker.tracks) == [1, 2]


def test_empty_first_frame_then_detections():
    from tracking.tracker import Tracker
    tracker = Tracker(DEFAULT_CONFIG)
    assert tracker.step(1, FrameDetections()).events == []
    assert tracker.step(2, _dets((0, 0, 20))).events == [events.birth(1, 1, 0)]


def test_track_dies_after_max_missed_frames():
    from tracking.tracker import Tracker
    tracker = Tracker(DEFAULT_CONFIG.with_overrides({"max_missed_frames": 2}))
    tracker.step(1, _dets((0, 0, 20)))
    assert tracker.step(2, FrameDetections()).events == []
    assert tracker.step(3, FrameDetections()).events == []
    assert tracker.step(4, FrameDetections()).events == [events.death(1, 1)]
    assert tracker.tracks == {}


def test_tracker_reproduces_synthetic_fixture_without_video():
    from tracking.tracker import Tracker
    records = json.loads((FIXTURES / "synthetic_detections.json").read_text())
    expected = json.loads((FIXTURES / "expected_tracking_log.json").read_text())
    tracker = Tracker(DEFAULT_CONFIG)
    got = [tracker.step(r["frame"], frame_detections_from_record(r, DEFAULT_CONFIG)).events
           for r in records]
    assert json.loads(json.dumps(got)) == [f["events"] for f in expected]


def test_video_shorter_than_detections_stops_at_last_frame(tmp_path):
    from tracking.track import track_from_detections_json
    _blank_video(tmp_path / "v.mp4", 3)
    log = tmp_path / "log.json"
    track_from_detections_json(str(tmp_path / "v.mp4"), str(FIXTURES / "synthetic_detections.json"),
                               str(tmp_path / "out.mp4"), str(log))
    assert [f["frame"] for f in json.loads(log.read_text())] == [1, 2, 3]


def test_missing_frame_key_falls_back_to_position(tmp_path):
    from tracking.track import track_from_detections_json
    records = json.loads((FIXTURES / "synthetic_detections.json").read_text())
    for record in records:
        record.pop("frame")
    (tmp_path / "dets.json").write_text(json.dumps(records))
    _blank_video(tmp_path / "v.mp4", 4)
    log = tmp_path / "log.json"
    track_from_detections_json(str(tmp_path / "v.mp4"), str(tmp_path / "dets.json"),
                               str(tmp_path / "out.mp4"), str(log))
    assert [f["frame"] for f in json.loads(log.read_text())] == [1, 2, 3, 4]
