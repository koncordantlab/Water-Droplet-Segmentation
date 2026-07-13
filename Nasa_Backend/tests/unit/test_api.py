"""End-to-end REST/SSE flow with process_video monkeypatched — the whole
task/queue/worker/SSE machinery runs for real, only the video pipeline is fake.
Verifies the final SSE payload carries every field the React app reads."""
import json
import os
import time

import numpy as np

from nasa_backend import pipeline
from nasa_backend.api import routes


def _drain_sse(app, task_id, deadline_s=10.0):
    """Collect SSE frames for a task until the closed frame or deadline."""
    frames = []
    with app.test_request_context(f"/api/events/{task_id}"):
        gen = routes.api_events(task_id).response
        t0 = time.time()
        for frame in gen:
            frames.append(frame)
            if frame.startswith('data: {"status": "closed"}'):
                break
            assert time.time() - t0 < deadline_s, f"SSE never closed: {frames!r}"
    return frames


def _data_events(frames):
    out = []
    for f in frames:
        if f.startswith("data: "):
            out.append(json.loads(f[len("data: "):].strip()))
    return out


def test_process_full_flow_final_payload(app, monkeypatch, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00fake")
    excel = tmp_path / "clip_detection_summary.xlsx"
    excel.write_bytes(b"PK\x03\x04fake")
    size_dist = {"interval": 5, "unit": "processed_frames", "bin_count": 30,
                 "y_max": {"water": 4, "ice": 2}, "checkpoints": []}
    captured = {}

    def fake_process_video(video_path, save_ovl, dist_interval=0, output_dir=None,
                           progress_callback=None, output_mode="full", um_per_px=None):
        captured.update(video_path=video_path, save_ovl=save_ovl,
                        dist_interval=dist_interval, output_mode=output_mode,
                        um_per_px=um_per_px)
        if progress_callback:
            progress_callback({"status": "progress", "progress": 50.0,
                               "np_value": np.float64(1.5), "nan_value": float("nan")})
        rows = [{"Frame": 1, "Water (%)": 12.5}]
        overlaps = {"ww": 1, "ii": 0, "wi": 2}
        charts = {"pct": {"x": [1], "water": [12.5], "ice": [3.0]},
                  "ov": {"x": [1], "ww": [1], "ii": [0], "wi": [2]},
                  "donuts": {"water_count": 3, "ice_count": 1, "void_pct_avg": 84.5, "avg_conf": 0.9}}
        return ("✅ Processing complete!", str(excel), rows, overlaps, charts, 1.23, size_dist)

    monkeypatch.setattr(pipeline, "process_video", fake_process_video)
    client = app.test_client()
    resp = client.post("/api/process", json={
        "video_path": str(video), "save_overlay": False,
        "dist_interval": 5, "output_mode": "basic", "um_per_px": 2.5,
    })
    assert resp.status_code == 202
    body = resp.get_json()
    assert body["status"] == "ok" and body["task_id"]

    events = _data_events(_drain_sse(app, body["task_id"]))

    # progress event passed through make_json_serializable: NaN -> null
    progress = [e for e in events if e.get("status") == "progress"]
    assert progress and progress[0]["np_value"] == 1.5 and progress[0]["nan_value"] is None

    finished = [e for e in events if e.get("status") == "finished"]
    assert len(finished) == 1
    data = finished[0]["data"]
    assert data["status"] == "ok"
    assert data["rows"] == [{"Frame": 1, "Water (%)": 12.5}]
    assert data["overlaps"] == {"ww": 1, "ii": 0, "wi": 2}
    assert data["charts"]["donuts"]["water_count"] == 3
    assert data["size_distribution"] == size_dist
    assert data["execution_time"] == 1.23
    from nasa_backend.api import tasks as tasks_mod
    assert "/api/download_summary?id=" in data["download_url"]
    did = data["download_url"].rsplit("id=", 1)[1]
    assert tasks_mod.downloads[did] == str(excel)
    # the served id round-trips to the file
    dl = app.test_client().get(f"/api/download_summary?id={did}")
    assert dl.status_code == 200 and dl.data == b"PK\x03\x04fake"
    assert events[-1] == {"status": "closed"}

    # api_process parameter parsing reached process_video intact
    assert captured == {"video_path": str(video), "save_ovl": False,
                        "dist_interval": 5, "output_mode": "basic", "um_per_px": 2.5}


def test_process_validation_errors(app):
    client = app.test_client()
    r1 = client.post("/api/process", json={})
    assert r1.status_code == 400
    assert r1.get_json() == {"status": "error", "message": "Missing video_path"}
    r2 = client.post("/api/process", json={"video_path": "/nonexistent/x.mp4"})
    assert r2.status_code == 400
    assert r2.get_json()["message"].startswith("Path is neither a file nor a directory")


def test_process_sanitizes_bad_params(app, monkeypatch, tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00")
    captured = {}

    def fake_process_video(video_path, save_ovl, dist_interval=0, output_dir=None,
                           progress_callback=None, output_mode="full", um_per_px=None):
        captured.update(dist_interval=dist_interval, output_mode=output_mode,
                        um_per_px=um_per_px)
        return ("✅ ok", None, None, None, None, 0.1, None)

    monkeypatch.setattr(pipeline, "process_video", fake_process_video)
    client = app.test_client()
    resp = client.post("/api/process", json={
        "video_path": str(video), "dist_interval": "junk",
        "output_mode": "WEIRD", "um_per_px": -3,
    })
    assert resp.status_code == 202
    _drain_sse(app, resp.get_json()["task_id"])
    assert captured == {"dist_interval": 0, "output_mode": "full", "um_per_px": None}


def test_download_summary_serves_only_registered_ids(app, tmp_path):
    from nasa_backend.api import tasks as tasks_mod
    f = tmp_path / "summary.xlsx"
    f.write_bytes(b"PK\x03\x04data")
    client = app.test_client()
    did = tasks_mod.register_download(str(f))
    ok = client.get(f"/api/download_summary?id={did}")
    assert ok.status_code == 200 and ok.data == b"PK\x03\x04data"
    assert client.get("/api/download_summary?id=deadbeef").status_code == 400
    assert client.get(f"/api/download_summary?path={f}").status_code == 400  # old contract rejected
    assert client.get("/api/download_summary").status_code == 400


def test_process_rejects_paths_outside_allowed_roots(app, monkeypatch, tmp_path):
    allowed = tmp_path / "allowed"; allowed.mkdir()
    outside = tmp_path / "outside"; outside.mkdir()
    vid = outside / "v.mp4"; vid.write_bytes(b"\x00")
    monkeypatch.setenv("NASA_VIDEO_ROOTS", str(allowed))
    r = app.test_client().post("/api/process", json={"video_path": str(vid)})
    assert r.status_code == 403
    assert "allowed video directories" in r.get_json()["message"]


def test_process_accepts_paths_inside_allowed_roots(app, monkeypatch, tmp_path):
    from nasa_backend import pipeline
    vid = tmp_path / "v.mp4"; vid.write_bytes(b"\x00")
    monkeypatch.setenv("NASA_VIDEO_ROOTS", f"/nonexistent-root:{tmp_path}")
    monkeypatch.setattr(pipeline, "process_video",
                        lambda *a, **k: ("✅ ok", None, None, None, None, 0.1, None))
    r = app.test_client().post("/api/process", json={"video_path": str(vid)})
    assert r.status_code == 202


def test_process_freezes_symlink_resolution_at_check_time(app, monkeypatch, tmp_path):
    from nasa_backend import pipeline
    real_dir = tmp_path / "real"; real_dir.mkdir()
    vid = real_dir / "v.mp4"; vid.write_bytes(b"\x00")
    link_dir = tmp_path / "link"
    link_dir.symlink_to(real_dir, target_is_directory=True)
    monkeypatch.setenv("NASA_VIDEO_ROOTS", str(tmp_path))
    captured = {}

    def fake_process_video(video_path, *a, **k):
        captured["path"] = video_path
        return ("✅ ok", None, None, None, None, 0.1, None)

    monkeypatch.setattr(pipeline, "process_video", fake_process_video)
    r = app.test_client().post("/api/process", json={"video_path": str(link_dir / "v.mp4")})
    assert r.status_code == 202
    for _ in range(50):
        if "path" in captured:
            break
        time.sleep(0.1)
    assert captured["path"] == os.path.realpath(str(link_dir / "v.mp4"))
    assert "/link/" not in captured["path"]


def test_batch_mode_skips_escapes_and_freezes_entries(app, monkeypatch, tmp_path):
    from nasa_backend import pipeline
    allowed = tmp_path / "allowed"; allowed.mkdir()
    outside = tmp_path / "outside"; outside.mkdir()
    (allowed / "in.mp4").write_bytes(b"\x00")
    (outside / "secret.mp4").write_bytes(b"\x00")
    (allowed / "escape.mp4").symlink_to(outside / "secret.mp4")
    monkeypatch.setenv("NASA_VIDEO_ROOTS", str(allowed))
    processed = []

    def fake_process_video(video_path, *a, **k):
        processed.append(video_path)
        return ("✅ ok", None, None, None, None, 0.1, None)

    monkeypatch.setattr(pipeline, "process_video", fake_process_video)
    r = app.test_client().post("/api/process", json={"video_path": str(allowed)})
    assert r.status_code == 202
    for _ in range(50):
        if processed:
            break
        time.sleep(0.1)
    time.sleep(0.3)  # let the batch loop finish
    assert processed == [os.path.realpath(str(allowed / "in.mp4"))]
