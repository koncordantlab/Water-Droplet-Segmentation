# nasa_backend/api/routes.py
"""The four REST/SSE routes consumed by the React frontend. Handler bodies are
verbatim monolith moves; process_video and the task registry are resolved as
module attributes at call time (monkeypatch-friendly, lazy model)."""
import json
import os
import threading
import traceback
import uuid
from queue import Empty, Queue

from flask import Blueprint, Response, jsonify, request, send_file, stream_with_context

from nasa_backend import pipeline
from nasa_backend.api import tasks as tasks_mod
from nasa_backend.pipeline import _list_videos_in_dir
from nasa_backend.serialization import make_json_serializable

api_bp = Blueprint("api", __name__, url_prefix="/api")


def _video_roots():
    """Allowed video roots from NASA_VIDEO_ROOTS (colon-separated); defaults to
    the invoking user's home directory, preserving the paste-a-path workflow
    while bounding it (CodeQL uncontrolled-path fix)."""
    raw = os.environ.get("NASA_VIDEO_ROOTS") or os.path.expanduser("~")
    return [os.path.realpath(r) for r in raw.split(":") if r.strip()]


def _path_allowed(p):
    rp = os.path.realpath(p)
    for root in _video_roots():
        try:
            if os.path.commonpath([rp, root]) == root:
                return True
        except ValueError:  # e.g. different drives; treat as not allowed
            continue
    return False


# REST API endpoints (synchronous) -------------------------------------------
@api_bp.route('/process', methods=['POST'])
def api_process():
    """Enqueue processing task and return task_id. Client should connect to /api/events/<task_id> for SSE."""
    print("Received /api/process request")

    data = request.get_json(force=True, silent=True) or {}
    video_path = data.get('video_path')
    save_ovl = data.get('save_overlay', True)
    try:
        dist_interval = int(data.get('dist_interval', 0) or 0)
    except (TypeError, ValueError):
        dist_interval = 0
    if dist_interval < 0:
        dist_interval = 0
    output_mode = str(data.get('output_mode', 'full')).strip().lower()
    if output_mode not in ("basic", "full"):
        output_mode = "full"
    try:
        um_per_px = float(data.get('um_per_px'))
        if um_per_px <= 0:
            um_per_px = None
    except (TypeError, ValueError):
        um_per_px = None
    if not video_path:
        return jsonify({"status": "error", "message": "Missing video_path"}), 400
    if not (os.path.isfile(video_path) or os.path.isdir(video_path)):
        return jsonify({"status": "error", "message": f"Path is neither a file nor a directory: {video_path}"}), 400
    # Resolve ONCE, before the gate, so the checked path and the used path are
    # the same resolution (no check/use divergence window); the worker closure
    # then captures this frozen canonical path (TOCTOU).
    video_path = os.path.realpath(video_path)
    if not _path_allowed(video_path):
        return jsonify({"status": "error",
                        "message": "Path is outside the allowed video directories (see NASA_VIDEO_ROOTS)"}), 403

    task_id = uuid.uuid4().hex
    task_queue = Queue()
    tasks_mod.tasks[task_id] = {"queue": task_queue, "completed": False, "status": "queued"}

    # url_for(_external=True) needs a Flask request context, which the worker
    # thread doesn't have. Capture the host here and build URLs by string concat.
    host_url = request.host_url.rstrip('/')

    def _build_download_url(excel_path):
        if not excel_path:
            return None
        return f"{host_url}/api/download_summary?id={tasks_mod.register_download(excel_path)}"

    def worker():
        try:
            def push(ev):
                try:
                    task_queue.put_nowait(make_json_serializable(ev))
                except Exception:
                    try:
                        task_queue.put_nowait({"message": str(ev)})
                    except Exception:
                        pass

            def _single_video_payload(msg, excel_path, rows, overlaps, charts, execution_time, size_distribution):
                return {
                    "status": "ok" if excel_path else "error",
                    "message": str(msg),
                    "charts": make_json_serializable(charts) if charts else None,
                    "rows": make_json_serializable(rows) if rows else None,
                    "overlaps": make_json_serializable(overlaps) if overlaps else None,
                    "excel_path": str(excel_path) if excel_path else None,
                    "download_url": _build_download_url(excel_path),
                    "execution_time": execution_time,
                    "size_distribution": make_json_serializable(size_distribution) if size_distribution else None,
                }

            if os.path.isdir(video_path):
                # Batch mode: process every video in the directory, sequentially.
                resolved = [os.path.realpath(v) for v in _list_videos_in_dir(video_path)]
                videos, disallowed = [], []
                for v in resolved:
                    (videos if _path_allowed(v) else disallowed).append(v)
                if disallowed:
                    print(f"⚠️  Skipping {len(disallowed)} entries resolving outside NASA_VIDEO_ROOTS: {disallowed}")
                print(f"📂 Batch mode: found {len(videos)} video(s) in {video_path}")
                for _v in videos:
                    print(f"   - {_v}")
                if not videos:
                    push({"status": "error", "message": f"No video files found in directory: {video_path}"})
                    task_queue.put_nowait({"status": "error", "message": f"No video files found in directory: {video_path}"})
                    return

                batch_results = []
                last_payload = None
                total = len(videos)
                for idx, vid in enumerate(videos, start=1):
                    print(f"▶️  [{idx}/{total}] Starting {os.path.basename(vid)}")
                    stem = os.path.splitext(os.path.basename(vid))[0]
                    out_dir = os.path.join(video_path, stem)
                    vid_name = os.path.basename(vid)
                    push({
                        "status": "video_started",
                        "message": f"Processing video {idx}/{total}: {vid_name}",
                        "video_index": idx,
                        "video_total": total,
                        "current_video": vid_name,
                    })

                    # Stamp every event with batch position, convert per-video
                    # progress into global progress, and demote intermediate
                    # "completed" events so the frontend doesn't close the SSE
                    # after the first video finishes.
                    def video_push(ev, _idx=idx, _total=total, _name=vid_name):
                        if not isinstance(ev, dict):
                            push(ev)
                            return
                        ev = dict(ev)
                        ev["video_index"] = _idx
                        ev["video_total"] = _total
                        ev["current_video"] = _name
                        if isinstance(ev.get("progress"), (int, float)):
                            per_video_pct = float(ev["progress"])
                            ev["progress"] = round(((_idx - 1) + per_video_pct / 100.0) / _total * 100.0, 2)
                        if ev.get("status") == "completed" and _idx < _total:
                            ev = {
                                "status": "video_completed",
                                "message": ev.get("message") or f"Finished {_name}",
                                "execution_time": ev.get("execution_time"),
                                "excel_path": ev.get("excel_path"),
                                "video_index": _idx,
                                "video_total": _total,
                                "current_video": _name,
                                "progress": round((_idx / _total) * 100.0, 2),
                            }
                        push(ev)

                    # Per-video try/except so one failure can't kill the whole batch.
                    # process_video has its own internal try/except for the main work,
                    # but the cv2 capture-open and frame-counting steps live outside
                    # that block and can still raise.
                    try:
                        msg, excel_path, rows, overlaps, charts, exec_time, size_dist = pipeline.process_video(
                            vid, save_ovl,
                            dist_interval=dist_interval,
                            output_dir=out_dir,
                            progress_callback=video_push,
                            output_mode=output_mode,
                            um_per_px=um_per_px,
                        )
                    except Exception as per_vid_err:
                        print(f"⚠️  Video {idx}/{total} ({vid_name}) raised: {per_vid_err!r}")
                        push({
                            "status": "error",
                            "message": f"Video {vid_name} failed: {per_vid_err}",
                            "video_index": idx,
                            "video_total": total,
                            "current_video": vid_name,
                        })
                        msg, excel_path = f"❌ {per_vid_err}", None
                        rows = overlaps = charts = size_dist = None
                        exec_time = None

                    batch_results.append({
                        "video": vid_name,
                        "video_path": vid,
                        "output_dir": out_dir,
                        "status": "ok" if excel_path else "error",
                        "message": str(msg),
                        "excel_path": str(excel_path) if excel_path else None,
                        "download_url": _build_download_url(excel_path),
                        "execution_time": exec_time,
                    })
                    if excel_path:
                        last_payload = _single_video_payload(msg, excel_path, rows, overlaps, charts, exec_time, size_dist)

                final_payload = last_payload or {"status": "error", "message": "All videos failed"}
                final_payload["batch_results"] = batch_results
                final_payload["batch_total"] = total
                task_queue.put_nowait({"status": "finished", "data": final_payload})
            else:
                # File mode (unchanged behavior).
                msg, excel_path, rows, overlaps, charts, execution_time, size_distribution = pipeline.process_video(
                    video_path, save_ovl, dist_interval=dist_interval, progress_callback=push,
                    output_mode=output_mode, um_per_px=um_per_px
                )
                task_queue.put_nowait({"status": "finished", "data": _single_video_payload(
                    msg, excel_path, rows, overlaps, charts, execution_time, size_distribution
                )})
        except Exception as e:
            print(f"❌ Worker fatal: {e!r}")
            traceback.print_exc()
            task_queue.put_nowait({"status": "error", "message": f"An error occurred: {e}"})
        finally:
            # The SSE stream may have already popped this task (e.g. the client
            # disconnected, or an earlier idle-timeout fired). Don't assume the
            # entry still exists — guard the lookup so a healthy worker can't
            # crash here after doing all its real work.
            tasks_mod._mark_task_completed(task_id)
            task_queue.put_nowait({"__done__": True})

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return jsonify({"status": "ok", "task_id": task_id}), 202


@api_bp.route('/events/<task_id>')
def api_events(task_id):
    """SSE endpoint to stream task progress and final result."""
    if task_id not in tasks_mod.tasks:
        return jsonify({"status": "error", "message": "Invalid task_id"}), 404

    task = tasks_mod.tasks[task_id]
    task_queue = task["queue"]

    def event_stream():
        while True:
            try:
                ev = task_queue.get(timeout=tasks_mod.SSE_IDLE_TIMEOUT)
            except Empty:
                # A healthy worker can go silent for well over 5 minutes during
                # the tail phases (saving chart PNGs and the per-frame xlsx
                # files), which push no progress events. Don't mistake that
                # silence for a dead worker and tear the task down underneath
                # it: while the task still exists and isn't finished, send an
                # SSE keep-alive comment (EventSource ignores comment lines) and
                # keep waiting. Only report a real timeout if the task is gone.
                if tasks_mod._sse_idle_decision(tasks_mod.tasks.get(task_id)) == "keep-alive":
                    yield ": keep-alive\n\n"
                    continue
                yield f"data: {json.dumps({'status': 'error', 'message': 'Timeout: No updates for 5 minutes'})}\n\n"
                break
            if ev is None:
                continue
            if isinstance(ev, dict) and ev.get("__done__"):
                yield f"data: {json.dumps({'status': 'closed'})}\n\n"
                break
            try:
                yield f"data: {json.dumps(ev)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'status': 'error', 'message': f'Error serializing event: {e}'})}\n\n"

        try:
            tasks_mod.tasks.pop(task_id, None)
        except Exception:
            pass
    return Response(stream_with_context(event_stream()), mimetype='text/event-stream')


@api_bp.route('/status', methods=['GET'])
def api_status():
    return jsonify({"status": "ok", "message": "API is running"}), 200


@api_bp.route('/download_summary')
def api_download_summary():
    did = request.args.get('id')
    path = tasks_mod.downloads.get(did) if did else None
    if not path or not os.path.isfile(path):
        return jsonify({"status": "error", "message": "Invalid or missing download id"}), 400
    return send_file(path, as_attachment=True)
