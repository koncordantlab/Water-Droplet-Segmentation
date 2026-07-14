"""SSE idle-timeout / keep-alive logic. Migrated from test_sse_timeout.py.
Covers _sse_idle_decision, _mark_task_completed, and the live event_stream
generator (heartbeat on idle, final payload not lost, real timeout when the
task vanishes)."""
from queue import Queue

import pytest

from droplet_backend.api import routes, tasks as tasks_mod


def test_idle_decision_keeps_alive_while_worker_runs():
    assert tasks_mod._sse_idle_decision({"completed": False}) == "keep-alive"
    assert tasks_mod._sse_idle_decision({}) == "keep-alive"


def test_idle_decision_times_out_when_gone_or_done():
    assert tasks_mod._sse_idle_decision(None) == "timeout"
    assert tasks_mod._sse_idle_decision({"completed": True}) == "timeout"


def test_mark_task_completed_present():
    tid = "test-mark-present"
    tasks_mod.tasks[tid] = {"queue": Queue(), "completed": False, "status": "queued"}
    try:
        assert tasks_mod._mark_task_completed(tid) is True
        assert tasks_mod.tasks[tid]["completed"] is True
    finally:
        tasks_mod.tasks.pop(tid, None)


def test_mark_task_completed_absent_does_not_raise():
    tasks_mod.tasks.pop("test-mark-absent", None)
    assert tasks_mod._mark_task_completed("test-mark-absent") is False


@pytest.fixture
def fast_sse(app, monkeypatch):
    monkeypatch.setattr(tasks_mod, "SSE_IDLE_TIMEOUT", 0.05)
    return app


def test_event_stream_heartbeat_then_final_payload(fast_sse):
    app = fast_sse
    tid = "test-stream-heartbeat"
    q = Queue()
    tasks_mod.tasks[tid] = {"queue": q, "completed": False, "status": "queued"}
    try:
        with app.test_request_context(f"/api/events/{tid}"):
            gen = routes.api_events(tid).response  # the event_stream() generator
            first = next(gen)
            assert first == ": keep-alive\n\n"
            q.put_nowait({"status": "finished", "data": {"ok": True}})
            tasks_mod._mark_task_completed(tid)
            q.put_nowait({"__done__": True})
            rest = list(gen)
    finally:
        tasks_mod.tasks.pop(tid, None)
    assert any('"status": "finished"' in f for f in rest)
    assert rest and rest[-1] == 'data: {"status": "closed"}\n\n'
    assert tid not in tasks_mod.tasks  # stream cleanup pops the task


def test_event_stream_real_timeout_when_task_vanishes(fast_sse):
    app = fast_sse
    tid = "test-stream-timeout"
    tasks_mod.tasks[tid] = {"queue": Queue(), "completed": False, "status": "queued"}
    try:
        with app.test_request_context(f"/api/events/{tid}"):
            gen = routes.api_events(tid).response
            tasks_mod.tasks.pop(tid, None)  # task vanishes mid-stream
            frame = next(gen)
    finally:
        tasks_mod.tasks.pop(tid, None)
    assert "Timeout: No updates for 5 minutes" in frame


def test_events_unknown_task_is_404(app):
    client = app.test_client()
    resp = client.get("/api/events/does-not-exist")
    assert resp.status_code == 404
    assert resp.get_json() == {"status": "error", "message": "Invalid task_id"}


def test_event_stream_gives_up_after_max_keepalives(app, monkeypatch):
    from droplet_backend.api import routes, tasks as tasks_mod
    from queue import Queue
    monkeypatch.setattr(tasks_mod, "SSE_IDLE_TIMEOUT", 0.02)
    monkeypatch.setattr(tasks_mod, "SSE_MAX_KEEPALIVES", 3)
    tid = "test-hung-worker"
    tasks_mod.tasks[tid] = {"queue": Queue(), "completed": False, "status": "queued"}
    try:
        with app.test_request_context(f"/api/events/{tid}"):
            frames = list(routes.api_events(tid).response)
    finally:
        tasks_mod.tasks.pop(tid, None)
    keepalives = [f for f in frames if f == ": keep-alive\n\n"]
    assert len(keepalives) == 3, "must stop keep-alive after the cap"
    assert "unresponsive" in frames[-1]
    assert tid not in tasks_mod.tasks, "hung task entry must be reclaimed"
