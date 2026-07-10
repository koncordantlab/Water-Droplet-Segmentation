"""SSE idle-timeout / keep-alive logic. Migrated from test_sse_timeout.py.
Covers _sse_idle_decision, _mark_task_completed, and the live event_stream
generator (heartbeat on idle, final payload not lost, real timeout when the
task vanishes)."""
from queue import Queue

import pytest


def test_idle_decision_keeps_alive_while_worker_runs(app_module):
    assert app_module._sse_idle_decision({"completed": False}) == "keep-alive"
    assert app_module._sse_idle_decision({}) == "keep-alive"


def test_idle_decision_times_out_when_gone_or_done(app_module):
    assert app_module._sse_idle_decision(None) == "timeout"
    assert app_module._sse_idle_decision({"completed": True}) == "timeout"


def test_mark_task_completed_present(app_module):
    tid = "test-mark-present"
    app_module.tasks[tid] = {"queue": Queue(), "completed": False, "status": "queued"}
    try:
        assert app_module._mark_task_completed(tid) is True
        assert app_module.tasks[tid]["completed"] is True
    finally:
        app_module.tasks.pop(tid, None)


def test_mark_task_completed_absent_does_not_raise(app_module):
    app_module.tasks.pop("test-mark-absent", None)
    assert app_module._mark_task_completed("test-mark-absent") is False


@pytest.fixture
def fast_sse(app_module, monkeypatch):
    monkeypatch.setattr(app_module, "SSE_IDLE_TIMEOUT", 0.05)
    return app_module


def test_event_stream_heartbeat_then_final_payload(fast_sse):
    mod = fast_sse
    tid = "test-stream-heartbeat"
    q = Queue()
    mod.tasks[tid] = {"queue": q, "completed": False, "status": "queued"}
    try:
        with mod.server.test_request_context(f"/api/events/{tid}"):
            gen = mod.api_events(tid).response  # the event_stream() generator
            first = next(gen)
            assert first == ": keep-alive\n\n"
            q.put_nowait({"status": "finished", "data": {"ok": True}})
            mod._mark_task_completed(tid)
            q.put_nowait({"__done__": True})
            rest = list(gen)
    finally:
        mod.tasks.pop(tid, None)
    assert any('"status": "finished"' in f for f in rest)
    assert rest and rest[-1] == 'data: {"status": "closed"}\n\n'
    assert tid not in mod.tasks  # stream cleanup pops the task


def test_event_stream_real_timeout_when_task_vanishes(fast_sse):
    mod = fast_sse
    tid = "test-stream-timeout"
    mod.tasks[tid] = {"queue": Queue(), "completed": False, "status": "queued"}
    try:
        with mod.server.test_request_context(f"/api/events/{tid}"):
            gen = mod.api_events(tid).response
            mod.tasks.pop(tid, None)  # task vanishes mid-stream
            frame = next(gen)
    finally:
        mod.tasks.pop(tid, None)
    assert "Timeout: No updates for 5 minutes" in frame


def test_events_unknown_task_is_404(app_module):
    client = app_module.server.test_client()
    resp = client.get("/api/events/does-not-exist")
    assert resp.status_code == 404
    assert resp.get_json() == {"status": "error", "message": "Invalid task_id"}
