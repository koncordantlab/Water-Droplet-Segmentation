"""Standalone checks for the SSE idle-timeout / keep-alive logic.

Reproduces the failure that crashed a real batch run: a healthy worker went
silent for >5 min while saving per-frame xlsx files, the SSE stream treated the
silence as a timeout and popped the task, and the worker then raised
``KeyError`` setting ``tasks[task_id]["completed"] = True``.

Covers:
  * ``_sse_idle_decision``      — keep-alive vs. real-timeout decision
  * ``_mark_task_completed``    — worker ``finally`` tolerates a popped task
  * the real ``event_stream``   — an idle gap on a live worker emits a heartbeat
                                  and the later final payload still arrives; a
                                  gap on a vanished task reports a real timeout

Run: python3 Nasa_Backend/test_sse_timeout.py
Exits non-zero on failure; prints ALL CHECKS PASSED on success.
No pytest dependency.
"""
import os
import sys
from queue import Queue

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# The module loads the YOLO model at import time using a relative weights path
# (app_root/weights_DP(*).pt), so we must import from within Nasa_Backend/.
os.chdir(_HERE)
import frontend_nasa13_apiV2 as mod


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_idle_decision_keeps_alive_while_worker_runs():
    # Record present and not finished -> the worker is alive, keep waiting.
    _check(mod._sse_idle_decision({"completed": False}) == "keep-alive",
           "not-completed task should keep-alive")
    # Missing 'completed' key is treated as not-yet-completed (still running).
    _check(mod._sse_idle_decision({}) == "keep-alive",
           "task without 'completed' key should keep-alive")


def test_idle_decision_times_out_when_gone_or_done():
    # Task already popped -> nothing left to wait for, report a real timeout.
    _check(mod._sse_idle_decision(None) == "timeout",
           "missing task should time out")
    # Worker marked completed -> finished; an idle gap now is a real timeout.
    _check(mod._sse_idle_decision({"completed": True}) == "timeout",
           "completed task should time out")


def test_mark_task_completed_present():
    tid = "test-mark-present"
    mod.tasks[tid] = {"queue": Queue(), "completed": False, "status": "queued"}
    try:
        _check(mod._mark_task_completed(tid) is True,
               "marking a present task should return True")
        _check(mod.tasks[tid]["completed"] is True,
               "present task should be flagged completed")
    finally:
        mod.tasks.pop(tid, None)


def test_mark_task_completed_absent_does_not_raise():
    tid = "test-mark-absent"
    mod.tasks.pop(tid, None)  # ensure absent
    # This is the exact line that crashed the worker's finally in the wild;
    # the helper must swallow the missing key instead of raising KeyError.
    _check(mod._mark_task_completed(tid) is False,
           "marking an absent task should return False, not raise")


def test_event_stream_heartbeat_then_final_payload():
    """An idle gap on a live worker yields a heartbeat, and the final payload
    delivered afterwards is NOT lost."""
    tid = "test-stream-heartbeat"
    q = Queue()
    mod.tasks[tid] = {"queue": q, "completed": False, "status": "queued"}
    saved = mod.SSE_IDLE_TIMEOUT
    mod.SSE_IDLE_TIMEOUT = 0.05  # shrink the idle wait so the test is fast
    try:
        with mod.server.test_request_context(f"/api/events/{tid}"):
            gen = mod.api_events(tid).response  # the event_stream() generator

            # Queue is empty and the worker isn't done -> first frame is a
            # keep-alive comment (EventSource silently ignores comment lines).
            first = next(gen)
            _check(first == ": keep-alive\n\n",
                   f"expected heartbeat comment, got {first!r}")

            # Worker now finishes: pushes its result, marks completed, signals
            # done -- mirroring process order in worker()'s try/finally.
            q.put_nowait({"status": "finished", "data": {"ok": True}})
            mod._mark_task_completed(tid)
            q.put_nowait({"__done__": True})

            rest = list(gen)  # drain to completion
    finally:
        mod.SSE_IDLE_TIMEOUT = saved
        mod.tasks.pop(tid, None)

    _check(any('"status": "finished"' in f for f in rest),
           f"final payload was dropped after the idle gap: {rest!r}")
    _check(rest and rest[-1] == 'data: {"status": "closed"}\n\n',
           f"stream should end with a 'closed' frame: {rest!r}")
    # event_stream's own cleanup pops the task after the loop ends.
    _check(tid not in mod.tasks, "task should be popped once the stream closes")


def test_event_stream_real_timeout_when_task_vanishes():
    """If the task is gone (client disconnected / cleaned up), an idle gap
    reports a real timeout instead of looping forever."""
    tid = "test-stream-timeout"
    q = Queue()
    mod.tasks[tid] = {"queue": q, "completed": False, "status": "queued"}
    saved = mod.SSE_IDLE_TIMEOUT
    mod.SSE_IDLE_TIMEOUT = 0.05
    try:
        with mod.server.test_request_context(f"/api/events/{tid}"):
            gen = mod.api_events(tid).response  # passes the initial 404 check
            mod.tasks.pop(tid, None)            # task vanishes mid-stream
            frame = next(gen)
    finally:
        mod.SSE_IDLE_TIMEOUT = saved
        mod.tasks.pop(tid, None)

    _check("Timeout: No updates for 5 minutes" in frame,
           f"expected a real timeout frame, got {frame!r}")


def main():
    tests = [
        test_idle_decision_keeps_alive_while_worker_runs,
        test_idle_decision_times_out_when_gone_or_done,
        test_mark_task_completed_present,
        test_mark_task_completed_absent_does_not_raise,
        test_event_stream_heartbeat_then_final_payload,
        test_event_stream_real_timeout_when_task_vanishes,
    ]
    for t in tests:
        t()
        print(f"  ok: {t.__name__}")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
