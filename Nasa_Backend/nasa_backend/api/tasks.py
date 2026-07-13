# nasa_backend/api/tasks.py
"""In-memory task registry shared by the worker threads and the SSE stream.
State is lost on restart; there is no persistence layer (spec non-goal)."""
import uuid

tasks = {}

# download_id -> absolute xlsx path; entries live for the process lifetime,
# matching the in-memory tasks dict (no persistence by design).
downloads = {}

# Seconds the SSE stream waits for an event before deciding keep-alive/timeout.
SSE_IDLE_TIMEOUT = 300


def register_download(path):
    did = uuid.uuid4().hex
    downloads[did] = str(path)
    return did


def _sse_idle_decision(task):
    """Decide what an SSE stream should do after an idle (queue-empty) gap.

    ``task`` is ``tasks.get(task_id)`` — the task record, or ``None`` if it was
    already popped. Returns ``"keep-alive"`` when the worker is still running
    (record present and not yet completed), meaning the stream should emit a
    heartbeat comment and keep waiting; returns ``"timeout"`` when the task is
    gone or finished, meaning the stream should report a real timeout and close.
    """
    if task is not None and not task.get("completed"):
        return "keep-alive"
    return "timeout"


def _mark_task_completed(task_id):
    """Mark a task completed, tolerating the SSE side having already popped it.

    Returns ``True`` if the record still existed and was updated, ``False`` if
    it was already removed (e.g. the client disconnected). This keeps the worker
    thread from raising ``KeyError`` in its ``finally`` after doing all its work.
    """
    if task_id in tasks:
        tasks[task_id]["completed"] = True
        return True
    return False
