"""Smoke: the api package builds a Flask app (via create_app) and exposes the
expected route surface."""
from droplet_backend.api.tasks import SSE_IDLE_TIMEOUT
from droplet_backend.config import SIZE_DIST_BINS


def test_module_imports_and_flask_surface(app):
    assert app is not None
    rules = {r.rule for r in app.url_map.iter_rules()}
    assert "/api/process" in rules
    assert "/api/events/<task_id>" in rules
    assert "/api/status" in rules
    assert "/api/download_summary" in rules


def test_status_endpoint(app):
    client = app.test_client()
    resp = client.get("/api/status")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok", "message": "API is running"}


def test_key_constants_unchanged():
    # Behavior-freeze tripwires: these values are part of the numeric contract.
    assert SIZE_DIST_BINS == 30
    assert SSE_IDLE_TIMEOUT == 300
