"""Smoke: the app module imports (stubbed or real) and exposes the API surface."""


def test_module_imports_and_flask_surface(app_module):
    assert app_module.server is not None
    rules = {r.rule for r in app_module.server.url_map.iter_rules()}
    assert "/api/process" in rules
    assert "/api/events/<task_id>" in rules
    assert "/api/status" in rules
    assert "/api/download_summary" in rules


def test_status_endpoint(app_module):
    client = app_module.server.test_client()
    resp = client.get("/api/status")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok", "message": "API is running"}


def test_key_constants_unchanged(app_module):
    # Behavior-freeze tripwires: these values are part of the numeric contract.
    assert app_module.SIZE_DIST_BINS == 30
    assert app_module.SSE_IDLE_TIMEOUT == 300
