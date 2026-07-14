# tests/unit/test_webui.py
"""Flask serves the built React bundle at / with an index.html catch-all;
API routes are never shadowed; a missing bundle degrades to a plain page."""
import pytest


@pytest.fixture
def webui(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("<html><body>REACT</body></html>")
    (tmp_path / "static").mkdir()
    (tmp_path / "static" / "app.js").write_text("console.log(1)")
    monkeypatch.setenv("NASA_WEBUI_DIR", str(tmp_path))
    return tmp_path


def _fresh_app():
    from nasa_backend.api import create_app
    return create_app()


def test_root_and_spa_paths_serve_index(webui):
    client = _fresh_app().test_client()
    assert b"REACT" in client.get("/").data
    assert b"REACT" in client.get("/summary").data          # SPA route
    assert client.get("/static/app.js").data == b"console.log(1)"


def test_api_routes_not_shadowed(webui):
    client = _fresh_app().test_client()
    assert client.get("/api/status").get_json()["status"] == "ok"
    assert client.get("/api/nonexistent").status_code == 404


def test_bare_api_is_json_404_not_index(webui):
    """GET /api (no trailing slash) must be an API-style JSON 404, not the SPA
    index.html — the startswith("api/") guard alone misses the bare path."""
    client = _fresh_app().test_client()
    r = client.get("/api")
    assert r.status_code == 404
    body = r.get_json()
    assert body == {"status": "error", "message": "Unknown API route"}


def test_missing_bundle_degrades_gracefully(monkeypatch, tmp_path):
    monkeypatch.setenv("NASA_WEBUI_DIR", str(tmp_path / "nowhere"))
    client = _fresh_app().test_client()
    r = client.get("/")
    assert r.status_code == 200
    assert b"UI not built" in r.data
    assert client.get("/api/status").status_code == 200


def test_cors_allows_configured_origin_only(monkeypatch):
    monkeypatch.setenv("NASA_CORS_ORIGINS", "http://localhost:3000")
    client = _fresh_app().test_client()
    ok = client.get("/api/status", headers={"Origin": "http://localhost:3000"})
    assert ok.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"
    bad = client.get("/api/status", headers={"Origin": "https://evil.example"})
    assert bad.headers.get("Access-Control-Allow-Origin") is None


def test_wildcard_cors_refuses_to_start(monkeypatch):
    """A literal '*' origin with supports_credentials=True would reflect any
    Origin with Allow-Credentials: true — the factory must fail fast instead."""
    monkeypatch.setenv("NASA_CORS_ORIGINS", "*")
    with pytest.raises(RuntimeError, match="NASA_CORS_ORIGINS must not contain"):
        _fresh_app()


def test_wildcard_cors_refused_even_among_other_origins(monkeypatch):
    monkeypatch.setenv("NASA_CORS_ORIGINS", "http://localhost:3000,*")
    with pytest.raises(RuntimeError):
        _fresh_app()
