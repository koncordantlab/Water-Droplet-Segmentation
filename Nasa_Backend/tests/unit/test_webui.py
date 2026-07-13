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
