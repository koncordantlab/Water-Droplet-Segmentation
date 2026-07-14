"""Shared pytest setup. The droplet_backend package has no import-time side
effects (the model loads lazily on first predict), so no YOLO stubbing is
needed — tier-1 tests simply never trigger a load. backend/ goes on
sys.path so `import droplet_backend` works uninstalled (CI included)."""
import os
import sys

import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


@pytest.fixture()
def app():
    from droplet_backend.api import create_app
    return create_app()


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture(autouse=True)
def _allow_tmp_videos(monkeypatch, tmp_path_factory):
    """Point the video-path allowlist at pytest's tmp root so route tests can
    submit tmp_path files; individual tests override DROPLET_VIDEO_ROOTS to test
    the enforcement itself."""
    monkeypatch.setenv("DROPLET_VIDEO_ROOTS", str(tmp_path_factory.getbasetemp()))
