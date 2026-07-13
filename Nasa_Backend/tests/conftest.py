"""Shared pytest setup. The nasa_backend package has no import-time side
effects (the model loads lazily on first predict), so no YOLO stubbing is
needed — tier-1 tests simply never trigger a load. Nasa_Backend/ goes on
sys.path so `import nasa_backend` works uninstalled (CI included)."""
import os
import sys

import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


@pytest.fixture()
def app():
    from nasa_backend.api import create_app
    return create_app()


@pytest.fixture()
def client(app):
    return app.test_client()
