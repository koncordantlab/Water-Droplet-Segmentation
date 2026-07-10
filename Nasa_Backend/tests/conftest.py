"""Shared pytest setup for the Nasa_Backend suite.

The app module (frontend_nasa13_apiV2) loads YOLO weights at import time via a
relative path. Importing it therefore needs: Nasa_Backend/ on sys.path, cwd =
Nasa_Backend/ during the import, and the weights file — which does not exist in
CI. When weights are absent (or NASA_FORCE_YOLO_STUB=1), ultralytics.YOLO is
replaced with a stub BEFORE the app module is first imported. Tier-1 tests
never touch the model, so the stub never leaks into behavior; on the lab box
the real weights load and `pytest -m local` exercises the real model.
"""
import os
import sys

import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(BACKEND_DIR, "app_root", "weights_DP(8).pt")

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

STUBBED = (
    os.environ.get("NASA_FORCE_YOLO_STUB") == "1"
    or not os.path.isfile(WEIGHTS_PATH)
)


class _StubYOLO:
    """Stand-in for ultralytics.YOLO covering only what the app module does at
    import time (construction + .to(device)). Any inference attempt fails
    loudly — tier-1 tests must never reach the model."""

    def __init__(self, path):
        self.path = path

    def to(self, device):
        return self

    def __call__(self, *args, **kwargs):
        raise RuntimeError("StubYOLO cannot run inference (no weights in this environment)")

    def predict(self, *args, **kwargs):
        raise RuntimeError("StubYOLO cannot run inference (no weights in this environment)")


def _import_app_module():
    if "frontend_nasa13_apiV2" in sys.modules:
        return sys.modules["frontend_nasa13_apiV2"]

    import ultralytics

    real_yolo = ultralytics.YOLO
    cwd = os.getcwd()
    try:
        if STUBBED:
            ultralytics.YOLO = _StubYOLO
        os.chdir(BACKEND_DIR)  # relative weights path must resolve during import
        import frontend_nasa13_apiV2  # noqa: F401
    finally:
        ultralytics.YOLO = real_yolo
        os.chdir(cwd)
    return sys.modules["frontend_nasa13_apiV2"]


@pytest.fixture(scope="session")
def app_module():
    """The imported app module; YOLO stubbed automatically when weights absent."""
    return _import_app_module()
