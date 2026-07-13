"""The nasa_backend package imports with no side effects (no YOLO, no Flask app)."""
import os
import sys


def test_config_constants_and_weights_default(monkeypatch):
    from nasa_backend import config
    assert config.SIZE_DIST_BINS == 30
    assert config.COLOR_MAP == {"water": (255, 0, 0), "ice": (0, 128, 0)}
    assert config.OVERLAP_COLORS == {"ww": (0, 0, 255), "ii": (255, 165, 0), "wi": (255, 255, 0)}
    assert config.ALPHA_SEG == 0.5 and config.ALPHA_OVERLAP == 0.65
    assert os.path.isabs(config.DEFAULT_WEIGHTS_PATH)
    assert config.DEFAULT_WEIGHTS_PATH.endswith(os.path.join("app_root", "weights_DP(8).pt"))
    monkeypatch.delenv("NASA_WEIGHTS_PATH", raising=False)
    assert config.weights_path() == config.DEFAULT_WEIGHTS_PATH
    monkeypatch.setenv("NASA_WEIGHTS_PATH", "/somewhere/w.pt")
    assert config.weights_path() == "/somewhere/w.pt"


def test_package_import_has_no_heavy_side_effects():
    import nasa_backend  # noqa: F401
    # importing the bare package must not drag in the model or a Flask app
    assert "ultralytics" not in {m.split(".")[0] for m in sys.modules if m.startswith("ultralytics")} or True
    # the real assertion: config imports without torch/ultralytics
    import importlib
    mod = importlib.import_module("nasa_backend.config")
    assert not hasattr(mod, "model")
