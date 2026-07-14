"""SegmentationModel is lazy: constructing it never loads weights; predict()
loads exactly once; inference params are the frozen constants."""
import threading
import time

import droplet_backend.model as model_mod


class _FakeYOLO:
    instances = 0

    def __init__(self, path):
        _FakeYOLO.instances += 1
        self.path = path
        self.calls = []

    def to(self, device):
        self.device = device
        return self

    def __call__(self, frames, **kw):
        self.calls.append(kw)
        return ["result"] * len(frames)


def test_construction_is_lazy_and_load_is_once(monkeypatch):
    monkeypatch.setattr(model_mod, "YOLO", _FakeYOLO)
    _FakeYOLO.instances = 0
    m = model_mod.SegmentationModel(weights_path="/w.pt")
    assert _FakeYOLO.instances == 0, "constructing must not load weights"
    out = m.predict(["f1", "f2"])
    assert out == ["result", "result"]
    m.predict(["f3"])
    assert _FakeYOLO.instances == 1, "weights must load exactly once"
    assert m._model.path == "/w.pt"


def test_predict_uses_frozen_inference_params(monkeypatch):
    monkeypatch.setattr(model_mod, "YOLO", _FakeYOLO)
    m = model_mod.SegmentationModel(weights_path="/w.pt")
    m.predict(["f"])
    assert m._model.calls == [{"imgsz": 640, "max_det": 2000, "verbose": False}]


def test_get_model_is_a_singleton(monkeypatch):
    monkeypatch.setattr(model_mod, "_instance", None)
    a = model_mod.get_model()
    b = model_mod.get_model()
    assert a is b
    monkeypatch.setattr(model_mod, "_instance", None)  # don't leak to other tests


def test_concurrent_first_requests_load_yolo_exactly_once(monkeypatch):
    """Two (here: eight) simultaneous first requests must not double-load YOLO:
    get_model() must hand every thread the SAME SegmentationModel, and the
    first load must be serialized so YOLO is constructed exactly once. The
    fake's 50 ms __init__ sleep holds the race window open (the GIL is
    released while sleeping), making the unsynchronized paths fail
    deterministically."""
    class _SlowFakeYOLO(_FakeYOLO):
        def __init__(self, path):
            time.sleep(0.05)  # widen the check-then-act window before counting
            super().__init__(path)

    monkeypatch.setattr(model_mod, "YOLO", _SlowFakeYOLO)
    monkeypatch.setattr(model_mod, "_instance", None)
    monkeypatch.setenv("DROPLET_WEIGHTS_PATH", "/w.pt")
    _FakeYOLO.instances = 0

    n = 8
    barrier = threading.Barrier(n)
    models, errors = [], []

    def first_request():
        try:
            barrier.wait(timeout=5)  # maximize contention: all threads start together
            m = model_mod.get_model()
            m.predict(["f"])
            models.append(m)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=first_request) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not errors, f"worker threads raised: {errors!r}"
    assert len(models) == n
    assert all(m is models[0] for m in models), "every thread must get the same model object"
    assert _FakeYOLO.instances == 1, f"YOLO constructed {_FakeYOLO.instances} times; must be exactly once"
    monkeypatch.setattr(model_mod, "_instance", None)  # don't leak to other tests


def test_default_weights_path_comes_from_config(monkeypatch):
    """The production path: no explicit weights_path -> config.weights_path()
    at load time, env override respected."""
    monkeypatch.setattr(model_mod, "YOLO", _FakeYOLO)
    monkeypatch.setenv("DROPLET_WEIGHTS_PATH", "/env/override.pt")
    m = model_mod.SegmentationModel()
    m.predict(["f"])
    assert m._model.path == "/env/override.pt"
