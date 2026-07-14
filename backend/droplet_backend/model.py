"""The segmentation-model seam (spec §6): nothing outside this module knows
what YOLO is. predict(frames) returns the ultralytics results list; consumers
rely only on the duck-typed surface .orig_shape / .names / .boxes(.cls, .conf,
.xyxy) / .masks.data — fakes in tier-1 tests implement exactly that.
Loading is lazy: import is side-effect free; weights load on first predict(),
and both get_model() and the first load are thread-safe (double-checked
locking), so concurrent first requests share one model and one YOLO load.
imgsz/max_det/verbose are frozen constants (goldens)."""
import threading

import torch
from ultralytics import YOLO

from nasa_backend import config


class SegmentationModel:
    def __init__(self, weights_path=None):
        self.weights_path = weights_path  # None -> resolve from config at load time
        self._model = None
        self.device = None
        self._load_lock = threading.Lock()

    def load(self):
        # Double-checked locking: the unlocked fast path keeps the loaded case
        # free, while the lock serializes the first load so concurrent
        # predict() calls can't construct YOLO twice. _model is published only
        # after .to(device), so the fast path never sees a half-initialized model.
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    path = self.weights_path or config.weights_path()
                    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    if torch.cuda.is_available():
                        print(f"GPU: {torch.cuda.get_device_name(0)}")
                    model = YOLO(path)
                    model.to(self.device)
                    self._model = model
        return self

    def predict(self, frames):
        self.load()
        return self._model(frames, imgsz=640, max_det=2000, verbose=False)


_instance = None
_instance_lock = threading.Lock()


def get_model():
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = SegmentationModel()
    return _instance
