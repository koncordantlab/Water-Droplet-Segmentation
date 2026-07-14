"""Constants shared across the package. Values are part of the behavior
freeze — see tests/golden/ before changing anything here."""
import os

# Droplet size-distribution histogram bin count (spec: global log-spaced bins).
SIZE_DIST_BINS = 30

# Overlay rendering (moved verbatim from the monolith header).
COLOR_MAP = {"water": (255, 0, 0), "ice": (0, 128, 0)}
OVERLAP_COLORS = {"ww": (0, 0, 255), "ii": (255, 165, 0), "wi": (255, 255, 0)}
ALPHA_SEG = 0.5
ALPHA_OVERLAP = 0.65

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS_PATH = os.path.join(_BACKEND_DIR, "app_root", "weights_DP(8).pt")


def weights_path():
    """Weights file location; override with NASA_WEIGHTS_PATH (spec §6)."""
    return os.environ.get("NASA_WEIGHTS_PATH") or DEFAULT_WEIGHTS_PATH
