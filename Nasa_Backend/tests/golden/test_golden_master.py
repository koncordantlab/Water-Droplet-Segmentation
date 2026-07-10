"""Golden-master gate: current code must reproduce the recorded outputs
value-for-value. Runs only on the lab box (weights + video): pytest -m local."""
import json
import os

import pytest

from tests.golden.record_goldens import CONFIGS, EXPECTED_DIR, GOLDEN_VIDEO, run_config

pytestmark = [pytest.mark.local, pytest.mark.weights, pytest.mark.gpu]


def _first_diff(a, b, path="$"):
    """Return a human-readable description of the first difference, or None."""
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} != {type(b).__name__} ({a!r} vs {b!r})"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                return f"{path}.{k}: missing in current"
            if k not in b:
                return f"{path}.{k}: missing in golden"
            d = _first_diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _first_diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    if a != b:
        return f"{path}: {a!r} != {b!r}"
    return None


@pytest.mark.parametrize(
    "cfg",
    [pytest.param(c, marks=pytest.mark.slow) if c["output_mode"] == "full" else c
     for c in CONFIGS],
    ids=lambda c: c["name"],
)
def test_golden_master(app_module, cfg):
    if not os.path.isfile(GOLDEN_VIDEO):
        pytest.skip(f"golden video not present: {GOLDEN_VIDEO}")
    expected_path = os.path.join(EXPECTED_DIR, f"{cfg['name']}.json")
    assert os.path.isfile(expected_path), (
        f"missing golden {expected_path} — run tests/golden/record_goldens.py"
    )
    with open(expected_path) as fh:
        expected = json.load(fh)
    # round-trip current through JSON so 'null', int/float, and key-string
    # semantics match the recorded file exactly
    current = json.loads(json.dumps(run_config(app_module, cfg), allow_nan=False, sort_keys=True))
    diff = _first_diff(current, expected)
    assert diff is None, f"golden mismatch [{cfg['name']}]: {diff}"
