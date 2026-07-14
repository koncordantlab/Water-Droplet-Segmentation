"""make_json_serializable: numpy/pandas coercion and the non-finite→null rule
(bare NaN/Infinity tokens would break the SSE frontend's JSON.parse)."""
import json

import numpy as np
import pandas as pd
import pytest

from nasa_backend import serialization


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf"), np.float64("nan")])
def test_non_finite_becomes_none(nonfinite):
    assert serialization.make_json_serializable(nonfinite) is None


def test_row_with_non_finite_fields():
    row = {
        "Water Avg Area (µm²)": float("nan"),
        "Resolution (pix/µm²)": np.float64("inf"),
        "Water (%)": 12.5,
    }
    clean = serialization.make_json_serializable([row])[0]
    assert clean["Water Avg Area (µm²)"] is None
    assert clean["Resolution (pix/µm²)"] is None
    assert clean["Water (%)"] == 12.5
    # strict round-trip — same strictness as browser JSON.parse
    json.loads(json.dumps(serialization.make_json_serializable([row]), allow_nan=False))


def test_numpy_and_pandas_coercions():
    ser = serialization.make_json_serializable
    assert ser(np.int64(7)) == 7 and isinstance(ser(np.int64(7)), int)
    assert ser(np.float32(1.5)) == 1.5 and isinstance(ser(np.float32(1.5)), float)
    assert ser(np.bool_(True)) is True
    assert ser(np.array([[1, 2], [3, 4]])) == [[1, 2], [3, 4]]
    assert ser(pd.NA) is None
    assert ser(pd.NaT) is None
    assert ser({"k": (np.int32(1), None)}) == {"k": [1, None]}
    assert ser((1, "a")) == [1, "a"]


def test_ndarray_elements_get_nan_scrubbed():
    import numpy as np
    out = serialization.make_json_serializable(np.array([1.0, float("nan"), 3.0]))
    assert out == [1.0, None, 3.0]
    nested = serialization.make_json_serializable({"a": np.array([[1.0, float("inf")]])})
    assert nested == {"a": [[1.0, None]]}
