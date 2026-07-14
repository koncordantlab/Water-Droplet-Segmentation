# nasa_backend/serialization.py
"""JSON coercion for numpy/pandas payloads (SSE events, golden recording).
Extend make_json_serializable rather than adding ad-hoc conversions."""
import math

import numpy as np
import pandas as pd


def make_json_serializable(obj):
    """Recursively convert numpy / pandas types to native Python types for JSON."""
    # handle None
    if obj is None:
        return None
    # primitives — non-finite floats become null: json.dumps would emit a bare
    # NaN/Infinity token, which JSON.parse rejects on the SSE frontend
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (str, int, bool)):
        return obj
    # numpy scalar types
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    # numpy arrays — route contents back through this function so NaN/inf
    # elements get scrubbed to null the same way scalars do (a bare ndarray
    # NaN would otherwise survive tolist() and break the SSE JSON.parse).
    if isinstance(obj, (np.ndarray,)):
        return make_json_serializable(obj.tolist())
    # pandas NA
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    # dict
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    # list/tuple
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    # fallback: try to convert using int/float or str
    try:
        return int(obj)
    except Exception:
        try:
            return float(obj)
        except Exception:
            return str(obj)
