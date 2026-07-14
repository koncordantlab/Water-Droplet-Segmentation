# droplet_backend/masks.py
"""GPU/CPU mask math pinned bit-identical to cv2 (INTER_NEAREST probe).
Every function here is covered by tests/unit/test_masks.py — rerun that
suite after ANY OpenCV or torch upgrade, and never swap the gather for
torch.nn.functional.interpolate (different nearest convention)."""
import cv2
import numpy as np
import torch


# Cache of cv2 INTER_NEAREST source-index maps, keyed by (src_len, dst_len). The
# map is tiny (length dst) and identical for every mask that shares a resize, so
# it is derived once per shape pair and reused.
_NN_INDEX_MAP_CACHE = {}


def _nn_resize_index_map(src_len, dst_len):
    """Source indices cv2.resize(..., INTER_NEAREST) picks along one axis.

    Derived by probing cv2 itself with a labelled row, so the mapping is
    *exactly* OpenCV's nearest-neighbour rule for the installed build rather than
    a re-derived formula that might disagree at boundaries. Returns an int64
    array of length ``dst_len`` with values in ``[0, src_len - 1]``.
    """
    key = (int(src_len), int(dst_len))
    cached = _NN_INDEX_MAP_CACHE.get(key)
    if cached is None:
        probe = np.arange(src_len, dtype=np.float32).reshape(1, src_len)
        picks = cv2.resize(probe, (int(dst_len), 1), interpolation=cv2.INTER_NEAREST)
        cached = picks.reshape(int(dst_len)).astype(np.int64)
        _NN_INDEX_MAP_CACHE[key] = cached
    return cached


def _threshold_masks(prob, thresh=0.3):
    """Binarise a stack of float masks on its own device. uint8, same shape."""
    return (prob > thresh).to(torch.uint8)


def _gather_resize_nn(binm, dst_h, dst_w):
    """Nearest-resize a uint8 mask stack to (dst_h, dst_w) via cv2's index map.

    Pure gather (index_select) using cv2's own nearest index map — NOT torch
    interpolation, whose 'nearest' convention differs from OpenCV's — so the
    result is bit-identical to cv2.resize(INTER_NEAREST) on GPU and CPU alike.
    """
    src_h, src_w = int(binm.shape[-2]), int(binm.shape[-1])
    y_map = torch.as_tensor(_nn_resize_index_map(src_h, dst_h), device=binm.device)
    x_map = torch.as_tensor(_nn_resize_index_map(src_w, dst_w), device=binm.device)
    return binm.index_select(-2, y_map).index_select(-1, x_map)


def _resize_bin_masks_nn(prob, dst_h, dst_w, thresh=0.3):
    """Threshold a stack of float masks and nearest-resize them to full res.

    ``prob`` is a torch tensor of shape (N, src_h, src_w) on any device. Returns
    a uint8 (N, dst_h, dst_w) tensor on the *same* device, bit-identical to
    ``stack(cv2.resize((prob[k] > thresh).astype(uint8), (dst_w, dst_h),
    INTER_NEAREST) for k)``. See tests/unit/test_masks.py.
    """
    return _gather_resize_nn(_threshold_masks(prob, thresh), dst_h, dst_w)


def _mask_areas_from_source(binm, dst_h, dst_w):
    """Per-instance full-resolution pixel areas, straight from the source masks.

    ``binm`` is a (N, src_h, src_w) uint8 tensor of 0/1 values. Returns an (N,)
    int64 numpy array exactly equal to
    ``[int(cv2.resize(binm[k], (dst_w, dst_h), INTER_NEAREST).sum()) for k]`` —
    *without* materialising the (N, dst_h, dst_w) masks. Nearest upsampling
    replicates each source pixel (i, j) exactly ``cy[i] * cx[j]`` times, where
    cy[i] / cx[j] are how many destination rows / cols map onto that source row /
    col (the bincount of cv2's nearest index map). Pure integer arithmetic on the
    masks' device, so it is exact (no float rounding) and only an (N,) vector
    crosses to the CPU — not the multi-GB full-res masks. See
    tests/unit/test_masks.py.
    """
    src_h, src_w = int(binm.shape[-2]), int(binm.shape[-1])
    cy = np.bincount(_nn_resize_index_map(src_h, dst_h), minlength=src_h).astype(np.int32)
    cx = np.bincount(_nn_resize_index_map(src_w, dst_w), minlength=src_w).astype(np.int32)
    cy_t = torch.as_tensor(cy, device=binm.device).view(1, src_h, 1)
    cx_t = torch.as_tensor(cx, device=binm.device).view(1, 1, src_w)
    weight = cy_t * cx_t                                   # (1, src_h, src_w) int32
    areas = (binm.to(torch.int32) * weight).sum(dim=(1, 2), dtype=torch.int64)
    return areas.cpu().numpy()


_OVERLAP_ROW_CHUNK = 1024  # rows per block; bounds the float32 workspace


def _overlap_exists_matrix(masks_2d):
    """(N, N) bool: whether masks k and j share at least one set pixel.

    ``masks_2d`` is an (N, P) torch tensor of 0/1 values. Computed as
    ``(M @ Mᵀ) > 0`` in float32: every product is an exact 0 or 1 and the
    accumulation is non-negative, so a pair that shares no pixel sums to exactly
    0.0 and a pair that shares any pixel sums to ≥ 1.0 — the ``> 0`` test is
    therefore exact regardless of accumulation order or TF32, and reproduces
    ``np.any(mask_k & mask_j)`` bit-for-bit (tests/unit/test_masks.py).

    The matmul is computed ``_OVERLAP_ROW_CHUNK`` rows at a time (read as a
    module attribute at call time so it can be monkeypatched) instead of as one
    (N, N) product: same dtype and op order per block, just fewer rows per
    call, so the result is bit-identical for any chunk size (see
    tests/unit/test_masks.py::test_overlap_exists_matrix_chunking_is_invariant)
    while bounding the transient float32 workspace to a (chunk, N) footprint —
    the returned matrix is bool either way.
    """
    n = int(masks_2d.shape[0])
    if n == 0:
        return torch.zeros((0, 0), dtype=torch.bool, device=masks_2d.device)
    m = masks_2d.to(torch.float32)
    mt = m.t()
    exists = torch.empty((n, n), dtype=torch.bool, device=masks_2d.device)
    chunk = _OVERLAP_ROW_CHUNK
    for i0 in range(0, n, chunk):
        exists[i0:i0 + chunk] = (m[i0:i0 + chunk] @ mt) > 0
    return exists


def _classify_overlaps(exists_matrix, class_names):
    """Tally (ww, ii, mixed) over unordered overlapping pairs.

    ``exists_matrix`` is an (N, N) bool array (numpy) whose [k, j] entry marks
    whether instances k and j overlap. Matches the original nested loop exactly:
    a pair counts as ``ww`` only if both classes are "water", ``ii`` only if both
    are "ice", and ``mixed`` for everything else (incl. any non-water/ice class).
    """
    n = len(class_names)
    if n == 0:
        return 0, 0, 0
    cls = np.array([1 if c == "water" else (2 if c == "ice" else 0) for c in class_names])
    iu, ju = np.triu_indices(n, k=1)
    ex = np.asarray(exists_matrix)[iu, ju].astype(bool)
    ci, cj = cls[iu], cls[ju]
    ww = int(np.count_nonzero(ex & (ci == 1) & (cj == 1)))
    ii = int(np.count_nonzero(ex & (ci == 2) & (cj == 2)))
    mixed = int(np.count_nonzero(ex) - ww - ii)
    return ww, ii, mixed
