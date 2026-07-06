"""Bit-exactness checks for the GPU mask path (Tier 1 / overlay).

The per-instance pixel work (threshold + nearest-resize of YOLO masks to full
resolution) is being moved from CPU (numpy + cv2.resize) onto the masks' torch
device (GPU when available). Every downstream number — pixel areas, overlap
counts, per-instance metrics, the overlay video — is derived from those
full-resolution binary masks, so the move is only safe if the masks come out
*bit-identical* to ``cv2.resize((m > 0.3).astype(uint8), (w, h), INTER_NEAREST)``.

These tests pin that equivalence. ``torch.nn.functional.interpolate(mode=...)``
is deliberately NOT used (it does not match cv2's nearest convention); the
helper reproduces cv2's exact index map and gathers, so results match on CPU and
GPU alike. Tests run on CPU tensors (the gather is device-independent).

Run: python3 Nasa_Backend/test_gpu_mask_equivalence.py
Exits non-zero on failure; prints ALL CHECKS PASSED on success. No pytest.
"""
import os
import sys

import numpy as np
import cv2
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.chdir(_HERE)  # module loads YOLO weights via a relative path at import time
import frontend_nasa13_apiV2 as mod


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _cv2_ref_full_masks(prob, dst_h, dst_w, thresh=0.3):
    """The exact current-code reference: threshold each mask then nearest-resize."""
    out = []
    for m in prob:
        binm = (m > thresh).astype(np.uint8)
        out.append(cv2.resize(binm, (dst_w, dst_h), interpolation=cv2.INTER_NEAREST))
    return np.stack(out, axis=0) if out else np.zeros((0, dst_h, dst_w), np.uint8)


def _ref_overlap_counts(full_masks, class_names):
    """The exact original nested-loop overlap classification (the reference)."""
    ww = ii = mixed = 0
    n = len(full_masks)
    for k in range(n):
        for j in range(k + 1, n):
            if not np.any(full_masks[k] & full_masks[j]):
                continue
            nk, nj = class_names[k], class_names[j]
            if nk == nj == "water":
                ww += 1
            elif nk == nj == "ice":
                ii += 1
            else:
                mixed += 1
    return ww, ii, mixed


# Mask source size, full-res target size: upscale, downscale, equal, non-square,
# odd dims, and 4K-ish — the shapes a real video actually hits.
_SHAPE_CASES = [
    (160, 160, 1080, 1920),
    (160, 160, 720, 1280),
    (104, 152, 513, 761),
    (200, 200, 100, 100),     # downscale
    (50, 80, 50, 80),         # identity
    (37, 41, 123, 256),       # odd / coprime
    (160, 160, 2160, 3840),   # 4K
    (96, 168, 480, 640),
]


def test_resize_bin_masks_matches_cv2_exactly():
    rng = np.random.default_rng(20260617)
    for (sh, sw, dh, dw) in _SHAPE_CASES:
        for n in (1, 3, 7):
            prob = rng.random((n, sh, sw), dtype=np.float32)
            # Force some values onto / around the 0.3 boundary to stress the
            # threshold, including the exact float32 representation of 0.3.
            prob.flat[0::11] = np.float32(0.3)
            prob.flat[1::11] = np.nextafter(np.float32(0.3), np.float32(1.0))
            prob.flat[2::11] = np.nextafter(np.float32(0.3), np.float32(0.0))

            expected = _cv2_ref_full_masks(prob, dh, dw, thresh=0.3)
            got = mod._resize_bin_masks_nn(
                torch.from_numpy(prob), dh, dw, thresh=0.3
            ).cpu().numpy()

            _check(got.shape == expected.shape,
                   f"shape {got.shape} != {expected.shape} for {(sh, sw, dh, dw)} n={n}")
            _check(got.dtype == np.uint8, f"dtype {got.dtype} != uint8")
            _check(np.array_equal(got, expected),
                   f"mask mismatch for src=({sh},{sw}) dst=({dh},{dw}) n={n}: "
                   f"{int((got != expected).sum())} differing pixels")


def test_threshold_matches_numpy_at_boundary():
    # numpy and torch must agree on '> 0.3' for the same float32 values,
    # including the exact boundary value, so the binary masks are identical.
    vals = np.array(
        [0.0, 0.29999998, 0.3, 0.30000001, 0.3000001, 0.5, 1.0, np.nextafter(np.float32(0.3), 1.0)],
        dtype=np.float32,
    ).reshape(1, 1, -1)
    expected = (vals > 0.3).astype(np.uint8)
    got = mod._resize_bin_masks_nn(
        torch.from_numpy(vals), vals.shape[1], vals.shape[2], thresh=0.3
    ).cpu().numpy()
    _check(np.array_equal(got, expected),
           f"threshold mismatch at boundary: got {got.ravel()} expected {expected.ravel()}")


def test_index_map_is_consistent_and_cached():
    a = mod._nn_resize_index_map(160, 1920)
    b = mod._nn_resize_index_map(160, 1920)
    _check(np.array_equal(a, b), "index map not stable across calls")
    _check(a.shape == (1920,), f"index map length {a.shape} != (1920,)")
    _check(a.min() >= 0 and a.max() <= 159, "index map out of [0, src-1] range")
    # Matches cv2's own 1-D nearest pick (probe a labelled row through cv2).
    probe = np.arange(160, dtype=np.float32).reshape(1, 160)
    cv2_pick = cv2.resize(probe, (1920, 1), interpolation=cv2.INTER_NEAREST).reshape(1920).astype(np.int64)
    _check(np.array_equal(a, cv2_pick), "index map disagrees with cv2 nearest pick")


def test_gpu_output_matches_cpu_and_cv2():
    """The real GPU path must equal the CPU path (and cv2). Skipped without CUDA."""
    if not torch.cuda.is_available():
        print("  (skip: no CUDA device)")
        return
    rng = np.random.default_rng(99)
    for (sh, sw, dh, dw) in [(160, 160, 1080, 1920), (104, 152, 513, 761), (200, 200, 100, 100)]:
        prob = rng.random((6, sh, sw), dtype=np.float32)
        prob.flat[0::11] = np.float32(0.3)
        expected = _cv2_ref_full_masks(prob, dh, dw, thresh=0.3)
        t = torch.from_numpy(prob)
        cpu_out = mod._resize_bin_masks_nn(t, dh, dw, thresh=0.3).cpu().numpy()
        gpu_out = mod._resize_bin_masks_nn(t.cuda(), dh, dw, thresh=0.3).cpu().numpy()
        _check(np.array_equal(gpu_out, cpu_out), f"GPU != CPU for {(sh, sw, dh, dw)}")
        _check(np.array_equal(gpu_out, expected), f"GPU != cv2 for {(sh, sw, dh, dw)}")

        # overlap-exists matrix must agree CPU vs GPU and vs np.any reference
        flat = torch.from_numpy(expected.reshape(expected.shape[0], -1))
        E_cpu = mod._overlap_exists_matrix(flat).cpu().numpy()
        E_gpu = mod._overlap_exists_matrix(flat.cuda()).cpu().numpy()
        _check(np.array_equal(E_cpu, E_gpu), f"overlap GPU != CPU for {(sh, sw, dh, dw)}")

        # GPU areas must equal CPU areas and the cv2 mask sums (binary input)
        binm = mod._threshold_masks(t, 0.3)
        ar_cpu = [int(v) for v in mod._mask_areas_from_source(binm, dh, dw)]
        ar_gpu = [int(v) for v in mod._mask_areas_from_source(binm.cuda(), dh, dw)]
        ar_ref = [int(expected[k].sum()) for k in range(expected.shape[0])]
        _check(ar_cpu == ar_gpu == ar_ref, f"areas GPU/CPU/cv2 disagree for {(sh, sw, dh, dw)}")
        for a in range(expected.shape[0]):
            for b in range(expected.shape[0]):
                ref = bool(np.any(expected[a] & expected[b]))
                _check(bool(E_gpu[a, b]) == ref, f"overlap GPU != np.any for {(sh, sw, dh, dw)}")


def test_overlap_exists_matrix_matches_np_any():
    # E[k, j] must equal np.any(mask_k & mask_j) for every pair, including
    # non-overlapping pairs (must be exactly 0, no float false positives) and
    # all-zero masks.
    rng = np.random.default_rng(1234)
    for trial in range(25):
        n = int(rng.integers(1, 12))
        p = int(rng.integers(1, 80))
        masks = (rng.random((n, p)) > 0.7).astype(np.uint8)  # sparse: some pairs miss
        if trial % 7 == 0 and n:
            masks[0] = 0  # force an all-zero mask
        E = mod._overlap_exists_matrix(torch.from_numpy(masks)).cpu().numpy()
        for k in range(n):
            for j in range(n):
                expect = bool(np.any(masks[k] & masks[j]))
                _check(bool(E[k, j]) == expect, f"E[{k},{j}] wrong (trial {trial})")


def test_overlap_source_res_equals_full_res_when_upscaling():
    # The integration computes overlap from the SOURCE-resolution masks when both
    # axes are upscaled; that must give the same overlap-exists (and the same
    # ww/ii/mixed counts) as the full-resolution masks the old code used.
    rng = np.random.default_rng(55)
    names = ["water", "ice", "water", "ice", "water", "ice"]
    for (sh, sw, dh, dw) in [(160, 160, 1080, 1920), (96, 168, 480, 640), (50, 80, 50, 80)]:
        n = 6
        src_bin = (rng.random((n, sh, sw)) > 0.88).astype(np.float32)  # sparse 0/1
        full = _cv2_ref_full_masks(src_bin, dh, dw, thresh=0.3)
        binm = (src_bin > 0.3).astype(np.uint8)
        E_src = mod._overlap_exists_matrix(torch.from_numpy(binm).reshape(n, -1)).cpu().numpy()
        E_full = mod._overlap_exists_matrix(torch.from_numpy(full).reshape(n, -1)).cpu().numpy()
        _check(np.array_equal(E_src, E_full),
               f"source-res vs full-res exists differ for {(sh, sw, dh, dw)}")
        ref = _ref_overlap_counts([full[k] for k in range(n)], names)
        _check(mod._classify_overlaps(E_src, names) == ref,
               f"counts via source-res differ from reference for {(sh, sw, dh, dw)}")


def test_classify_overlaps_matches_reference_loop():
    # _classify_overlaps must tally exactly like the original nested loop,
    # including a third 'other' class falling into 'mixed' and n == 0.
    rng = np.random.default_rng(7)
    pool = ["water", "ice", "other"]
    for trial in range(40):
        n = int(rng.integers(0, 11))
        cls = [pool[int(rng.integers(0, 3))] for _ in range(n)]
        E = rng.random((n, n)) > 0.5
        E = E | E.T
        ww = ii = mixed = 0
        for k in range(n):
            for j in range(k + 1, n):
                if not E[k, j]:
                    continue
                if cls[k] == cls[j] == "water":
                    ww += 1
                elif cls[k] == cls[j] == "ice":
                    ii += 1
                else:
                    mixed += 1
        _check(mod._classify_overlaps(E, cls) == (ww, ii, mixed),
               f"classify mismatch (trial {trial}): cls={cls}")


class _FakeConf:
    def __init__(self, v):
        self._v = v

    def item(self):
        return self._v


class _FakeBox:
    """Minimal stand-in for an ultralytics Boxes element."""
    def __init__(self, conf, xyxy=(1.0, 2.0, 30.0, 40.0)):
        self.conf = _FakeConf(conf)
        self.xyxy = torch.tensor([list(xyxy)], dtype=torch.float32)


def test_mask_areas_from_source_matches_cv2_sum():
    # Full-res pixel area computed from the source masks (multiplicity trick)
    # must equal int(cv2.resize(...).sum()) for every instance — upscale,
    # downscale, identity, 4K, odd dims.
    rng = np.random.default_rng(2024)
    for (sh, sw, dh, dw) in _SHAPE_CASES:
        for n in (1, 4):
            prob = rng.random((n, sh, sw), dtype=np.float32)
            prob.flat[0::7] = np.float32(0.3)
            binm = torch.from_numpy((prob > 0.3).astype(np.uint8))
            full = _cv2_ref_full_masks(prob, dh, dw, thresh=0.3)
            expected = [int(full[k].sum()) for k in range(n)]
            got = [int(v) for v in mod._mask_areas_from_source(binm, dh, dw)]
            _check(got == expected,
                   f"area mismatch for {(sh, sw, dh, dw)} n={n}: {got} vs {expected}")


def test_per_instance_basic_areas_equals_masks_path():
    # In basic mode, passing precomputed GPU areas (masks=None) must yield the
    # exact same rows as the original masks-based path — including area==0 skips.
    rng = np.random.default_rng(11)
    n, sh, sw, dh, dw = 8, 60, 80, 200, 240
    prob = rng.random((n, sh, sw), dtype=np.float32)
    prob[2] = 0.0  # force an empty instance (area 0 -> skipped)
    binm = torch.from_numpy((prob > 0.3).astype(np.uint8))
    full = _cv2_ref_full_masks(prob, dh, dw, thresh=0.3)
    full_list = [full[k] for k in range(n)]
    class_names = ["water", "ice", "water", "ice", "water", "ice", "water", "ice"]
    boxes = [_FakeBox(0.5 + 0.01 * k) for k in range(n)]
    areas = mod._mask_areas_from_source(binm, dh, dw)
    ref = mod._per_instance_metrics(full_list, boxes, class_names, (dh, dw), mode="basic")
    new = mod._per_instance_metrics(None, boxes, class_names, (dh, dw), mode="basic", areas=areas)
    _check(ref == new, f"basic per-instance differs with areas: {ref} vs {new}")


def test_per_instance_full_mode_areas_match_recompute():
    # In full mode, passing GPU areas must give identical rows to recomputing
    # int(fm.sum()) internally (areas == mask sums, so all derived metrics match).
    rng = np.random.default_rng(13)
    n, sh, sw, dh, dw = 6, 60, 80, 200, 240
    prob = rng.random((n, sh, sw), dtype=np.float32)
    binm = torch.from_numpy((prob > 0.3).astype(np.uint8))
    full = _cv2_ref_full_masks(prob, dh, dw, thresh=0.3)
    full_list = [full[k] for k in range(n)]
    class_names = ["water", "ice", "water", "ice", "water", "ice"]
    boxes = [_FakeBox(0.5) for _ in range(n)]
    areas = mod._mask_areas_from_source(binm, dh, dw)
    ref = mod._per_instance_metrics(full_list, boxes, class_names, (dh, dw), mode="full")
    new = mod._per_instance_metrics(full_list, boxes, class_names, (dh, dw), mode="full", areas=areas)
    _check(ref == new, "full-mode per-instance differs when GPU areas are passed")


def test_apply_full_overlay_precomputed_equals_internal():
    # The overlay must produce an identical frame whether it resizes the masks
    # itself (old path) or is handed the already-GPU-resized full masks (new).
    rng = np.random.default_rng(7)
    h, w = 240, 320
    img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    sh, sw, n = 60, 80, 5
    masks_np = rng.random((n, sh, sw), dtype=np.float32)
    class_names = ["water", "ice", "water", "ice", "water"]

    ref = mod.apply_full_overlay(img.copy(), masks_np, class_names)
    full = _cv2_ref_full_masks(masks_np, h, w, thresh=0.3)
    new = mod.apply_full_overlay(img.copy(), None, class_names,
                                 full_masks=[full[k] for k in range(n)])
    _check(np.array_equal(ref, new),
           f"overlay differs with precomputed masks: {int((ref != new).sum())} px")


def main():
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)} (tests run on CPU tensors)")
    else:
        print("No GPU (tests run on CPU tensors — gather is device-independent)")
    tests = [
        test_index_map_is_consistent_and_cached,
        test_threshold_matches_numpy_at_boundary,
        test_resize_bin_masks_matches_cv2_exactly,
        test_overlap_exists_matrix_matches_np_any,
        test_overlap_source_res_equals_full_res_when_upscaling,
        test_classify_overlaps_matches_reference_loop,
        test_mask_areas_from_source_matches_cv2_sum,
        test_per_instance_basic_areas_equals_masks_path,
        test_per_instance_full_mode_areas_match_recompute,
        test_gpu_output_matches_cpu_and_cv2,
        test_apply_full_overlay_precomputed_equals_internal,
    ]
    for t in tests:
        t()
        print(f"  ok: {t.__name__}")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
