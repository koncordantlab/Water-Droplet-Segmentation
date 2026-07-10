"""Record golden-master outputs from the CURRENT code.

Run ONCE on the lab box (real weights + real video), from Nasa_Backend/:

    ~/miniconda3/envs/droplets/bin/python tests/golden/record_goldens.py

Writes tests/golden/expected/<config>.json (committed to git). Re-record ONLY
when a numeric behavior change is intended and reviewed — the whole point is
that phase-0 refactors must reproduce these values exactly.

Normalization: execution_time and absolute paths are excluded; everything
passes through make_json_serializable (NaN -> null) before writing.
"""
import argparse
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(os.path.dirname(HERE))
REPO_ROOT = os.path.dirname(BACKEND_DIR)
EXPECTED_DIR = os.path.join(HERE, "expected")
GOLDEN_VIDEO = os.environ.get(
    "NASA_GOLDEN_VIDEO", os.path.join(REPO_ROOT, "Test_videos", "10 seconds.mp4")
)

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Config set deliberately excludes full/no-um: on the dense golden clip
# (~1,045 instances/checkpoint frame) each FULL-mode run costs ~10 min warm
# (first cold run ~2 h) — the contour + pairwise-overlap work, which is real
# production behavior — and full/no-um's only delta vs full_um (NaN µm
# columns) is already pinned by the unit tests and by basic_noum. One slow
# full config + two fast basic configs.
CONFIGS = [
    {"name": "full_um", "output_mode": "full", "um_per_px": 2.5, "dist_interval": 5},   # ~10 min warm — 'slow' golden
    {"name": "basic_um", "output_mode": "basic", "um_per_px": 2.5, "dist_interval": 5},
    {"name": "basic_noum", "output_mode": "basic", "um_per_px": None, "dist_interval": 5},
]

MAX_GOLDEN_BYTES = 5 * 1024 * 1024  # refuse to commit runaway fixtures


def _xlsx_to_dict(mod, path):
    """Every sheet of a workbook -> {sheet: {column: [values...]}} (JSON-safe)."""
    import pandas as pd

    sheets = pd.read_excel(path, sheet_name=None)
    return {
        name: mod.make_json_serializable(df.to_dict(orient="list"))
        for name, df in sheets.items()
    }


def run_config(mod, cfg):
    """Run process_video on the golden clip with cfg; return the normalized
    comparable payload (no volatile fields)."""
    with tempfile.TemporaryDirectory() as out_dir:
        msg, excel_path, rows, overlap_totals, charts, _exec_time, size_dist = mod.process_video(
            GOLDEN_VIDEO,
            save_ovl=False,
            dist_interval=cfg["dist_interval"],
            output_dir=out_dir,
            output_mode=cfg["output_mode"],
            um_per_px=cfg["um_per_px"],
        )
        assert excel_path and os.path.isfile(excel_path), f"no summary xlsx: {msg}"
        base = os.path.splitext(os.path.basename(GOLDEN_VIDEO))[0]
        per_frame_dir = os.path.join(out_dir, f"{base}_per_frame_xlsx")
        per_frame = {}
        if os.path.isdir(per_frame_dir):
            for fname in sorted(os.listdir(per_frame_dir)):
                per_frame[fname] = _xlsx_to_dict(mod, os.path.join(per_frame_dir, fname))
        assert per_frame, "no per-frame xlsx written (dist_interval>0 should produce some)"
        return {
            "config": {k: cfg[k] for k in ("output_mode", "um_per_px", "dist_interval")},
            "video": os.path.basename(GOLDEN_VIDEO),
            "rows": mod.make_json_serializable(rows),
            "overlap_totals": mod.make_json_serializable(overlap_totals),
            "charts": mod.make_json_serializable(charts),
            "size_distribution": mod.make_json_serializable(size_dist),
            "summary_xlsx": _xlsx_to_dict(mod, excel_path),
            "per_frame_xlsx": per_frame,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only", help="comma-separated config names to record (default: all)"
    )
    args = parser.parse_args()
    selected = CONFIGS
    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        unknown = wanted - {c["name"] for c in CONFIGS}
        assert not unknown, f"unknown config name(s): {sorted(unknown)}"
        selected = [c for c in CONFIGS if c["name"] in wanted]

    assert os.path.isfile(GOLDEN_VIDEO), f"golden video missing: {GOLDEN_VIDEO}"
    import frontend_nasa13_apiV2 as mod  # real weights load (run from Nasa_Backend/)

    os.makedirs(EXPECTED_DIR, exist_ok=True)
    written = []
    for cfg in selected:
        print(f"▶ recording {cfg['name']} ...")
        payload = run_config(mod, cfg)
        out = os.path.join(EXPECTED_DIR, f"{cfg['name']}.json")
        with open(out, "w") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True, allow_nan=False)
        size = os.path.getsize(out)
        assert size < MAX_GOLDEN_BYTES, f"{out} is {size} bytes — too big to commit"
        written.append((out, size))

    print("\nInventory of golden files written:")
    for path, size in written:
        assert os.path.isfile(path)
        print(f"  - {path} ({size} bytes)")
    print("DONE — commit tests/golden/expected/*.json")


if __name__ == "__main__":
    main()
