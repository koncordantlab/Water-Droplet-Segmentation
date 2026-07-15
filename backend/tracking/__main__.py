"""python -m tracking -- CLI over the two-phase pipeline. Mirrors the old
monolith __main__ flow; every path defaults to tracking.config and is
overridable per invocation."""
import argparse

from tracking import config


def _add_common(p):
    p.add_argument("--video", default=config.VIDEO_PATH)
    p.add_argument("--detections", default=config.DETECTIONS_PATH)


def main():
    parser = argparse.ArgumentParser(prog="tracking")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("detect", help="run YOLO once, write detections.json")
    _add_common(p)
    p.add_argument("--model", default=config.MODEL_PATH)

    p = sub.add_parser("track", help="consume detections.json, write tracking_log.json + annotated mp4")
    _add_common(p)
    p.add_argument("--output", default=config.OUTPUT_PATH)
    p.add_argument("--log", default=config.TRACK_LOG_PATH)

    p = sub.add_parser("all", help="detect then track (the old script's default flow)")
    _add_common(p)
    p.add_argument("--model", default=config.MODEL_PATH)
    p.add_argument("--output", default=config.OUTPUT_PATH)
    p.add_argument("--log", default=config.TRACK_LOG_PATH)

    args = parser.parse_args()
    if args.cmd in ("detect", "all"):
        from tracking.detect import export_detections_json, load_model
        print("Loading YOLO model...")
        model = load_model(args.model)
        print("Exporting detections...")
        export_detections_json(model, args.video, args.detections)
    if args.cmd in ("track", "all"):
        from tracking.track import track_from_detections_json
        print("Tracking from detections JSON...")
        track_from_detections_json(args.video, args.detections,
                                   getattr(args, "output", config.OUTPUT_PATH),
                                   getattr(args, "log", config.TRACK_LOG_PATH))
    print("Done.")


if __name__ == "__main__":
    main()
