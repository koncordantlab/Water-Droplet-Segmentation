"""Standalone droplet tracking pipeline (ported from branch
1-object-tracking-algorithm-v01 @ 9c24e1c). Two-phase design: detect.py runs
YOLO once -> detections.json; track.py steps tracker.Tracker over it ->
tracking_log.json + annotated mp4. Settings are one TrackingConfig per run
(config.py; override with --config); the per-frame algorithm is
tracker.Tracker.step -> associate / matched / unmatched stages, whose
matchers (matching.py) are heavily interdependent -- prefer tuning settings
over editing logic. Entry point: python -m tracking {config,detect,track,analyze,plots,all}."""
