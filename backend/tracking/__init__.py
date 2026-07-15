"""Standalone droplet tracking pipeline (ported verbatim from branch
1-object-tracking-algorithm-v01 @ 9c24e1c and split one-module-per-concern).
Two-phase design: detect.py runs YOLO once -> detections.json; track.py
consumes it -> tracking_log.json + annotated mp4. config.py holds the ~110
tuning constants -- the intended tuning surface; the matchers in matching.py
are heavily interdependent, prefer tuning constants over editing logic.
Entry point: python -m tracking {detect,track,analyze,plots,all}."""
