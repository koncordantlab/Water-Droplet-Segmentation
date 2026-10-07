"""Per-frame detections as one record of parallel columns, and the parser
that rebuilds them from a detections.json frame record (confidence,
segment-validity and shape filters)."""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from tracking.config import TrackingConfig
from tracking.geometry import detection_area, is_relevant_circular_segment


@dataclass
class FrameDetections:
    """The kept detections of one frame. Index i of every column is detection
    i; append() is the only writer, so the columns always have equal length."""
    segments: List[Optional[np.ndarray]] = field(default_factory=list)
    areas: List[float] = field(default_factory=list)
    confs: List[float] = field(default_factory=list)
    classes: List[int] = field(default_factory=list)
    det_ids: List[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.segments)

    def append(self, segment: Optional[np.ndarray], area: float, conf: float, cls: int, det_id: int) -> None:
        self.segments.append(segment)
        self.areas.append(area)
        self.confs.append(conf)
        self.classes.append(cls)
        self.det_ids.append(det_id)


def frame_detections_from_record(frame_record, cfg: TrackingConfig) -> FrameDetections:
    seg_cfg = cfg.segments
    dets = FrameDetections()
    dropped_low_conf = 0
    dropped_non_circular = 0
    kept_non_circular = 0
    total = 0
    for det_idx, det in enumerate(frame_record.get("detections", [])):
        total += 1
        conf = float(det.get("conf", 0.0))
        cls = int(det.get("cls", 0))
        det_id = int(det.get("det_id", det_idx))
        if conf < seg_cfg.min_confidence:
            dropped_low_conf += 1
            continue
        segment_raw = det.get("segment")
        segment = None
        if isinstance(segment_raw, list):
            points = np.asarray(segment_raw, dtype=np.float32)
            if points.ndim == 2 and points.shape[0] >= 3 and points.shape[1] == 2:
                segment = points
        if seg_cfg.require_segmentation and segment is None:
            frame_id = frame_record.get("frame", "?")
            raise RuntimeError(
                f"Detections JSON is missing valid segment data (frame={frame_id}, det_id={det_id}). "
                "Regenerate detections with segmentation enabled."
            )
        det_area = float(det.get("area", 0.0))
        if det_area <= 0:
            det_area = detection_area(segment)
        if seg_cfg.require_segmentation and det_area <= 0:
            frame_id = frame_record.get("frame", "?")
            raise RuntimeError(
                f"Detections JSON has non-positive segmented area (frame={frame_id}, det_id={det_id})."
            )
        if segment is not None and not is_relevant_circular_segment(segment, cfg):
            if seg_cfg.keep_non_circular_for_association:
                kept_non_circular += 1
            else:
                dropped_non_circular += 1
                continue
        dets.append(segment, det_area, conf, cls, det_id)
    if dropped_low_conf > 0 or dropped_non_circular > 0 or kept_non_circular > 0:
        frame_id = frame_record.get("frame", "?")
        print(
            f"Warning: filtered detections in JSON "
            f"(frame={frame_id}, low_conf={dropped_low_conf}, conf_threshold={seg_cfg.min_confidence}, "
            f"non_circular_dropped={dropped_non_circular}, non_circular_kept={kept_non_circular}, "
            f"kept={len(dets)}, total={total})"
        )
    return dets
