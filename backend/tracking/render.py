"""Annotated-video rendering: mp4 writer, mask blending, and the per-frame
tracking overlay (union fills + track labels). Verbatim from the monolith."""
from typing import TYPE_CHECKING, Dict, Tuple

import cv2
import numpy as np

from tracking.geometry import segment_label_origin

if TYPE_CHECKING:  # typing-only imports
    from tracking.config import TrackingConfig
    from tracking.tracks import Track

def create_video_writer(output_path, fps, width, height):
    """Instantiate a video writer which allows the program to generate new video frames.

    Args:
        output_path (str)
        fps (int)
        width (int)
        height (int)

    Returns:
        VideoWriter
    """
    # Source - https://stackoverflow.com/questions/30103077/what-is-the-codec-for-mp4-videos-in-python-opencv
    # Posted by Gonzalo Garcia, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-01-21, License - CC BY-SA 4.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def blend_mask(base_img: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float) -> np.ndarray:
    idx = mask.astype(bool)
    if np.any(idx):
        base_img[idx] = base_img[idx] * (1.0 - alpha) + np.asarray(color, dtype=np.float32) * alpha
    return base_img

def apply_tracking_overlay(frame: np.ndarray, tracked_objects: Dict[int, "Track"], merged_children: set,
                           cfg: "TrackingConfig") -> None:
    """Apply frontend-style union mask overlays for active/lost/merged tracks."""
    r = cfg.render
    h, w = frame.shape[:2]
    active_union = np.zeros((h, w), dtype=np.uint8)
    lost_union = np.zeros((h, w), dtype=np.uint8)
    merged_union = np.zeros((h, w), dtype=np.uint8)
    label_items = []

    for obj_id, track in tracked_objects.items():
        segment = track.segment
        if segment is None or segment.ndim != 2 or segment.shape[0] < 3:
            continue
        is_merged_track = (track.gen > 1) or (obj_id in merged_children)
        if r.show_only_merged and not is_merged_track:
            continue
        contour = np.round(segment).astype(np.int32).reshape(-1, 1, 2)
        if track.missed > 0:
            if r.show_only_merged and not r.show_only_merged_include_lost:
                continue
            cv2.fillPoly(lost_union, [contour], 1)
            label_color = r.lost_color
        elif is_merged_track:
            cv2.fillPoly(merged_union, [contour], 1)
            label_color = r.merged_color
        else:
            cv2.fillPoly(active_union, [contour], 1)
            label_color = r.active_color
        label_items.append((obj_id, track, contour, label_color))

    base = frame.astype(np.float32)
    base = blend_mask(base, active_union, r.active_color, r.fill_alpha)
    base = blend_mask(base, merged_union, r.merged_color, r.fill_alpha)
    base = blend_mask(base, lost_union, r.lost_color, r.fill_alpha)
    frame[:] = np.clip(base, 0, 255).astype(np.uint8)

    for obj_id, track, contour, label_color in label_items:
        segment = track.segment
        if r.draw_border:
            cv2.polylines(frame, [contour], isClosed=True, color=label_color, thickness=2)
        x, y = segment_label_origin(segment)
        if track.missed > 0:
            label = f"ID {obj_id} G{track.gen} (lost {track.missed})"
        else:
            label = f"ID {obj_id} G{track.gen}"
        cv2.putText(frame, label, (x, y - 5), r.font, r.font_scale, label_color, r.label_thickness)
