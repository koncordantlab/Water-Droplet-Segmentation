"""Annotated-video rendering: mp4 writer, mask blending, and the per-frame
tracking overlay (union fills + track labels). Verbatim from the monolith."""
from typing import TYPE_CHECKING, Dict, Tuple

import cv2
import numpy as np

from tracking.config import (
    DRAW_SEGMENT_BORDER,
    FONT,
    FONT_SCALE,
    GREEN_BOX_COLOR,
    RED_BOX_COLOR,
    SEGMENT_FILL_ALPHA,
    SHOW_ONLY_MERGED_INCLUDE_LOST,
    SHOW_ONLY_MERGED_INSTANCES,
    THICKNESS,
    YELLOW_BOX_COLOR,
)
from tracking.geometry import segment_label_origin

if TYPE_CHECKING:  # typing-only: Track lands in tracking.tracks (Task 4)
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

def apply_tracking_overlay(frame: np.ndarray, tracked_objects: Dict[int, "Track"], merged_children: set) -> None:
    """Apply frontend-style union mask overlays for active/lost/merged tracks."""
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
        if SHOW_ONLY_MERGED_INSTANCES and not is_merged_track:
            continue
        contour = np.round(segment).astype(np.int32).reshape(-1, 1, 2)
        if track.missed > 0:
            if SHOW_ONLY_MERGED_INSTANCES and not SHOW_ONLY_MERGED_INCLUDE_LOST:
                continue
            cv2.fillPoly(lost_union, [contour], 1)
            label_color = RED_BOX_COLOR
        elif is_merged_track:
            cv2.fillPoly(merged_union, [contour], 1)
            label_color = YELLOW_BOX_COLOR
        else:
            cv2.fillPoly(active_union, [contour], 1)
            label_color = GREEN_BOX_COLOR
        label_items.append((obj_id, track, contour, label_color))

    base = frame.astype(np.float32)
    base = blend_mask(base, active_union, GREEN_BOX_COLOR, SEGMENT_FILL_ALPHA)
    base = blend_mask(base, merged_union, YELLOW_BOX_COLOR, SEGMENT_FILL_ALPHA)
    base = blend_mask(base, lost_union, RED_BOX_COLOR, SEGMENT_FILL_ALPHA)
    frame[:] = np.clip(base, 0, 255).astype(np.uint8)

    for obj_id, track, contour, label_color in label_items:
        segment = track.segment
        if DRAW_SEGMENT_BORDER:
            cv2.polylines(frame, [contour], isClosed=True, color=label_color, thickness=2)
        x, y = segment_label_origin(segment)
        if track.missed > 0:
            label = f"ID {obj_id} G{track.gen} (lost {track.missed})"
        else:
            label = f"ID {obj_id} G{track.gen}"
        cv2.putText(frame, label, (x, y - 5), FONT, FONT_SCALE, label_color, THICKNESS)
