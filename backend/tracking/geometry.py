"""Pure segment/box geometry: areas, centers, bounds, circularity filters,
rasterized segment IoU, and the torch bbox-IoU prefilter kernel. Verbatim
from the monolith."""
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from tracking.config import (
    FILTER_NON_CIRCULAR_SEGMENTS,
    MIN_SEGMENT_AXIS_RATIO,
    MIN_SEGMENT_CIRCULARITY,
    SEGMENT_IOU_MAX_RASTER_PIXELS,
)

def segment_area(segment: Optional[np.ndarray]) -> float:
    if segment is None:
        return 0.0
    if not isinstance(segment, np.ndarray) or segment.ndim != 2 or segment.shape[0] < 3:
        return 0.0
    return float(abs(cv2.contourArea(segment.astype(np.float32))))

def segment_center(segment: Optional[np.ndarray]) -> Tuple[float, float]:
    if segment is None or segment.ndim != 2 or segment.shape[0] == 0:
        return 0.0, 0.0
    return float(np.mean(segment[:, 0])), float(np.mean(segment[:, 1]))

def segment_max_dim(segment: Optional[np.ndarray]) -> float:
    if segment is None or segment.ndim != 2 or segment.shape[0] == 0:
        return 1.0
    x_min = float(np.min(segment[:, 0]))
    y_min = float(np.min(segment[:, 1]))
    x_max = float(np.max(segment[:, 0]))
    y_max = float(np.max(segment[:, 1]))
    return max(1.0, x_max - x_min, y_max - y_min)

def segment_distance(seg_a: Optional[np.ndarray], seg_b: Optional[np.ndarray]) -> float:
    ax, ay = segment_center(seg_a)
    bx, by = segment_center(seg_b)
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)

def segment_distance_norm(seg_a: Optional[np.ndarray], seg_b: Optional[np.ndarray]) -> float:
    return segment_distance(seg_a, seg_b) / max(segment_max_dim(seg_a), segment_max_dim(seg_b))

def segment_bounds(segment: Optional[np.ndarray]) -> Tuple[float, float, float, float]:
    if segment is None or segment.ndim != 2 or segment.shape[0] == 0:
        return 0.0, 0.0, 0.0, 0.0
    x_min = float(np.min(segment[:, 0]))
    y_min = float(np.min(segment[:, 1]))
    x_max = float(np.max(segment[:, 0]))
    y_max = float(np.max(segment[:, 1]))
    return x_min, y_min, x_max, y_max

def segment_label_origin(segment: Optional[np.ndarray]) -> Tuple[int, int]:
    if segment is None or segment.ndim != 2 or segment.shape[0] == 0:
        return 0, 0
    x = int(np.min(segment[:, 0]))
    y = int(np.min(segment[:, 1]))
    return x, y

def detection_area(segment: Optional[np.ndarray]) -> float:
    return segment_area(segment)

def segment_circularity(segment: Optional[np.ndarray]) -> float:
    area = segment_area(segment)
    if area <= 0:
        return 0.0
    if segment is None or segment.ndim != 2 or segment.shape[0] < 3:
        return 0.0
    contour = segment.astype(np.float32).reshape(-1, 1, 2)
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 0:
        return 0.0
    return float((4.0 * np.pi * area) / (perimeter * perimeter))

def segment_axis_ratio(segment: Optional[np.ndarray]) -> float:
    x_min, y_min, x_max, y_max = segment_bounds(segment)
    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)
    return float(min(w, h) / max(w, h))

def is_relevant_circular_segment(segment: Optional[np.ndarray]) -> bool:
    if not FILTER_NON_CIRCULAR_SEGMENTS:
        return True
    circularity = segment_circularity(segment)
    if circularity < MIN_SEGMENT_CIRCULARITY:
        return False
    axis_ratio = segment_axis_ratio(segment)
    if axis_ratio < MIN_SEGMENT_AXIS_RATIO:
        return False
    return True

def _rasterized_segment_mask(
    segment: np.ndarray,
    min_x: float,
    min_y: float,
    scale: float,
    width: int,
    height: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.round((segment - np.array([min_x, min_y])) / scale).astype(np.int32)
    if pts.shape[0] < 3:
        return mask
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 1)
    return mask

def segment_iou(seg_a: Optional[np.ndarray], seg_b: Optional[np.ndarray]) -> Optional[float]:
    if seg_a is None or seg_b is None:
        return None
    if seg_a.ndim != 2 or seg_b.ndim != 2:
        return None
    if seg_a.shape[0] < 3 or seg_b.shape[0] < 3:
        return None

    min_x = float(min(np.min(seg_a[:, 0]), np.min(seg_b[:, 0])))
    min_y = float(min(np.min(seg_a[:, 1]), np.min(seg_b[:, 1])))
    max_x = float(max(np.max(seg_a[:, 0]), np.max(seg_b[:, 0])))
    max_y = float(max(np.max(seg_a[:, 1]), np.max(seg_b[:, 1])))

    width = max(1, int(np.ceil(max_x - min_x)) + 1)
    height = max(1, int(np.ceil(max_y - min_y)) + 1)
    pixel_count = width * height
    scale = 1.0
    if pixel_count > SEGMENT_IOU_MAX_RASTER_PIXELS:
        scale = float(np.sqrt(pixel_count / SEGMENT_IOU_MAX_RASTER_PIXELS))
        width = max(1, int(np.ceil(width / scale)))
        height = max(1, int(np.ceil(height / scale)))

    mask_a = _rasterized_segment_mask(seg_a, min_x, min_y, scale, width, height)
    mask_b = _rasterized_segment_mask(seg_b, min_x, min_y, scale, width, height)
    inter = int(np.count_nonzero(mask_a & mask_b))
    if inter <= 0:
        return 0.0
    union = int(np.count_nonzero(mask_a | mask_b))
    if union <= 0:
        return 0.0
    return float(inter / union)

def _bbox_iou_matrix_torch(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between box arrays [N,4] and [M,4]."""
    x1 = torch.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = torch.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = torch.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = torch.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter_w = (x2 - x1).clamp_min(0.0)
    inter_h = (y2 - y1).clamp_min(0.0)
    inter = inter_w * inter_h

    area_a = ((boxes_a[:, 2] - boxes_a[:, 0]).clamp_min(0.0) * (boxes_a[:, 3] - boxes_a[:, 1]).clamp_min(0.0))[:, None]
    area_b = ((boxes_b[:, 2] - boxes_b[:, 0]).clamp_min(0.0) * (boxes_b[:, 3] - boxes_b[:, 1]).clamp_min(0.0))[None, :]
    union = area_a + area_b - inter
    return torch.where(union > 0.0, inter / union, torch.zeros_like(inter))
