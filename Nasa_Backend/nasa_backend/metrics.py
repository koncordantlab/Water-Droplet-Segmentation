# nasa_backend/metrics.py
"""Per-instance and average droplet-size metrics. The seven avg-size summary
columns and the per-instance xlsx column set derive from these helpers —
header strings and NaN-when-no-scale semantics are golden-pinned."""
import cv2
import numpy as np


def _per_instance_metrics(full_bin_masks, boxes, class_names, frame_shape, mode="full", areas=None):
    """Compute per-instance shape descriptors for one frame.

    mode: "full" (default) returns the rich descriptor set below. "basic" returns
        only instance_id, class, confidence, pixel_count, eq_diameter_px and skips
        all contour/ellipse/overlap work, so it is also faster.

    full_bin_masks: list of (H, W) uint8 binary masks, already resized to the
        original frame resolution (same objects used for overlap counting). May be
        ``None`` in basic mode when ``areas`` is supplied — basic mode only needs
        the per-instance pixel area, so the full-res masks are never built.
    boxes: ultralytics Boxes object — provides per-instance confidence + bbox.
    class_names: list of lowercased class strings, parallel to the detections.
    frame_shape: (H, W) tuple of the original frame.
    areas: optional precomputed per-instance pixel areas (e.g. from
        _mask_areas_from_source). When omitted, areas are summed from
        full_bin_masks, reproducing the original behaviour exactly.

    Returns a list of dicts (one per non-empty instance). Mask area is the
    full pixel sum; contour-based metrics (perimeter, circularity, feret, etc.)
    are computed on the largest connected component, which is the dominant blob
    for any well-formed YOLO instance mask.
    """
    H, W = frame_shape
    n = len(boxes)
    if areas is None:
        areas = [int(full_bin_masks[k].sum()) for k in range(n)]

    overlap_info = [{"count": 0, "classes": set(), "pixels": 0} for _ in range(n)]
    if mode != "basic":
        for i in range(n):
            for j in range(i + 1, n):
                inter = full_bin_masks[i] & full_bin_masks[j]
                inter_px = int(inter.sum())
                if inter_px == 0:
                    continue
                overlap_info[i]["count"] += 1
                overlap_info[i]["pixels"] += inter_px
                overlap_info[i]["classes"].add(class_names[j])
                overlap_info[j]["count"] += 1
                overlap_info[j]["pixels"] += inter_px
                overlap_info[j]["classes"].add(class_names[i])

    rows = []
    for idx in range(n):
        area = int(areas[idx])
        box = boxes[idx]
        if area == 0:
            continue

        eq_d = float(np.sqrt(4.0 * area / np.pi))

        if mode == "basic":
            rows.append({
                "instance_id": len(rows) + 1,
                "class": class_names[idx],
                "confidence": round(float(box.conf.item()), 4),
                "pixel_count": area,
                "eq_diameter_px": round(eq_d, 3),
            })
            continue

        fm = full_bin_masks[idx]
        ys, xs = np.where(fm > 0)
        cx, cy = float(xs.mean()), float(ys.mean())
        bx1, by1, bx2, by2 = [float(v) for v in box.xyxy[0].cpu().numpy()]
        bw, bh = bx2 - bx1, by2 - by1
        bbox_area = bw * bh

        contours, _ = cv2.findContours(fm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = circularity = solidity = feret_max = None
        ellipse_major = ellipse_minor = ellipse_ecc = ellipse_angle = None
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            perimeter = float(cv2.arcLength(cnt, True))
            if perimeter > 0:
                circularity = float(4.0 * np.pi * area / (perimeter ** 2))
            hull = cv2.convexHull(cnt)
            hull_area = float(cv2.contourArea(hull))
            if hull_area > 0:
                solidity = float(area / hull_area)
            hull_pts = hull.reshape(-1, 2)
            if len(hull_pts) >= 2:
                diffs = hull_pts[:, None, :] - hull_pts[None, :, :]
                feret_max = float(np.sqrt((diffs ** 2).sum(-1).max()))
            if len(cnt) >= 5:
                _, (ax_a, ax_b), e_angle = cv2.fitEllipse(cnt)
                ellipse_major = float(max(ax_a, ax_b))
                ellipse_minor = float(min(ax_a, ax_b))
                ellipse_angle = float(e_angle)
                if ellipse_major > 0:
                    ratio = ellipse_minor / ellipse_major
                    ellipse_ecc = float(np.sqrt(max(0.0, 1.0 - ratio * ratio)))

        extent = float(area / bbox_area) if bbox_area > 0 else None
        touches_border = bool(
            xs.min() == 0 or ys.min() == 0 or xs.max() == W - 1 or ys.max() == H - 1
        )
        oi = overlap_info[idx]
        rows.append({
            "instance_id": len(rows) + 1,
            "class": class_names[idx],
            "confidence": round(float(box.conf.item()), 4),
            "pixel_count": area,
            "eq_diameter_px": round(eq_d, 3),
            "centroid_x": round(cx, 2),
            "centroid_y": round(cy, 2),
            "bbox_x1": round(bx1, 2),
            "bbox_y1": round(by1, 2),
            "bbox_x2": round(bx2, 2),
            "bbox_y2": round(by2, 2),
            "bbox_width": round(bw, 2),
            "bbox_height": round(bh, 2),
            "bbox_area": round(bbox_area, 2),
            "bbox_aspect_ratio": round(bw / bh, 3) if bh > 0 else None,
            "extent": round(extent, 3) if extent is not None else None,
            "perimeter_px": round(perimeter, 2) if perimeter is not None else None,
            "circularity": round(circularity, 3) if circularity is not None else None,
            "solidity": round(solidity, 3) if solidity is not None else None,
            "feret_diameter_max_px": round(feret_max, 2) if feret_max is not None else None,
            "ellipse_major_axis_px": round(ellipse_major, 2) if ellipse_major is not None else None,
            "ellipse_minor_axis_px": round(ellipse_minor, 2) if ellipse_minor is not None else None,
            "ellipse_eccentricity": round(ellipse_ecc, 3) if ellipse_ecc is not None else None,
            "ellipse_angle_deg": round(ellipse_angle, 2) if ellipse_angle is not None else None,
            "touches_border": touches_border,
            "overlap_count": oi["count"],
            "overlap_classes": ",".join(sorted(oi["classes"])) if oi["classes"] else "",
            "overlap_pixels_total": oi["pixels"],
        })
    return rows


def _stats_row(class_name, values):
    """One row of summary stats for a class's eq-diameter list."""
    if not values:
        return {
            "class": class_name, "count": 0,
            "min": None, "max": None, "mean": None, "median": None, "std": None,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "class": class_name,
        "count": int(arr.size),
        "min": round(float(arr.min()), 3),
        "max": round(float(arr.max()), 3),
        "mean": round(float(arr.mean()), 3),
        "median": round(float(np.median(arr)), 3),
        "std": round(float(arr.std()), 3),
    }


def _apply_metric(rows, um_per_px):
    """Add eq_diameter_um and area_um2 to each instance row, in place.

    eq_diameter_um = eq_diameter_px * um_per_px
    area_um2       = pixel_count * um_per_px**2
    Both are np.nan when um_per_px is missing or <= 0. Returns the same list.
    """
    valid = isinstance(um_per_px, (int, float)) and not isinstance(um_per_px, bool) and um_per_px > 0
    for r in rows:
        if valid:
            r["eq_diameter_um"] = round(r["eq_diameter_px"] * um_per_px, 3)
            r["area_um2"] = round(r["pixel_count"] * (um_per_px ** 2), 3)
        else:
            r["eq_diameter_um"] = np.nan
            r["area_um2"] = np.nan
    return rows


def _avg_size_metrics(areas_px, um_per_px):
    """Mean droplet area (µm²) and mean equivalent-circular diameter (µm) over a
    list of per-instance pixel areas. Diameter is computed per droplet
    (sqrt(4*a/pi)) then averaged. Returns (nan, nan) when the scale is
    missing/≤0 or the list is empty."""
    if not areas_px or not um_per_px or um_per_px <= 0:
        return float("nan"), float("nan")
    arr = np.asarray(areas_px, dtype=float)
    avg_area_um2 = float(arr.mean()) * (um_per_px ** 2)
    avg_dia_um = float(np.sqrt(4.0 * arr / np.pi).mean()) * um_per_px
    return avg_area_um2, avg_dia_um


def _resolution_pix_per_um2(um_per_px):
    """Calibration constant: pixels per square micron = 1/um_per_px².
    NaN when the scale is missing/≤0."""
    if not um_per_px or um_per_px <= 0:
        return float("nan")
    return 1.0 / (um_per_px ** 2)

