"""Annotation visualization, statistics, and validation utilities."""
from __future__ import annotations

import json
import logging
import os
from typing import Any
import hashlib

import cv2
import numpy as np
from numpy.typing import NDArray

from core.image_io import read_image, get_image_info
from utils.helpers import load_json

logger = logging.getLogger(__name__)


COLORS: list[tuple[int, int, int]] = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 128), (128, 128, 0),
    (0, 128, 128), (255, 128, 0),
]

# Drawing constants
BOX_THICKNESS = 2
TEXT_FONT_SCALE = 0.5
TEXT_FONT_SCALE_SMALL = 0.4
TEXT_THICKNESS = 1

# LabelMe point constants
POINT_RADIUS = 6
POINT_OUTER_RADIUS = 8
CROSSHAIR_LEN = 10

# YOLO format constants
YOLO_FIELD_COUNT = 5

YoloBox = dict[str, float | int]


def parse_yolo_file(txt_path: str) -> list[YoloBox]:
    """Parse a YOLO annotation file. Returns list of dicts with cls, xc, yc, bw, bh."""
    if not os.path.exists(txt_path):
        return []
    boxes: list[YoloBox] = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                boxes.append({
                    "cls": int(parts[0]),
                    "xc": float(parts[1]),
                    "yc": float(parts[2]),
                    "bw": float(parts[3]),
                    "bh": float(parts[4]),
                })
            except ValueError:
                continue
    return boxes


def draw_yolo_boxes(img: NDArray[np.uint8], txt_path: str,
                    class_names: list[str] | None = None) -> NDArray[np.uint8]:
    """Draw YOLO-format bounding boxes on an image."""
    h, w = img.shape[:2]
    result = img.copy()
    for box in parse_yolo_file(txt_path):
        cls = int(box["cls"])
        x1 = int((box["xc"] - box["bw"] / 2) * w)
        y1 = int((box["yc"] - box["bh"] / 2) * h)
        x2 = int((box["xc"] + box["bw"] / 2) * w)
        y2 = int((box["yc"] + box["bh"] / 2) * h)
        color = COLORS[cls % len(COLORS)]
        cv2.rectangle(result, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        label = class_names[cls] if class_names and cls < len(class_names) else str(cls)
        cv2.putText(result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, TEXT_FONT_SCALE, color, TEXT_THICKNESS)
    return result


def draw_coco_boxes(img: NDArray[np.uint8], coco_path: str | dict[str, Any],
                    image_id: int | None = None,
                    image_name: str | None = None) -> NDArray[np.uint8]:
    """Draw COCO-format annotations on an image."""
    data: dict[str, Any] = load_json(coco_path) if isinstance(coco_path, str) else coco_path
    result = img.copy()
    img_info = None
    for img_ in data["images"]:
        if image_id is not None and img_["id"] == image_id:
            img_info = img_
            break
        if image_name and img_["file_name"] == image_name:
            img_info = img_
            break
    if img_info is None:
        return result
    cat_map: dict[int, str] = {c["id"]: c["name"] for c in data.get("categories", [])}
    for ann in data["annotations"]:
        if ann["image_id"] != img_info["id"]:
            continue
        x, y, bw, bh = map(int, ann["bbox"])
        color = COLORS[ann["category_id"] % len(COLORS)]
        cv2.rectangle(result, (x, y), (x + bw, y + bh), color, BOX_THICKNESS)
        label = cat_map.get(ann["category_id"], str(ann["category_id"]))
        cv2.putText(result, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, TEXT_FONT_SCALE, color, TEXT_THICKNESS)
    return result


def validate_yolo_annotations(txt_path: str, img_w: int, img_h: int) -> list[str]:
    """Check a YOLO annotation file for errors. Returns list of issues."""
    issues: list[str] = []
    if not os.path.exists(txt_path):
        return []
    with open(txt_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < YOLO_FIELD_COUNT:
                issues.append(f"Line {i}: expected {YOLO_FIELD_COUNT}+ fields, got {len(parts)}")
                continue
            try:
                xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                issues.append(f"Line {i}: non-numeric values")
                continue
            if xc < 0 or xc > 1 or yc < 0 or yc > 1:
                issues.append(f"Line {i}: normalized center out of [0,1]")
            if bw <= 0 or bh <= 0 or bw > 1 or bh > 1:
                issues.append(f"Line {i}: invalid box dimensions (bw={bw}, bh={bh})")
            x1 = (xc - bw / 2) * img_w
            y1 = (yc - bh / 2) * img_h
            x2 = (xc + bw / 2) * img_w
            y2 = (yc + bh / 2) * img_h
            if x2 <= x1 or y2 <= y1:
                issues.append(f"Line {i}: zero-area or inverted box")
            if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
                issues.append(f"Line {i}: box exceeds image bounds")
    return issues


def annotation_statistics(ann_dir: str, img_dir: str,
                          format_type: str = "yolo") -> dict[str, Any]:
    """Compute statistics: class distribution, bbox size distribution."""
    from collections import Counter
    class_counts: Counter[int] = Counter()
    box_areas: list[float] = []
    aspect_ratios: list[float] = []
    total_boxes = 0

    for fname in sorted(os.listdir(ann_dir)):
        if format_type == "yolo" and fname.endswith(".txt"):
            base = os.path.splitext(fname)[0]
            img_path = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                p = os.path.join(img_dir, base + ext)
                if os.path.exists(p):
                    img_path = p
                    break
            if img_path is None:
                continue
            info = get_image_info(img_path)
            if info is None:
                continue
            w, h = info[0], info[1]
            for box in parse_yolo_file(os.path.join(ann_dir, fname)):
                class_counts[box["cls"]] += 1
                box_areas.append(box["bw"] * box["bh"] * w * h)
                bw_px = box["bw"] * w
                bh_px = box["bh"] * h
                aspect_ratios.append(bw_px / bh_px if bh_px > 0 else 0)
                total_boxes += 1
    return {"class_counts": dict(class_counts), "total_boxes": total_boxes,
            "mean_area": np.mean(box_areas) if box_areas else 0,
            "median_area": np.median(box_areas) if box_areas else 0,
            "mean_aspect_ratio": np.mean(aspect_ratios) if aspect_ratios else 0}


def crop_roi_from_yolo(img_path: str, txt_path: str, output_dir: str,
                       class_names: list[str] | None = None,
                       padding: int = 0) -> int:
    """Crop bounding box regions from an image and save them."""
    from core.image_io import write_image
    from utils.helpers import ensure_dir
    img = read_image(img_path)
    if img is None:
        return 0
    h, w = img.shape[:2]
    base = os.path.splitext(os.path.basename(img_path))[0]
    count = 0
    for i, box in enumerate(parse_yolo_file(txt_path)):
        x1 = max(0, int((box["xc"] - box["bw"] / 2) * w) - padding)
        y1 = max(0, int((box["yc"] - box["bh"] / 2) * h) - padding)
        x2 = min(w, int((box["xc"] + box["bw"] / 2) * w) + padding)
        y2 = min(h, int((box["yc"] + box["bh"] / 2) * h) + padding)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img[y1:y2, x1:x2]
        cls_name = (class_names[box["cls"]] if class_names and box["cls"] < len(class_names)
                    else f"class_{box['cls']}")
        out_dir = os.path.join(output_dir, cls_name)
        ensure_dir(out_dir)
        out_path = os.path.join(out_dir, f"{base}_roi{i}.png")
        write_image(out_path, crop)
        count += 1
    return count


def parse_labelme(json_path: str) -> dict[str, Any] | None:
    """Parse a LabelMe JSON file. Returns the parsed dict or None."""
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to parse LabelMe JSON %s: %s", json_path, e)
        return None


def draw_labelme_shapes(img: NDArray[np.uint8],
                        json_path: str) -> NDArray[np.uint8]:
    """Draw LabelMe annotation shapes on an image and return it."""
    data = parse_labelme(json_path)
    if data is None:
        return img
    result = img.copy()
    shapes = data.get("shapes", [])
    for shape in shapes:
        label = shape.get("label", "")
        shape_type = shape.get("shape_type", "")
        points = shape.get("points", [])
        color = COLORS[int(hashlib.md5(label.encode()).hexdigest(), 16) % len(COLORS)]
        if shape_type == "point" and len(points) >= 1:
            x, y = int(points[0][0]), int(points[0][1])
            cv2.circle(result, (x, y), POINT_RADIUS, color, -1)
            cv2.circle(result, (x, y), POINT_OUTER_RADIUS, (255, 255, 255), TEXT_THICKNESS)
            cv2.line(result, (x - CROSSHAIR_LEN, y), (x + CROSSHAIR_LEN, y), color, TEXT_THICKNESS)
            cv2.line(result, (x, y - CROSSHAIR_LEN), (x, y + CROSSHAIR_LEN), color, TEXT_THICKNESS)
            if label:
                cv2.putText(result, label, (x + CROSSHAIR_LEN, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, TEXT_FONT_SCALE_SMALL, color, TEXT_THICKNESS)
        elif shape_type == "rectangle" and len(points) >= 2:
            x1, y1 = int(points[0][0]), int(points[0][1])
            x2, y2 = int(points[1][0]), int(points[1][1])
            cv2.rectangle(result, (x1, y1), (x2, y2), color, BOX_THICKNESS)
            if label:
                cv2.putText(result, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, TEXT_FONT_SCALE, color, TEXT_THICKNESS)
        elif shape_type == "polygon" and len(points) >= 3:
            pts = np.array([[int(p[0]), int(p[1])] for p in points], np.int32)
            cv2.polylines(result, [pts], True, color, BOX_THICKNESS)
            if label:
                cv2.putText(result, label, (pts[0][0], pts[0][1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, TEXT_FONT_SCALE, color, TEXT_THICKNESS)
        elif shape_type == "circle" and len(points) >= 2:
            cx, cy = int(points[0][0]), int(points[0][1])
            ex, ey = int(points[1][0]), int(points[1][1])
            r = int(((ex - cx) ** 2 + (ey - cy) ** 2) ** 0.5)
            cv2.circle(result, (cx, cy), r, color, BOX_THICKNESS)
            if label:
                cv2.putText(result, label, (cx - r, cy - r - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, TEXT_FONT_SCALE, color, TEXT_THICKNESS)
        elif shape_type == "line" and len(points) >= 2:
            x1, y1 = int(points[0][0]), int(points[0][1])
            x2, y2 = int(points[1][0]), int(points[1][1])
            cv2.line(result, (x1, y1), (x2, y2), color, BOX_THICKNESS)
    return result
