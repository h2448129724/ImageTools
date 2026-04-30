"""Annotation visualization, statistics, and validation utilities."""
import os
import cv2
import numpy as np
from utils.helpers import load_json


COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
          (255, 0, 255), (0, 255, 255), (128, 0, 128), (128, 128, 0),
          (0, 128, 128), (255, 128, 0)]


def draw_yolo_boxes(img, txt_path, class_names=None):
    """Draw YOLO-format bounding boxes on an image."""
    h, w = img.shape[:2]
    result = img.copy()
    if not os.path.exists(txt_path):
        return result
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            xc, yc, bw_abs, bh_abs = map(float, parts[1:5])
            x1 = int((xc - bw_abs / 2) * w)
            y1 = int((yc - bh_abs / 2) * h)
            x2 = int((xc + bw_abs / 2) * w)
            y2 = int((yc + bh_abs / 2) * h)
            color = COLORS[cls % len(COLORS)]
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            label = class_names[cls] if class_names and cls < len(class_names) else str(cls)
            cv2.putText(result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return result


def draw_coco_boxes(img, coco_path, image_id=None, image_name=None):
    """Draw COCO-format annotations on an image."""
    data = load_json(coco_path) if isinstance(coco_path, str) else coco_path
    result = img.copy()
    h, w = img.shape[:2]
    img_info = None
    for img_ in data["images"]:
        if image_id and img_["id"] == image_id:
            img_info = img_
            break
        if image_name and img_["file_name"] == image_name:
            img_info = img_
            break
    if img_info is None:
        return result
    cat_map = {c["id"]: c["name"] for c in data.get("categories", [])}
    for ann in data["annotations"]:
        if ann["image_id"] != img_info["id"]:
            continue
        x, y, bw, bh = map(int, ann["bbox"])
        color = COLORS[ann["category_id"] % len(COLORS)]
        cv2.rectangle(result, (x, y), (x + bw, y + bh), color, 2)
        label = cat_map.get(ann["category_id"], str(ann["category_id"]))
        cv2.putText(result, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return result


def validate_yolo_annotations(txt_path, img_w, img_h):
    """Check a YOLO annotation file for errors. Returns list of issues."""
    issues = []
    if not os.path.exists(txt_path):
        return []
    with open(txt_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                issues.append(f"Line {i}: expected 5+ fields, got {len(parts)}")
                continue
            try:
                cls = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:5])
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


def annotation_statistics(ann_dir, img_dir, format_type="yolo"):
    """Compute statistics: class distribution, bbox size distribution."""
    from collections import Counter
    class_counts = Counter()
    box_areas = []
    aspect_ratios = []
    total_boxes = 0

    for fname in sorted(os.listdir(ann_dir)):
        if format_type == "yolo" and fname.endswith(".txt"):
            base = os.path.splitext(fname)[0]
            img_path = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                p = os.path.join(img_dir, base + ext)
                if os.path.exists(p):
                    img_path = p
                    break
            if img_path is None:
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            with open(os.path.join(ann_dir, fname), "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(parts[0])
                    _, _, bw_abs, bh_abs = map(float, parts[1:5])
                    class_counts[cls] += 1
                    box_areas.append(bw_abs * bh_abs * w * h)
                    aspect_ratios.append((bw_abs * w) / (bh_abs * h) if bh_abs * h > 0 else 0)
                    total_boxes += 1
    return {"class_counts": dict(class_counts), "total_boxes": total_boxes,
            "mean_area": np.mean(box_areas) if box_areas else 0,
            "median_area": np.median(box_areas) if box_areas else 0,
            "mean_aspect_ratio": np.mean(aspect_ratios) if aspect_ratios else 0}


def crop_roi_from_yolo(img_path, txt_path, output_dir, class_names=None, padding=0):
    """Crop bounding box regions from an image and save them."""
    from core.image_io import read_image, write_image
    from utils.helpers import ensure_dir
    img = read_image(img_path)
    if img is None:
        return 0
    h, w = img.shape[:2]
    base = os.path.splitext(os.path.basename(img_path))[0]
    count = 0
    with open(txt_path, "r") as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            xc, yc, bw_abs, bh_abs = map(float, parts[1:5])
            x1 = max(0, int((xc - bw_abs / 2) * w) - padding)
            y1 = max(0, int((yc - bh_abs / 2) * h) - padding)
            x2 = min(w, int((xc + bw_abs / 2) * w) + padding)
            y2 = min(h, int((yc + bh_abs / 2) * h) + padding)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img[y1:y2, x1:x2]
            cls_name = class_names[cls] if class_names and cls < len(class_names) else f"class_{cls}"
            out_dir = os.path.join(output_dir, cls_name)
            ensure_dir(out_dir)
            out_path = os.path.join(out_dir, f"{base}_roi{i}.png")
            write_image(out_path, crop)
            count += 1
    return count
