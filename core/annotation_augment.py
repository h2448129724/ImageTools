"""Annotation augmentation: transform images and their annotations in lockstep."""
from __future__ import annotations

import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import cv2
import numpy as np
from core.image_io import read_image, write_image, get_image_info, resize_image
from core.basic_processing import rotate_image, flip_image, crop_image
from core.annotation import parse_yolo_file
from utils.helpers import ensure_dir, get_image_files, save_json

logger = logging.getLogger(__name__)


def _rotate_points(points, img_w, img_h, angle, keep_size=True):
    """Apply rotation transform to a list of [x, y] absolute coordinate pairs."""
    center = (img_w / 2, img_h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    if keep_size:
        new_w, new_h = img_w, img_h
    else:
        cos_a = abs(M[0, 0])
        sin_a = abs(M[0, 1])
        new_w = int(img_h * sin_a + img_w * cos_a)
        new_h = int(img_h * cos_a + img_w * sin_a)
        M[0, 2] += new_w / 2 - center[0]
        M[1, 2] += new_h / 2 - center[1]
    pts = np.array(points, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.hstack([pts, ones])
    transformed = (M @ pts_h.T).T
    return transformed.tolist(), new_w, new_h


def _flip_points(points, img_w, img_h, direction):
    """Apply flip transform to a list of [x, y] absolute coordinate pairs."""
    result = []
    for x, y in points:
        if direction == "horizontal":
            result.append([img_w - x, y])
        elif direction == "vertical":
            result.append([x, img_h - y])
        else:
            result.append([img_w - x, img_h - y])
    return result


def transform_yolo_bbox(boxes, img_w, img_h, transform_type, **kwargs):
    """Transform YOLO bounding boxes in lockstep with a geometric image transform.

    Strategy: decode each box to absolute corners, apply transform, re-encode.

    Args:
        boxes: list of dicts with keys cls, xc, yc, bw, bh (normalized YOLO format)
        img_w, img_h: original image dimensions
        transform_type: "rotate", "flip", "resize", or "crop"
        **kwargs: transform-specific parameters

    Returns:
        (transformed_boxes, new_w, new_h)
    """
    new_w, new_h = img_w, img_h
    transformed = []

    for box in boxes:
        # Decode to absolute corners
        xc = box["xc"] * img_w
        yc = box["yc"] * img_h
        bw = box["bw"] * img_w
        bh = box["bh"] * img_h
        corners = [
            [xc - bw / 2, yc - bh / 2],
            [xc + bw / 2, yc - bh / 2],
            [xc + bw / 2, yc + bh / 2],
            [xc - bw / 2, yc + bh / 2],
        ]

        if transform_type == "rotate":
            angle = kwargs.get("angle", 90)
            keep_size = kwargs.get("keep_size", True)
            corners, new_w, new_h = _rotate_points(corners, img_w, img_h, angle, keep_size)
        elif transform_type == "flip":
            direction = kwargs.get("direction", "horizontal")
            corners = _flip_points(corners, img_w, img_h, direction)
        elif transform_type == "resize":
            new_w = kwargs.get("new_w", img_w)
            new_h = kwargs.get("new_h", img_h)
            sx = new_w / img_w
            sy = new_h / img_h
            corners = [[x * sx, y * sy] for x, y in corners]
        elif transform_type == "crop":
            cx, cy = kwargs.get("x", 0), kwargs.get("y", 0)
            cw, ch = kwargs.get("w", img_w), kwargs.get("h", img_h)
            corners = [[x - cx, y - cy] for x, y in corners]
            new_w, new_h = cw, ch

        # Re-encode as axis-aligned bbox
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        x1 = max(0, min(xs))
        y1 = max(0, min(ys))
        x2 = min(new_w, max(xs))
        y2 = min(new_h, max(ys))

        bw_new = x2 - x1
        bh_new = y2 - y1
        if bw_new <= 0 or bh_new <= 0:
            continue  # Box fully outside after transform
        xc_new = (x1 + x2) / 2
        yc_new = (y1 + y2) / 2

        transformed.append({
            "cls": box["cls"],
            "xc": xc_new / new_w,
            "yc": yc_new / new_h,
            "bw": bw_new / new_w,
            "bh": bh_new / new_h,
        })

    return transformed, new_w, new_h


def transform_labelme_shapes(shapes, img_h, img_w, transform_type, **kwargs):
    """Transform Labelme polygon annotation points in lockstep with an image transform.

    Args:
        shapes: list of Labelme shape dicts
        img_h, img_w: original image dimensions
        transform_type: "rotate", "flip", "resize", or "crop"
        **kwargs: transform-specific parameters

    Returns:
        (transformed_shapes, new_w, new_h)
    """
    new_w, new_h = img_w, img_h
    transformed = []

    for shape in shapes:
        points = [list(p) for p in shape["points"]]

        if transform_type == "rotate":
            angle = kwargs.get("angle", 90)
            keep_size = kwargs.get("keep_size", True)
            points, new_w, new_h = _rotate_points(points, img_w, img_h, angle, keep_size)
            points = [[round(x, 2), round(y, 2)] for x, y in points]
        elif transform_type == "flip":
            direction = kwargs.get("direction", "horizontal")
            points = _flip_points(points, img_w, img_h, direction)
        elif transform_type == "resize":
            new_w = kwargs.get("new_w", img_w)
            new_h = kwargs.get("new_h", img_h)
            sx = new_w / img_w
            sy = new_h / img_h
            points = [[round(x * sx, 2), round(y * sy, 2)] for x, y in points]
        elif transform_type == "crop":
            cx, cy = kwargs.get("x", 0), kwargs.get("y", 0)
            cw, ch = kwargs.get("w", img_w), kwargs.get("h", img_h)
            points = [[round(x - cx, 2), round(y - cy, 2)] for x, y in points]
            new_w, new_h = cw, ch

        new_shape = dict(shape)
        new_shape["points"] = points
        transformed.append(new_shape)

    return transformed, new_w, new_h


def _transform_image(img, transform_type, **kwargs):
    """Apply a geometric transform to an image."""
    if transform_type == "rotate":
        return rotate_image(img, kwargs.get("angle", 90),
                            keep_size=kwargs.get("keep_size", True))
    if transform_type == "flip":
        return flip_image(img, kwargs.get("direction", "horizontal"))
    if transform_type == "resize":
        return resize_image(img, kwargs.get("new_w"), kwargs.get("new_h"))
    if transform_type == "crop":
        return crop_image(img, kwargs.get("x", 0), kwargs.get("y", 0),
                          kwargs.get("w"), kwargs.get("h"))
    return img


def augment_batch(input_dir, ann_dir, output_dir, transform_type, ann_format="yolo",
                  class_names=None, progress_callback: Callable[[int, int], None] | None = None,
                  max_workers: int = 4, **kwargs):
    """Batch augment images and their annotations.

    Args:
        input_dir: directory of source images
        ann_dir: directory of annotation files
        output_dir: output directory (images/ and annotations/ subdirs)
        transform_type: "rotate", "flip", "resize", or "crop"
        ann_format: "yolo" or "labelme"
        class_names: optional class name list (for YOLO output)
        progress_callback: optional callback(current, total)
        max_workers: parallel thread count
        **kwargs: transform-specific parameters

    Returns:
        dict with processing statistics
    """
    out_img_dir = os.path.join(output_dir, "images")
    out_ann_dir = os.path.join(output_dir, "annotations")
    ensure_dir(out_img_dir)
    ensure_dir(out_ann_dir)

    files = get_image_files(input_dir)

    def process_one(img_path: str) -> bool:
        base = os.path.splitext(os.path.basename(img_path))[0]
        img = read_image(img_path)
        if img is None:
            return False
        h, w = img.shape[:2]

        img_out = _transform_image(img, transform_type, **kwargs)
        new_h, new_w = img_out.shape[:2]
        write_image(os.path.join(out_img_dir, os.path.basename(img_path)), img_out)

        if ann_format == "yolo":
            txt_path = os.path.join(ann_dir, base + ".txt")
            if not os.path.exists(txt_path):
                return False
            boxes = parse_yolo_file(txt_path)
            boxes_out, _, _ = transform_yolo_bbox(boxes, w, h, transform_type, **kwargs)
            out_txt = os.path.join(out_ann_dir, base + ".txt")
            with open(out_txt, "w") as f:
                for box in boxes_out:
                    f.write(f"{box['cls']} {box['xc']:.6f} {box['yc']:.6f} "
                            f"{box['bw']:.6f} {box['bh']:.6f}\n")
        else:  # labelme
            json_path = os.path.join(ann_dir, base + ".json")
            if not os.path.exists(json_path):
                return False
            with open(json_path, "r", encoding="utf-8") as f:
                ann = json.load(f)
            shapes = ann.get("shapes", [])
            shapes_out, _, _ = transform_labelme_shapes(shapes, h, w, transform_type, **kwargs)
            ann_out = dict(ann)
            ann_out["shapes"] = shapes_out
            ann_out["imageHeight"] = new_h
            ann_out["imageWidth"] = new_w
            ann_out["imagePath"] = os.path.basename(img_path)
            save_json(ann_out, os.path.join(out_ann_dir, base + ".json"))
        return True

    processed = 0
    skipped = 0
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(process_one, f): f for f in files}
        for future in as_completed(futures):
            done += 1
            if future.result():
                processed += 1
            else:
                skipped += 1
            if progress_callback:
                progress_callback(done, len(files))

    result = {"total_files": len(files), "processed": processed, "skipped": skipped}
    logger.info("标注增强(%s): 处理 %d, 跳过 %d", transform_type, processed, skipped)
    return result
