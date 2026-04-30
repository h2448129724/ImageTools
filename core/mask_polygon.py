"""Convert between binary mask images and Labelme polygon JSON annotations."""
from __future__ import annotations

import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import cv2
import numpy as np
from core.image_io import read_image, write_image, get_image_info
from utils.helpers import ensure_dir, get_image_files

logger = logging.getLogger(__name__)


def mask_to_polygons(mask, label="", epsilon_factor=0.001):
    """Extract polygon shapes from a binary mask image.

    Args:
        mask: numpy array (H, W), uint8, binary (0/255 or 0/1)
        label: label string for all extracted shapes
        epsilon_factor: contour simplification factor (relative to arc length)

    Returns:
        list of Labelme-compatible shape dicts with keys: label, points, shape_type, etc.
    """
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []
    for contour in contours:
        if len(contour) < 3:
            continue
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]
        if len(points) < 3:
            continue
        shapes.append({
            "label": label,
            "points": points,
            "shape_type": "polygon",
            "flags": {},
            "group_id": None,
            "description": "",
            "difficult": False,
            "attributes": {},
            "kie_linking": [],
        })
    return shapes


def polygons_to_mask(shapes, img_h, img_w):
    """Render Labelme polygon shapes onto a binary mask.

    Args:
        shapes: list of Labelme shape dicts (each with "points" key)
        img_h, img_w: output mask dimensions

    Returns:
        numpy array (img_h, img_w), uint8, values 0 or 255
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for shape in shapes:
        if shape.get("shape_type", "polygon") not in ("polygon", "rectangle"):
            continue
        pts = np.array(shape["points"], dtype=np.float32).reshape((-1, 1, 2))
        pts = pts.astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def batch_mask_to_labelme(mask_dir, output_dir, label="", epsilon_factor=0.001,
                          progress_callback: Callable[[int, int], None] | None = None,
                          max_workers: int = 4):
    """Convert a directory of binary mask images to Labelme JSON files.

    Returns:
        dict with keys: total_files, converted, output_dir
    """
    ensure_dir(output_dir)
    files = get_image_files(mask_dir)

    def convert_one(f: str) -> bool:
        img = read_image(f, flags=cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        base = os.path.splitext(os.path.basename(f))[0]
        shapes = mask_to_polygons(img, label, epsilon_factor)
        ann = {
            "version": "5.0.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": os.path.basename(f),
            "imageData": None,
            "imageHeight": img.shape[0],
            "imageWidth": img.shape[1],
        }
        out_path = os.path.join(output_dir, base + ".json")
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(ann, fp, indent=2, ensure_ascii=False)
        return True

    converted = 0
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(convert_one, f): f for f in files}
        for future in as_completed(futures):
            if future.result():
                converted += 1
            done += 1
            if progress_callback:
                progress_callback(done, len(files))

    result = {"total_files": len(files), "converted": converted, "output_dir": output_dir}
    logger.info("Mask→多边形: %d/%d 文件已转换", converted, len(files))
    return result


def batch_labelme_to_mask(ann_dir, output_dir, image_dir=None,
                          default_h=None, default_w=None,
                          progress_callback: Callable[[int, int], None] | None = None,
                          max_workers: int = 4):
    """Convert a directory of Labelme JSON files to binary mask images.

    Returns:
        dict with keys: total_files, converted, output_dir
    """
    ensure_dir(output_dir)
    json_files = sorted(f for f in os.listdir(ann_dir) if f.endswith(".json"))

    def convert_one(fname: str) -> bool:
        with open(os.path.join(ann_dir, fname), "r", encoding="utf-8") as fp:
            ann = json.load(fp)
        shapes = ann.get("shapes", [])
        if not shapes:
            return False
        h = ann.get("imageHeight")
        w = ann.get("imageWidth")
        if (h is None or w is None) and image_dir:
            img_name = ann.get("imagePath", os.path.splitext(fname)[0])
            img_path = os.path.join(image_dir, img_name)
            if os.path.exists(img_path):
                info = get_image_info(img_path)
                if info:
                    w, h = info[0], info[1]
        if h is None:
            h = default_h or 1080
        if w is None:
            w = default_w or 1920
        mask = polygons_to_mask(shapes, h, w)
        base = os.path.splitext(fname)[0]
        write_image(os.path.join(output_dir, base + ".png"), mask)
        return True

    converted = 0
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(convert_one, f): f for f in json_files}
        for future in as_completed(futures):
            if future.result():
                converted += 1
            done += 1
            if progress_callback:
                progress_callback(done, len(json_files))

    result = {"total_files": len(json_files), "converted": converted, "output_dir": output_dir}
    logger.info("多边形→Mask: %d/%d 文件已转换", converted, len(json_files))
    return result
