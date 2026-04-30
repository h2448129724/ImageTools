"""Shared test helpers for cross-file fixtures."""
from __future__ import annotations

import json
import os

import numpy as np
from lxml import etree

from core.image_io import write_image


def _make_test_img(w=100, h=80, channels=3):
    """Create a deterministic test image."""
    np.random.seed(42)
    if channels == 1:
        return np.random.randint(0, 255, (h, w), dtype=np.uint8)
    return np.random.randint(0, 255, (h, w, channels), dtype=np.uint8)


def _write_voc_xml(path, filename, w, h, objects):
    """Write a Pascal VOC XML annotation file.

    objects: list of (name, xmin, ymin, xmax, ymax)
    """
    root = etree.Element("annotation")
    etree.SubElement(root, "folder").text = "images"
    etree.SubElement(root, "filename").text = filename
    size_el = etree.SubElement(root, "size")
    etree.SubElement(size_el, "width").text = str(w)
    etree.SubElement(size_el, "height").text = str(h)
    etree.SubElement(size_el, "depth").text = "3"
    for name, xmin, ymin, xmax, ymax in objects:
        obj_el = etree.SubElement(root, "object")
        etree.SubElement(obj_el, "name").text = name
        etree.SubElement(obj_el, "difficult").text = "0"
        bndbox = etree.SubElement(obj_el, "bndbox")
        etree.SubElement(bndbox, "xmin").text = str(xmin)
        etree.SubElement(bndbox, "ymin").text = str(ymin)
        etree.SubElement(bndbox, "xmax").text = str(xmax)
        etree.SubElement(bndbox, "ymax").text = str(ymax)
    tree = etree.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True, pretty_print=True)


def _write_xanylabeling_json(path, shapes, img_w=200, img_h=150, image_path="img1.jpg"):
    """Write an X-AnyLabeling/LabelMe JSON file."""
    data = {
        "version": "2.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path,
        "imageData": None,
        "imageWidth": img_w,
        "imageHeight": img_h,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _make_xanylabeling_dataset(tmp_path, images_data=None):
    """Create a test dataset with images and X-AnyLabeling JSON files.

    images_data: list of (filename, shapes, w, h)
    """
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    if images_data is None:
        images_data = [
            ("img1.jpg", [
                {"label": "cat", "points": [[20, 20], [80, 80]], "shape_type": "rectangle"},
            ], 200, 150),
            ("img2.jpg", [
                {"label": "dog", "points": [[30, 30], [90, 60]], "shape_type": "rectangle"},
                {"label": "cat", "points": [[100, 50], [180, 120]], "shape_type": "rectangle"},
            ], 200, 150),
            ("img3.png", [
                {"label": "cat", "points": [[10, 10], [50, 10], [50, 50], [10, 50]], "shape_type": "polygon"},
            ], 100, 100),
        ]

    for fname, shapes, w, h in images_data:
        img = _make_test_img(w, h)
        write_image(str(src_dir / fname), img)
        json_name = os.path.splitext(fname)[0] + ".json"
        _write_xanylabeling_json(str(src_dir / json_name), shapes, w, h)

    return src_dir
