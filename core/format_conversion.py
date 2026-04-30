"""Dataset format conversion: YOLO <-> COCO <-> VOC <-> X-AnyLabeling."""
from __future__ import annotations

import json
import logging
import os
import random
import shutil
from core.image_io import get_image_info
from core.annotation import parse_yolo_file
from utils.helpers import load_json, save_json, ensure_dir, get_image_files

logger = logging.getLogger(__name__)


# ---------- YOLO to COCO ----------
def yolo_to_coco(yolo_dir: str, image_dir: str, output_path: str, categories) -> dict:
    """Convert YOLO txt annotations to COCO JSON format.
    yolo_dir: directory containing .txt files
    image_dir: directory containing images
    output_path: path for output COCO JSON
    categories: list of category names or list of {"id": int, "name": str}
    """
    images, annotations = [], []
    ann_id = 0
    if isinstance(categories[0], str):
        categories = [{"id": i, "name": n} for i, n in enumerate(categories)]

    for fname in sorted(os.listdir(yolo_dir)):
        if not fname.endswith(".txt"):
            continue
        base = os.path.splitext(fname)[0]
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            p = os.path.join(image_dir, base + ext)
            if os.path.exists(p):
                img_path = p
                break
        if img_path is None:
            continue
        info = get_image_info(img_path)
        if info is None:
            continue
        w, h = info[0], info[1]
        img_id = len(images) + 1
        images.append({"id": img_id, "file_name": os.path.basename(img_path), "width": w, "height": h})

        for box in parse_yolo_file(os.path.join(yolo_dir, fname)):
            x = (box["xc"] - box["bw"] / 2) * w
            y = (box["yc"] - box["bh"] / 2) * h
            bw_abs = box["bw"] * w
            bh_abs = box["bh"] * h
            ann_id += 1
            annotations.append({"id": ann_id, "image_id": img_id, "category_id": box["cls"],
                                "bbox": [round(x, 2), round(y, 2), round(bw_abs, 2), round(bh_abs, 2)],
                                "area": round(bw_abs * bh_abs, 2), "iscrowd": 0})

    coco = {"images": images, "annotations": annotations, "categories": categories}
    save_json(coco, output_path)
    return coco


# ---------- COCO to YOLO ----------
def coco_to_yolo(coco_path: str, output_dir: str) -> None:
    """Convert COCO JSON to YOLO txt format."""
    ensure_dir(output_dir)
    data = load_json(coco_path)
    img_map = {img["id"]: (img["file_name"], img["width"], img["height"]) for img in data["images"]}
    anns_by_image = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    for img_id, anns in anns_by_image.items():
        if img_id not in img_map:
            continue
        fname, w, h = img_map[img_id]
        base = os.path.splitext(fname)[0]
        out_path = os.path.join(output_dir, base + ".txt")
        with open(out_path, "w") as f:
            for a in anns:
                x, y, bw, bh = a["bbox"]
                xc = (x + bw / 2) / w
                yc = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h
                f.write(f"{a['category_id']} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")


# ---------- VOC XML to YOLO ----------
def voc_to_yolo(voc_dir: str, output_dir: str, class_names: list[str]) -> None:
    """Convert Pascal VOC XML annotations to YOLO txt format."""
    ensure_dir(output_dir)
    name_to_id = {n: i for i, n in enumerate(class_names)}
    from lxml import etree

    for fname in sorted(os.listdir(voc_dir)):
        if not fname.endswith(".xml"):
            continue
        try:
            tree = etree.parse(os.path.join(voc_dir, fname))
        except (etree.XMLSyntaxError, OSError) as e:
            logger.warning("Skipping VOC file %s: %s", fname, e)
            continue
        root = tree.getroot()
        size = root.find("size")
        if size is None:
            logger.warning("Skipping %s: missing <size> element", fname)
            continue
        w_el, h_el = size.find("width"), size.find("height")
        if w_el is None or h_el is None or w_el.text is None or h_el.text is None:
            logger.warning("Skipping %s: missing width/height in <size>", fname)
            continue
        w, h = int(w_el.text), int(h_el.text)
        out_path = os.path.join(output_dir, os.path.splitext(fname)[0] + ".txt")
        with open(out_path, "w") as out:
            for obj in root.findall("object"):
                name_el = obj.find("name")
                if name_el is None or name_el.text is None:
                    continue
                name = name_el.text
                if name not in name_to_id:
                    continue
                bbox = obj.find("bndbox")
                if bbox is None:
                    continue
                try:
                    xmin = float(bbox.find("xmin").text)
                    ymin = float(bbox.find("ymin").text)
                    xmax = float(bbox.find("xmax").text)
                    ymax = float(bbox.find("ymax").text)
                except (AttributeError, TypeError, ValueError):
                    continue
                xc = ((xmin + xmax) / 2) / w
                yc = ((ymin + ymax) / 2) / h
                bw = (xmax - xmin) / w
                bh = (ymax - ymin) / h
                out.write(f"{name_to_id[name]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


# ---------- YOLO to VOC XML ----------
def yolo_to_voc(yolo_dir: str, image_dir: str, output_dir: str, class_names: list[str]) -> None:
    """Convert YOLO txt annotations to Pascal VOC XML format."""
    from lxml import etree
    ensure_dir(output_dir)
    for fname in sorted(os.listdir(yolo_dir)):
        if not fname.endswith(".txt"):
            continue
        base = os.path.splitext(fname)[0]
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            p = os.path.join(image_dir, base + ext)
            if os.path.exists(p):
                img_path = p
                break
        if img_path is None:
            continue
        info = get_image_info(img_path)
        if info is None:
            continue
        w, h = info[0], info[1]
        filename = os.path.basename(img_path)

        root = etree.Element("annotation")
        etree.SubElement(root, "folder").text = "images"
        etree.SubElement(root, "filename").text = filename
        size_el = etree.SubElement(root, "size")
        etree.SubElement(size_el, "width").text = str(w)
        etree.SubElement(size_el, "height").text = str(h)
        etree.SubElement(size_el, "depth").text = "3"
        for box in parse_yolo_file(os.path.join(yolo_dir, fname)):
            cls_id = box["cls"]
            cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            xmin = (box["xc"] - box["bw"] / 2) * w
            ymin = (box["yc"] - box["bh"] / 2) * h
            xmax = (box["xc"] + box["bw"] / 2) * w
            ymax = (box["yc"] + box["bh"] / 2) * h
            obj_el = etree.SubElement(root, "object")
            etree.SubElement(obj_el, "name").text = cls_name
            etree.SubElement(obj_el, "difficult").text = "0"
            bndbox = etree.SubElement(obj_el, "bndbox")
            etree.SubElement(bndbox, "xmin").text = str(int(xmin))
            etree.SubElement(bndbox, "ymin").text = str(int(ymin))
            etree.SubElement(bndbox, "xmax").text = str(int(xmax))
            etree.SubElement(bndbox, "ymax").text = str(int(ymax))
        tree = etree.ElementTree(root)
        tree.write(os.path.join(output_dir, base + ".xml"), encoding="utf-8", xml_declaration=True,
                   pretty_print=True)


# ---------- VOC XML to COCO ----------
def voc_to_coco(voc_dir: str, image_dir: str, output_path: str, class_names: list[str]) -> None:
    """Convert Pascal VOC XML to COCO JSON."""
    from lxml import etree
    name_to_id = {n: i for i, n in enumerate(class_names)}
    categories = [{"id": i, "name": n} for i, n in enumerate(class_names)]
    images, annotations = [], []
    ann_id = 0

    for fname in sorted(os.listdir(voc_dir)):
        if not fname.endswith(".xml"):
            continue
        try:
            tree = etree.parse(os.path.join(voc_dir, fname))
        except (etree.XMLSyntaxError, OSError) as e:
            logger.warning("Skipping VOC file %s: %s", fname, e)
            continue
        root = tree.getroot()
        fn_el = root.find("filename")
        if fn_el is None or fn_el.text is None:
            logger.warning("Skipping %s: missing <filename>", fname)
            continue
        basename = fn_el.text
        img_path = os.path.join(image_dir, basename)
        if not os.path.exists(img_path):
            logger.debug("Skipping %s: image not found at %s", fname, img_path)
            continue
        info = get_image_info(img_path)
        if info is None:
            logger.warning("Skipping %s: cannot read image info for %s", fname, img_path)
            continue
        w, h = info[0], info[1]
        img_id = len(images) + 1
        images.append({"id": img_id, "file_name": basename, "width": w, "height": h})

        for obj in root.findall("object"):
            name_el = obj.find("name")
            if name_el is None or name_el.text is None:
                continue
            name = name_el.text
            if name not in name_to_id:
                continue
            bbox = obj.find("bndbox")
            if bbox is None:
                continue
            try:
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)
            except (AttributeError, TypeError, ValueError):
                continue
            bw = xmax - xmin
            bh = ymax - ymin
            ann_id += 1
            annotations.append({"id": ann_id, "image_id": img_id, "category_id": name_to_id[name],
                                "bbox": [xmin, ymin, bw, bh], "area": bw * bh, "iscrowd": 0})
    save_json({"images": images, "annotations": annotations, "categories": categories}, output_path)


# ---------- COCO to VOC ----------
def coco_to_voc(coco_path: str, output_dir: str) -> None:
    """Convert COCO JSON to Pascal VOC XML format."""
    from lxml import etree
    ensure_dir(output_dir)
    data = load_json(coco_path)
    cat_map = {c["id"]: c["name"] for c in data.get("categories", [])}
    img_map = {img["id"]: img for img in data["images"]}
    anns_by_image = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    for img_id, anns in anns_by_image.items():
        img = img_map[img_id]
        base = os.path.splitext(img["file_name"])[0]
        root = etree.Element("annotation")
        etree.SubElement(root, "folder").text = "images"
        etree.SubElement(root, "filename").text = img["file_name"]
        size_el = etree.SubElement(root, "size")
        etree.SubElement(size_el, "width").text = str(img["width"])
        etree.SubElement(size_el, "height").text = str(img["height"])
        etree.SubElement(size_el, "depth").text = "3"
        for a in anns:
            x, y, w, h = a["bbox"]
            cat_name = cat_map.get(a["category_id"], str(a["category_id"]))
            obj_el = etree.SubElement(root, "object")
            etree.SubElement(obj_el, "name").text = cat_name
            etree.SubElement(obj_el, "difficult").text = "0"
            bndbox = etree.SubElement(obj_el, "bndbox")
            etree.SubElement(bndbox, "xmin").text = str(int(x))
            etree.SubElement(bndbox, "ymin").text = str(int(y))
            etree.SubElement(bndbox, "xmax").text = str(int(x + w))
            etree.SubElement(bndbox, "ymax").text = str(int(y + h))
        tree = etree.ElementTree(root)
        tree.write(os.path.join(output_dir, base + ".xml"), encoding="utf-8", xml_declaration=True,
                   pretty_print=True)


# ---------- Classification dataset ----------
def create_classification_dataset(input_dir: str, output_dir: str) -> str:
    """Copy images from class subdirectories to output directory preserving structure.

    Input directory is expected to contain one subdirectory per class,
    with image files inside each subdirectory.
    """
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    total = 0
    for entry in sorted(os.listdir(input_dir)):
        src = os.path.join(input_dir, entry)
        if not os.path.isdir(src):
            continue
        dst_cls_dir = os.path.join(output_dir, entry)
        ensure_dir(dst_cls_dir)
        for fname in sorted(os.listdir(src)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in img_exts:
                shutil.copy2(os.path.join(src, fname), os.path.join(dst_cls_dir, fname))
                total += 1
    return f"复制图片 {total} 张"


# ---------- X-AnyLabeling / LabelMe to YOLO ----------
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _yolo_box_from_points(points, width: int, height: int) -> tuple[float, float, float, float]:
    """Convert shape points to YOLO normalized bbox (xc, yc, w, h)."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min = max(0.0, min(xs))
    x_max = min(float(width), max(xs))
    y_min = max(0.0, min(ys))
    y_max = min(float(height), max(ys))
    box_w = max(0.0, x_max - x_min)
    box_h = max(0.0, y_max - y_min)
    xc = (x_min + box_w / 2.0) / width
    yc = (y_min + box_h / 2.0) / height
    return xc, yc, box_w / width, box_h / height


def _load_xanylabeling_json(json_path: str, class_to_id: dict[str, int]) -> tuple[list[str], int, int]:
    """Load a single X-AnyLabeling/LabelMe JSON and return YOLO lines."""
    data = load_json(json_path)
    width = int(data.get("imageWidth", 0))
    height = int(data.get("imageHeight", 0))
    if width <= 0 or height <= 0:
        return [], width, height

    lines = []
    for shape in data.get("shapes", []):
        label = shape.get("label", "")
        if label not in class_to_id:
            continue
        points = shape.get("points", [])
        if len(points) < 2:
            continue
        xc, yc, w, h = _yolo_box_from_points(points, width, height)
        if w <= 0 or h <= 0:
            continue
        lines.append(f"{class_to_id[label]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines, width, height


def collect_xanylabeling_labels(src_dir: str) -> list[str]:
    """Scan all JSON files under src_dir and collect unique label names."""
    labels = set()
    for root, _, files in os.walk(src_dir):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            json_path = os.path.join(root, fname)
            try:
                data = load_json(json_path)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Skipping corrupted JSON %s: %s", json_path, e)
                continue
            for shape in data.get("shapes", []):
                label = shape.get("label", "")
                if label:
                    labels.add(label)
    return sorted(labels)


def xanylabeling_to_yolo(src_dir: str, output_dir: str, val_ratio: float = 0.2, seed: int = 42, categories: list[str] | None = None) -> str:
    """Convert X-AnyLabeling/LabelMe dataset to YOLO format with train/val split.

    Scans src_dir for images and JSON files, converts annotations to YOLO format,
    splits into train/val sets, and generates data.yaml.

    Output structure:
    output_dir/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── data.yaml

    Args:
        src_dir: source directory containing images and JSON files
        output_dir: output directory for YOLO dataset
        val_ratio: validation set ratio (default 0.2)
        seed: random seed for split
        categories: optional list of class names; if None, auto-collected from JSONs
    """
    src_dir = os.path.abspath(src_dir)
    output_dir = os.path.abspath(output_dir)

    # Collect images
    images = []
    for root, _, files in os.walk(src_dir):
        for fname in sorted(files):
            if os.path.splitext(fname)[1].lower() in _IMAGE_SUFFIXES:
                images.append(os.path.join(root, fname))
    if not images:
        raise ValueError(f"未找到图片文件: {src_dir}")

    # Collect or use provided categories
    if categories:
        names = list(categories)
    else:
        names = collect_xanylabeling_labels(src_dir)
    if not names:
        raise ValueError(f"未找到标注类别: {src_dir}")
    class_to_id = {n: i for i, n in enumerate(names)}

    # Split train/val
    rng = random.Random(seed)
    shuffled = images[:]
    rng.shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_ratio))
    val_set = set(shuffled[:val_count])

    # Create output directories
    for split in ("train", "val"):
        ensure_dir(os.path.join(output_dir, "images", split))
        ensure_dir(os.path.join(output_dir, "labels", split))

    # Process each image
    missing_json = []
    for img_path in images:
        base = os.path.splitext(os.path.basename(img_path))[0]
        split = "val" if img_path in val_set else "train"

        # Copy image
        dst_img = os.path.join(output_dir, "images", split, os.path.basename(img_path))
        shutil.copy2(img_path, dst_img)

        # Find and convert JSON annotation
        json_path = os.path.splitext(img_path)[0] + ".json"
        if os.path.exists(json_path):
            label_lines, _, _ = _load_xanylabeling_json(json_path, class_to_id)
        else:
            label_lines = []
            missing_json.append(img_path)

        # Write YOLO label file
        dst_lbl = os.path.join(output_dir, "labels", split, base + ".txt")
        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines) + ("\n" if label_lines else ""))

    # Generate data.yaml
    yaml_path = os.path.join(output_dir, "data.yaml")
    _write_yolo_yaml(yaml_path, output_dir, names)

    train_count = len(images) - len(val_set)
    result = (f"转换完成: {len(images)} 张图片, {len(names)} 个类别\n"
              f"训练集: {train_count}, 验证集: {len(val_set)}\n"
              f"类别: {', '.join(names)}")
    if missing_json:
        result += f"\n无标注JSON的图片({len(missing_json)}张)将生成空标签文件"
    return result


def _write_yolo_yaml(yaml_path: str, dataset_root: str, names: list[str]) -> None:
    """Write YOLO data.yaml configuration file."""
    lines = [
        f"path: {dataset_root}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for idx, name in enumerate(names):
        lines.append(f"  {idx}: {name}")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
