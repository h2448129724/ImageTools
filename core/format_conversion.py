"""Dataset format conversion: YOLO <-> COCO <-> VOC."""
import os
import cv2
import numpy as np
from utils.helpers import load_json, save_json, ensure_dir


# ---------- YOLO to COCO ----------
def yolo_to_coco(yolo_dir, image_dir, output_path, categories):
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
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        img_id = len(images) + 1
        images.append({"id": img_id, "file_name": os.path.basename(img_path), "width": w, "height": h})

        with open(os.path.join(yolo_dir, fname), "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:5])
                x = (xc - bw / 2) * w
                y = (yc - bh / 2) * h
                bw_abs = bw * w
                bh_abs = bh * h
                ann_id += 1
                annotations.append({"id": ann_id, "image_id": img_id, "category_id": cls,
                                    "bbox": [round(x, 2), round(y, 2), round(bw_abs, 2), round(bh_abs, 2)],
                                    "area": round(bw_abs * bh_abs, 2), "iscrowd": 0})

    coco = {"images": images, "annotations": annotations, "categories": categories}
    save_json(coco, output_path)
    return coco


# ---------- COCO to YOLO ----------
def coco_to_yolo(coco_path, output_dir):
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
def voc_to_yolo(voc_dir, output_dir, class_names):
    """Convert Pascal VOC XML annotations to YOLO txt format."""
    ensure_dir(output_dir)
    name_to_id = {n: i for i, n in enumerate(class_names)}
    from lxml import etree

    for fname in sorted(os.listdir(voc_dir)):
        if not fname.endswith(".xml"):
            continue
        tree = etree.parse(os.path.join(voc_dir, fname))
        root = tree.getroot()
        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)
        out_path = os.path.join(output_dir, os.path.splitext(fname)[0] + ".txt")
        with open(out_path, "w") as out:
            for obj in root.findall("object"):
                name = obj.find("name").text
                if name not in name_to_id:
                    continue
                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)
                xc = ((xmin + xmax) / 2) / w
                yc = ((ymin + ymax) / 2) / h
                bw = (xmax - xmin) / w
                bh = (ymax - ymin) / h
                out.write(f"{name_to_id[name]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


# ---------- VOC XML to COCO ----------
def voc_to_coco(voc_dir, image_dir, output_path, class_names):
    """Convert Pascal VOC XML to COCO JSON."""
    from lxml import etree
    name_to_id = {n: i for i, n in enumerate(class_names)}
    categories = [{"id": i, "name": n} for i, n in enumerate(class_names)]
    images, annotations = [], []
    ann_id = 0

    for fname in sorted(os.listdir(voc_dir)):
        if not fname.endswith(".xml"):
            continue
        tree = etree.parse(os.path.join(voc_dir, fname))
        root = tree.getroot()
        basename = root.find("filename").text
        img_path = os.path.join(image_dir, basename)
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        img_id = len(images) + 1
        images.append({"id": img_id, "file_name": basename, "width": w, "height": h})

        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in name_to_id:
                continue
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            bw = xmax - xmin
            bh = ymax - ymin
            ann_id += 1
            annotations.append({"id": ann_id, "image_id": img_id, "category_id": name_to_id[name],
                                "bbox": [xmin, ymin, bw, bh], "area": bw * bh, "iscrowd": 0})
    save_json({"images": images, "annotations": annotations, "categories": categories}, output_path)


# ---------- COCO to VOC ----------
def coco_to_voc(coco_path, output_dir):
    """Convert COCO JSON to Pascal VOC XML format."""
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
        xml = f"""<annotation>
    <folder>images</folder>
    <filename>{img['file_name']}</filename>
    <size>
        <width>{img['width']}</width>
        <height>{img['height']}</height>
        <depth>3</depth>
    </size>"""
        for a in anns:
            x, y, w, h = a["bbox"]
            cat_name = cat_map.get(a["category_id"], str(a["category_id"]))
            xml += f"""
    <object>
        <name>{cat_name}</name>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{int(x)}</xmin>
            <ymin>{int(y)}</ymin>
            <xmax>{int(x + w)}</xmax>
            <ymax>{int(y + h)}</ymax>
        </bndbox>
    </object>"""
        xml += "\n</annotation>\n"
        with open(os.path.join(output_dir, base + ".xml"), "w", encoding="utf-8") as f:
            f.write(xml)


# ---------- Classification dataset ----------
def create_classification_dataset(input_dir, output_dir):
    """Convert folder-per-class structure to a standard classification layout."""
    for cls_name in sorted(os.listdir(input_dir)):
        cls_path = os.path.join(input_dir, cls_name)
        if not os.path.isdir(cls_path):
            continue
        dst = os.path.join(output_dir, cls_name)
        ensure_dir(dst)
        from utils.helpers import get_image_files
        for img_path in get_image_files(cls_path):
            import shutil
            shutil.copy2(img_path, os.path.join(dst, os.path.basename(img_path)))
