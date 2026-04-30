"""Tests for core.format_conversion module."""
import os
import json
import numpy as np
import pytest
from lxml import etree
from core.image_io import write_image
from core.format_conversion import (
    yolo_to_coco,
    coco_to_yolo,
    voc_to_yolo,
    yolo_to_voc,
    create_classification_dataset,
    xanylabeling_to_yolo,
    collect_xanylabeling_labels,
)
from tests.helpers import _make_test_img, _write_voc_xml, _write_xanylabeling_json, _make_xanylabeling_dataset


def _write_yolo_txt(path, lines):
    """Write a YOLO annotation file with the given text lines."""
    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")


def _setup_yolo_images_and_annotations(tmp_path, annotations=None):
    """Create image files and matching YOLO annotation files.

    annotations: dict mapping base_name -> list of YOLO annotation strings.
                 Default: one image "img1" with one box.
    """
    img_dir = tmp_path / "images"
    ann_dir = tmp_path / "annotations"
    img_dir.mkdir()
    ann_dir.mkdir()

    if annotations is None:
        annotations = {"img1": ["0 0.5 0.5 0.2 0.3"]}

    for base, lines in annotations.items():
        img = _make_test_img(200, 150)
        write_image(str(img_dir / (base + ".jpg")), img)
        _write_yolo_txt(str(ann_dir / (base + ".txt")), lines)

    return str(ann_dir), str(img_dir)


# ---------- yolo_to_coco ----------

class TestYoloToCoco:
    def test_basic_conversion(self, tmp_path):
        ann_dir, img_dir = _setup_yolo_images_and_annotations(tmp_path)
        output_path = str(tmp_path / "coco_output.json")

        result = yolo_to_coco(ann_dir, img_dir, output_path, categories=["cat", "dog"])

        # Check returned dict structure
        assert "images" in result
        assert "annotations" in result
        assert "categories" in result
        assert len(result["images"]) == 1
        assert len(result["annotations"]) == 1
        assert result["categories"] == [{"id": 0, "name": "cat"}, {"id": 1, "name": "dog"}]

        # Check written JSON file
        with open(output_path) as f:
            loaded = json.load(f)
        assert loaded["images"] == result["images"]
        assert loaded["annotations"] == result["annotations"]

    def test_bbox_values_reasonable(self, tmp_path):
        ann_dir, img_dir = _setup_yolo_images_and_annotations(tmp_path)
        output_path = str(tmp_path / "coco_output.json")

        result = yolo_to_coco(ann_dir, img_dir, output_path, categories=["cat"])
        ann = result["annotations"][0]
        bbox = ann["bbox"]
        # All bbox values should be positive and reasonable for a 200x150 image
        assert bbox[0] > 0  # x
        assert bbox[1] > 0  # y
        assert bbox[2] > 0  # width
        assert bbox[3] > 0  # height
        assert ann["area"] > 0

    def test_multiple_images(self, tmp_path):
        anns = {
            "img1": ["0 0.5 0.5 0.2 0.2"],
            "img2": ["1 0.3 0.3 0.1 0.1", "0 0.7 0.7 0.15 0.15"],
        }
        ann_dir, img_dir = _setup_yolo_images_and_annotations(tmp_path, annotations=anns)
        output_path = str(tmp_path / "coco_output.json")

        result = yolo_to_coco(ann_dir, img_dir, output_path, categories=["cat", "dog"])
        assert len(result["images"]) == 2
        assert len(result["annotations"]) == 3

    def test_dict_categories(self, tmp_path):
        """Test passing categories as list of dicts with id and name."""
        ann_dir, img_dir = _setup_yolo_images_and_annotations(tmp_path)
        output_path = str(tmp_path / "coco_output.json")
        cats = [{"id": 5, "name": "person"}, {"id": 10, "name": "car"}]

        result = yolo_to_coco(ann_dir, img_dir, output_path, categories=cats)
        assert result["categories"] == cats

    def test_empty_annotation_dir(self, tmp_path):
        ann_dir = tmp_path / "annotations"
        img_dir = tmp_path / "images"
        ann_dir.mkdir()
        img_dir.mkdir()
        output_path = str(tmp_path / "coco_output.json")

        result = yolo_to_coco(str(ann_dir), str(img_dir), output_path, categories=["cat"])
        assert result["images"] == []
        assert result["annotations"] == []


# ---------- coco_to_yolo ----------

class TestCocoToYolo:
    def test_basic_conversion(self, tmp_path):
        output_dir = str(tmp_path / "yolo_output")
        coco_path = str(tmp_path / "coco.json")

        coco_data = {
            "images": [{"id": 1, "file_name": "img1.jpg", "width": 200, "height": 150}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 0, "bbox": [80, 45, 40, 45]},
            ],
            "categories": [{"id": 0, "name": "cat"}],
        }
        with open(coco_path, "w") as f:
            json.dump(coco_data, f)

        coco_to_yolo(coco_path, output_dir)

        # Should create img1.txt
        out_txt = os.path.join(output_dir, "img1.txt")
        assert os.path.exists(out_txt)

        with open(out_txt) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 1
        parts = lines[0].split()
        assert len(parts) == 5
        assert parts[0] == "0"  # category_id
        # Verify normalized values are in [0, 1]
        for val in parts[1:]:
            assert 0 <= float(val) <= 1

    def test_round_trip_yolo_coco_yolo(self, tmp_path):
        """YOLO -> COCO -> YOLO round trip: verify class IDs survive."""
        ann_dir, img_dir = _setup_yolo_images_and_annotations(
            tmp_path, annotations={"img1": ["2 0.5 0.5 0.2 0.3"]}
        )
        coco_path = str(tmp_path / "coco_rt.json")
        yolo_out = str(tmp_path / "yolo_rt")

        cats = ["a", "b", "c", "d"]
        yolo_to_coco(ann_dir, img_dir, coco_path, categories=cats)
        coco_to_yolo(coco_path, yolo_out)

        out_txt = os.path.join(yolo_out, "img1.txt")
        assert os.path.exists(out_txt)
        with open(out_txt) as f:
            parts = f.readline().strip().split()
        assert parts[0] == "2"  # class ID preserved
        # Normalized coordinates should be close to original
        assert abs(float(parts[1]) - 0.5) < 0.05
        assert abs(float(parts[2]) - 0.5) < 0.05


# ---------- voc_to_yolo ----------

class TestVocToYolo:
    def test_basic_conversion(self, tmp_path):
        voc_dir = tmp_path / "voc"
        voc_dir.mkdir()
        output_dir = str(tmp_path / "yolo_output")

        _write_voc_xml(
            str(voc_dir / "img1.xml"),
            filename="img1.jpg",
            w=200, h=150,
            objects=[("cat", 50, 30, 90, 75)],
        )

        voc_to_yolo(str(voc_dir), output_dir, class_names=["cat", "dog"])

        out_txt = os.path.join(output_dir, "img1.txt")
        assert os.path.exists(out_txt)
        with open(out_txt) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 1
        parts = lines[0].split()
        assert parts[0] == "0"  # cat is index 0

        # Verify normalized values
        xc = float(parts[1])
        yc = float(parts[2])
        bw = float(parts[3])
        bh = float(parts[4])
        # Box (50,30)-(90,75) in 200x150: xc=(50+90)/2/200=0.35, yc=(30+75)/2/150=0.35
        assert abs(xc - 0.35) < 0.01
        assert abs(yc - 0.35) < 0.01
        assert abs(bw - 0.2) < 0.01
        assert abs(bh - 0.3) < 0.01

    def test_unknown_class_skipped(self, tmp_path):
        voc_dir = tmp_path / "voc"
        voc_dir.mkdir()
        output_dir = str(tmp_path / "yolo_output")

        _write_voc_xml(
            str(voc_dir / "img1.xml"),
            filename="img1.jpg",
            w=200, h=150,
            objects=[("bird", 10, 10, 50, 50)],  # "bird" not in class_names
        )

        voc_to_yolo(str(voc_dir), output_dir, class_names=["cat", "dog"])

        out_txt = os.path.join(output_dir, "img1.txt")
        if os.path.exists(out_txt):
            with open(out_txt) as f:
                lines = [l.strip() for l in f if l.strip()]
            assert len(lines) == 0

    def test_multiple_objects(self, tmp_path):
        voc_dir = tmp_path / "voc"
        voc_dir.mkdir()
        output_dir = str(tmp_path / "yolo_output")

        _write_voc_xml(
            str(voc_dir / "img1.xml"),
            filename="img1.jpg",
            w=200, h=150,
            objects=[
                ("cat", 10, 10, 50, 50),
                ("dog", 100, 80, 180, 140),
            ],
        )

        voc_to_yolo(str(voc_dir), output_dir, class_names=["cat", "dog"])
        out_txt = os.path.join(output_dir, "img1.txt")
        with open(out_txt) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 2


# ---------- yolo_to_voc ----------

class TestYoloToVoc:
    def test_basic_conversion(self, tmp_path):
        ann_dir, img_dir = _setup_yolo_images_and_annotations(
            tmp_path, annotations={"img1": ["0 0.5 0.5 0.2 0.3"]}
        )
        output_dir = str(tmp_path / "voc_output")

        yolo_to_voc(ann_dir, img_dir, output_dir, class_names=["cat", "dog"])

        out_xml = os.path.join(output_dir, "img1.xml")
        assert os.path.exists(out_xml)

        # Parse with lxml to verify valid XML structure
        tree = etree.parse(out_xml)
        root = tree.getroot()
        assert root.tag == "annotation"
        assert root.find("filename").text == "img1.jpg"
        size = root.find("size")
        assert size.find("width").text == "200"
        assert size.find("height").text == "150"

        objects = root.findall("object")
        assert len(objects) == 1
        assert objects[0].find("name").text == "cat"
        bndbox = objects[0].find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        # Box: xc=0.5, yc=0.5, bw=0.2, bh=0.3 in 200x150
        # xmin=(0.5-0.1)*200=80, xmax=(0.5+0.1)*200=120
        # ymin=(0.5-0.15)*150=52, ymax=(0.5+0.15)*150=97
        assert 75 <= xmin <= 85
        assert 115 <= xmax <= 125
        assert 47 <= ymin <= 57
        assert 92 <= ymax <= 102

    def test_xml_is_valid_lxml_output(self, tmp_path):
        """Verify the output is proper XML (lxml-generated), not string-concatenated."""
        ann_dir, img_dir = _setup_yolo_images_and_annotations(tmp_path)
        output_dir = str(tmp_path / "voc_output")

        yolo_to_voc(ann_dir, img_dir, output_dir, class_names=["cat"])

        out_xml = os.path.join(output_dir, "img1.xml")
        # Should parse without error
        tree = etree.parse(out_xml)
        # Should have XML declaration
        with open(out_xml, "rb") as f:
            header = f.read(50)
        assert b"<?xml" in header

    def test_no_matching_image_skips(self, tmp_path):
        """If annotation has no matching image, it should be skipped."""
        ann_dir = tmp_path / "annotations"
        img_dir = tmp_path / "images"
        output_dir = str(tmp_path / "voc_output")
        ann_dir.mkdir()
        img_dir.mkdir()

        # Only annotation, no image
        _write_yolo_txt(str(ann_dir / "orphan.txt"), ["0 0.5 0.5 0.2 0.3"])

        yolo_to_voc(str(ann_dir), str(img_dir), output_dir, class_names=["cat"])
        assert not os.path.exists(os.path.join(output_dir, "orphan.xml"))


# ---------- create_classification_dataset ----------

class TestCreateClassificationDataset:
    def test_folder_per_class(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        # Create folder-per-class structure
        for cls in ["cat", "dog"]:
            cls_dir = input_dir / cls
            cls_dir.mkdir(parents=True)
            img = _make_test_img()
            write_image(str(cls_dir / "photo1.jpg"), img)
            write_image(str(cls_dir / "photo2.png"), img)

        create_classification_dataset(str(input_dir), str(output_dir))

        assert (output_dir / "cat").is_dir()
        assert (output_dir / "dog").is_dir()
        assert len(list((output_dir / "cat").glob("*"))) == 2
        assert len(list((output_dir / "dog").glob("*"))) == 2

    def test_empty_input(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        create_classification_dataset(str(input_dir), str(output_dir))
        # Output dir may not exist yet if no classes were found, which is fine
        assert not output_dir.exists() or len(list(output_dir.iterdir())) == 0

    def test_ignores_non_directory_files(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # A loose file in input_dir (not a directory, so should be skipped)
        (input_dir / "readme.txt").write_text("not a class")

        # Also create a valid class folder
        cls_dir = input_dir / "cat"
        cls_dir.mkdir()
        write_image(str(cls_dir / "photo.jpg"), _make_test_img())

        create_classification_dataset(str(input_dir), str(output_dir))
        assert (output_dir / "cat").is_dir()
        assert len(list((output_dir / "cat").glob("*"))) == 1


# ---------- xanylabeling_to_yolo ----------

class TestCollectXanylabelingLabels:
    def test_auto_collect(self, tmp_path):
        src_dir = _make_xanylabeling_dataset(tmp_path)
        labels = collect_xanylabeling_labels(str(src_dir))
        assert labels == ["cat", "dog"]

    def test_empty_dir(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        labels = collect_xanylabeling_labels(str(empty_dir))
        assert labels == []

    def test_ignores_non_json(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "readme.txt").write_text("not json")
        labels = collect_xanylabeling_labels(str(src_dir))
        assert labels == []


class TestXanylabelingToYolo:
    def test_basic_conversion(self, tmp_path):
        src_dir = _make_xanylabeling_dataset(tmp_path)
        output_dir = str(tmp_path / "output")

        result = xanylabeling_to_yolo(str(src_dir), output_dir, val_ratio=0.5, seed=0)

        # Check output structure exists
        assert os.path.isdir(os.path.join(output_dir, "images", "train"))
        assert os.path.isdir(os.path.join(output_dir, "images", "val"))
        assert os.path.isdir(os.path.join(output_dir, "labels", "train"))
        assert os.path.isdir(os.path.join(output_dir, "labels", "val"))
        assert os.path.exists(os.path.join(output_dir, "data.yaml"))

        # Check specific images are in train or val
        all_imgs = set()
        for split in ("train", "val"):
            img_dir = os.path.join(output_dir, "images", split)
            for f in os.listdir(img_dir):
                if f in ("img1.jpg", "img2.jpg", "img3.png"):
                    all_imgs.add(f)
        assert all_imgs == {"img1.jpg", "img2.jpg", "img3.png"}

        # Check corresponding label files exist
        all_lbls = set()
        for split in ("train", "val"):
            lbl_dir = os.path.join(output_dir, "labels", split)
            for f in os.listdir(lbl_dir):
                if f in ("img1.txt", "img2.txt", "img3.txt"):
                    all_lbls.add(f)
        assert all_lbls == {"img1.txt", "img2.txt", "img3.txt"}

        assert "3 张图片" in result
        assert "2 个类别" in result

    def test_yolo_format_correctness(self, tmp_path):
        """Verify YOLO bbox values match expected conversions."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        # Single image with a known rectangle: (20,20)-(80,80) in 200x150
        img = _make_test_img(200, 150)
        write_image(str(src_dir / "img.jpg"), img)
        _write_xanylabeling_json(
            str(src_dir / "img.json"),
            [{"label": "obj", "points": [[20, 20], [80, 80]], "shape_type": "rectangle"}],
            img_w=200, img_h=150,
        )
        output_dir = str(tmp_path / "output")
        xanylabeling_to_yolo(str(src_dir), output_dir, val_ratio=1.0, seed=0)

        # With val_ratio=1.0 all go to val
        lbl_dir = os.path.join(output_dir, "labels", "val")
        lbl_file = os.path.join(lbl_dir, "img.txt")
        assert os.path.exists(lbl_file)
        with open(lbl_file) as f:
            parts = f.readline().strip().split()
        assert len(parts) == 5
        cls_id, xc, yc, bw, bh = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        assert cls_id == "0"
        # Expected: xc=(20+80)/2/200=0.25, yc=(20+80)/2/150=0.3333, w=60/200=0.3, h=60/150=0.4
        assert abs(xc - 0.25) < 0.01
        assert abs(yc - 0.3333) < 0.01
        assert abs(bw - 0.3) < 0.01
        assert abs(bh - 0.4) < 0.01

    def test_polygon_conversion(self, tmp_path):
        """Polygon points should be converted to bounding box."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        img = _make_test_img(100, 100)
        write_image(str(src_dir / "img.png"), img)
        _write_xanylabeling_json(
            str(src_dir / "img.json"),
            [{"label": "obj", "points": [[10, 10], [50, 10], [50, 50], [10, 50]], "shape_type": "polygon"}],
            img_w=100, img_h=100,
        )
        output_dir = str(tmp_path / "output")
        xanylabeling_to_yolo(str(src_dir), output_dir, val_ratio=1.0, seed=0)

        with open(os.path.join(output_dir, "labels", "val", "img.txt")) as f:
            parts = f.readline().strip().split()
        xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        # Bounding box of polygon: x=[10,50], y=[10,50] -> xc=30/100=0.3, yc=30/100=0.3, w=40/100=0.4, h=40/100=0.4
        assert abs(xc - 0.3) < 0.01
        assert abs(yc - 0.3) < 0.01
        assert abs(bw - 0.4) < 0.01
        assert abs(bh - 0.4) < 0.01

    def test_custom_categories(self, tmp_path):
        src_dir = _make_xanylabeling_dataset(tmp_path)
        output_dir = str(tmp_path / "output")

        # Provide explicit category list with different order
        xanylabeling_to_yolo(str(src_dir), output_dir, val_ratio=1.0, seed=0,
                             categories=["dog", "cat"])

        # Check that category mapping is dog=0, cat=1
        all_labels = []
        for split in ("train", "val"):
            lbl_dir = os.path.join(output_dir, "labels", split)
            if os.path.isdir(lbl_dir):
                for fname in os.listdir(lbl_dir):
                    with open(os.path.join(lbl_dir, fname)) as f:
                        for line in f:
                            all_labels.append(int(line.split()[0]))
        # dog=0 should exist (img2 has dog)
        assert 0 in all_labels
        # cat=1 should exist
        assert 1 in all_labels

    def test_missing_json_creates_empty_label(self, tmp_path):
        """Image without JSON should get an empty label file."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        img = _make_test_img(100, 100)
        write_image(str(src_dir / "no_label.jpg"), img)
        output_dir = str(tmp_path / "output")

        result = xanylabeling_to_yolo(str(src_dir), output_dir, val_ratio=1.0, seed=0,
                                      categories=["cat"])

        # Label file should exist but be empty
        lbl_file = os.path.join(output_dir, "labels", "val", "no_label.txt")
        assert os.path.exists(lbl_file)
        with open(lbl_file) as f:
            content = f.read()
        assert content == ""
        assert "无标注JSON" in result

    def test_data_yaml_content(self, tmp_path):
        src_dir = _make_xanylabeling_dataset(tmp_path)
        output_dir = str(tmp_path / "output")
        xanylabeling_to_yolo(str(src_dir), output_dir, val_ratio=0.2, seed=42)

        yaml_path = os.path.join(output_dir, "data.yaml")
        assert os.path.exists(yaml_path)
        with open(yaml_path) as f:
            content = f.read()
        assert "train: images/train" in content
        assert "val: images/val" in content
        assert "0: cat" in content
        assert "1: dog" in content

    def test_no_images_raises(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="未找到图片"):
            xanylabeling_to_yolo(str(empty_dir), str(tmp_path / "out"))

    def test_no_labels_raises(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        img = _make_test_img(100, 100)
        write_image(str(src_dir / "img.jpg"), img)
        # No JSON file
        with pytest.raises(ValueError, match="未找到标注类别"):
            xanylabeling_to_yolo(str(src_dir), str(tmp_path / "out"))

    def test_train_val_split_ratio(self, tmp_path):
        """Verify split respects val_ratio approximately."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        # Create 10 images
        expected_imgs = set()
        for i in range(10):
            fname = f"img{i:02d}.jpg"
            expected_imgs.add(fname)
            img = _make_test_img(100, 100)
            write_image(str(src_dir / fname), img)
            _write_xanylabeling_json(
                str(src_dir / f"img{i:02d}.json"),
                [{"label": "obj", "points": [[10, 10], [90, 90]], "shape_type": "rectangle"}],
                img_w=100, img_h=100,
            )
        output_dir = str(tmp_path / "output")
        xanylabeling_to_yolo(str(src_dir), output_dir, val_ratio=0.3, seed=42)

        # Count only our expected images
        val_count = 0
        train_count = 0
        for split, counter in (("val", "val"), ("train", "train")):
            img_dir = os.path.join(output_dir, "images", split)
            for f in os.listdir(img_dir):
                if f in expected_imgs:
                    if split == "val":
                        val_count += 1
                    else:
                        train_count += 1
        assert val_count + train_count == 10
        # val_ratio=0.3, so ~3 images in val
        assert 2 <= val_count <= 4
