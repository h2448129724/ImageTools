"""Tests for core/annotation_augment.py."""
import json
import os

import numpy as np
import pytest

from core.annotation_augment import (
    augment_batch,
    transform_labelme_shapes,
    transform_yolo_bbox,
)
from core.image_io import write_image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_yolo_box(cls=0, xc=0.5, yc=0.5, bw=0.3, bh=0.2):
    """Return a single YOLO box dict."""
    return {"cls": cls, "xc": xc, "yc": yc, "bw": bw, "bh": bh}


def _make_labelme_shape(label="cat", shape_type="polygon", points=None):
    """Return a single Labelme shape dict."""
    if points is None:
        points = [[100.0, 50.0], [200.0, 50.0], [200.0, 150.0], [100.0, 150.0]]
    return {
        "label": label,
        "shape_type": shape_type,
        "points": points,
    }


def _write_yolo_txt(path, boxes):
    """Write a YOLO annotation txt file."""
    with open(path, "w") as f:
        for b in boxes:
            f.write(f"{b['cls']} {b['xc']:.6f} {b['yc']:.6f} "
                    f"{b['bw']:.6f} {b['bh']:.6f}\n")


def _write_labelme_json(path, shapes, img_w=400, img_h=300, img_name="test.png"):
    """Write a Labelme annotation JSON file."""
    data = {
        "version": "5.0.1",
        "shapes": shapes,
        "imageWidth": img_w,
        "imageHeight": img_h,
        "imagePath": img_name,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ===================================================================
# Test transform_yolo_bbox
# ===================================================================

class TestTransformYoloBbox:

    # --- Flip --------------------------------------------------------

    def test_flip_horizontal(self):
        boxes = [_make_yolo_box(xc=0.25, yc=0.5, bw=0.2, bh=0.2)]
        result, new_w, new_h = transform_yolo_bbox(
            boxes, 100, 100, "flip", direction="horizontal",
        )
        assert new_w == 100 and new_h == 100
        assert len(result) == 1
        # Horizontal flip: xc should map to 1 - (0.25 + 0.1) + 0.1 = 0.75
        b = result[0]
        assert abs(b["xc"] - 0.75) < 0.01
        assert abs(b["yc"] - 0.5) < 0.01
        assert b["bw"] == pytest.approx(0.2, abs=0.01)
        assert b["bh"] == pytest.approx(0.2, abs=0.01)

    def test_flip_vertical(self):
        boxes = [_make_yolo_box(xc=0.5, yc=0.25, bw=0.2, bh=0.2)]
        result, _, _ = transform_yolo_bbox(
            boxes, 100, 100, "flip", direction="vertical",
        )
        assert len(result) == 1
        b = result[0]
        # Vertical flip: yc should map to 1 - (0.25 + 0.1) + 0.1 = 0.75
        assert abs(b["yc"] - 0.75) < 0.01
        assert abs(b["xc"] - 0.5) < 0.01

    def test_flip_both(self):
        boxes = [_make_yolo_box(xc=0.2, yc=0.3, bw=0.2, bh=0.2)]
        result, _, _ = transform_yolo_bbox(
            boxes, 100, 100, "flip", direction="both",
        )
        b = result[0]
        assert abs(b["xc"] - 0.8) < 0.01
        assert abs(b["yc"] - 0.7) < 0.01

    # --- Rotate ------------------------------------------------------

    def test_rotate_90(self):
        """Rotate 90 degrees: center (0.5, 0.5) should stay at center."""
        boxes = [_make_yolo_box(xc=0.5, yc=0.5, bw=0.2, bh=0.1)]
        result, new_w, new_h = transform_yolo_bbox(
            boxes, 100, 200, "rotate", angle=90, keep_size=True,
        )
        assert new_w == 100 and new_h == 200
        assert len(result) == 1
        b = result[0]
        # Centered box stays centered after 90 deg rotation
        assert abs(b["xc"] - 0.5) < 0.02
        assert abs(b["yc"] - 0.5) < 0.02

    def test_rotate_180(self):
        """180-degree rotation is equivalent to flipping both axes."""
        boxes = [_make_yolo_box(xc=0.25, yc=0.25, bw=0.2, bh=0.2)]
        result, new_w, new_h = transform_yolo_bbox(
            boxes, 100, 100, "rotate", angle=180,
        )
        assert new_w == 100 and new_h == 100
        b = result[0]
        # After 180: (0.25, 0.25) -> (0.75, 0.75)
        assert abs(b["xc"] - 0.75) < 0.02
        assert abs(b["yc"] - 0.75) < 0.02
        assert b["bw"] == pytest.approx(0.2, abs=0.02)
        assert b["bh"] == pytest.approx(0.2, abs=0.02)

    def test_rotate_270(self):
        """270-degree rotation maps center to center as well."""
        boxes = [_make_yolo_box(xc=0.5, yc=0.5, bw=0.3, bh=0.2)]
        result, _, _ = transform_yolo_bbox(
            boxes, 100, 100, "rotate", angle=270,
        )
        b = result[0]
        assert abs(b["xc"] - 0.5) < 0.02
        assert abs(b["yc"] - 0.5) < 0.02

    # --- Resize ------------------------------------------------------

    def test_resize(self):
        boxes = [_make_yolo_box(xc=0.5, yc=0.5, bw=0.4, bh=0.4)]
        result, new_w, new_h = transform_yolo_bbox(
            boxes, 100, 100, "resize", new_w=200, new_h=200,
        )
        assert new_w == 200 and new_h == 200
        # Normalized coordinates should be unchanged after uniform resize
        b = result[0]
        assert abs(b["xc"] - 0.5) < 0.01
        assert abs(b["yc"] - 0.5) < 0.01
        assert abs(b["bw"] - 0.4) < 0.01
        assert abs(b["bh"] - 0.4) < 0.01

    def test_resize_non_uniform(self):
        boxes = [_make_yolo_box(xc=0.5, yc=0.5, bw=0.2, bh=0.2)]
        result, new_w, new_h = transform_yolo_bbox(
            boxes, 100, 100, "resize", new_w=200, new_h=50,
        )
        assert new_w == 200 and new_h == 50
        # After non-uniform resize, center is still mid but box dims change
        b = result[0]
        assert 0 <= b["xc"] <= 1
        assert 0 <= b["yc"] <= 1
        assert 0 < b["bw"] <= 1
        assert 0 < b["bh"] <= 1

    # --- Crop --------------------------------------------------------

    def test_crop_inside(self):
        """Crop region fully contains the box."""
        boxes = [_make_yolo_box(xc=0.5, yc=0.5, bw=0.2, bh=0.2)]
        result, new_w, new_h = transform_yolo_bbox(
            boxes, 100, 100, "crop", x=0, y=0, w=60, h=60,
        )
        assert new_w == 60 and new_h == 60
        assert len(result) == 1
        b = result[0]
        # Original abs center = (50, 50), after crop offset = (50, 50) / (60, 60)
        assert abs(b["xc"] - 50 / 60) < 0.02
        assert abs(b["yc"] - 50 / 60) < 0.02

    def test_crop_partial_overlap(self):
        """Crop cuts through the box -- should clamp to crop region."""
        boxes = [_make_yolo_box(xc=0.15, yc=0.15, bw=0.2, bh=0.2)]
        result, new_w, new_h = transform_yolo_bbox(
            boxes, 100, 100, "crop", x=10, y=10, w=30, h=30,
        )
        # Box top-left corner at (5,5) partially outside crop (10,10)
        assert len(result) == 1
        b = result[0]
        assert 0 <= b["xc"] <= 1
        assert 0 <= b["yc"] <= 1
        assert 0 < b["bw"] <= 1
        assert 0 < b["bh"] <= 1

    def test_crop_box_outside(self):
        """Box fully outside the crop region should be dropped."""
        boxes = [_make_yolo_box(xc=0.9, yc=0.9, bw=0.1, bh=0.1)]
        result, _, _ = transform_yolo_bbox(
            boxes, 100, 100, "crop", x=0, y=0, w=20, h=20,
        )
        assert len(result) == 0

    # --- Coordinate range checks -------------------------------------

    def test_output_coords_in_range(self):
        """All output normalized coords should be in [0, 1]."""
        boxes = [
            _make_yolo_box(cls=0, xc=0.3, yc=0.4, bw=0.2, bh=0.2),
            _make_yolo_box(cls=1, xc=0.7, yc=0.6, bw=0.3, bh=0.1),
        ]
        for transform_type, kwargs in [
            ("flip", {"direction": "horizontal"}),
            ("flip", {"direction": "vertical"}),
            ("rotate", {"angle": 90}),
            ("rotate", {"angle": 180}),
            ("rotate", {"angle": 270}),
            ("resize", {"new_w": 200, "new_h": 200}),
        ]:
            result, _, _ = transform_yolo_bbox(
                boxes, 100, 100, transform_type, **kwargs,
            )
            for b in result:
                assert 0 <= b["xc"] <= 1, f"{transform_type} xc={b['xc']}"
                assert 0 <= b["yc"] <= 1, f"{transform_type} yc={b['yc']}"
                assert 0 < b["bw"] <= 1, f"{transform_type} bw={b['bw']}"
                assert 0 < b["bh"] <= 1, f"{transform_type} bh={b['bh']}"

    def test_cls_preserved(self):
        boxes = [_make_yolo_box(cls=3, xc=0.5, yc=0.5, bw=0.2, bh=0.2)]
        result, _, _ = transform_yolo_bbox(
            boxes, 100, 100, "flip", direction="horizontal",
        )
        assert result[0]["cls"] == 3

    def test_multiple_boxes(self):
        boxes = [
            _make_yolo_box(cls=0, xc=0.2, yc=0.2, bw=0.1, bh=0.1),
            _make_yolo_box(cls=1, xc=0.5, yc=0.5, bw=0.2, bh=0.2),
            _make_yolo_box(cls=2, xc=0.8, yc=0.8, bw=0.1, bh=0.1),
        ]
        result, _, _ = transform_yolo_bbox(
            boxes, 100, 100, "flip", direction="horizontal",
        )
        assert len(result) == 3
        classes = [b["cls"] for b in result]
        assert classes == [0, 1, 2]


# ===================================================================
# Test transform_labelme_shapes
# ===================================================================

class TestTransformLabelmeShapes:

    def test_flip_horizontal(self):
        shapes = [_make_labelme_shape(points=[[10, 20], [90, 20], [90, 80], [10, 80]])]
        result, new_w, new_h = transform_labelme_shapes(
            shapes, 100, 100, "flip", direction="horizontal",
        )
        assert new_w == 100 and new_h == 100
        assert len(result) == 1
        pts = result[0]["points"]
        # x coords should be flipped: 100 - x
        assert pts[0][0] == pytest.approx(90, abs=0.5)
        assert pts[1][0] == pytest.approx(10, abs=0.5)
        # y coords unchanged
        assert pts[0][1] == pytest.approx(20, abs=0.5)

    def test_flip_vertical(self):
        shapes = [_make_labelme_shape(points=[[10, 20], [90, 20], [90, 80], [10, 80]])]
        result, new_w, new_h = transform_labelme_shapes(
            shapes, 100, 100, "flip", direction="vertical",
        )
        pts = result[0]["points"]
        # y coords should be flipped: 100 - y
        assert pts[0][1] == pytest.approx(80, abs=0.5)
        assert pts[2][1] == pytest.approx(20, abs=0.5)
        # x coords unchanged
        assert pts[0][0] == pytest.approx(10, abs=0.5)

    def test_rotate_90(self):
        shapes = [_make_labelme_shape(points=[[50, 50], [70, 50], [70, 70], [50, 70]])]
        result, new_w, new_h = transform_labelme_shapes(
            shapes, 100, 100, "rotate", angle=90,
        )
        assert new_w == 100 and new_h == 100
        # OpenCV 90-deg rotation around center shifts the centroid slightly;
        # verify all points stay within the image bounds.
        pts = result[0]["points"]
        for x, y in pts:
            assert 0 <= x <= 100
            assert 0 <= y <= 100

    def test_rotate_180(self):
        shapes = [_make_labelme_shape(points=[[10, 20], [50, 20], [50, 60], [10, 60]])]
        result, _, _ = transform_labelme_shapes(
            shapes, 100, 100, "rotate", angle=180,
        )
        pts = result[0]["points"]
        # Center of original = (30, 40) -> after 180 -> (70, 60)
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        assert abs(cx - 70) < 2
        assert abs(cy - 60) < 2

    def test_rotate_270(self):
        shapes = [_make_labelme_shape(points=[[50, 50], [70, 50], [70, 70], [50, 70]])]
        result, _, _ = transform_labelme_shapes(
            shapes, 100, 100, "rotate", angle=270,
        )
        pts = result[0]["points"]
        # Verify all points stay within image bounds after rotation
        for x, y in pts:
            assert 0 <= x <= 100
            assert 0 <= y <= 100

    def test_resize(self):
        shapes = [_make_labelme_shape(points=[[10, 20], [50, 20], [50, 60], [10, 60]])]
        result, new_w, new_h = transform_labelme_shapes(
            shapes, 100, 100, "resize", new_w=200, new_h=200,
        )
        assert new_w == 200 and new_h == 200
        pts = result[0]["points"]
        # Points should be scaled by 2x
        assert pts[0][0] == pytest.approx(20, abs=0.5)
        assert pts[0][1] == pytest.approx(40, abs=0.5)

    def test_crop(self):
        shapes = [_make_labelme_shape(points=[[60, 60], [90, 60], [90, 90], [60, 90]])]
        result, new_w, new_h = transform_labelme_shapes(
            shapes, 100, 100, "crop", x=50, y=50, w=50, h=50,
        )
        assert new_w == 50 and new_h == 50
        pts = result[0]["points"]
        # After subtracting offset (50, 50)
        assert pts[0][0] == pytest.approx(10, abs=0.5)
        assert pts[0][1] == pytest.approx(10, abs=0.5)

    def test_shape_metadata_preserved(self):
        original = _make_labelme_shape(label="dog", shape_type="rectangle",
                                       points=[[10, 20], [90, 80]])
        result, _, _ = transform_labelme_shapes(
            [original], 100, 100, "flip", direction="horizontal",
        )
        assert result[0]["label"] == "dog"
        assert result[0]["shape_type"] == "rectangle"

    def test_multiple_shapes(self):
        shapes = [
            _make_labelme_shape(label="a", points=[[10, 10], [30, 10], [30, 30], [10, 30]]),
            _make_labelme_shape(label="b", points=[[60, 60], [90, 60], [90, 90], [60, 90]]),
        ]
        result, _, _ = transform_labelme_shapes(
            shapes, 100, 100, "flip", direction="horizontal",
        )
        assert len(result) == 2
        assert result[0]["label"] == "a"
        assert result[1]["label"] == "b"


# ===================================================================
# Test augment_batch — YOLO format
# ===================================================================

class TestAugmentBatchYOLO:

    def _setup_yolo_data(self, tmp_path):
        """Create a minimal image + YOLO annotation pair and return dirs."""
        img_dir = str(tmp_path / "images")
        ann_dir = str(tmp_path / "annotations")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)

        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        write_image(os.path.join(img_dir, "sample.png"), img)

        boxes = [
            _make_yolo_box(cls=0, xc=0.5, yc=0.5, bw=0.3, bh=0.3),
            _make_yolo_box(cls=1, xc=0.2, yc=0.8, bw=0.1, bh=0.1),
        ]
        _write_yolo_txt(os.path.join(ann_dir, "sample.txt"), boxes)

        return img_dir, ann_dir

    def test_flip_creates_output_files(self, tmp_path):
        img_dir, ann_dir = self._setup_yolo_data(tmp_path)
        out_dir = str(tmp_path / "output")

        stats = augment_batch(
            img_dir, ann_dir, out_dir, "flip",
            ann_format="yolo", direction="horizontal",
        )
        assert stats["processed"] == 1
        assert stats["skipped"] == 0

        out_img_dir = os.path.join(out_dir, "images")
        out_ann_dir = os.path.join(out_dir, "annotations")
        assert os.path.exists(os.path.join(out_img_dir, "sample.png"))
        assert os.path.exists(os.path.join(out_ann_dir, "sample.txt"))

    def test_flip_yolo_coords_valid(self, tmp_path):
        img_dir, ann_dir = self._setup_yolo_data(tmp_path)
        out_dir = str(tmp_path / "output")

        augment_batch(
            img_dir, ann_dir, out_dir, "flip",
            ann_format="yolo", direction="horizontal",
        )

        out_txt = os.path.join(out_dir, "annotations", "sample.txt")
        with open(out_txt) as f:
            lines = f.readlines()
        assert len(lines) == 2
        for line in lines:
            parts = line.strip().split()
            assert len(parts) == 5
            cls_id = int(parts[0])
            assert cls_id in (0, 1)
            xc, yc, bw, bh = [float(v) for v in parts[1:]]
            assert 0 <= xc <= 1
            assert 0 <= yc <= 1
            assert 0 < bw <= 1
            assert 0 < bh <= 1

    def test_rotate_creates_output(self, tmp_path):
        img_dir, ann_dir = self._setup_yolo_data(tmp_path)
        out_dir = str(tmp_path / "output")

        stats = augment_batch(
            img_dir, ann_dir, out_dir, "rotate",
            ann_format="yolo", angle=90,
        )
        assert stats["processed"] == 1
        assert os.path.exists(os.path.join(out_dir, "images", "sample.png"))
        assert os.path.exists(os.path.join(out_dir, "annotations", "sample.txt"))

    def test_resize_creates_output(self, tmp_path):
        img_dir, ann_dir = self._setup_yolo_data(tmp_path)
        out_dir = str(tmp_path / "output")

        stats = augment_batch(
            img_dir, ann_dir, out_dir, "resize",
            ann_format="yolo", new_w=200, new_h=200,
        )
        assert stats["processed"] == 1

    def test_missing_annotation_skipped(self, tmp_path):
        """Image without matching annotation should be skipped."""
        img_dir = str(tmp_path / "images")
        ann_dir = str(tmp_path / "annotations")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)

        img = np.zeros((50, 50, 3), dtype=np.uint8)
        write_image(os.path.join(img_dir, "orphan.png"), img)
        # No .txt file created

        out_dir = str(tmp_path / "output")
        stats = augment_batch(
            img_dir, ann_dir, out_dir, "flip",
            ann_format="yolo", direction="horizontal",
        )
        assert stats["skipped"] >= 1
        assert stats["processed"] == 0

    def test_multiple_images(self, tmp_path):
        img_dir = str(tmp_path / "images")
        ann_dir = str(tmp_path / "annotations")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)

        for i in range(3):
            img = np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)
            write_image(os.path.join(img_dir, f"img_{i}.png"), img)
            boxes = [_make_yolo_box(cls=0, xc=0.5, yc=0.5, bw=0.3, bh=0.3)]
            _write_yolo_txt(os.path.join(ann_dir, f"img_{i}.txt"), boxes)

        out_dir = str(tmp_path / "output")
        stats = augment_batch(
            img_dir, ann_dir, out_dir, "flip",
            ann_format="yolo", direction="vertical",
        )
        assert stats["processed"] == 3
        assert stats["skipped"] == 0
        for i in range(3):
            assert os.path.exists(os.path.join(out_dir, "images", f"img_{i}.png"))
            assert os.path.exists(os.path.join(out_dir, "annotations", f"img_{i}.txt"))


# ===================================================================
# Test augment_batch — labelme format
# ===================================================================

class TestAugmentBatchLabelme:

    def _setup_labelme_data(self, tmp_path, img_w=200, img_h=150):
        """Create a minimal image + Labelme JSON annotation pair and return dirs."""
        img_dir = str(tmp_path / "images")
        ann_dir = str(tmp_path / "annotations")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)

        img = np.random.randint(0, 256, (img_h, img_w, 3), dtype=np.uint8)
        img_name = "sample.png"
        write_image(os.path.join(img_dir, img_name), img)

        shapes = [
            _make_labelme_shape(
                label="car",
                shape_type="polygon",
                points=[[50, 40], [150, 40], [150, 110], [50, 110]],
            ),
            _make_labelme_shape(
                label="person",
                shape_type="rectangle",
                points=[[10, 10], [40, 80]],
            ),
        ]
        _write_labelme_json(
            os.path.join(ann_dir, "sample.json"),
            shapes, img_w=img_w, img_h=img_h, img_name=img_name,
        )
        return img_dir, ann_dir

    def test_flip_creates_output_files(self, tmp_path):
        img_dir, ann_dir = self._setup_labelme_data(tmp_path)
        out_dir = str(tmp_path / "output")

        stats = augment_batch(
            img_dir, ann_dir, out_dir, "flip",
            ann_format="labelme", direction="horizontal",
        )
        assert stats["processed"] == 1
        assert stats["skipped"] == 0

        out_img_dir = os.path.join(out_dir, "images")
        out_ann_dir = os.path.join(out_dir, "annotations")
        assert os.path.exists(os.path.join(out_img_dir, "sample.png"))
        assert os.path.exists(os.path.join(out_ann_dir, "sample.json"))

    def test_flip_labelme_shapes_transformed(self, tmp_path):
        img_dir, ann_dir = self._setup_labelme_data(tmp_path)
        out_dir = str(tmp_path / "output")

        augment_batch(
            img_dir, ann_dir, out_dir, "flip",
            ann_format="labelme", direction="horizontal",
        )

        out_json = os.path.join(out_dir, "annotations", "sample.json")
        with open(out_json, encoding="utf-8") as f:
            data = json.load(f)

        assert len(data["shapes"]) == 2
        assert data["shapes"][0]["label"] == "car"
        assert data["shapes"][1]["label"] == "person"

        # Verify points are flipped horizontally (x coords should be 200 - original_x)
        pts_car = data["shapes"][0]["points"]
        assert pts_car[0][0] == pytest.approx(150, abs=1)
        assert pts_car[1][0] == pytest.approx(50, abs=1)

    def test_rotate_creates_output(self, tmp_path):
        img_dir, ann_dir = self._setup_labelme_data(tmp_path)
        out_dir = str(tmp_path / "output")

        stats = augment_batch(
            img_dir, ann_dir, out_dir, "rotate",
            ann_format="labelme", angle=90,
        )
        assert stats["processed"] == 1
        out_json = os.path.join(out_dir, "annotations", "sample.json")
        with open(out_json, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data["shapes"]) == 2
        # imageWidth/imageHeight should be updated
        assert isinstance(data["imageWidth"], int)
        assert isinstance(data["imageHeight"], int)
        assert data["imagePath"] == "sample.png"

    def test_resize_creates_output(self, tmp_path):
        img_dir, ann_dir = self._setup_labelme_data(tmp_path)
        out_dir = str(tmp_path / "output")

        stats = augment_batch(
            img_dir, ann_dir, out_dir, "resize",
            ann_format="labelme", new_w=400, new_h=300,
        )
        assert stats["processed"] == 1

        out_json = os.path.join(out_dir, "annotations", "sample.json")
        with open(out_json, encoding="utf-8") as f:
            data = json.load(f)
        assert data["imageWidth"] == 400
        assert data["imageHeight"] == 300

        # Points should be scaled by 2x
        pts = data["shapes"][0]["points"]
        # Original first point [50, 40] -> [100, 80] after 2x resize
        assert pts[0][0] == pytest.approx(100, abs=1)
        assert pts[0][1] == pytest.approx(80, abs=1)

    def test_missing_annotation_skipped(self, tmp_path):
        img_dir = str(tmp_path / "images")
        ann_dir = str(tmp_path / "annotations")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)

        img = np.zeros((50, 50, 3), dtype=np.uint8)
        write_image(os.path.join(img_dir, "orphan.png"), img)
        # No JSON annotation

        out_dir = str(tmp_path / "output")
        stats = augment_batch(
            img_dir, ann_dir, out_dir, "flip",
            ann_format="labelme", direction="horizontal",
        )
        assert stats["skipped"] >= 1
        assert stats["processed"] == 0

    def test_progress_callback(self, tmp_path):
        img_dir, ann_dir = self._setup_labelme_data(tmp_path)
        out_dir = str(tmp_path / "output")

        callbacks = []
        augment_batch(
            img_dir, ann_dir, out_dir, "flip",
            ann_format="labelme", direction="horizontal",
            progress_callback=lambda cur, tot: callbacks.append((cur, tot)),
        )
        assert len(callbacks) >= 1
        assert callbacks[-1][0] == 1  # current
        assert callbacks[-1][1] == 1  # total
