"""Tests for core.annotation module."""
import os
import json
import numpy as np
import pytest
from core.image_io import write_image
from core.annotation import (
    parse_yolo_file,
    draw_yolo_boxes,
    draw_coco_boxes,
    validate_yolo_annotations,
    annotation_statistics,
    crop_roi_from_yolo,
)
from tests.helpers import _make_test_img


def _write_yolo_txt(path, lines):
    """Write a YOLO annotation file with the given text lines."""
    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")


# ---------- parse_yolo_file ----------

class TestParseYoloFile:
    def test_valid_file(self, tmp_path):
        txt = str(tmp_path / "ann.txt")
        _write_yolo_txt(txt, [
            "0 0.5 0.5 0.3 0.4",
            "1 0.2 0.3 0.1 0.2",
        ])
        boxes = parse_yolo_file(txt)
        assert len(boxes) == 2
        assert boxes[0]["cls"] == 0
        assert boxes[0]["xc"] == pytest.approx(0.5)
        assert boxes[0]["yc"] == pytest.approx(0.5)
        assert boxes[0]["bw"] == pytest.approx(0.3)
        assert boxes[0]["bh"] == pytest.approx(0.4)
        assert boxes[1]["cls"] == 1

    def test_empty_file(self, tmp_path):
        txt = str(tmp_path / "empty.txt")
        _write_yolo_txt(txt, [])
        boxes = parse_yolo_file(txt)
        assert boxes == []

    def test_nonexistent_file(self):
        boxes = parse_yolo_file("/nonexistent/path/ann.txt")
        assert boxes == []

    def test_malformed_lines_skipped(self, tmp_path):
        txt = str(tmp_path / "bad.txt")
        _write_yolo_txt(txt, [
            "0 0.5 0.5 0.3 0.4",   # valid
            "only_two_fields",       # too few fields
            "a b c d e",             # non-numeric
            "1 0.1 0.2 0.3",        # only 4 fields
            "",                      # empty line
            "2 0.1 0.2 0.3 0.4",    # valid
        ])
        boxes = parse_yolo_file(txt)
        assert len(boxes) == 2
        assert boxes[0]["cls"] == 0
        assert boxes[1]["cls"] == 2


# ---------- draw_yolo_boxes ----------

class TestDrawYoloBoxes:
    def test_output_same_shape(self, tmp_path):
        img = _make_test_img(200, 150)
        txt = str(tmp_path / "ann.txt")
        _write_yolo_txt(txt, ["0 0.5 0.5 0.3 0.4"])
        result = draw_yolo_boxes(img, txt)
        assert result.shape == img.shape

    def test_with_class_names(self, tmp_path):
        img = _make_test_img(200, 150)
        txt = str(tmp_path / "ann.txt")
        _write_yolo_txt(txt, ["0 0.5 0.5 0.3 0.4"])
        result = draw_yolo_boxes(img, txt, class_names=["cat", "dog"])
        assert result.shape == img.shape
        # The result should differ from the input because rectangles were drawn
        assert not np.array_equal(result, img)

    def test_no_annotation_file(self, tmp_path):
        img = _make_test_img(200, 150)
        result = draw_yolo_boxes(img, str(tmp_path / "missing.txt"))
        # Should return a copy when no boxes found
        assert result.shape == img.shape
        assert np.array_equal(result, img)


# ---------- draw_coco_boxes ----------

class TestDrawCocoBoxes:
    def _make_coco_data(self, image_id=1, category_name="cat"):
        """Build a minimal COCO dict in memory."""
        return {
            "images": [{"id": image_id, "file_name": "test.png", "width": 100, "height": 80}],
            "annotations": [
                {"id": 1, "image_id": image_id, "category_id": 0, "bbox": [10, 10, 30, 20]},
            ],
            "categories": [{"id": 0, "name": category_name}],
        }

    def test_with_mock_coco_dict(self):
        img = _make_test_img(100, 80)
        coco = self._make_coco_data(image_id=1)
        result = draw_coco_boxes(img, coco, image_id=1)
        assert result.shape == img.shape
        assert not np.array_equal(result, img)

    def test_with_coco_json_file(self, tmp_path):
        """Test with a COCO JSON file path (string) instead of a dict."""
        img = _make_test_img(100, 80)
        coco_data = self._make_coco_data(image_id=1)
        json_path = str(tmp_path / "coco.json")
        with open(json_path, "w") as f:
            json.dump(coco_data, f)
        result = draw_coco_boxes(img, json_path, image_id=1)
        assert result.shape == img.shape
        assert not np.array_equal(result, img)

    def test_image_id_zero_works(self):
        """Critical test: image_id=0 must be matched (old bug used `if image_id`)."""
        img = _make_test_img(100, 80)
        coco = self._make_coco_data(image_id=0)
        result = draw_coco_boxes(img, coco, image_id=0)
        assert result.shape == img.shape
        # Boxes should have been drawn, so result differs from input
        assert not np.array_equal(result, img)

    def test_image_id_zero_no_match_returns_copy(self):
        """image_id=0 passed but no matching image in COCO data should return unmodified copy."""
        img = _make_test_img(100, 80)
        coco = self._make_coco_data(image_id=1)
        result = draw_coco_boxes(img, coco, image_id=0)
        assert result.shape == img.shape
        assert np.array_equal(result, img)

    def test_match_by_image_name(self):
        img = _make_test_img(100, 80)
        coco = self._make_coco_data(image_id=1)
        result = draw_coco_boxes(img, coco, image_name="test.png")
        assert result.shape == img.shape
        assert not np.array_equal(result, img)

    def test_no_match_returns_copy(self):
        img = _make_test_img(100, 80)
        coco = self._make_coco_data(image_id=99)
        result = draw_coco_boxes(img, coco, image_id=1)
        assert np.array_equal(result, img)


# ---------- validate_yolo_annotations ----------

class TestValidateYoloAnnotations:
    def test_valid_annotations_no_issues(self, tmp_path):
        txt = str(tmp_path / "valid.txt")
        _write_yolo_txt(txt, ["0 0.5 0.5 0.3 0.4"])
        issues = validate_yolo_annotations(txt, img_w=100, img_h=80)
        assert issues == []

    def test_out_of_bounds_center(self, tmp_path):
        txt = str(tmp_path / "oob.txt")
        _write_yolo_txt(txt, ["0 1.5 0.5 0.1 0.1"])
        issues = validate_yolo_annotations(txt, img_w=100, img_h=80)
        assert any("normalized center out of [0,1]" in issue for issue in issues)

    def test_box_exceeds_image_bounds(self, tmp_path):
        txt = str(tmp_path / "exceed.txt")
        # Center at (0.05, 0.05) with box 0.3x0.3 means left/top edges go negative
        _write_yolo_txt(txt, ["0 0.05 0.05 0.3 0.3"])
        issues = validate_yolo_annotations(txt, img_w=100, img_h=80)
        assert any("box exceeds image bounds" in issue for issue in issues)

    def test_too_few_fields(self, tmp_path):
        txt = str(tmp_path / "short.txt")
        _write_yolo_txt(txt, ["0 0.5 0.5"])
        issues = validate_yolo_annotations(txt, img_w=100, img_h=80)
        assert any("expected 5+ fields" in issue for issue in issues)

    def test_non_numeric_values(self, tmp_path):
        txt = str(tmp_path / "nan.txt")
        _write_yolo_txt(txt, ["0 abc 0.5 0.3 0.4"])
        issues = validate_yolo_annotations(txt, img_w=100, img_h=80)
        assert any("non-numeric" in issue for issue in issues)

    def test_invalid_box_dimensions(self, tmp_path):
        txt = str(tmp_path / "neg.txt")
        _write_yolo_txt(txt, ["0 0.5 0.5 -0.1 0.2"])
        issues = validate_yolo_annotations(txt, img_w=100, img_h=80)
        assert any("invalid box dimensions" in issue for issue in issues)

    def test_nonexistent_file(self):
        issues = validate_yolo_annotations("/nonexistent/file.txt", 100, 80)
        assert issues == []


# ---------- annotation_statistics ----------

class TestAnnotationStatistics:
    def test_with_sample_files(self, tmp_path):
        ann_dir = tmp_path / "annotations"
        img_dir = tmp_path / "images"
        ann_dir.mkdir()
        img_dir.mkdir()

        # Create a test image and write to disk
        img = _make_test_img(200, 150)
        img_path = str(img_dir / "photo.jpg")
        write_image(img_path, img)

        # Create matching annotation
        txt_path = str(ann_dir / "photo.txt")
        _write_yolo_txt(txt_path, [
            "0 0.5 0.5 0.2 0.2",
            "1 0.3 0.3 0.1 0.1",
        ])

        stats = annotation_statistics(str(ann_dir), str(img_dir))
        assert stats["total_boxes"] == 2
        assert 0 in stats["class_counts"]
        assert 1 in stats["class_counts"]
        assert stats["class_counts"][0] == 1
        assert stats["class_counts"][1] == 1
        assert stats["mean_area"] > 0
        assert stats["median_area"] > 0

    def test_empty_directory(self, tmp_path):
        ann_dir = tmp_path / "annotations"
        img_dir = tmp_path / "images"
        ann_dir.mkdir()
        img_dir.mkdir()

        stats = annotation_statistics(str(ann_dir), str(img_dir))
        assert stats["total_boxes"] == 0
        assert stats["class_counts"] == {}
        assert stats["mean_area"] == 0
        assert stats["median_area"] == 0


# ---------- crop_roi_from_yolo ----------

class TestCropRoiFromYolo:
    def test_creates_output_files(self, tmp_path):
        img_dir = tmp_path / "images"
        ann_dir = tmp_path / "annotations"
        output_dir = tmp_path / "crops"
        img_dir.mkdir()
        ann_dir.mkdir()

        # Write a test image
        img = _make_test_img(200, 150)
        img_path = str(img_dir / "photo.png")
        write_image(img_path, img)

        # Write annotation with two boxes well inside the image
        txt_path = str(ann_dir / "photo.txt")
        _write_yolo_txt(txt_path, [
            "0 0.5 0.5 0.2 0.2",
            "1 0.3 0.3 0.1 0.1",
        ])

        count = crop_roi_from_yolo(img_path, txt_path, str(output_dir))
        assert count == 2

        # Check that class subdirectories were created
        class0_dir = output_dir / "class_0"
        class1_dir = output_dir / "class_1"
        assert class0_dir.is_dir()
        assert class1_dir.is_dir()
        # Each should contain one crop file
        assert len(list(class0_dir.glob("*.png"))) == 1
        assert len(list(class1_dir.glob("*.png"))) == 1

    def test_with_class_names(self, tmp_path):
        img_dir = tmp_path / "images"
        ann_dir = tmp_path / "annotations"
        output_dir = tmp_path / "crops"
        img_dir.mkdir()
        ann_dir.mkdir()

        img = _make_test_img(200, 150)
        img_path = str(img_dir / "photo.png")
        write_image(img_path, img)

        txt_path = str(ann_dir / "photo.txt")
        _write_yolo_txt(txt_path, ["0 0.5 0.5 0.2 0.2"])

        count = crop_roi_from_yolo(
            img_path, txt_path, str(output_dir),
            class_names=["cat", "dog"],
        )
        assert count == 1
        assert (output_dir / "cat").is_dir()

    def test_no_annotations(self, tmp_path):
        img_dir = tmp_path / "images"
        ann_dir = tmp_path / "annotations"
        output_dir = tmp_path / "crops"
        img_dir.mkdir()
        ann_dir.mkdir()

        img = _make_test_img(200, 150)
        img_path = str(img_dir / "photo.png")
        write_image(img_path, img)

        txt_path = str(ann_dir / "photo.txt")
        _write_yolo_txt(txt_path, [])

        count = crop_roi_from_yolo(img_path, txt_path, str(output_dir))
        assert count == 0

    def test_with_padding(self, tmp_path):
        img_dir = tmp_path / "images"
        ann_dir = tmp_path / "annotations"
        output_dir = tmp_path / "crops"
        img_dir.mkdir()
        ann_dir.mkdir()

        img = _make_test_img(200, 150)
        img_path = str(img_dir / "photo.png")
        write_image(img_path, img)

        txt_path = str(ann_dir / "photo.txt")
        _write_yolo_txt(txt_path, ["0 0.5 0.5 0.2 0.2"])

        count_no_pad = crop_roi_from_yolo(img_path, txt_path, str(output_dir / "nopad"))
        count_pad = crop_roi_from_yolo(
            img_path, txt_path, str(output_dir / "withpad"), padding=10,
        )
        assert count_no_pad == 1
        assert count_pad == 1
