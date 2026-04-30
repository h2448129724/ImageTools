"""Tests for core.mask_polygon module."""
import json
import os

import cv2
import numpy as np
import pytest

from core.image_io import write_image
from core.mask_polygon import (
    batch_labelme_to_mask,
    batch_mask_to_labelme,
    mask_to_polygons,
    polygons_to_mask,
)


def _make_rect_mask(h, w, x1, y1, x2, y2):
    """Create a binary mask with a white rectangle."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def _make_circle_mask(h, w, cx, cy, radius):
    """Create a binary mask with a white filled circle."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, thickness=-1)
    return mask


# ---------- mask_to_polygons ----------


class TestMaskToPolygons:
    def test_rectangle_mask_extracts_polygon(self):
        """A solid rectangle mask should produce at least one polygon."""
        mask = _make_rect_mask(100, 100, 10, 10, 60, 60)
        shapes = mask_to_polygons(mask, label="rect")
        assert len(shapes) >= 1
        for s in shapes:
            assert s["label"] == "rect"
            assert s["shape_type"] == "polygon"
            assert len(s["points"]) >= 3

    def test_circle_mask_extracts_polygon(self):
        """A solid circle mask should produce at least one polygon."""
        mask = _make_circle_mask(100, 100, 50, 50, 30)
        shapes = mask_to_polygons(mask, label="circle")
        assert len(shapes) >= 1
        for s in shapes:
            assert s["label"] == "circle"
            assert s["shape_type"] == "polygon"
            assert len(s["points"]) >= 3

    def test_two_disjoint_shapes(self):
        """A mask with two disjoint rectangles should produce at least two polygons."""
        mask = _make_rect_mask(100, 200, 10, 10, 50, 50)
        mask[10:50, 110:150] = 255  # second rectangle
        shapes = mask_to_polygons(mask, label="obj")
        assert len(shapes) >= 2

    def test_empty_mask_returns_empty(self):
        """An all-zero mask should produce no shapes."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        shapes = mask_to_polygons(mask)
        assert shapes == []

    def test_binary_0_1_mask(self):
        """A mask with values 0/1 (not 0/255) should be handled correctly."""
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[20:60, 20:60] = 1
        shapes = mask_to_polygons(mask, label="small")
        assert len(shapes) >= 1

    def test_shape_dict_has_required_keys(self):
        """Each shape dict should contain all Labelme-required keys."""
        mask = _make_rect_mask(60, 60, 5, 5, 55, 55)
        shapes = mask_to_polygons(mask, label="test")
        assert len(shapes) >= 1
        required_keys = {"label", "points", "shape_type", "flags", "group_id"}
        for s in shapes:
            assert required_keys.issubset(s.keys())


# ---------- polygons_to_mask ----------


class TestPolygonsToMask:
    def test_single_polygon_produces_nonzero_mask(self):
        """A known polygon should produce a mask with non-zero pixels."""
        shapes = [
            {
                "label": "rect",
                "points": [[10, 10], [50, 10], [50, 50], [10, 50]],
                "shape_type": "polygon",
            }
        ]
        mask = polygons_to_mask(shapes, 100, 100)
        assert mask.shape == (100, 100)
        assert mask.dtype == np.uint8
        assert np.any(mask > 0)
        # Inside the rectangle should be 255
        assert mask[30, 30] == 255
        # Outside should be 0
        assert mask[0, 0] == 0

    def test_empty_shapes_produces_zero_mask(self):
        """An empty shapes list should produce an all-zero mask."""
        mask = polygons_to_mask([], 50, 50)
        assert mask.shape == (50, 50)
        assert np.all(mask == 0)

    def test_rectangle_shape_type(self):
        """Shapes with shape_type='rectangle' should be rendered."""
        shapes = [
            {
                "label": "box",
                "points": [[5, 5], [40, 5], [40, 40], [5, 40]],
                "shape_type": "rectangle",
            }
        ]
        mask = polygons_to_mask(shapes, 60, 60)
        assert np.any(mask > 0)

    def test_non_polygon_shape_type_skipped(self):
        """Shapes with unsupported shape_type (e.g., 'line') should be skipped."""
        shapes = [
            {
                "label": "line",
                "points": [[0, 0], [100, 100]],
                "shape_type": "line",
            }
        ]
        mask = polygons_to_mask(shapes, 100, 100)
        assert np.all(mask == 0)

    def test_mask_values_are_0_or_255(self):
        """All mask values should be either 0 or 255."""
        shapes = [
            {
                "label": "tri",
                "points": [[10, 10], [90, 10], [50, 90]],
                "shape_type": "polygon",
            }
        ]
        mask = polygons_to_mask(shapes, 100, 100)
        unique = np.unique(mask)
        assert set(unique).issubset({0, 255})


# ---------- batch_mask_to_labelme ----------


class TestBatchMaskToLabelme:
    def test_creates_json_files(self, tmp_path):
        """Mask images in input dir should produce Labelme JSON files in output dir."""
        mask_dir = tmp_path / "masks"
        output_dir = tmp_path / "annotations"
        mask_dir.mkdir()

        mask = _make_rect_mask(100, 100, 10, 10, 90, 90)
        write_image(str(mask_dir / "mask1.png"), mask)

        result = batch_mask_to_labelme(str(mask_dir), str(output_dir), label="obj")

        assert result["converted"] == 1
        assert result["total_files"] == 1
        assert os.path.isfile(str(output_dir / "mask1.json"))

    def test_json_structure(self, tmp_path):
        """Each output JSON should have the correct Labelme annotation structure."""
        mask_dir = tmp_path / "masks"
        output_dir = tmp_path / "annotations"
        mask_dir.mkdir()

        mask = _make_circle_mask(80, 80, 40, 40, 25)
        write_image(str(mask_dir / "circle.png"), mask)

        batch_mask_to_labelme(str(mask_dir), str(output_dir), label="circle")

        with open(str(output_dir / "circle.json"), "r") as f:
            ann = json.load(f)

        assert "version" in ann
        assert "shapes" in ann
        assert "imagePath" in ann
        assert "imageHeight" in ann
        assert "imageWidth" in ann
        assert ann["imageHeight"] == 80
        assert ann["imageWidth"] == 80
        assert ann["imagePath"] == "circle.png"
        assert len(ann["shapes"]) >= 1
        assert ann["shapes"][0]["label"] == "circle"

    def test_multiple_mask_files(self, tmp_path):
        """Multiple mask files should all be converted."""
        mask_dir = tmp_path / "masks"
        output_dir = tmp_path / "annotations"
        mask_dir.mkdir()

        for i in range(3):
            mask = _make_rect_mask(60, 60, 5, 5, 55, 55)
            write_image(str(mask_dir / f"m{i}.png"), mask)

        result = batch_mask_to_labelme(str(mask_dir), str(output_dir))

        assert result["converted"] == 3
        for i in range(3):
            assert os.path.isfile(str(output_dir / f"m{i}.json"))

    def test_empty_mask_directory(self, tmp_path):
        """An empty directory should produce zero conversions."""
        mask_dir = tmp_path / "masks"
        output_dir = tmp_path / "annotations"
        mask_dir.mkdir()

        result = batch_mask_to_labelme(str(mask_dir), str(output_dir))

        assert result["converted"] == 0
        assert result["total_files"] == 0

    def test_progress_callback(self, tmp_path):
        """Progress callback should be called for each file."""
        mask_dir = tmp_path / "masks"
        output_dir = tmp_path / "annotations"
        mask_dir.mkdir()

        for i in range(2):
            mask = _make_rect_mask(50, 50, 5, 5, 45, 45)
            write_image(str(mask_dir / f"m{i}.png"), mask)

        calls = []
        batch_mask_to_labelme(
            str(mask_dir), str(output_dir),
            progress_callback=lambda cur, total: calls.append((cur, total)),
        )

        assert len(calls) == 2
        assert calls[-1] == (2, 2)


# ---------- batch_labelme_to_mask ----------


class TestBatchLabelmeToMask:
    def _make_labelme_json(self, path, base_name, h, w, shapes):
        """Write a Labelme annotation JSON file."""
        ann = {
            "version": "5.0.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": f"{base_name}.png",
            "imageData": None,
            "imageHeight": h,
            "imageWidth": w,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ann, f, indent=2)

    def test_creates_mask_files(self, tmp_path):
        """Labelme JSON files should produce mask PNG files."""
        ann_dir = tmp_path / "annotations"
        output_dir = tmp_path / "masks"
        ann_dir.mkdir()

        shapes = [
            {
                "label": "obj",
                "points": [[10, 10], [50, 10], [50, 50], [10, 50]],
                "shape_type": "polygon",
            }
        ]
        self._make_labelme_json(str(ann_dir / "test1.json"), "test1", 100, 100, shapes)

        result = batch_labelme_to_mask(str(ann_dir), str(output_dir))

        assert result["converted"] == 1
        assert os.path.isfile(str(output_dir / "test1.png"))

    def test_mask_has_correct_dimensions(self, tmp_path):
        """Output mask should match imageHeight/imageWidth from the JSON."""
        ann_dir = tmp_path / "annotations"
        output_dir = tmp_path / "masks"
        ann_dir.mkdir()

        shapes = [
            {
                "label": "obj",
                "points": [[5, 5], [40, 5], [40, 40], [5, 40]],
                "shape_type": "polygon",
            }
        ]
        self._make_labelme_json(str(ann_dir / "dim.json"), "dim", 80, 120, shapes)

        batch_labelme_to_mask(str(ann_dir), str(output_dir))

        mask = cv2.imread(str(output_dir / "dim.png"), cv2.IMREAD_GRAYSCALE)
        assert mask is not None
        assert mask.shape == (80, 120)

    def test_mask_has_nonzero_pixels(self, tmp_path):
        """Output mask should have non-zero pixels where the polygon is."""
        ann_dir = tmp_path / "annotations"
        output_dir = tmp_path / "masks"
        ann_dir.mkdir()

        shapes = [
            {
                "label": "rect",
                "points": [[10, 10], [60, 10], [60, 60], [10, 60]],
                "shape_type": "polygon",
            }
        ]
        self._make_labelme_json(str(ann_dir / "fill.json"), "fill", 100, 100, shapes)

        batch_labelme_to_mask(str(ann_dir), str(output_dir))

        mask = cv2.imread(str(output_dir / "fill.png"), cv2.IMREAD_GRAYSCALE)
        assert mask is not None
        assert np.any(mask > 0)
        assert mask[30, 30] > 0

    def test_empty_shapes_skipped(self, tmp_path):
        """JSON with empty shapes list should be skipped (no mask file created)."""
        ann_dir = tmp_path / "annotations"
        output_dir = tmp_path / "masks"
        ann_dir.mkdir()

        self._make_labelme_json(str(ann_dir / "empty.json"), "empty", 100, 100, [])

        result = batch_labelme_to_mask(str(ann_dir), str(output_dir))

        assert result["converted"] == 0
        assert not os.path.isfile(str(output_dir / "empty.png"))

    def test_multiple_json_files(self, tmp_path):
        """Multiple Labelme JSON files should all be converted."""
        ann_dir = tmp_path / "annotations"
        output_dir = tmp_path / "masks"
        ann_dir.mkdir()

        for i in range(3):
            shapes = [
                {
                    "label": "obj",
                    "points": [[10, 10], [40, 10], [40, 40], [10, 40]],
                    "shape_type": "polygon",
                }
            ]
            self._make_labelme_json(
                str(ann_dir / f"ann{i}.json"), f"ann{i}", 60, 60, shapes
            )

        result = batch_labelme_to_mask(str(ann_dir), str(output_dir))

        assert result["converted"] == 3
        for i in range(3):
            assert os.path.isfile(str(output_dir / f"ann{i}.png"))

    def test_progress_callback(self, tmp_path):
        """Progress callback should be called for each file."""
        ann_dir = tmp_path / "annotations"
        output_dir = tmp_path / "masks"
        ann_dir.mkdir()

        for i in range(2):
            shapes = [
                {
                    "label": "obj",
                    "points": [[5, 5], [30, 5], [30, 30], [5, 30]],
                    "shape_type": "polygon",
                }
            ]
            self._make_labelme_json(
                str(ann_dir / f"cb{i}.json"), f"cb{i}", 50, 50, shapes
            )

        calls = []
        batch_labelme_to_mask(
            str(ann_dir), str(output_dir),
            progress_callback=lambda cur, total: calls.append((cur, total)),
        )

        assert len(calls) == 2
        assert calls[-1] == (2, 2)
