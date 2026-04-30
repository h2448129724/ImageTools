"""Tests for gui.function_handlers module."""
import numpy as np
import pytest
from gui.function_handlers import apply_simple, apply_complex, is_batch_function


def _make_img(w=100, h=80, channels=3):
    return np.random.randint(0, 255, (h, w, channels), dtype=np.uint8)


def _make_gray(w=100, h=80):
    return np.random.randint(0, 255, (h, w), dtype=np.uint8)


class TestColorConversion:
    def test_bgr2rgb(self):
        img = _make_img()
        result = apply_simple("color_bgr2rgb", img, {})
        assert result is not None
        assert result.shape == img.shape

    def test_bgr2gray(self):
        img = _make_img()
        result = apply_simple("color_bgr2gray", img, {})
        assert result is not None
        assert len(result.shape) == 2

    def test_bgr2hsv(self):
        img = _make_img()
        result = apply_simple("color_bgr2hsv", img, {})
        assert result is not None
        assert result.shape == img.shape


class TestBasicProcessing:
    def test_resize(self):
        img = _make_img(200, 150)
        result = apply_simple("resize", img, {"width": 100, "height": 80, "keep_aspect": False, "scale": 0})
        assert result is not None
        assert result.shape[:2] == (80, 100)

    def test_crop(self):
        img = _make_img(200, 150)
        result = apply_simple("crop", img, {"x": 10, "y": 10, "w": 50, "h": 40})
        assert result is not None
        assert result.shape == (40, 50, 3)

    def test_rotate(self):
        img = _make_img(100, 80)
        result = apply_simple("rotate", img, {"angle": 90, "keep_size": True})
        assert result is not None
        assert result.shape == img.shape

    def test_flip(self):
        img = _make_img()
        result = apply_simple("flip", img, {"direction": "horizontal"})
        assert result is not None
        assert result.shape == img.shape

    def test_brightness_contrast(self):
        img = _make_img()
        result = apply_simple("brightness_contrast", img, {"brightness": 20, "contrast": 1.2})
        assert result is not None
        assert result.shape == img.shape

    def test_saturation(self):
        img = _make_img()
        result = apply_simple("saturation", img, {"factor": 1.5})
        assert result is not None
        assert result.shape == img.shape

    def test_pad(self):
        img = _make_img(100, 80)
        result = apply_simple("pad", img, {"top": 5, "bottom": 5, "left": 5, "right": 5, "mode": "constant"})
        assert result is not None
        assert result.shape == (90, 110, 3)

    def test_threshold(self):
        img = _make_img()
        result = apply_simple("threshold", img, {"method": "otsu", "thresh": 127, "maxval": 255, "block_size": 11})
        assert result is not None


class TestFilters:
    def test_gaussian(self):
        img = _make_img()
        result = apply_simple("filter_gaussian", img, {"ksize": 5})
        assert result is not None
        assert result.shape == img.shape

    def test_sharpen(self):
        img = _make_img()
        result = apply_simple("filter_sharpen", img, {})
        assert result is not None
        assert result.shape == img.shape


class TestEdgeDetection:
    def test_canny(self):
        img = _make_img()
        result = apply_simple("edge_canny", img, {"threshold1": 50, "threshold2": 150})
        assert result is not None


class TestAddBorder:
    def test_add_border(self):
        img = _make_img(100, 80)
        result = apply_simple("batch_add_border", img, {"border_size": 10, "color": "black"})
        assert result is not None
        assert result.shape == (100, 120, 3)


class TestUnknownFunction:
    def test_unknown_returns_img(self):
        img = _make_img()
        result = apply_simple("nonexistent_func", img, {})
        assert result is not None
        np.testing.assert_array_equal(result, img)


class TestComplexHandlers:
    def test_tile_fixed_preview(self):
        img = _make_img(200, 200)
        result = apply_complex("tile_fixed", img, {"tile_w": 100, "tile_h": 100, "overlap": 0, "discard_incomplete": True})
        assert result is not None
        assert result.shape[:2] == (100, 100)

    def test_tile_grid_preview(self):
        img = _make_img(200, 200)
        result = apply_complex("tile_grid", img, {"rows": 2, "cols": 2})
        assert result is not None

    def test_annot_draw_yolo_no_txt(self):
        img = _make_img()
        result = apply_complex("annot_draw_yolo", img, {"txt_dir": "", "categories": ""}, filepath="/fake/path.jpg")
        assert result is not None
        np.testing.assert_array_equal(result, img)

    def test_batch_roi_crop(self):
        img = _make_img(200, 200)
        result = apply_complex("batch_roi_crop", img, {"x": 10, "y": 10, "w": 50, "h": 50})
        assert result is not None
        assert result.shape == (50, 50, 3)

    def test_batch_roi_crop_out_of_bounds(self):
        img = _make_img(100, 100)
        result = apply_complex("batch_roi_crop", img, {"x": 90, "y": 90, "w": 200, "h": 200})
        assert result is not None
        assert result.shape[0] <= 100 and result.shape[1] <= 100


class TestIsBatchFunction:
    def test_batch_prefix(self):
        assert is_batch_function("batch_rename") is True
        assert is_batch_function("batch_resize") is True

    def test_dataset_prefix(self):
        assert is_batch_function("dataset_random_split") is True

    def test_format_prefix(self):
        assert is_batch_function("format_yolo2coco") is True

    def test_format_convert_not_batch(self):
        assert is_batch_function("format_convert") is False

    def test_annot_prefix(self):
        assert is_batch_function("annot_draw_yolo") is True

    def test_seg_tile(self):
        assert is_batch_function("seg_tile") is True

    def test_augment_prefix(self):
        assert is_batch_function("augment_yolo") is True

    def test_simple_not_batch(self):
        assert is_batch_function("resize") is False
        assert is_batch_function("flip") is False
        assert is_batch_function("filter_gaussian") is False
