"""Tests for core.basic_processing module."""
import numpy as np
import pytest
from core.basic_processing import (
    crop_image, center_crop, pad_image, rotate_image, flip_image,
    adjust_brightness_contrast, apply_filter, edge_detect, threshold_image,
    morphology_op, remove_alpha, add_alpha
)


def _make_img(w=100, h=80, channels=3):
    if channels == 1:
        return np.random.randint(0, 255, (h, w), dtype=np.uint8)
    return np.random.randint(0, 255, (h, w, channels), dtype=np.uint8)


class TestCrop:
    def test_crop_dimensions(self):
        img = _make_img(100, 80)
        result = crop_image(img, 10, 10, 40, 30)
        assert result.shape == (30, 40, 3)

    def test_center_crop(self):
        img = _make_img(100, 80)
        result = center_crop(img, 40, 30)
        assert result.shape == (30, 40, 3)


class TestPad:
    def test_pad_increases_size(self):
        img = _make_img(100, 80)
        result = pad_image(img, 5, 5, 5, 5)
        assert result.shape == (90, 110, 3)


class TestRotate:
    def test_rotate_90(self):
        img = _make_img(100, 80)
        result = rotate_image(img, 90)
        assert result.shape[:2] == img.shape[:2]  # keep_size=True

    def test_rotate_no_keep_size(self):
        img = _make_img(100, 80)
        result = rotate_image(img, 45, keep_size=False)
        # rotated image should be larger to contain the whole rotated image
        assert result.shape[0] >= 80 and result.shape[1] >= 100


class TestFlip:
    def test_horizontal(self):
        img = _make_img()
        result = flip_image(img, "horizontal")
        assert result.shape == img.shape
        # Verify flip reverses columns
        assert np.array_equal(result[:, 0], img[:, -1])

    def test_vertical(self):
        img = _make_img()
        result = flip_image(img, "vertical")
        assert result.shape == img.shape
        # Verify flip reverses rows
        assert np.array_equal(result[0], img[-1])


class TestBrightnessContrast:
    def test_brightness_increase(self):
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        result = adjust_brightness_contrast(img, brightness=50)
        assert result[0, 0, 0] == 150

    def test_contrast_zero(self):
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        result = adjust_brightness_contrast(img, contrast=0)
        assert np.all(result == 0)


class TestFilter:
    @pytest.mark.parametrize("filter_type", ["blur", "gaussian", "median", "bilateral", "sharpen"])
    def test_filters_preserve_shape(self, filter_type):
        img = _make_img()
        result = apply_filter(img, filter_type, ksize=3)
        assert result.shape == img.shape


class TestEdgeDetect:
    @pytest.mark.parametrize("method", ["canny", "sobel", "laplacian"])
    def test_edge_methods(self, method):
        img = _make_img()
        result = edge_detect(img, method=method)
        assert result.shape[:2] == img.shape[:2]


class TestThreshold:
    def test_binary(self):
        img = _make_img(channels=1)
        result = threshold_image(img, method="binary", thresh=127)
        assert set(np.unique(result)).issubset({0, 255})

    def test_otsu(self):
        img = _make_img(channels=1)
        result = threshold_image(img, method="otsu")
        assert set(np.unique(result)).issubset({0, 255})


class TestAlpha:
    def test_add_alpha(self):
        img = _make_img(channels=3)
        result = add_alpha(img)
        assert result.shape[2] == 4

    def test_remove_alpha(self):
        img = _make_img(channels=4)
        result = remove_alpha(img)
        assert result.shape[2] == 3

    def test_add_alpha_grayscale(self):
        img = _make_img(channels=1)
        result = add_alpha(img)
        assert result.shape[2] == 4


class TestMorphology:
    @pytest.mark.parametrize("op", ["erode", "dilate", "open", "close"])
    def test_morphology_ops(self, op):
        img = _make_img(channels=1)
        result = morphology_op(img, op_type=op)
        assert result.shape == img.shape
