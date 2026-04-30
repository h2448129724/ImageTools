"""Tests for core.color_conversion module."""
import numpy as np
import pytest
from core.color_conversion import (
    CONVERSION_CODES,
    convert_color,
    to_grayscale,
    split_channels,
    merge_channels,
    extract_channel,
)


def _make_bgr(h=10, w=10):
    """Create a simple BGR test image."""
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _make_gray(h=10, w=10):
    """Create a single-channel grayscale test image."""
    return np.random.randint(0, 255, (h, w), dtype=np.uint8)


def _make_rgba(h=10, w=10):
    """Create a BGRA test image."""
    return np.random.randint(0, 255, (h, w, 4), dtype=np.uint8)


# ---------- convert_color ----------

class TestConvertColor:
    def test_bgr_conversion_keys_produce_valid_output(self):
        """BGR-native conversions should work on a BGR image."""
        img = _make_bgr()
        bgr_keys = {"BGR → RGB", "BGR → HSV", "BGR → LAB", "BGR → YUV",
                     "BGR → GRAY", "BGR → YCrCb", "BGR → HLS"}
        for key in bgr_keys:
            result = convert_color(img, key)
            assert result is not None
            assert result.size > 0

    def test_gray_to_bgr_produces_3channel(self):
        gray = _make_gray()
        result = convert_color(gray, "GRAY → BGR")
        assert len(result.shape) == 3
        assert result.shape[2] == 3

    def test_bgr_to_rgb_channels_flipped(self):
        """BGR-to-RGB should flip channel order for a known-colour image."""
        img = np.full((4, 4, 3), [100, 150, 200], dtype=np.uint8)  # B=100 G=150 R=200
        result = convert_color(img, "BGR → RGB")
        np.testing.assert_array_equal(result[0, 0], [200, 150, 100])

    def test_bgr_to_gray_reduces_channels(self):
        img = _make_bgr()
        result = convert_color(img, "BGR → GRAY")
        assert len(result.shape) == 2

    def test_unknown_key_raises_value_error(self):
        img = _make_bgr()
        with pytest.raises(ValueError, match="Unknown conversion"):
            convert_color(img, "NONEXISTENT → FOO")


# ---------- to_grayscale ----------

class TestToGrayscale:
    def test_bgr_input(self):
        img = _make_bgr()
        result = to_grayscale(img)
        assert result is not None
        assert len(result.shape) == 2
        assert result.shape == (img.shape[0], img.shape[1])

    def test_already_gray_passthrough(self):
        gray = _make_gray()
        result = to_grayscale(gray)
        assert result is gray  # same object, no copy

    def test_rgba_input_strips_alpha(self):
        """RGBA input should have its alpha stripped and be converted to gray."""
        img = _make_rgba()
        result = to_grayscale(img)
        assert result is not None
        assert len(result.shape) == 2
        assert result.shape == (img.shape[0], img.shape[1])


# ---------- split_channels / merge_channels ----------

class TestSplitMergeChannels:
    def test_split_returns_three_arrays(self):
        img = _make_bgr()
        channels = split_channels(img)
        assert len(channels) == 3
        for ch in channels:
            assert ch.shape == (img.shape[0], img.shape[1])

    def test_split_channel_values_match_original(self):
        """Each split channel should equal the corresponding slice of the image."""
        img = _make_bgr()
        b, g, r = split_channels(img)
        np.testing.assert_array_equal(b, img[:, :, 0])
        np.testing.assert_array_equal(g, img[:, :, 1])
        np.testing.assert_array_equal(r, img[:, :, 2])

    def test_merge_roundtrip_equals_original(self):
        """split_channels -> merge_channels should reconstruct the original image."""
        img = _make_bgr()
        channels = split_channels(img)
        merged = merge_channels(channels)
        np.testing.assert_array_equal(merged, img)


# ---------- extract_channel ----------

class TestExtractChannel:
    def test_extract_channel_0(self):
        img = _make_bgr()
        ch = extract_channel(img, 0)
        np.testing.assert_array_equal(ch, img[:, :, 0])

    def test_extract_channel_1(self):
        img = _make_bgr()
        ch = extract_channel(img, 1)
        np.testing.assert_array_equal(ch, img[:, :, 1])

    def test_extract_channel_2(self):
        img = _make_bgr()
        ch = extract_channel(img, 2)
        np.testing.assert_array_equal(ch, img[:, :, 2])

    def test_out_of_range_returns_none(self):
        img = _make_bgr()  # 3 channels
        assert extract_channel(img, 3) is None
        assert extract_channel(img, 99) is None

    def test_grayscale_passthrough(self):
        gray = _make_gray()
        result = extract_channel(gray, 0)
        assert result is gray  # same object for grayscale input
