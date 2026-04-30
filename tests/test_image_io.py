"""Tests for core.image_io module."""
import os
import struct
import tempfile
import numpy as np
import cv2
import pytest
from core.image_io import read_image, write_image, resize_image, get_image_info, convert_format
from core.image_io import _read_exif_orientation, _apply_exif_orientation
from tests.helpers import _make_test_img


class TestReadWrite:
    def test_write_and_read_png(self, tmp_path):
        img = _make_test_img()
        path = str(tmp_path / "test.png")
        write_image(path, img)
        result = read_image(path)
        assert result is not None
        assert result.shape == img.shape

    def test_write_and_read_jpg(self, tmp_path):
        img = _make_test_img()
        path = str(tmp_path / "test.jpg")
        write_image(path, img, quality=95)
        result = read_image(path)
        assert result is not None
        assert result.shape[:2] == img.shape[:2]

    def test_write_and_read_webp(self, tmp_path):
        img = _make_test_img()
        path = str(tmp_path / "test.webp")
        write_image(path, img)
        result = read_image(path)
        assert result is not None

    def test_unicode_path(self, tmp_path):
        img = _make_test_img()
        path = str(tmp_path / "测试图片_中文路径.png")
        write_image(path, img)
        result = read_image(path)
        assert result is not None
        assert result.shape == img.shape

    def test_read_nonexistent(self):
        assert read_image("/nonexistent/path.png") is None

    def test_write_creates_subdirectory(self, tmp_path):
        img = _make_test_img()
        path = str(tmp_path / "sub" / "dir" / "test.png")
        write_image(path, img)
        assert os.path.exists(path)

    def test_grayscale_write_read(self, tmp_path):
        img = _make_test_img(channels=1)
        path = str(tmp_path / "gray.png")
        write_image(path, img)
        result = read_image(path, flags=cv2.IMREAD_UNCHANGED)
        assert result is not None


class TestResize:
    def test_resize_by_width(self):
        img = _make_test_img(200, 100)
        result = resize_image(img, width=100)
        assert result.shape[1] == 100

    def test_resize_by_height(self):
        img = _make_test_img(200, 100)
        result = resize_image(img, height=50)
        assert result.shape[0] == 50

    def test_resize_by_scale(self):
        img = _make_test_img(200, 100)
        result = resize_image(img, scale=0.5)
        assert result.shape[1] == 100
        assert result.shape[0] == 50

    def test_resize_keep_aspect(self):
        img = _make_test_img(200, 100)
        result = resize_image(img, width=100, height=100, keep_aspect=True)
        assert result.shape[1] == 100
        assert result.shape[0] == 50

    def test_resize_no_args_returns_original(self):
        img = _make_test_img()
        result = resize_image(img)
        assert result.shape == img.shape


class TestGetImageInfo:
    def test_info(self, tmp_path):
        img = _make_test_img(120, 80, 3)
        path = str(tmp_path / "info.png")
        write_image(path, img)
        info = get_image_info(path)
        assert info is not None
        w, h, c = info
        assert w == 120 and h == 80 and c == 3

    def test_info_nonexistent(self):
        assert get_image_info("/nonexistent.png") is None


class TestConvertFormat:
    def test_png_to_jpg(self, tmp_path):
        img = _make_test_img()
        src = str(tmp_path / "src.png")
        dst = str(tmp_path / "dst.jpg")
        write_image(src, img)
        assert convert_format(src, dst) is True
        assert os.path.exists(dst)


def _make_jpeg_with_exif_orientation(orientation: int, img: np.ndarray) -> bytes:
    """Build a minimal JPEG with EXIF APP1 containing the given orientation tag."""
    # Encode image to JPEG bytes
    success, jpeg_buf = cv2.imencode(".jpg", img)
    assert success
    jpeg_data = bytes(jpeg_buf)

    # Build EXIF APP1 segment
    # TIFF header: "II" (little-endian), 0x002A, IFD offset=8
    tiff_header = b"II" + struct.pack("<H", 0x002A) + struct.pack("<I", 8)
    # IFD: 1 entry, orientation tag
    num_entries = struct.pack("<H", 1)
    # Tag: 0x0112 (Orientation), Type: 3 (SHORT), Count: 1, Value: orientation
    ifd_entry = struct.pack("<HHII", 0x0112, 3, 1, orientation)
    # Next IFD offset: 0 (no more IFDs)
    next_ifd = struct.pack("<I", 0)
    tiff_data = tiff_header + num_entries + ifd_entry + next_ifd

    # APP1: "Exif\x00\x00" + TIFF data
    exif_payload = b"Exif\x00\x00" + tiff_data
    app1_length = len(exif_payload) + 2
    app1_segment = b"\xFF\xE1" + struct.pack(">H", app1_length) + exif_payload

    # Insert APP1 after SOI (first 2 bytes)
    return jpeg_data[:2] + app1_segment + jpeg_data[2:]


class TestExifOrientation:
    def test_orientation_1_no_change(self, tmp_path):
        """Orientation 1 (normal) should leave image unchanged."""
        img = _make_test_img(60, 40)
        jpeg_bytes = _make_jpeg_with_exif_orientation(1, img)
        path = str(tmp_path / "orient1.jpg")
        with open(path, "wb") as f:
            f.write(jpeg_bytes)
        result = read_image(path)
        assert result is not None
        assert result.shape[:2] == (40, 60)

    def test_orientation_3_rotate_180(self, tmp_path):
        """Orientation 3 should rotate 180 degrees."""
        img = np.zeros((40, 60, 3), dtype=np.uint8)
        img[:20, :, :] = 255  # top half white
        jpeg_bytes = _make_jpeg_with_exif_orientation(3, img)
        path = str(tmp_path / "orient3.jpg")
        with open(path, "wb") as f:
            f.write(jpeg_bytes)
        result = read_image(path)
        assert result is not None
        assert result.shape[:2] == (40, 60)
        # After 180 rotation, bottom half should now be white (was top)
        assert result[20:, :, :].mean() > 200

    def test_orientation_6_rotate_90cw(self, tmp_path):
        """Orientation 6 should rotate 90° clockwise (h,w swap)."""
        img = _make_test_img(60, 40)
        jpeg_bytes = _make_jpeg_with_exif_orientation(6, img)
        path = str(tmp_path / "orient6.jpg")
        with open(path, "wb") as f:
            f.write(jpeg_bytes)
        result = read_image(path)
        assert result is not None
        # 60x40 → rotate 90 CW → 40x60 (h=60, w=40)
        assert result.shape[:2] == (60, 40)

    def test_orientation_8_rotate_90ccw(self, tmp_path):
        """Orientation 8 should rotate 90° counter-clockwise."""
        img = _make_test_img(60, 40)
        jpeg_bytes = _make_jpeg_with_exif_orientation(8, img)
        path = str(tmp_path / "orient8.jpg")
        with open(path, "wb") as f:
            f.write(jpeg_bytes)
        result = read_image(path)
        assert result is not None
        assert result.shape[:2] == (60, 40)

    def test_png_no_exif_unchanged(self, tmp_path):
        """PNG files have no EXIF, should stay unchanged."""
        img = _make_test_img(60, 40)
        path = str(tmp_path / "test.png")
        write_image(path, img)
        result = read_image(path)
        assert result is not None
        assert result.shape[:2] == (40, 60)

    def test_auto_orient_disabled(self, tmp_path):
        """With auto_orient=False, EXIF rotation should NOT be applied."""
        img = _make_test_img(60, 40)
        jpeg_bytes = _make_jpeg_with_exif_orientation(6, img)
        path = str(tmp_path / "noorient.jpg")
        with open(path, "wb") as f:
            f.write(jpeg_bytes)
        result = read_image(path, auto_orient=False)
        assert result is not None
        # Without auto-orient, shape should be original (40, 60)
        assert result.shape[:2] == (40, 60)

    def test_read_exif_from_bytes(self):
        """Unit test for _read_exif_orientation."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        for orient in [1, 2, 3, 4, 5, 6, 7, 8]:
            jpeg_bytes = _make_jpeg_with_exif_orientation(orient, img)
            result = _read_exif_orientation(jpeg_bytes)
            assert result == orient, f"Expected {orient}, got {result}"

    def test_non_jpeg_returns_zero(self):
        """Non-JPEG data should return orientation 0."""
        assert _read_exif_orientation(b"\x89PNG\r\n\x1a\n") == 0
        assert _read_exif_orientation(b"\x00\x00") == 0
