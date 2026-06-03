"""Tests for core.batch_crop."""
import os
import numpy as np
import cv2
import pytest
from core.batch_crop import batch_crop, crop_single_image


def _write_test_images(tmp, count=3, w=100, h=80):
    for i in range(count):
        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(tmp, f"img_{i:03d}.png"), img)


class TestBatchCrop:
    def test_single_rect(self, tmp_path):
        src = str(tmp_path / "src")
        out = str(tmp_path / "out")
        os.makedirs(src)
        _write_test_images(src, 3, 100, 80)

        total = batch_crop(src, [(10, 20, 60, 70)], 100, 80, out)
        assert total == 3
        assert os.path.exists(os.path.join(out, "img_000_roi_1.png"))

    def test_multiple_rects(self, tmp_path):
        src = str(tmp_path / "src")
        out = str(tmp_path / "out")
        os.makedirs(src)
        _write_test_images(src, 2, 100, 80)

        rects = [(0, 0, 50, 40), (50, 40, 100, 80)]
        total = batch_crop(src, rects, 100, 80, out)
        assert total == 4
        assert os.path.exists(os.path.join(out, "img_000_roi_1.png"))
        assert os.path.exists(os.path.join(out, "img_000_roi_2.png"))

    def test_scaled_images(self, tmp_path):
        src = str(tmp_path / "src")
        out = str(tmp_path / "out")
        os.makedirs(src)
        # ref image is 100x80, actual image is 200x160
        img = np.zeros((160, 200, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(src, "big.png"), img)

        total = batch_crop(src, [(10, 10, 50, 40)], 100, 80, out)
        assert total == 1
        # scaled: (20, 20, 100, 80)
        cropped = cv2.imread(os.path.join(out, "big_roi_1.png"))
        assert cropped is not None
        assert cropped.shape[0] == 60  # 80 - 20
        assert cropped.shape[1] == 80  # 100 - 20

    def test_empty_input(self, tmp_path):
        src = str(tmp_path / "src")
        out = str(tmp_path / "out")
        os.makedirs(src)
        total = batch_crop(src, [(0, 0, 50, 50)], 100, 100, out)
        assert total == 0

    def test_single_image_crop(self, tmp_path):
        src = str(tmp_path / "src")
        out = str(tmp_path / "out")
        os.makedirs(src)
        _write_test_images(src, 1, 100, 80)

        image_path = os.path.join(src, "img_000.png")
        total = crop_single_image(image_path, [(10, 10, 50, 40), (50, 40, 90, 70)], 100, 80, out)

        assert total == 2
        assert os.path.exists(os.path.join(out, "img_000_roi_1.png"))
        assert os.path.exists(os.path.join(out, "img_000_roi_2.png"))
