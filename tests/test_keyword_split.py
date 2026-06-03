"""Tests for core.keyword_split."""
import os
import pytest
from core.keyword_split import classify_by_keywords
from utils.helpers import get_image_files


def _write_images(tmp, names):
    import numpy as np
    import cv2
    for name in names:
        path = os.path.join(tmp, name)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(path, img)


class TestClassifyByKeywords:
    def test_basic_split(self, tmp_path):
        src = str(tmp_path / "src")
        out = str(tmp_path / "out")
        os.makedirs(src)
        _write_images(src, ["top_001.png", "bottom_001.png", "top_002.png"])

        counts = classify_by_keywords(src, ["top", "bottom"], out)
        assert counts["top"] == 2
        assert counts["bottom"] == 1
        assert counts["_unsorted"] == 0
        assert os.path.exists(os.path.join(out, "top", "top_001.png"))
        assert os.path.exists(os.path.join(out, "bottom", "bottom_001.png"))

    def test_unsorted(self, tmp_path):
        src = str(tmp_path / "src")
        out = str(tmp_path / "out")
        os.makedirs(src)
        _write_images(src, ["unknown_001.png"])

        counts = classify_by_keywords(src, ["top", "bottom"], out)
        assert counts["_unsorted"] == 1
        assert os.path.exists(os.path.join(out, "_unsorted", "unknown_001.png"))

    def test_case_insensitive(self, tmp_path):
        src = str(tmp_path / "src")
        out = str(tmp_path / "out")
        os.makedirs(src)
        _write_images(src, ["TOP_001.png"])

        counts = classify_by_keywords(src, ["top"], out)
        assert counts["top"] == 1

    def test_dry_run(self, tmp_path):
        src = str(tmp_path / "src")
        out = str(tmp_path / "out")
        os.makedirs(src)
        _write_images(src, ["top_001.png", "bottom_001.png"])

        counts = classify_by_keywords(src, ["top", "bottom"], out, dry_run=True)
        assert counts["top"] == 1
        assert counts["bottom"] == 1
        assert not os.path.exists(out)
