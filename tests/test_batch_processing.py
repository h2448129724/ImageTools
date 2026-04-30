"""Tests for core.batch_processing module."""
import os
import numpy as np
import pytest
from core.image_io import write_image
from core.batch_processing import (
    batch_rename, batch_resize, batch_convert_format,
    deduplicate_images, batch_add_border
)


def _make_img(w=50, h=50):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


@pytest.fixture
def img_dir(tmp_path):
    """Create a directory with 5 test images."""
    d = tmp_path / "images"
    d.mkdir()
    for i in range(5):
        write_image(str(d / f"img_{i:03d}.png"), _make_img())
    return str(d)


class TestBatchRename:
    def test_rename(self, img_dir, tmp_path):
        out = str(tmp_path / "out")
        results = batch_rename(img_dir, out, prefix="pic_", start_index=1)
        assert len(results) == 5
        assert os.path.exists(results[0]["dest"])

    def test_rename_progress(self, img_dir, tmp_path):
        progress = []
        batch_rename(img_dir, str(tmp_path / "out"), progress_callback=lambda c, t: progress.append((c, t)))
        assert len(progress) == 5
        assert progress[-1] == (5, 5)

    def test_rename_cancel(self, img_dir, tmp_path):
        cancel_at = [False]

        def cancel():
            if cancel_at[0]:
                return True
            cancel_at[0] = True
            return False

        results = batch_rename(img_dir, str(tmp_path / "out"), cancel_check=cancel)
        assert len(results) < 5


class TestBatchResize:
    def test_resize(self, img_dir, tmp_path):
        out = str(tmp_path / "out")
        count = batch_resize(img_dir, out, width=25)
        assert count == 5

    def test_resize_progress(self, img_dir, tmp_path):
        progress = []
        batch_resize(img_dir, str(tmp_path / "out"), width=25, progress_callback=lambda c, t: progress.append((c, t)))
        assert len(progress) == 5


class TestBatchConvertFormat:
    def test_convert_to_jpg(self, img_dir, tmp_path):
        out = str(tmp_path / "out")
        count = batch_convert_format(img_dir, out, fmt="jpg")
        assert count == 5
        files = os.listdir(out)
        assert all(f.endswith(".jpg") for f in files)


class TestDeduplicate:
    def test_no_duplicates(self, img_dir):
        dupes = deduplicate_images(img_dir)
        assert len(dupes) == 0

    def test_with_duplicates(self, tmp_path):
        d = tmp_path / "dupes"
        d.mkdir()
        img = _make_img()
        write_image(str(d / "a.png"), img)
        write_image(str(d / "b.png"), img)  # same content
        write_image(str(d / "c.png"), _make_img())  # different
        dupes = deduplicate_images(str(d))
        assert len(dupes) == 1


class TestDeduplicatePerceptual:
    def test_exact_copy_detected(self, tmp_path):
        d = tmp_path / "percep"
        d.mkdir()
        img = _make_img()
        write_image(str(d / "a.png"), img)
        write_image(str(d / "b.png"), img)
        dupes = deduplicate_images(str(d), mode="perceptual")
        assert len(dupes) == 1

    def test_resized_copy_detected(self, tmp_path):
        import cv2
        d = tmp_path / "percep"
        d.mkdir()
        # Use a structured image so dHash is consistent across sizes
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:50, :, :] = 255  # top half white
        write_image(str(d / "orig.png"), img)
        resized = cv2.resize(img, (50, 50))
        write_image(str(d / "small.png"), resized)
        dupes = deduplicate_images(str(d), mode="perceptual", similarity_threshold=20)
        assert len(dupes) >= 1

    def test_different_images_not_matched(self, tmp_path):
        d = tmp_path / "percep"
        d.mkdir()
        write_image(str(d / "a.png"), _make_img())
        write_image(str(d / "b.png"), _make_img())
        write_image(str(d / "c.png"), _make_img())
        dupes = deduplicate_images(str(d), mode="perceptual", similarity_threshold=5)
        assert len(dupes) == 0

    def test_cancel_check(self, tmp_path):
        d = tmp_path / "percep"
        d.mkdir()
        for i in range(3):
            write_image(str(d / f"img_{i}.png"), _make_img())
        dupes = deduplicate_images(str(d), mode="perceptual", cancel_check=lambda: True)
        assert len(dupes) == 0


class TestBatchAddBorder:
    def test_add_border(self, img_dir, tmp_path):
        out = str(tmp_path / "out")
        count = batch_add_border(img_dir, out, border_size=10)
        assert count == 5
