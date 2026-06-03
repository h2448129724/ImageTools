"""Tests for core.auto_tile_crop."""
import os
import numpy as np
import cv2
import pytest
from core.auto_tile_crop import compute_tile_grid, batch_tile_crop


class TestComputeTileGrid:
    def test_exact_division(self):
        tiles = compute_tile_grid(512, 512, 256, 256)
        assert len(tiles) == 4
        assert tiles[0] == (0, 0, 256, 256, 0, 0)
        assert tiles[3] == (256, 256, 256, 256, 1, 1)

    def test_non_exact_with_overlap(self):
        tiles = compute_tile_grid(300, 300, 256, 256)
        assert len(tiles) == 4
        # last col shifted to 300-256=44
        assert tiles[1][0] == 44
        # last row shifted to 300-256=44
        assert tiles[2][1] == 44

    def test_smaller_than_tile(self):
        tiles = compute_tile_grid(200, 200, 256, 256)
        assert len(tiles) == 1
        assert tiles[0] == (0, 0, 256, 256, 0, 0)

    def test_single_tile_exact(self):
        tiles = compute_tile_grid(256, 256, 256, 256)
        assert len(tiles) == 1


class TestBatchTileCrop:
    def test_basic(self, tmp_path):
        src = str(tmp_path / "src")
        out = str(tmp_path / "out")
        os.makedirs(src)
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(src, "test.png"), img)

        total = batch_tile_crop(src, out, 256, 256)
        assert total == 4
        assert os.path.exists(os.path.join(out, "test_tile_0_0.png"))
        assert os.path.exists(os.path.join(out, "test_tile_1_1.png"))

    def test_non_exact(self, tmp_path):
        src = str(tmp_path / "src")
        out = str(tmp_path / "out")
        os.makedirs(src)
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(src, "test.png"), img)

        total = batch_tile_crop(src, out, 256, 256)
        assert total == 4

    def test_non_exact_without_overlap_skips_small_tiles(self, tmp_path):
        src = str(tmp_path / "src")
        out = str(tmp_path / "out")
        os.makedirs(src)
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(src, "test.png"), img)

        total = batch_tile_crop(src, out, 256, 256, allow_overlap=False)
        assert total == 1
        assert os.path.exists(os.path.join(out, "test_tile_0_0.png"))
        assert not os.path.exists(os.path.join(out, "test_tile_0_1.png"))

    def test_empty_dir(self, tmp_path):
        src = str(tmp_path / "src")
        out = str(tmp_path / "out")
        os.makedirs(src)
        total = batch_tile_crop(src, out, 256, 256)
        assert total == 0
