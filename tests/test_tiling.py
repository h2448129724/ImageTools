"""Tests for core.tiling module."""
import json
import os
import numpy as np
import pytest
from core.image_io import write_image
from core.tiling import tile_image, tile_image_file, grid_tile, tile_directory


def _make_img(w=100, h=100, channels=3):
    return np.random.randint(0, 255, (h, w, channels), dtype=np.uint8)


# ---------- tile_image ----------

class TestTileImage:
    def test_correct_number_of_tiles_no_overlap(self):
        """A 100x100 image with 50x50 tiles should produce 4 tiles."""
        img = _make_img(100, 100)
        tiles = tile_image(img, 50, 50, overlap=0)
        assert len(tiles) == 4

    def test_overlap_produces_more_tiles(self):
        """With overlap > 0, the number of tiles should increase."""
        img = _make_img(300, 300)
        tiles_no_overlap = tile_image(img, 50, 50, overlap=0)
        tiles_with_overlap = tile_image(img, 50, 50, overlap=20)
        assert len(tiles_with_overlap) > len(tiles_no_overlap)

    def test_overlap_tile_positions(self):
        """With step=40 (tile=50, overlap=10), second tile should start at x=40."""
        img = _make_img(100, 100)
        tiles = tile_image(img, 50, 50, overlap=10)
        # Collect x positions from first row
        xs = sorted({t[1] for t in tiles})
        assert xs[0] == 0
        assert xs[1] == 40

    def test_discard_incomplete_true(self):
        """Image not evenly divisible should discard edge tiles."""
        img = _make_img(90, 90)  # 90 / 50 = 1 full tile, 40px remainder discarded
        tiles = tile_image(img, 50, 50, discard_incomplete=True)
        assert len(tiles) == 1  # only one full 50x50 tile in each dimension

    def test_discard_incomplete_false_pads(self):
        """Image not evenly divisible should pad edge tiles when discard=False."""
        img = _make_img(90, 90)
        tiles = tile_image(img, 50, 50, discard_incomplete=False)
        assert len(tiles) == 4  # 2x2 grid with padding
        # The last tile in each dimension should be padded to 50x50
        for tile_img, x, y, tw, th in tiles:
            assert tw == 50
            assert th == 50

    def test_tile_content_matches_original_region(self):
        """Pixel values of each tile should match the corresponding region in the source."""
        img = _make_img(100, 100)
        tiles = tile_image(img, 50, 50, overlap=0)
        for tile_img, x, y, tw, th in tiles:
            np.testing.assert_array_equal(tile_img, img[y:y + th, x:x + tw])

    def test_single_tile_covers_small_image(self):
        """An image smaller than tile size produces one padded tile when discard=False."""
        img = _make_img(30, 30)
        tiles = tile_image(img, 50, 50, discard_incomplete=False)
        assert len(tiles) == 1
        assert tiles[0][3] == 50  # w
        assert tiles[0][4] == 50  # h

    def test_grayscale_image(self):
        """Should work with single-channel (grayscale) images."""
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        tiles = tile_image(img, 50, 50)
        assert len(tiles) == 4
        for tile_img, x, y, tw, th in tiles:
            assert len(tile_img.shape) == 2


# ---------- tile_image_file ----------

class TestTileImageFile:
    def test_output_files_created(self, tmp_path):
        img = _make_img(100, 100)
        src = str(tmp_path / "source.png")
        write_image(src, img)

        result = tile_image_file(src, str(tmp_path / "tiles"), 50, 50)

        assert "error" not in result
        assert result["tiles"] == 4
        output_dir = result["output_dir"]
        # Check that tile image files exist
        files = [f for f in os.listdir(output_dir) if f.endswith(".png")]
        assert len(files) == 4

    def test_coords_json_created(self, tmp_path):
        img = _make_img(100, 100)
        src = str(tmp_path / "source.png")
        write_image(src, img)

        result = tile_image_file(src, str(tmp_path / "tiles"), 50, 50)

        assert result["coords"]
        assert os.path.isfile(result["coords"])
        with open(result["coords"]) as f:
            data = json.load(f)
        assert data["source"] == src
        assert data["tile_w"] == 50
        assert data["tile_h"] == 50
        assert len(data["tiles"]) == 4
        # Check each tile entry has required keys
        for entry in data["tiles"]:
            assert "file" in entry
            assert "x" in entry
            assert "y" in entry
            assert "w" in entry
            assert "h" in entry

    def test_failed_read_returns_error(self, tmp_path):
        result = tile_image_file("/nonexistent/image.png", str(tmp_path / "tiles"), 50, 50)
        assert "error" in result

    def test_with_prefix(self, tmp_path):
        img = _make_img(100, 100)
        src = str(tmp_path / "source.png")
        write_image(src, img)

        result = tile_image_file(src, str(tmp_path / "tiles"), 50, 50, prefix="tile_")

        output_dir = result["output_dir"]
        files = [f for f in os.listdir(output_dir) if f.endswith(".png")]
        for f in files:
            assert f.startswith("tile_")


# ---------- grid_tile ----------

class TestGridTile:
    def test_correct_row_col_count(self):
        img = _make_img(100, 100)
        tiles = grid_tile(img, rows=2, cols=3)
        assert len(tiles) == 2 * 3  # 6 tiles

    def test_tile_dimensions(self):
        img = _make_img(100, 100)
        tiles = grid_tile(img, rows=2, cols=5)
        # Each tile should be 50x20
        for tile_img, x, y, tw, th in tiles:
            assert th == 50
            assert tw == 20

    def test_single_grid_tile(self):
        img = _make_img(100, 100)
        tiles = grid_tile(img, rows=1, cols=1)
        assert len(tiles) == 1
        tile_img, x, y, tw, th = tiles[0]
        assert tw == 100
        assert th == 100

    def test_discard_incomplete_false_keeps_partial(self):
        """If image dimensions aren't evenly divisible, partial tiles are kept."""
        img = _make_img(90, 90)
        # 90 / 4 = 22 (integer division), last row/col gets 90 - 3*22 = 24 pixels
        tiles = grid_tile(img, rows=4, cols=4, discard_incomplete=False)
        # With discard_incomplete=False, partial tiles at edges are kept
        assert len(tiles) > 0

    def test_tile_positions_form_grid(self):
        img = _make_img(100, 100)
        tiles = grid_tile(img, rows=2, cols=2)
        positions = set((x, y) for _, x, y, _, _ in tiles)
        assert positions == {(0, 0), (0, 50), (50, 0), (50, 50)}


# ---------- tile_directory ----------

class TestTileDirectory:
    def test_all_files_processed(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        os.makedirs(str(input_dir))

        # Create 3 test images
        for i in range(3):
            write_image(str(input_dir / f"img_{i}.png"), _make_img(100, 100))

        result = tile_directory(str(input_dir), str(output_dir), 50, 50)

        assert result["total_files"] == 3
        assert result["total_tiles"] == 12  # 3 images * 4 tiles each

    def test_empty_directory(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        os.makedirs(str(input_dir))

        result = tile_directory(str(input_dir), str(output_dir), 50, 50)

        assert result["total_files"] == 0
        assert result["total_tiles"] == 0

    def test_with_progress_callback(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        os.makedirs(str(input_dir))

        for i in range(4):
            write_image(str(input_dir / f"img_{i}.png"), _make_img(100, 100))

        progress_calls = []
        def callback(current, total):
            progress_calls.append((current, total))

        tile_directory(str(input_dir), str(output_dir), 50, 50,
                       progress_callback=callback)

        assert len(progress_calls) == 4
        assert progress_calls[-1] == (4, 4)
        # Should be called with monotonically increasing current
        currents = [c[0] for c in progress_calls]
        assert currents == sorted(currents)
