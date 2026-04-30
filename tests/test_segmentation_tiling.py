"""Tests for core.segmentation_tiling module."""
import json
import os

import numpy as np
import pytest

from core.image_io import write_image
from core.segmentation_tiling import tile_segmentation_dataset, tile_segmentation_single


def _make_img(h=200, w=200, channels=3):
    """Create a random test image."""
    if channels == 1:
        return np.random.randint(0, 255, (h, w), dtype=np.uint8)
    return np.random.randint(0, 255, (h, w, channels), dtype=np.uint8)


def _make_labelme_json(path, base_name, h, w, shapes):
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


def _sample_shapes():
    """Return a list of sample polygon shapes that sit in the top-left quadrant."""
    return [
        {
            "label": "obj",
            "points": [[10, 10], [80, 10], [80, 80], [10, 80]],
            "shape_type": "polygon",
            "flags": {},
            "group_id": None,
        }
    ]


def _count_files(directory, ext):
    """Count files with a given extension in a directory (non-recursive)."""
    if not os.path.isdir(directory):
        return 0
    return sum(1 for f in os.listdir(directory) if f.endswith(ext))


# ---------- tile_segmentation_dataset ----------


class TestTileSegmentationDataset:
    def test_basic_tiling_creates_output(self, tmp_path):
        """Tiling a 200x200 image with 100x100 tiles should produce 4 tiles."""
        img_dir = tmp_path / "images"
        ann_dir = tmp_path / "annotations"
        output_dir = tmp_path / "output"
        img_dir.mkdir()
        ann_dir.mkdir()

        img = _make_img(200, 200)
        write_image(str(img_dir / "sample.png"), img)
        _make_labelme_json(
            str(ann_dir / "sample.json"), "sample", 200, 200, _sample_shapes()
        )

        result = tile_segmentation_dataset(
            str(img_dir), str(ann_dir), str(output_dir), tile_w=100, tile_h=100
        )

        assert result["total_pairs"] == 1
        assert result["total_tiles"] == 4
        assert _count_files(result["output_image_dir"], ".png") == 4
        assert _count_files(result["output_ann_dir"], ".json") == 4

    def test_tile_annotation_has_shapes(self, tmp_path):
        """Tiles that overlap the annotation should contain shapes in their JSON."""
        img_dir = tmp_path / "images"
        ann_dir = tmp_path / "annotations"
        output_dir = tmp_path / "output"
        img_dir.mkdir()
        ann_dir.mkdir()

        img = _make_img(200, 200)
        write_image(str(img_dir / "annotated.png"), img)
        # Shape in top-left corner (0-80 range), should appear in tile (0,0)
        _make_labelme_json(
            str(ann_dir / "annotated.json"),
            "annotated",
            200,
            200,
            _sample_shapes(),
        )

        result = tile_segmentation_dataset(
            str(img_dir), str(ann_dir), str(output_dir), tile_w=100, tile_h=100
        )

        ann_out = result["output_ann_dir"]
        # The top-left tile should contain shapes
        top_left_json = os.path.join(ann_out, "annotated_x0000_y0000_w100_h100.json")
        assert os.path.isfile(top_left_json)
        with open(top_left_json) as f:
            ann = json.load(f)
        assert len(ann["shapes"]) >= 1
        assert ann["shapes"][0]["label"] == "obj"

    def test_tile_annotation_structure(self, tmp_path):
        """Each tile annotation should have the correct Labelme structure."""
        img_dir = tmp_path / "images"
        ann_dir = tmp_path / "annotations"
        output_dir = tmp_path / "output"
        img_dir.mkdir()
        ann_dir.mkdir()

        img = _make_img(100, 100)
        write_image(str(img_dir / "struct.png"), img)
        _make_labelme_json(
            str(ann_dir / "struct.json"), "struct", 100, 100, _sample_shapes()
        )

        tile_segmentation_dataset(
            str(img_dir), str(ann_dir), str(output_dir), tile_w=100, tile_h=100
        )

        ann_out = os.path.join(str(output_dir), "annotations")
        json_files = [f for f in os.listdir(ann_out) if f.endswith(".json")]
        assert len(json_files) >= 1

        with open(os.path.join(ann_out, json_files[0])) as f:
            ann = json.load(f)

        assert "version" in ann
        assert "shapes" in ann
        assert "imagePath" in ann
        assert "imageHeight" in ann
        assert "imageWidth" in ann
        assert ann["imageHeight"] == 100
        assert ann["imageWidth"] == 100

    def test_multiple_pairs(self, tmp_path):
        """Multiple image-annotation pairs should all be tiled."""
        img_dir = tmp_path / "images"
        ann_dir = tmp_path / "annotations"
        output_dir = tmp_path / "output"
        img_dir.mkdir()
        ann_dir.mkdir()

        for i in range(3):
            img = _make_img(100, 100)
            write_image(str(img_dir / f"pair{i}.png"), img)
            _make_labelme_json(
                str(ann_dir / f"pair{i}.json"),
                f"pair{i}",
                100,
                100,
                _sample_shapes(),
            )

        result = tile_segmentation_dataset(
            str(img_dir), str(ann_dir), str(output_dir), tile_w=100, tile_h=100
        )

        assert result["total_pairs"] == 3
        assert result["total_tiles"] == 3  # each 100x100 image -> 1 tile

    def test_unpaired_files_ignored(self, tmp_path):
        """Images without a matching JSON or vice versa should be ignored."""
        img_dir = tmp_path / "images"
        ann_dir = tmp_path / "annotations"
        output_dir = tmp_path / "output"
        img_dir.mkdir()
        ann_dir.mkdir()

        # Only image, no annotation
        img = _make_img(100, 100)
        write_image(str(img_dir / "no_ann.png"), img)

        result = tile_segmentation_dataset(
            str(img_dir), str(ann_dir), str(output_dir), tile_w=100, tile_h=100
        )

        assert result["total_pairs"] == 0
        assert result["total_tiles"] == 0


class TestTileSegmentationDatasetDiscardEmpty:
    def test_discard_empty_skips_tiles_without_annotations(self, tmp_path):
        """Tiles without annotations should be skipped when discard_empty=True."""
        img_dir = tmp_path / "images"
        ann_dir = tmp_path / "annotations"
        output_dir = tmp_path / "output"
        img_dir.mkdir()
        ann_dir.mkdir()

        # 200x200 image with a shape only in the top-left quadrant
        img = _make_img(200, 200)
        write_image(str(img_dir / "sparse.png"), img)
        _make_labelme_json(
            str(ann_dir / "sparse.json"),
            "sparse",
            200,
            200,
            _sample_shapes(),  # shape at (10,10)-(80,80), only in top-left tile
        )

        result = tile_segmentation_dataset(
            str(img_dir),
            str(ann_dir),
            str(output_dir),
            tile_w=100,
            tile_h=100,
            discard_empty=True,
        )

        # Only the top-left tile has annotations; the other 3 should be skipped
        assert result["skipped_empty"] == 3
        assert result["total_tiles"] == 1
        assert _count_files(result["output_image_dir"], ".png") == 1

    def test_discard_empty_false_keeps_all(self, tmp_path):
        """Without discard_empty, all tiles should be kept even if empty."""
        img_dir = tmp_path / "images"
        ann_dir = tmp_path / "annotations"
        output_dir = tmp_path / "output"
        img_dir.mkdir()
        ann_dir.mkdir()

        img = _make_img(200, 200)
        write_image(str(img_dir / "all.png"), img)
        _make_labelme_json(
            str(ann_dir / "all.json"),
            "all",
            200,
            200,
            _sample_shapes(),
        )

        result = tile_segmentation_dataset(
            str(img_dir),
            str(ann_dir),
            str(output_dir),
            tile_w=100,
            tile_h=100,
            discard_empty=False,
        )

        assert result["skipped_empty"] == 0
        assert result["total_tiles"] == 4


class TestTileSegmentationDatasetDiscardIncomplete:
    def test_discard_incomplete_skips_edge_tiles(self, tmp_path):
        """Edge tiles smaller than tile_w x tile_h should be skipped."""
        img_dir = tmp_path / "images"
        ann_dir = tmp_path / "annotations"
        output_dir = tmp_path / "output"
        img_dir.mkdir()
        ann_dir.mkdir()

        # 150x150 image with 100x100 tiles: produces 1 full tile + 2 partial edge tiles
        img = _make_img(150, 150)
        write_image(str(img_dir / "incomplete.png"), img)
        # Place shape in center so all tiles potentially have annotations
        shapes = [
            {
                "label": "obj",
                "points": [[50, 50], [100, 50], [100, 100], [50, 100]],
                "shape_type": "polygon",
                "flags": {},
                "group_id": None,
            }
        ]
        _make_labelme_json(
            str(ann_dir / "incomplete.json"), "incomplete", 150, 150, shapes
        )

        result = tile_segmentation_dataset(
            str(img_dir),
            str(ann_dir),
            str(output_dir),
            tile_w=100,
            tile_h=100,
            discard_incomplete=True,
        )

        assert result["skipped_incomplete"] > 0
        # Only full 100x100 tiles at (0,0) should be kept
        assert result["total_tiles"] >= 1

    def test_discard_incomplete_false_keeps_edge_tiles(self, tmp_path):
        """Without discard_incomplete, edge tiles should be kept (padded)."""
        img_dir = tmp_path / "images"
        ann_dir = tmp_path / "annotations"
        output_dir = tmp_path / "output"
        img_dir.mkdir()
        ann_dir.mkdir()

        img = _make_img(150, 150)
        write_image(str(img_dir / "padded.png"), img)
        _make_labelme_json(
            str(ann_dir / "padded.json"),
            "padded",
            150,
            150,
            _sample_shapes(),
        )

        result = tile_segmentation_dataset(
            str(img_dir),
            str(ann_dir),
            str(output_dir),
            tile_w=100,
            tile_h=100,
            discard_incomplete=False,
        )

        assert result["skipped_incomplete"] == 0
        assert result["total_tiles"] == 4


class TestTileSegmentationDatasetOverlap:
    def test_overlap_produces_more_tiles(self, tmp_path):
        """Using overlap should produce more tiles than without."""
        img_dir = tmp_path / "images"
        ann_dir = tmp_path / "annotations"
        img_dir.mkdir()
        ann_dir.mkdir()

        img = _make_img(200, 200)
        write_image(str(img_dir / "overlap.png"), img)
        _make_labelme_json(
            str(ann_dir / "overlap.json"), "overlap", 200, 200, _sample_shapes()
        )

        result_no_overlap = tile_segmentation_dataset(
            str(img_dir),
            str(ann_dir),
            str(tmp_path / "out1"),
            tile_w=100,
            tile_h=100,
            overlap=0,
        )
        result_with_overlap = tile_segmentation_dataset(
            str(img_dir),
            str(ann_dir),
            str(tmp_path / "out2"),
            tile_w=100,
            tile_h=100,
            overlap=30,
        )

        assert result_with_overlap["total_tiles"] > result_no_overlap["total_tiles"]


class TestTileSegmentationDatasetProgress:
    def test_progress_callback(self, tmp_path):
        """Progress callback should be invoked for each pair."""
        img_dir = tmp_path / "images"
        ann_dir = tmp_path / "annotations"
        output_dir = tmp_path / "output"
        img_dir.mkdir()
        ann_dir.mkdir()

        for i in range(3):
            img = _make_img(100, 100)
            write_image(str(img_dir / f"prog{i}.png"), img)
            _make_labelme_json(
                str(ann_dir / f"prog{i}.json"),
                f"prog{i}",
                100,
                100,
                _sample_shapes(),
            )

        calls = []
        tile_segmentation_dataset(
            str(img_dir),
            str(ann_dir),
            str(output_dir),
            tile_w=100,
            tile_h=100,
            progress_callback=lambda cur, total: calls.append((cur, total)),
        )

        assert len(calls) == 3
        assert calls[-1] == (3, 3)


# ---------- tile_segmentation_single ----------


class TestTileSegmentationSingle:
    def test_single_pair_tiling(self, tmp_path):
        """Tile a single image+annotation pair and verify output."""
        img_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        img_dir.mkdir()

        img = _make_img(200, 200)
        img_path = str(img_dir / "single.png")
        write_image(img_path, img)

        ann_path = str(img_dir / "single.json")
        _make_labelme_json(ann_path, "single", 200, 200, _sample_shapes())

        result = tile_segmentation_single(
            img_path, ann_path, str(output_dir), tile_w=100, tile_h=100
        )

        assert result["total_pairs"] == 1
        assert result["total_tiles"] == 4
        assert _count_files(result["output_image_dir"], ".png") == 4
        assert _count_files(result["output_ann_dir"], ".json") == 4

    def test_single_with_discard_empty(self, tmp_path):
        """Single tiling with discard_empty should skip tiles without annotations."""
        img_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        img_dir.mkdir()

        img = _make_img(200, 200)
        img_path = str(img_dir / "sparse.png")
        write_image(img_path, img)

        ann_path = str(img_dir / "sparse.json")
        _make_labelme_json(ann_path, "sparse", 200, 200, _sample_shapes())

        result = tile_segmentation_single(
            img_path,
            ann_path,
            str(output_dir),
            tile_w=100,
            tile_h=100,
            discard_empty=True,
        )

        assert result["skipped_empty"] == 3
        assert result["total_tiles"] == 1

    def test_single_grayscale_image(self, tmp_path):
        """Should work correctly with grayscale (single-channel) images."""
        img_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        img_dir.mkdir()

        img = _make_img(100, 100, channels=1)
        img_path = str(img_dir / "gray.png")
        write_image(img_path, img)

        ann_path = str(img_dir / "gray.json")
        _make_labelme_json(ann_path, "gray", 100, 100, _sample_shapes())

        result = tile_segmentation_single(
            img_path, ann_path, str(output_dir), tile_w=100, tile_h=100
        )

        assert result["total_tiles"] >= 1
