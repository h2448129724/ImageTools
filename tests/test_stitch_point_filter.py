import json

import numpy as np

from core.image_io import write_image
from gui.stitch_point_filter import collect_filter_items, load_points_from_json, move_item_pair


def test_collect_filter_items_skips_unmatched(tmp_path):
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()

    write_image(str(image_dir / "keep.png"), np.zeros((12, 12, 3), dtype=np.uint8))
    write_image(str(image_dir / "skip.png"), np.zeros((12, 12, 3), dtype=np.uint8))
    (label_dir / "keep.json").write_text('{"points":[{"x":1,"y":2}]}', encoding="utf-8")

    items = collect_filter_items(image_dir, label_dir)

    assert len(items) == 1
    assert items[0].image_path.name == "keep.png"
    assert items[0].label_path.name == "keep.json"


def test_load_points_from_json_supports_labelme(tmp_path):
    json_path = tmp_path / "sample.json"
    json_path.write_text(
        json.dumps(
            {
                "imageWidth": 100,
                "imageHeight": 80,
                "shapes": [
                    {"label": "sew", "shape_type": "point", "points": [[10, 20]]},
                    {"label": "ignore", "shape_type": "point", "points": [[30, 40]]},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    points = load_points_from_json(json_path)

    assert points == [{"id": 0, "x": 10.0, "y": 20.0, "score": 1.0, "source": "labelme_point"}]


def test_load_points_from_json_supports_keypoint_label(tmp_path):
    json_path = tmp_path / "sample_keypoint.json"
    json_path.write_text(
        json.dumps(
            {
                "shapes": [
                    {"label": "keypoint", "shape_type": "point", "points": [[42, 38]], "score": 0.9},
                    {"label": "keypoint", "shape_type": "point", "points": [[14, 39]], "score": 0.8},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    points = load_points_from_json(json_path)

    assert points == [
        {"id": 0, "x": 42.0, "y": 38.0, "score": 0.9, "source": "labelme_point"},
        {"id": 1, "x": 14.0, "y": 39.0, "score": 0.8, "source": "labelme_point"},
    ]


def test_move_item_pair_avoids_overwrite(tmp_path):
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    dest_dir = tmp_path / "saved"
    image_dir.mkdir()
    label_dir.mkdir()
    dest_dir.mkdir()

    image_path = image_dir / "sample.png"
    label_path = label_dir / "sample.json"
    write_image(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))
    label_path.write_text('{"points":[{"x":1,"y":2}]}', encoding="utf-8")
    (dest_dir / "sample.png").write_bytes(b"existing-image")
    (dest_dir / "sample.json").write_text('{"existing":true}', encoding="utf-8")

    items = collect_filter_items(image_dir, label_dir)
    moved_image, moved_label = move_item_pair(items[0], dest_dir)

    assert moved_image.name == "sample_1.png"
    assert moved_label.name == "sample_1.json"
    assert moved_image.exists()
    assert moved_label.exists()
    assert not image_path.exists()
    assert not label_path.exists()
