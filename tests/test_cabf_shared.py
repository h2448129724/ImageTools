import json
import os
import tempfile
from pathlib import Path

# Ensure the shared layer is importable
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project_modules" / "cabf_pipeline" / "shared"))

import pytest
from cabf.constants import IMAGE_SUFFIXES, MASTER_SCHEMA_VERSION, POINT_LABEL_ALIASES
from cabf.io import read_json, write_json, iter_image_files, iter_json_files
from cabf.schema import (
    make_empty_master_annotation,
    is_labelme_point_annotation,
    load_labelme_points,
    convert_labelme_to_master,
    master_to_labelme,
)
from cabf.normalize import (
    normalize_points_for_editor,
    normalize_edges_for_editor,
    normalize_master_annotation,
)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_schema_version(self):
        assert MASTER_SCHEMA_VERSION == "1.2"

    def test_image_suffixes_contains_common_formats(self):
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]:
            assert ext in IMAGE_SUFFIXES

    def test_point_label_aliases(self):
        assert "sew" in POINT_LABEL_ALIASES
        assert "keypoint" in POINT_LABEL_ALIASES


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------

class TestIO:
    def test_read_write_json(self, tmp_path):
        data = {"hello": "world", "num": 42}
        path = tmp_path / "test.json"
        write_json(path, data)
        assert read_json(path) == data

    def test_write_json_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "a" / "b" / "c" / "test.json"
        write_json(path, {"x": 1})
        assert path.exists()
        assert read_json(path) == {"x": 1}

    def test_iter_image_files(self, tmp_path):
        (tmp_path / "a.png").write_bytes(b"\x89PNG")
        (tmp_path / "b.jpg").write_bytes(b"\xff\xd8\xff")
        (tmp_path / "c.txt").write_text("not an image")
        images = iter_image_files(tmp_path)
        stems = [p.stem for p in images]
        assert "a" in stems
        assert "b" in stems
        assert "c" not in stems

    def test_iter_json_files(self, tmp_path):
        (tmp_path / "a.json").write_text("{}")
        (tmp_path / "b.json").write_text("[]")
        (tmp_path / "c.txt").write_text("not json")
        jsons = iter_json_files(tmp_path)
        stems = [p.stem for p in jsons]
        assert "a" in stems
        assert "b" in stems
        assert "c" not in stems

    def test_iter_image_files_missing_dir(self):
        with pytest.raises(FileNotFoundError):
            iter_image_files("/nonexistent_dir_xyz")

    def test_iter_json_files_missing_dir(self):
        with pytest.raises(FileNotFoundError):
            iter_json_files("/nonexistent_dir_xyz")


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

class TestSchema:
    def test_make_empty_master_annotation(self):
        ann = make_empty_master_annotation("img.png", 256, 256, "sample_001")
        assert ann["schema_version"] == "1.2"
        assert ann["sample_id"] == "sample_001"
        assert ann["image_path"] == "img.png"
        assert ann["image_size"] == {"width": 256, "height": 256}
        assert ann["roi"] is None
        assert ann["spacing_hint"] is None
        assert ann["points"] == []
        assert ann["edges"] == []
        assert ann["segments"] == []
        assert ann["metadata"] == {}

    def test_is_labelme_point_annotation_true(self):
        data = {"shapes": [{"shape_type": "point", "points": [[10, 20]]}]}
        assert is_labelme_point_annotation(data) is True

    def test_is_labelme_point_annotation_false_no_shapes(self):
        assert is_labelme_point_annotation({}) is False
        assert is_labelme_point_annotation("not a dict") is False

    def test_load_labelme_points(self):
        data = {
            "shapes": [
                {"shape_type": "point", "label": "sew", "points": [[10.5, 20.3]], "score": 0.9},
                {"shape_type": "point", "label": "keypoint", "points": [[30, 40]]},
                {"shape_type": "rectangle", "label": "sew", "points": [[0, 0], [100, 100]]},
            ]
        }
        points = load_labelme_points(data)
        assert len(points) == 2
        assert points[0]["x"] == 10.5
        assert points[0]["source"] == "labelme_point"
        assert points[1]["x"] == 30.0

    def test_convert_labelme_to_master(self):
        data = {
            "imageWidth": 320,
            "imageHeight": 240,
            "shapes": [
                {"shape_type": "point", "label": "sew", "points": [[10, 20]]},
            ],
        }
        ann = convert_labelme_to_master(data, "test.png", "test")
        assert ann["image_size"] == {"width": 320, "height": 240}
        assert len(ann["points"]) == 1
        assert ann["metadata"]["source"] == "labelme_point"

    def test_master_to_labelme_roundtrip(self):
        master = make_empty_master_annotation("img.png", 256, 256, "s1")
        master["points"] = [
            {"id": 0, "x": 10.0, "y": 20.0, "score": 1.0, "source": "manual"},
            {"id": 1, "x": 30.0, "y": 40.0, "score": 0.8, "source": "manual"},
        ]
        labelme = master_to_labelme(master)
        assert labelme["imageWidth"] == 256
        assert labelme["imageHeight"] == 256
        assert len(labelme["shapes"]) == 2
        assert labelme["shapes"][0]["shape_type"] == "point"


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_normalize_points_for_editor_basic(self):
        raw = [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0, "id": 10}]
        result = normalize_points_for_editor(raw)
        assert len(result) == 2
        assert result[0]["id"] == 0
        assert result[1]["id"] == 10
        assert result[0]["x"] == 1.0

    def test_normalize_points_for_editor_fills_defaults(self):
        raw = [{"x": 5, "y": 6}]
        result = normalize_points_for_editor(raw)
        assert result[0]["score"] == 1.0
        assert result[0]["source"] == "manual"

    def test_normalize_edges_for_editor_basic(self):
        raw = [{"src": 0, "dst": 1}, {"src": 1, "dst": 2}]
        result = normalize_edges_for_editor(raw)
        assert len(result) == 2
        assert result[0]["label"] == 1

    def test_normalize_edges_for_editor_dedup(self):
        raw = [{"src": 0, "dst": 1}, {"src": 1, "dst": 0}]
        result = normalize_edges_for_editor(raw)
        assert len(result) == 1

    def test_normalize_edges_for_editor_skip_self_loop(self):
        raw = [{"src": 0, "dst": 0}, {"src": 1, "dst": 2}]
        result = normalize_edges_for_editor(raw)
        assert len(result) == 1
        assert result[0]["src"] == 1

    def test_normalize_edges_for_editor_skip_invalid(self):
        raw = [{"src": -1, "dst": 1}, {"src": 0, "dst": -1}]
        result = normalize_edges_for_editor(raw)
        assert len(result) == 0

    def test_normalize_master_annotation_basic(self):
        raw = {
            "points": [{"x": 1, "y": 2}, {"x": 3, "y": 4}],
            "edges": [{"src": 0, "dst": 1}],
        }
        ann, issues = normalize_master_annotation(raw, sample_id="s1", image_path="s1.png")
        assert ann["sample_id"] == "s1"
        assert len(ann["points"]) == 2
        assert len(ann["edges"]) == 1
        assert ann["schema_version"] == "1.2"
        assert issues == []

    def test_normalize_master_annotation_labelme_auto_convert(self):
        labelme = {
            "shapes": [{"shape_type": "point", "label": "sew", "points": [[10, 20]]}],
            "imageWidth": 256,
            "imageHeight": 256,
        }
        ann, issues = normalize_master_annotation(labelme, sample_id="s1", image_path="s1.png")
        assert len(ann["points"]) == 1
        assert ann["points"][0]["source"] == "labelme_point"

    def test_normalize_master_annotation_dedup_point_ids(self):
        raw = {
            "points": [{"id": 0, "x": 1, "y": 2}, {"id": 0, "x": 3, "y": 4}],
            "edges": [],
        }
        ann, issues = normalize_master_annotation(raw, sample_id="s1", image_path="s1.png")
        assert len(ann["points"]) == 2
        ids = [p["id"] for p in ann["points"]]
        assert len(set(ids)) == 2
        assert any("重复" in i for i in issues)

    def test_normalize_master_annotation_skip_dangling_edge(self):
        raw = {
            "points": [{"id": 0, "x": 1, "y": 2}],
            "edges": [{"src": 0, "dst": 99}],
        }
        ann, issues = normalize_master_annotation(raw, sample_id="s1", image_path="s1.png")
        assert len(ann["edges"]) == 0
        assert any("不存在" in i for i in issues)

    def test_normalize_master_annotation_skip_self_loop(self):
        raw = {
            "points": [{"id": 0, "x": 1, "y": 2}],
            "edges": [{"src": 0, "dst": 0}],
        }
        ann, issues = normalize_master_annotation(raw, sample_id="s1", image_path="s1.png")
        assert len(ann["edges"]) == 0
        assert any("自环" in i for i in issues)

    def test_normalize_master_annotation_skip_duplicate_edge(self):
        raw = {
            "points": [{"id": 0, "x": 1, "y": 2}, {"id": 1, "x": 3, "y": 4}],
            "edges": [{"src": 0, "dst": 1}, {"src": 1, "dst": 0}],
        }
        ann, issues = normalize_master_annotation(raw, sample_id="s1", image_path="s1.png")
        assert len(ann["edges"]) == 1
        assert any("重复" in i for i in issues)

    def test_normalize_master_annotation_fills_image_path(self):
        raw = {"points": [], "edges": []}
        ann, issues = normalize_master_annotation(raw, sample_id="s1")
        assert ann["image_path"] == "s1.png"
        assert any("回填" in i for i in issues)

    def test_normalize_master_annotation_invalid_input(self):
        with pytest.raises(ValueError, match="有效 JSON"):
            normalize_master_annotation("not a dict", sample_id="s1")


# ---------------------------------------------------------------------------
# config_model
# ---------------------------------------------------------------------------

class TestConfigModel:
    def test_apply_defaults_fills_missing_keys(self):
        from project_modules.cabf_pipeline.config_model import apply_defaults
        partial = {"img_tools_root": "/some/path"}
        result = apply_defaults(partial)
        assert result["img_tools_root"] == "/some/path"
        assert "weights" in result
        assert result["weights"]["sew_point_onnx"] == ""
        assert "outputs" in result

    def test_apply_defaults_preserves_nested(self):
        from project_modules.cabf_pipeline.config_model import apply_defaults
        data = {
            "img_tools_root": "/x",
            "weights": {"sew_point_onnx": "/model.onnx"},
        }
        result = apply_defaults(data)
        assert result["weights"]["sew_point_onnx"] == "/model.onnx"
        assert result["weights"]["sew_point_connector_pth"] == ""

    def test_load_config_missing_file(self, tmp_path):
        from project_modules.cabf_pipeline.config_model import load_config
        cfg = load_config(tmp_path / "nonexistent.json")
        assert "weights" in cfg
        assert "outputs" in cfg

    def test_get_set_nested(self):
        from project_modules.cabf_pipeline.config_model import get_nested, set_nested
        data = {"weights": {"sew_point_onnx": ""}}
        assert get_nested(data, "weights.sew_point_onnx") == ""
        set_nested(data, "weights.sew_point_onnx", "/path/to/model.onnx")
        assert data["weights"]["sew_point_onnx"] == "/path/to/model.onnx"
