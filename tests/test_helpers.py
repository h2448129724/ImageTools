"""Tests for utils.helpers module."""
import json
import os
import numpy as np
import pytest
from core.image_io import write_image
from utils.helpers import get_image_files, ensure_dir, file_hash, get_output_path, load_json, save_json


class TestGetImageFiles:
    def test_finds_images(self, tmp_path):
        for name in ["a.png", "b.jpg", "c.bmp"]:
            write_image(str(tmp_path / name), np.zeros((10, 10, 3), dtype=np.uint8))
        (tmp_path / "readme.txt").write_text("not an image")
        files = get_image_files(str(tmp_path))
        assert len(files) == 3

    def test_single_file(self, tmp_path):
        p = str(tmp_path / "single.png")
        write_image(p, np.zeros((10, 10, 3), dtype=np.uint8))
        files = get_image_files(p)
        assert len(files) == 1

    def test_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        write_image(str(sub / "deep.png"), np.zeros((10, 10, 3), dtype=np.uint8))
        files = get_image_files(str(tmp_path))
        assert len(files) == 1


class TestEnsureDir:
    def test_creates_nested(self, tmp_path):
        path = str(tmp_path / "a" / "b" / "c")
        ensure_dir(path)
        assert os.path.isdir(path)

    def test_existing_dir(self, tmp_path):
        ensure_dir(str(tmp_path))  # should not raise


class TestFileHash:
    def test_same_content_same_hash(self, tmp_path):
        data = b"hello world"
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(data)
        f2.write_bytes(data)
        assert file_hash(str(f1)) == file_hash(str(f2))

    def test_different_content(self, tmp_path):
        (tmp_path / "a.bin").write_bytes(b"aaa")
        (tmp_path / "b.bin").write_bytes(b"bbb")
        assert file_hash(str(tmp_path / "a.bin")) != file_hash(str(tmp_path / "b.bin"))


class TestGetOutputPath:
    def test_basic(self):
        result = get_output_path("/input/img.png", "/output")
        assert result == "/output/img.png"

    def test_with_suffix(self):
        result = get_output_path("/input/img.png", "/output", suffix="_resized")
        assert result == "/output/img_resized.png"

    def test_with_ext(self):
        result = get_output_path("/input/img.png", "/output", ext=".jpg")
        assert result == "/output/img.jpg"


class TestJsonIO:
    def test_save_and_load(self, tmp_path):
        data = {"key": "值", "num": 42}
        path = str(tmp_path / "test.json")
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data
