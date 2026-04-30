"""Tests for core.dataset_split module."""
import os
import numpy as np
import pytest
from core.image_io import write_image
from core.dataset_split import random_split, stratified_split, kfold_split


def _make_img(w=32, h=32, channels=3):
    return np.random.randint(0, 255, (h, w, channels), dtype=np.uint8)


def _populate_class_dir(base_dir, class_name, count, prefix="img"):
    """Create <count> test images under base_dir/class_name/."""
    cls_dir = os.path.join(str(base_dir), class_name)
    os.makedirs(cls_dir, exist_ok=True)
    paths = []
    for i in range(count):
        fname = f"{prefix}_{i:03d}.png"
        p = os.path.join(cls_dir, fname)
        write_image(p, _make_img())
        paths.append(p)
    return paths


def _populate_flat_dir(base_dir, count, prefix="img"):
    """Create <count> test images directly in base_dir (no class subfolders)."""
    d = str(base_dir)
    os.makedirs(d, exist_ok=True)
    paths = []
    for i in range(count):
        fname = f"{prefix}_{i:03d}.png"
        p = os.path.join(d, fname)
        write_image(p, _make_img())
        paths.append(p)
    return paths


def _populate_labels(label_dir, img_paths):
    """Create a dummy .txt label file for each image."""
    d = str(label_dir)
    os.makedirs(d, exist_ok=True)
    for p in img_paths:
        base = os.path.splitext(os.path.basename(p))[0]
        lbl_path = os.path.join(d, base + ".txt")
        with open(lbl_path, "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")


def _count_files(directory):
    """Count files recursively under directory."""
    total = 0
    for _, _, files in os.walk(directory):
        total += len(files)
    return total


# ---------- random_split ----------

class TestRandomSplit:
    def test_correct_counts(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_flat_dir(input_dir, 20)

        splits = random_split(str(input_dir), str(output_dir), ratios=(0.7, 0.2, 0.1))

        total_files = sum(len(v) for v in splits.values())
        assert total_files == 20
        assert len(splits["train"]) == 14
        assert len(splits["val"]) == 4
        assert len(splits["test"]) == 2

    def test_seed_reproducibility(self, tmp_path):
        input_dir = tmp_path / "input"
        out_a = tmp_path / "out_a"
        out_b = tmp_path / "out_b"
        _populate_flat_dir(input_dir, 20)

        splits_a = random_split(str(input_dir), str(out_a), ratios=(0.7, 0.2, 0.1), seed=123)
        splits_b = random_split(str(input_dir), str(out_b), ratios=(0.7, 0.2, 0.1), seed=123)

        for split_name in ("train", "val", "test"):
            assert splits_a[split_name] == splits_b[split_name]

    def test_different_seeds_produce_different_splits(self, tmp_path):
        input_dir = tmp_path / "input"
        out_a = tmp_path / "out_a"
        out_b = tmp_path / "out_b"
        _populate_flat_dir(input_dir, 20)

        splits_a = random_split(str(input_dir), str(out_a), seed=1)
        splits_b = random_split(str(input_dir), str(out_b), seed=2)

        # With 20 images it's extremely unlikely the two seeds produce identical ordering
        assert splits_a["train"] != splits_b["train"]

    def test_with_label_dir(self, tmp_path):
        input_dir = tmp_path / "input"
        label_dir = tmp_path / "labels"
        output_dir = tmp_path / "output"
        paths = _populate_flat_dir(input_dir, 10)
        _populate_labels(label_dir, paths)

        splits = random_split(str(input_dir), str(output_dir),
                              ratios=(0.6, 0.2, 0.2), label_dir=str(label_dir))

        total_files = _count_files(str(output_dir))
        # 10 images + 10 labels = 20 files
        assert total_files == 20

    def test_output_directories_created(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_flat_dir(input_dir, 10)

        random_split(str(input_dir), str(output_dir))

        assert os.path.isdir(str(output_dir / "train"))
        assert os.path.isdir(str(output_dir / "val"))
        assert os.path.isdir(str(output_dir / "test"))


# ---------- stratified_split ----------

class TestStratifiedSplit:
    def test_class_balance_maintained(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_class_dir(input_dir, "cat", 10)
        _populate_class_dir(input_dir, "dog", 10)

        splits = stratified_split(str(input_dir), str(output_dir),
                                  ratios=(0.6, 0.2, 0.2))

        # Each class should have roughly proportional representation
        for split_name in ("train", "val", "test"):
            cats = [f for f, cls in splits[split_name] if cls == "cat"]
            dogs = [f for f, cls in splits[split_name] if cls == "dog"]
            # Both classes should be present in each split
            assert len(cats) > 0, f"No cats in {split_name}"
            assert len(dogs) > 0, f"No dogs in {split_name}"

    def test_all_files_accounted_for(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_class_dir(input_dir, "a", 5)
        _populate_class_dir(input_dir, "b", 5)
        _populate_class_dir(input_dir, "c", 5)

        splits = stratified_split(str(input_dir), str(output_dir),
                                  ratios=(0.6, 0.2, 0.2))

        total = sum(len(v) for v in splits.values())
        assert total == 15

    def test_output_class_subdirectories(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_class_dir(input_dir, "cat", 5)
        _populate_class_dir(input_dir, "dog", 5)

        stratified_split(str(input_dir), str(output_dir),
                         ratios=(0.6, 0.2, 0.2))

        # Each split directory should have class subdirectories
        for split_name in ("train", "val", "test"):
            split_path = output_dir / split_name
            assert os.path.isdir(str(split_path / "cat"))
            assert os.path.isdir(str(split_path / "dog"))


# ---------- kfold_split ----------

class TestKFoldSplit:
    def test_correct_number_of_folds(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_flat_dir(input_dir, 20)

        folds = kfold_split(str(input_dir), str(output_dir), k=5)

        assert len(folds) == 5
        for fold in folds:
            assert "fold" in fold
            assert "train" in fold
            assert "val" in fold

    def test_folds_cover_all_files(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_flat_dir(input_dir, 20)

        folds = kfold_split(str(input_dir), str(output_dir), k=5)

        for fold in folds:
            # train + val should equal total (20)
            assert fold["train"] + fold["val"] == 20

    def test_fold_directories_created(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_flat_dir(input_dir, 12)

        kfold_split(str(input_dir), str(output_dir), k=3)

        for i in range(3):
            fold_dir = output_dir / f"fold_{i}"
            assert os.path.isdir(str(fold_dir / "train"))
            assert os.path.isdir(str(fold_dir / "val"))

    def test_kfold_reproducibility_with_seed(self, tmp_path):
        input_dir = tmp_path / "input"
        out_a = tmp_path / "out_a"
        out_b = tmp_path / "out_b"
        _populate_flat_dir(input_dir, 15)

        folds_a = kfold_split(str(input_dir), str(out_a), k=3, seed=42)
        folds_b = kfold_split(str(input_dir), str(out_b), k=3, seed=42)

        for a, b in zip(folds_a, folds_b):
            assert a["train"] == b["train"]
            assert a["val"] == b["val"]
