import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from utils.helpers import ensure_dir, get_image_files


def _copy_files(filelist, src_dir, dst_dir, label_dir=None):
    """Copy files from src to dst, optionally copying label files."""
    for src in filelist:
        rel = os.path.relpath(src, src_dir) if src_dir in src else os.path.basename(src)
        dst = os.path.join(dst_dir, rel)
        ensure_dir(os.path.dirname(dst))
        shutil.copy2(src, dst)
        if label_dir:
            for lbl in _find_label_files(src, label_dir):
                lbl_dst = os.path.join(dst_dir, os.path.relpath(lbl, src_dir) if src_dir in lbl else os.path.basename(lbl))
                ensure_dir(os.path.dirname(lbl_dst))
                shutil.copy2(lbl, lbl_dst)


def _find_label_files(img_path, label_dir):
    """Find matching label files (.txt, .xml, .json) for an image."""
    base = os.path.splitext(os.path.basename(img_path))[0]
    labels = []
    for ext in [".txt", ".xml", ".json"]:
        # Look in the same relative location in label_dir
        lbl_path = os.path.join(label_dir, base + ext)
        if os.path.exists(lbl_path):
            labels.append(lbl_path)
    return labels


def random_split(input_dir, output_dir, ratios=(0.7, 0.2, 0.1), label_dir=None, seed=42):
    """Randomly split images into train/val/test sets."""
    if len(ratios) == 2:
        ratios = (ratios[0], ratios[1], 0)
    files = get_image_files(input_dir)
    random.seed(seed)
    random.shuffle(files)
    n = len(files)
    train_end = max(1, int(n * ratios[0]))
    val_end = train_end + max(1, int(n * ratios[1]))
    splits = {"train": files[:train_end], "val": files[train_end:val_end]}
    if ratios[2] > 0:
        splits["test"] = files[val_end:]
    for name, flist in splits.items():
        dst = os.path.join(output_dir, name)
        _copy_files(flist, input_dir, dst, label_dir)
    return splits


def stratified_split(input_dir, output_dir, ratios=(0.7, 0.2, 0.1), seed=42):
    """Split by subfolder (each subfolder = one class), preserving class distribution."""
    if len(ratios) == 2:
        ratios = (ratios[0], ratios[1], 0)
    random.seed(seed)
    splits = {"train": [], "val": []}
    if ratios[2] > 0:
        splits["test"] = []

    for cls_name in sorted(os.listdir(input_dir)):
        cls_path = os.path.join(input_dir, cls_name)
        if not os.path.isdir(cls_path):
            continue
        files = get_image_files(cls_path)
        random.shuffle(files)
        n = len(files)
        train_end = max(1, int(n * ratios[0]))
        val_end = train_end + max(1, int(n * ratios[1]))

        for f in files[:train_end]:
            splits["train"].append((f, cls_name))
        for f in files[train_end:val_end]:
            splits["val"].append((f, cls_name))
        if ratios[2] > 0:
            for f in files[val_end:]:
                splits["test"].append((f, cls_name))

    for name, flist in splits.items():
        for f, cls in flist:
            dst = os.path.join(output_dir, name, cls)
            ensure_dir(dst)
            shutil.copy2(f, os.path.join(dst, os.path.basename(f)))
    return splits


def kfold_split(input_dir, output_dir, k=5, seed=42):
    """Generate K-fold cross-validation splits."""
    files = get_image_files(input_dir)
    random.seed(seed)
    random.shuffle(files)
    n = len(files)
    fold_size = n // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n
        val_files = files[start:end]
        train_files = files[:start] + files[end:]
        fold_dir = os.path.join(output_dir, f"fold_{i}")
        _copy_files(train_files, input_dir, os.path.join(fold_dir, "train"))
        _copy_files(val_files, input_dir, os.path.join(fold_dir, "val"))
        folds.append({"fold": i, "train": len(train_files), "val": len(val_files)})
    return folds
