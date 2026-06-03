from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .constants import IMAGE_SUFFIXES


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_image_size(image_path: str | Path) -> tuple[int, int]:
    image_path = Path(image_path)
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    h, w = image.shape[:2]
    return int(w), int(h)


def iter_image_files(image_dir: str | Path) -> list[Path]:
    folder = Path(image_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"图片目录不存在: {folder}")
    files = []
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            files.append(path)
    return files


def iter_json_files(annotation_dir: str | Path) -> list[Path]:
    folder = Path(annotation_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"标注目录不存在: {folder}")
    return sorted([path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".json"])
