from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def get_image_files(path: str, extensions: set[str] | None = None) -> list[str]:
    """Recursively find all image files in a directory."""
    extensions = extensions or {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    files: list[str] = []
    p = Path(path)
    if p.is_file():
        return [str(p)] if p.suffix.lower() in extensions else []
    for root, _, filenames in os.walk(path):
        for f in filenames:
            if Path(f).suffix.lower() in extensions:
                files.append(os.path.join(root, f))
    return sorted(files)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def file_hash(filepath: str) -> str:
    """MD5 hash of a file for deduplication."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_output_path(input_path: str, output_dir: str, suffix: str = "", ext: str | None = None) -> str:
    """Generate output file path in the output directory."""
    base = os.path.splitext(os.path.basename(input_path))[0]
    new_ext = ext if ext else os.path.splitext(input_path)[1]
    name = f"{base}{suffix}{new_ext}"
    return os.path.join(output_dir, name).replace("\\", "/")


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
