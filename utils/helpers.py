import os
import json
import hashlib
from pathlib import Path


def get_image_files(path, extensions=None):
    """Recursively find all image files in a directory."""
    extensions = extensions or {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    files = []
    p = Path(path)
    if p.is_file():
        return [str(p)] if p.suffix.lower() in extensions else []
    for root, _, filenames in os.walk(path):
        for f in filenames:
            if Path(f).suffix.lower() in extensions:
                files.append(os.path.join(root, f))
    return sorted(files)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def file_hash(filepath):
    """MD5 hash of a file for deduplication."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_output_path(input_path, output_dir, suffix="", ext=None):
    """Generate output file path preserving relative structure."""
    base = os.path.splitext(os.path.basename(input_path))[0]
    new_ext = ext if ext else os.path.splitext(input_path)[1]
    name = f"{base}{suffix}{new_ext}"
    return os.path.join(output_dir, name)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
