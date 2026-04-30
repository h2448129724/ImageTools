import os
import shutil
from pathlib import Path
from utils.helpers import get_image_files, ensure_dir, file_hash


def batch_rename(input_dir, output_dir, prefix="img_", start_index=1, digits=4, keep_ext=True):
    """Rename images sequentially with prefix and zero-padded index."""
    files = get_image_files(input_dir)
    ensure_dir(output_dir)
    results = []
    for i, src in enumerate(files, start=start_index):
        ext = os.path.splitext(src)[1] if keep_ext else ".png"
        new_name = f"{prefix}{str(i).zfill(digits)}{ext}"
        dst = os.path.join(output_dir, new_name)
        shutil.copy2(src, dst)
        results.append({"source": src, "dest": dst})
    return results


def batch_resize(input_dir, output_dir, width=None, height=None, scale=None, keep_aspect=True):
    """Resize all images in a directory."""
    from core.image_io import read_image, write_image, resize_image
    files = get_image_files(input_dir)
    ensure_dir(output_dir)
    for f in files:
        img = read_image(f)
        if img is not None:
            resized = resize_image(img, width, height, scale, keep_aspect)
            rel = os.path.relpath(f, input_dir)
            out = os.path.join(output_dir, rel)
            write_image(out, resized)
    return len(files)


def batch_convert_format(input_dir, output_dir, fmt="png", quality=95):
    """Convert all images to a specified format."""
    from core.image_io import read_image, write_image
    files = get_image_files(input_dir)
    ensure_dir(output_dir)
    count = 0
    for f in files:
        img = read_image(f)
        if img is not None:
            base = os.path.splitext(os.path.basename(f))[0]
            out = os.path.join(output_dir, f"{base}.{fmt}")
            write_image(out, img, quality)
            count += 1
    return count


def deduplicate_images(input_dir):
    """Find and list duplicate images by MD5 hash."""
    files = get_image_files(input_dir)
    seen = {}
    dupes = []
    for f in files:
        h = file_hash(f)
        if h in seen:
            dupes.append((f, seen[h]))
        else:
            seen[h] = f
    return dupes


def batch_add_border(input_dir, output_dir, border_size=10, color=(0, 0, 0)):
    """Add border to all images."""
    import cv2
    from core.image_io import read_image, write_image
    files = get_image_files(input_dir)
    ensure_dir(output_dir)
    for f in files:
        img = read_image(f)
        if img is not None:
            bordered = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size,
                                          cv2.BORDER_CONSTANT, value=color)
            out = os.path.join(output_dir, os.path.basename(f))
            write_image(out, bordered)
    return len(files)
