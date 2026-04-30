from __future__ import annotations

import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from utils.helpers import get_image_files, ensure_dir, file_hash

logger = logging.getLogger(__name__)


def _resize_one(f: str, input_dir: str, output_dir: str, width: int | None, height: int | None,
                scale: float | None, keep_aspect: bool, quality: int) -> bool:
    from core.image_io import read_image, write_image, resize_image
    img = read_image(f)
    if img is None:
        logger.warning("Failed to read image: %s", f)
        return False
    resized = resize_image(img, width, height, scale, keep_aspect)
    rel = os.path.relpath(f, input_dir)
    out = os.path.join(output_dir, rel)
    write_image(out, resized)
    return True


def batch_rename(input_dir: str, output_dir: str, prefix: str = "img_", start_index: int = 1,
                 digits: int = 4, keep_ext: bool = True,
                 progress_callback: Callable[[int, int], None] | None = None,
                 cancel_check: Callable[[], bool] | None = None) -> list[dict[str, str]]:
    """Rename images sequentially with prefix and zero-padded index."""
    files = get_image_files(input_dir)
    ensure_dir(output_dir)
    results: list[dict[str, str]] = []
    for i, src in enumerate(files, start=start_index):
        if cancel_check and cancel_check():
            break
        ext = os.path.splitext(src)[1] if keep_ext else ".png"
        new_name = f"{prefix}{str(i).zfill(digits)}{ext}"
        dst = os.path.join(output_dir, new_name)
        shutil.copy2(src, dst)
        results.append({"source": src, "dest": dst})
        if progress_callback:
            progress_callback(i - start_index + 1, len(files))
    return results


def batch_resize(input_dir: str, output_dir: str, width: int | None = None,
                 height: int | None = None, scale: float | None = None,
                 keep_aspect: bool = True, max_workers: int = 4,
                 progress_callback: Callable[[int, int], None] | None = None,
                 cancel_check: Callable[[], bool] | None = None) -> int:
    """Resize all images in a directory using parallel threads."""
    from core.image_io import read_image, write_image, resize_image
    files = get_image_files(input_dir)
    ensure_dir(output_dir)
    count = 0
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_resize_one, f, input_dir, output_dir,
                               width, height, scale, keep_aspect, 95): f for f in files}
        for future in as_completed(futures):
            if cancel_check and cancel_check():
                pool.shutdown(wait=False, cancel_futures=True)
                break
            if future.result():
                count += 1
            done += 1
            if progress_callback:
                progress_callback(done, len(files))
    return count


def batch_convert_format(input_dir: str, output_dir: str, fmt: str = "png", quality: int = 95,
                         max_workers: int = 4,
                         progress_callback: Callable[[int, int], None] | None = None,
                         cancel_check: Callable[[], bool] | None = None) -> int:
    """Convert all images to a specified format using parallel threads."""
    from core.image_io import read_image, write_image
    files = get_image_files(input_dir)
    ensure_dir(output_dir)

    def convert_one(f: str) -> bool:
        img = read_image(f)
        if img is None:
            logger.warning("Failed to read image: %s", f)
            return False
        base = os.path.splitext(os.path.relpath(f, input_dir))[0]
        out = os.path.join(output_dir, f"{base}.{fmt}")
        ensure_dir(os.path.dirname(out))
        write_image(out, img, quality)
        return True

    count = 0
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(convert_one, f): f for f in files}
        for future in as_completed(futures):
            if cancel_check and cancel_check():
                pool.shutdown(wait=False, cancel_futures=True)
                break
            if future.result():
                count += 1
            done += 1
            if progress_callback:
                progress_callback(done, len(files))
    return count


def _compute_dhash(img_data: bytes, hash_size: int = 8) -> int:
    """Compute difference hash (dHash) from raw image bytes.

    Resizes to (hash_size+1, hash_size), converts to grayscale, then compares
    adjacent horizontal pixels. Returns a hash_size*hash_size bit integer.
    """
    import cv2
    import numpy as np
    arr = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    resized = cv2.resize(img, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    flat = diff.flatten()
    hash_val = 0
    for bit in flat:
        hash_val = (hash_val << 1) | int(bit)
    return hash_val


def _hamming_distance(h1: int, h2: int) -> int:
    """Count differing bits between two integer hashes."""
    return bin(h1 ^ h2).count("1")


def deduplicate_images(input_dir: str,
                       mode: str = "exact",
                       similarity_threshold: int = 10,
                       progress_callback: Callable[[int, int], None] | None = None,
                       cancel_check: Callable[[], bool] | None = None) -> list[tuple[str, str]]:
    """Find duplicate or near-duplicate images.

    Args:
        mode: "exact" for byte-identical dedup, "perceptual" for visually similar images.
        similarity_threshold: Max Hamming distance for perceptual mode (0-64, lower=stricter).
    """
    files = get_image_files(input_dir)
    if mode == "perceptual":
        return _deduplicate_perceptual(files, similarity_threshold, progress_callback, cancel_check)
    return _deduplicate_exact(files, progress_callback, cancel_check)


def _deduplicate_exact(files: list[str],
                       progress_callback: Callable[[int, int], None] | None,
                       cancel_check: Callable[[], bool] | None) -> list[tuple[str, str]]:

    """Find exact duplicate images using two-stage hashing: file size + partial hash, then full hash."""
    # Stage 1: Group by file size
    size_groups: dict[int, list[str]] = {}
    for f in files:
        if cancel_check and cancel_check():
            return []
        sz = os.path.getsize(f)
        size_groups.setdefault(sz, []).append(f)
        if progress_callback:
            progress_callback(len(size_groups), len(files))

    # Stage 2: For same-size groups, compute partial hash (first 8KB)
    partial_groups: dict[tuple[int, str], list[str]] = {}
    checked = 0
    for sz, group in size_groups.items():
        if len(group) < 2:
            checked += len(group)
            continue
        for f in group:
            if cancel_check and cancel_check():
                return []
            with open(f, "rb") as fh:
                partial = fh.read(8192)
            h = _quick_hash(partial)
            partial_groups.setdefault((sz, h), []).append(f)
            checked += 1
            if progress_callback:
                progress_callback(checked, len(files))

    # Stage 3: For same partial-hash groups, compute full hash
    seen: dict[str, str] = {}
    dupes: list[tuple[str, str]] = []
    for key, group in partial_groups.items():
        if len(group) < 2:
            continue
        for f in group:
            if cancel_check and cancel_check():
                return dupes
            h = file_hash(f)
            if h in seen:
                dupes.append((f, seen[h]))
            else:
                seen[h] = f
    return dupes


def _quick_hash(data: bytes) -> str:
    import hashlib
    return hashlib.md5(data).hexdigest()


def _deduplicate_perceptual(files: list[str], threshold: int,
                            progress_callback: Callable[[int, int], None] | None,
                            cancel_check: Callable[[], bool] | None) -> list[tuple[str, str]]:
    """Find near-duplicate images using dHash + Hamming distance.

    Uses multi-index hashing for performance: splits 64-bit hashes into 8 segments
    of 8 bits each, buckets by segment value, and only compares hashes that share
    at least one segment. This gives O(n) expected time for typical datasets where
    most images are distinct.
    """
    # Compute dHash for every image
    hashes: list[tuple[str, int]] = []
    for i, f in enumerate(files):
        if cancel_check and cancel_check():
            return []
        with open(f, "rb") as fh:
            data = fh.read()
        h = _compute_dhash(data)
        hashes.append((f, h))
        if progress_callback:
            progress_callback(i + 1, len(files))

    if not hashes:
        return []

    dupes: list[tuple[str, str]] = []
    matched: set[int] = set()

    # Fast path: group by exact hash — identical hashes are distance 0
    exact_groups: dict[int, list[int]] = {}
    for idx, (_, h) in enumerate(hashes):
        exact_groups.setdefault(h, []).append(idx)
    for group in exact_groups.values():
        if len(group) < 2:
            continue
        canonical = group[0]
        for idx in group[1:]:
            dupes.append((hashes[idx][0], hashes[canonical][0]))
            matched.add(idx)

    if threshold <= 0:
        return dupes

    # Multi-index: split 64-bit hash into 8 segments of 8 bits each
    # Two hashes with Hamming distance ≤ 10 must share at least one segment value.
    NUM_SEGMENTS = 8
    SEG_BITS = 64 // NUM_SEGMENTS  # 8 bits per segment
    SEG_MASK = (1 << SEG_BITS) - 1  # 0xFF

    # Build segment buckets: segment_index -> segment_value -> list of image indices
    seg_buckets: list[dict[int, list[int]]] = [{} for _ in range(NUM_SEGMENTS)]
    for idx, (_, h) in enumerate(hashes):
        if idx in matched:
            continue
        for s in range(NUM_SEGMENTS):
            seg_val = (h >> (s * SEG_BITS)) & SEG_MASK
            seg_buckets[s].setdefault(seg_val, []).append(idx)

    # For each unmatched image, collect candidates from all segment buckets
    remaining = [i for i in range(len(hashes)) if i not in matched]
    for i in remaining:
        if cancel_check and cancel_check():
            break
        if i in matched:
            continue
        h_i = hashes[i][1]
        candidates: set[int] = set()
        for s in range(NUM_SEGMENTS):
            seg_val = (h_i >> (s * SEG_BITS)) & SEG_MASK
            for j in seg_buckets[s].get(seg_val, []):
                if j != i and j not in matched:
                    candidates.add(j)
        for j in sorted(candidates):
            if j in matched:
                continue
            if _hamming_distance(h_i, hashes[j][1]) <= threshold:
                dupes.append((hashes[j][0], hashes[i][0]))
                matched.add(j)
    return dupes


def batch_add_border(input_dir: str, output_dir: str, border_size: int = 10,
                     color: tuple[int, int, int] = (0, 0, 0),
                     max_workers: int = 4,
                     progress_callback: Callable[[int, int], None] | None = None,
                     cancel_check: Callable[[], bool] | None = None) -> int:
    """Add border to all images using parallel threads."""
    import cv2
    from core.image_io import read_image, write_image
    files = get_image_files(input_dir)
    ensure_dir(output_dir)

    def border_one(f: str) -> bool:
        img = read_image(f)
        if img is None:
            logger.warning("Failed to read image: %s", f)
            return False
        bordered = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size,
                                      cv2.BORDER_CONSTANT, value=color)
        rel = os.path.relpath(f, input_dir)
        out = os.path.join(output_dir, rel)
        ensure_dir(os.path.dirname(out))
        write_image(out, bordered)
        return True

    count = 0
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(border_one, f): f for f in files}
        for future in as_completed(futures):
            if cancel_check and cancel_check():
                pool.shutdown(wait=False, cancel_futures=True)
                break
            if future.result():
                count += 1
            done += 1
            if progress_callback:
                progress_callback(done, len(files))
    return count
