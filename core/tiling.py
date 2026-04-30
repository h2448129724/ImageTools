"""Image tiling utilities for splitting large images into fixed-size tiles."""
from __future__ import annotations

import os
import json
from typing import Any, Callable

import cv2
import numpy as np
from numpy.typing import NDArray

from utils.helpers import ensure_dir
from core.image_io import read_image, write_image

TileResult = tuple[NDArray[np.uint8], int, int, int, int]


def tile_image(img: NDArray[np.uint8], tile_w: int, tile_h: int, overlap: int = 0,
               discard_incomplete: bool = True,
               pad_value: int | tuple[int, ...] = 0) -> list[TileResult]:
    """Split image into fixed-size tiles.

    Returns list of (tile_img, x, y, w, h).
    """
    if tile_w <= 0 or tile_h <= 0:
        raise ValueError(f"tile_w 和 tile_h 必须大于 0，当前为 {tile_w}x{tile_h}")
    ih, iw = img.shape[:2]
    tiles: list[TileResult] = []
    step_y = max(1, tile_h - overlap)
    step_x = max(1, tile_w - overlap)

    y = 0
    while y < ih:
        x = 0
        while x < iw:
            tile = img[y:min(y + tile_h, ih), x:min(x + tile_w, iw)].copy()
            th, tw = tile.shape[:2]

            if th < tile_h or tw < tile_w:
                if discard_incomplete:
                    x += step_x
                    continue
                padded = np.full((tile_h, tile_w, img.shape[2]) if len(img.shape) > 2 else (tile_h, tile_w),
                                 pad_value, dtype=img.dtype)
                padded[:th, :tw] = tile
                tile = padded
                th, tw = tile_h, tile_w

            tiles.append((tile, x, y, tw, th))
            if x + tile_w >= iw:
                break
            x += step_x
        if y + tile_h >= ih:
            break
        y += step_y
    return tiles


def tile_image_file(input_path: str, output_dir: str, tile_w: int, tile_h: int,
                    overlap: int = 0, discard_incomplete: bool = True,
                    prefix: str = "") -> dict[str, Any]:
    """Tile a single image file and save tiles."""
    img = read_image(input_path)
    if img is None:
        return {"error": f"Failed to read {input_path}"}
    basename = os.path.splitext(os.path.basename(input_path))[0]
    tiles_out = os.path.join(output_dir, basename)
    ensure_dir(tiles_out)

    tiles = tile_image(img, tile_w, tile_h, overlap, discard_incomplete)
    coords: list[dict[str, Any]] = []
    for i, (tile, x, y, tw, th) in enumerate(tiles):
        fname = f"{prefix}{basename}_x{x:04d}_y{y:04d}.png"
        out_path = os.path.join(tiles_out, fname)
        write_image(out_path, tile)
        coords.append({"file": fname, "x": x, "y": y, "w": tw, "h": th})

    coord_path = os.path.join(tiles_out, f"{basename}_coords.json")
    with open(coord_path, "w") as f:
        json.dump({"source": input_path, "tile_w": tile_w, "tile_h": tile_h,
                   "overlap": overlap, "tiles": coords}, f, indent=2)
    return {"tiles": len(tiles), "output_dir": tiles_out, "coords": coord_path}


def tile_directory(input_dir: str, output_dir: str, tile_w: int, tile_h: int,
                   overlap: int = 0, discard_incomplete: bool = True,
                   progress_callback: Callable[[int, int], None] | None = None) -> dict[str, int]:
    """Tile all images in a directory."""
    from utils.helpers import get_image_files
    files = get_image_files(input_dir)
    total = 0
    for i, f in enumerate(files):
        result = tile_image_file(f, output_dir, tile_w, tile_h, overlap, discard_incomplete)
        total += result.get("tiles", 0)
        if progress_callback:
            progress_callback(i + 1, len(files))
    return {"total_files": len(files), "total_tiles": total}


def grid_tile(img: NDArray[np.uint8], rows: int, cols: int,
              discard_incomplete: bool = True) -> list[TileResult]:
    """Split image into a grid of rows x cols equal tiles."""
    ih, iw = img.shape[:2]
    tile_h = ih // rows
    tile_w = iw // cols
    tiles: list[TileResult] = []
    for r in range(rows):
        for c in range(cols):
            y, x = r * tile_h, c * tile_w
            tile = img[y:y + tile_h, x:x + tile_w].copy()
            if tile.shape[0] == tile_h and tile.shape[1] == tile_w:
                tiles.append((tile, x, y, tile_w, tile_h))
            elif not discard_incomplete:
                tiles.append((tile, x, y, tile_w, tile_h))
    return tiles
