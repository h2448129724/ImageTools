from __future__ import annotations

import os
import logging
import struct
from typing import Callable

import cv2
import numpy as np
from numpy.typing import NDArray

from utils.helpers import ensure_dir

logger = logging.getLogger(__name__)


# EXIF orientation tag values → OpenCV transform
# 1=normal, 2=flip-h, 3=rotate-180, 4=flip-v, 5=transpose, 6=rotate-90-cw,
# 7=transverse, 8=rotate-90-ccw
_EXIF_ORIENT_MAP = {
    2: (cv2.ROTATE_180, True),   # flip horizontal → flip(1)
    3: (cv2.ROTATE_180, False),
    4: (None, True),             # flip vertical → flip(0)
    5: (cv2.ROTATE_90_COUNTERCLOCKWISE, True),
    6: (cv2.ROTATE_90_CLOCKWISE, False),
    7: (cv2.ROTATE_90_CLOCKWISE, True),
    8: (cv2.ROTATE_90_COUNTERCLOCKWISE, False),
}


def _read_exif_orientation(data: bytes) -> int:
    """Extract EXIF orientation from JPEG bytes. Returns 0 if not found."""
    # Only JPEG files have EXIF in this context
    if data[:2] != b"\xff\xd8":
        return 0
    i = 2
    while i < len(data) - 8:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        # APP1 marker (0xE1) contains EXIF
        if marker == 0xE1:
            length = (data[i + 2] << 8) | data[i + 3]
            payload = data[i + 4: i + 2 + length]
            # Check for "Exif\x00\x00" header
            if payload[:6] == b"Exif\x00\x00":
                return _parse_exif_tiff_orientation(payload[6:])
            return 0
        if marker == 0xDA:  # SOS — no more metadata
            break
        if marker == 0xD9:  # EOI
            break
        length = (data[i + 2] << 8) | data[i + 3]
        i += 2 + length
    return 0


def _parse_exif_tiff_orientation(tiff_header: bytes) -> int:
    """Parse TIFF/IFD structure to find orientation tag (0x0112)."""
    if len(tiff_header) < 14:
        return 0
    byte_order = tiff_header[:2]
    if byte_order == b"II":
        endian = "<"
    elif byte_order == b"MM":
        endian = ">"
    else:
        return 0

    ifd_offset = struct.unpack(endian + "I", tiff_header[4:8])[0]
    if ifd_offset + 2 > len(tiff_header):
        return 0
    num_entries = struct.unpack(endian + "H", tiff_header[ifd_offset:ifd_offset + 2])[0]

    for j in range(num_entries):
        entry_off = ifd_offset + 2 + j * 12
        if entry_off + 12 > len(tiff_header):
            break
        tag = struct.unpack(endian + "H", tiff_header[entry_off:entry_off + 2])[0]
        if tag == 0x0112:  # Orientation
            val = struct.unpack(endian + "H", tiff_header[entry_off + 8:entry_off + 10])[0]
            return val
    return 0


def _apply_exif_orientation(img: NDArray[np.uint8], raw_data: bytes) -> NDArray[np.uint8]:
    """Apply EXIF orientation correction to an image."""
    orientation = _read_exif_orientation(raw_data)
    if orientation <= 1 or orientation > 8:
        return img
    mapping = _EXIF_ORIENT_MAP.get(orientation)
    if mapping is None:
        return img
    rotate_code, flip = mapping
    if orientation == 2:
        return cv2.flip(img, 1)
    if orientation == 4:
        return cv2.flip(img, 0)
    if rotate_code is not None:
        img = cv2.rotate(img, rotate_code)
    if flip:
        flip_code = 1 if orientation in (5, 7) else 0
        img = cv2.flip(img, flip_code)
    return img


def read_image(path: str, flags: int = cv2.IMREAD_UNCHANGED,
               auto_orient: bool = True) -> NDArray[np.uint8] | None:
    """Read image with OpenCV. Supports paths with unicode.

    Args:
        path: Image file path.
        flags: OpenCV imread flags.
        auto_orient: If True, auto-rotate based on EXIF orientation tag.
    """
    try:
        with open(path, "rb") as stream:
            data = bytearray(stream.read())
        arr = np.asarray(data, dtype=np.uint8)
        result = cv2.imdecode(arr, flags)
        if result is None:
            logger.warning("Failed to decode image: %s", path)
            return None
        if auto_orient:
            result = _apply_exif_orientation(result, bytes(data))
        return result
    except OSError as e:
        logger.error("Failed to read image: %s - %s", path, e)
        return None


def write_image(path: str, img: NDArray[np.uint8], quality: int = 95) -> None:
    """Write image to disk. Supports Unicode paths via imencode."""
    ensure_dir(os.path.dirname(path) or ".")
    ext = os.path.splitext(path)[1].lower()
    if ext in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, max(0, min(9, 9 - quality // 11))]
    elif ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, quality]
    else:
        params = []
    success, buf = cv2.imencode(ext, img, params)
    if not success:
        logger.warning("Failed to encode image: %s", path)
        return
    try:
        with open(path, "wb") as f:
            f.write(buf.tobytes())
    except OSError as e:
        logger.error("Failed to write image: %s - %s", path, e)


def get_image_info(path: str) -> tuple[int, int, int] | None:
    """Return (width, height, channels) for an image, reading header only."""
    try:
        with open(path, "rb") as f:
            header = f.read(128)
        if len(header) < 24:
            return None
        # PNG: width/height at bytes 16-24
        if header[:8] == b"\x89PNG\r\n\x1a\n":
            w, h = struct.unpack(">II", header[16:24])
            # channels: color type 2=RGB(3), 6=RGBA(4), 0=gray(1)
            ct = header[25]
            c = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(ct, 3)
            return w, h, c
        # JPEG: SOF marker contains dimensions
        if header[:2] == b"\xff\xd8":
            # Read up to 64KB — SOF marker is always in the first few KB
            with open(path, "rb") as f:
                data = bytearray(f.read(65536))
            i = 2
            while i < len(data) - 9:
                if data[i] != 0xFF:
                    i += 1
                    continue
                marker = data[i + 1]
                if marker in (0xC0, 0xC1, 0xC2):
                    h = (data[i + 5] << 8) | data[i + 6]
                    w = (data[i + 7] << 8) | data[i + 8]
                    c = data[i + 9]
                    return w, h, c
                if marker == 0xD9:
                    break
                length = (data[i + 2] << 8) | data[i + 3]
                i += 2 + length
        # Fallback: full decode for BMP, TIFF, WEBP, etc.
        img = read_image(path, auto_orient=False)
        if img is None:
            return None
        h, w = img.shape[:2]
        c = img.shape[2] if len(img.shape) > 2 else 1
        return w, h, c
    except (OSError, ValueError, struct.error) as e:
        logger.warning("Failed to get image info for %s: %s", path, e)
        return None


def convert_format(input_path: str, output_path: str) -> bool:
    """Convert image format based on output_path extension."""
    img = read_image(input_path)
    if img is None:
        return False
    write_image(output_path, img)
    return True


def resize_image(img: NDArray[np.uint8], width: int | None = None, height: int | None = None,
                 scale: float | None = None, keep_aspect: bool = True,
                 interp: int = cv2.INTER_LINEAR) -> NDArray[np.uint8]:
    """Resize image by width, height, or scale factor."""
    h, w = img.shape[:2]
    if scale:
        if scale <= 0:
            return img
        new_w, new_h = int(w * scale), int(h * scale)
    elif width and height:
        if width <= 0 or height <= 0:
            return img
        if keep_aspect:
            ratio = min(width / w, height / h)
            new_w, new_h = int(w * ratio), int(h * ratio)
        else:
            new_w, new_h = width, height
    elif width and width > 0:
        ratio = width / w
        new_w, new_h = width, int(h * ratio)
    elif height and height > 0:
        ratio = height / h
        new_w, new_h = int(w * ratio), height
    else:
        return img
    new_w = max(1, new_w)
    new_h = max(1, new_h)
    return cv2.resize(img, (new_w, new_h), interpolation=interp)
