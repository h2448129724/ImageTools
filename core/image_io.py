import cv2
import os
from PIL import Image
import numpy as np
from utils.helpers import ensure_dir


def read_image(path, flags=cv2.IMREAD_UNCHANGED):
    """Read image with OpenCV. Supports paths with unicode."""
    stream = open(path, "rb")
    data = bytearray(stream.read())
    arr = np.asarray(data, dtype=np.uint8)
    return cv2.imdecode(arr, flags)


def write_image(path, img, quality=95):
    """Write image to disk. Auto-creates directories."""
    ensure_dir(os.path.dirname(path) or ".")
    ext = os.path.splitext(path)[1].lower()
    params = []
    if ext in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, max(0, min(9, 9 - quality // 11))]
    elif ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, quality]
    cv2.imwrite(path, img, params)


def get_image_info(path):
    """Return (width, height, channels) for an image."""
    img = read_image(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape) > 2 else 1
    return w, h, c


def convert_format(input_path, output_path, fmt):
    """Convert image to specified format."""
    img = read_image(input_path)
    if img is None:
        return False
    write_image(output_path, img)
    return True


def resize_image(img, width=None, height=None, scale=None, keep_aspect=True, interp=cv2.INTER_LINEAR):
    """Resize image by width, height, or scale factor."""
    h, w = img.shape[:2]
    if scale:
        new_w, new_h = int(w * scale), int(h * scale)
    elif width and height:
        if keep_aspect:
            ratio = min(width / w, height / h)
            new_w, new_h = int(w * ratio), int(h * ratio)
        else:
            new_w, new_h = width, height
    elif width:
        ratio = width / w
        new_w, new_h = width, int(h * ratio)
    elif height:
        ratio = height / h
        new_w, new_h = int(w * ratio), height
    else:
        return img
    return cv2.resize(img, (new_w, new_h), interpolation=interp)
