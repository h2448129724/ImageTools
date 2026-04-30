from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


CONVERSION_CODES: dict[str, int] = {
    "BGR → RGB": cv2.COLOR_BGR2RGB,
    "RGB → BGR": cv2.COLOR_RGB2BGR,
    "BGR → HSV": cv2.COLOR_BGR2HSV,
    "HSV → BGR": cv2.COLOR_HSV2BGR,
    "BGR → LAB": cv2.COLOR_BGR2LAB,
    "LAB → BGR": cv2.COLOR_LAB2BGR,
    "BGR → YUV": cv2.COLOR_BGR2YUV,
    "YUV → BGR": cv2.COLOR_YUV2BGR,
    "BGR → GRAY": cv2.COLOR_BGR2GRAY,
    "GRAY → BGR": cv2.COLOR_GRAY2BGR,
    "BGR → YCrCb": cv2.COLOR_BGR2YCrCb,
    "BGR → HLS": cv2.COLOR_BGR2HLS,
}


def convert_color(img: NDArray[np.uint8], conversion_key: str) -> NDArray[np.uint8]:
    """Convert image between color spaces using the named conversion key."""
    code = CONVERSION_CODES.get(conversion_key)
    if code is None:
        raise ValueError(f"Unknown conversion: {conversion_key}")
    return cv2.cvtColor(img, code)


def to_grayscale(img: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if len(img.shape) == 2:
        return img
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def split_channels(img: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], ...]:
    return cv2.split(img)


def merge_channels(channels: tuple[NDArray[np.uint8], ...] | list[NDArray[np.uint8]]) -> NDArray[np.uint8]:
    return cv2.merge(channels)


def extract_channel(img: NDArray[np.uint8], index: int) -> NDArray[np.uint8] | None:
    """Extract a single channel (0=B, 1=G, 2=R, 3=A)."""
    if len(img.shape) == 2:
        return img
    if index < img.shape[2]:
        return img[:, :, index].copy()
    return None
