"""Composable image augmentation pipeline for deep learning dataset preparation."""
from __future__ import annotations

import logging
import random
from typing import Any, Callable

logger = logging.getLogger(__name__)

import cv2
import numpy as np
from numpy.typing import NDArray


class Transform:
    """Base class for all augmentation transforms."""

    def __init__(self, p: float = 1.0):
        self.p = p

    def __repr__(self) -> str:
        params = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        args = ', '.join(f'{k}={v!r}' for k, v in params.items())
        return f'{self.__class__.__name__}({args})'

    def __call__(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if random.random() > self.p:
            return img
        return self.apply(img)

    def apply(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        raise NotImplementedError


class Compose(Transform):
    """Chain multiple transforms together."""

    def __init__(self, transforms: list[Transform], p: float = 1.0):
        super().__init__(p)
        self.transforms = transforms

    def apply(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        for t in self.transforms:
            img = t(img)
        return img


# ---------- Geometric ----------

class RandomHorizontalFlip(Transform):
    def apply(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        return cv2.flip(img, 1)


class RandomVerticalFlip(Transform):
    def apply(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        return cv2.flip(img, 0)


class RandomRotate(Transform):
    """Rotate by a random angle within [-limit, limit]."""

    def __init__(self, limit: float = 15.0, border_mode: int = cv2.BORDER_REFLECT_101,
                 border_value: tuple[int, int, int] = (0, 0, 0), p: float = 0.5):
        super().__init__(p)
        self.limit = limit
        self.border_mode = border_mode
        self.border_value = border_value

    def apply(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        angle = random.uniform(-self.limit, self.limit)
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=self.border_mode,
                              borderValue=self.border_value)


class RandomScale(Transform):
    """Scale by a random factor within [scale_limit[0], scale_limit[1]]."""

    def __init__(self, scale_limit: tuple[float, float] = (0.8, 1.2),
                 interpolation: int = cv2.INTER_LINEAR, p: float = 0.5):
        super().__init__(p)
        self.scale_limit = scale_limit
        self.interpolation = interpolation

    def apply(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        scale = random.uniform(*self.scale_limit)
        h, w = img.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w < 1 or new_h < 1:
            return img
        return cv2.resize(img, (new_w, new_h), interpolation=self.interpolation)


class RandomCrop(Transform):
    """Crop a random region of size (crop_h, crop_w). Pads if image is smaller."""

    def __init__(self, crop_h: int, crop_w: int, pad_mode: str = "constant",
                 pad_value: int = 0, p: float = 1.0):
        super().__init__(p)
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.pad_mode = pad_mode
        self.pad_value = pad_value

    def apply(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        h, w = img.shape[:2]
        # Pad if needed
        pad_top = max(0, (self.crop_h - h) // 2 + 1)
        pad_left = max(0, (self.crop_w - w) // 2 + 1)
        if pad_top > 0 or pad_left > 0:
            border = cv2.BORDER_CONSTANT if self.pad_mode == "constant" else cv2.BORDER_REFLECT_101
            value = self.pad_value if border == cv2.BORDER_CONSTANT else None
            img = cv2.copyMakeBorder(img, pad_top, pad_top, pad_left, pad_left, border, value=value)
            h, w = img.shape[:2]
        y = random.randint(0, h - self.crop_h)
        x = random.randint(0, w - self.crop_w)
        return img[y:y + self.crop_h, x:x + self.crop_w]


class LetterboxResize(Transform):
    """Resize to target size with padding (letterbox). Standard for YOLO-style models."""

    def __init__(self, target_h: int, target_w: int, fill_value: int = 114,
                 interpolation: int = cv2.INTER_LINEAR, p: float = 1.0):
        super().__init__(p)
        self.target_h = target_h
        self.target_w = target_w
        self.fill_value = fill_value
        self.interpolation = interpolation

    def apply(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        h, w = img.shape[:2]
        scale = min(self.target_w / w, self.target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=self.interpolation)

        # Handle grayscale
        if len(resized.shape) == 2:
            canvas = np.full((self.target_h, self.target_w), self.fill_value, dtype=np.uint8)
        else:
            canvas = np.full((self.target_h, self.target_w, resized.shape[2]), self.fill_value,
                             dtype=np.uint8)

        pad_top = (self.target_h - new_h) // 2
        pad_left = (self.target_w - new_w) // 2
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
        return canvas


# ---------- Color ----------

class ColorJitter(Transform):
    """Random brightness, contrast, saturation, and hue shifts."""

    def __init__(self, brightness: float = 0.2, contrast: float = 0.2,
                 saturation: float = 0.2, hue: float = 0.1, p: float = 0.5):
        super().__init__(p)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def apply(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if self.brightness == 0 and self.contrast == 0 and self.saturation == 0 and self.hue == 0:
            return img
        if len(img.shape) == 2:
            # Only brightness/contrast for grayscale
            if self.brightness > 0:
                b = random.uniform(-self.brightness, self.brightness) * 255
                img = np.clip(img.astype(np.float32) + b, 0, 255).astype(np.uint8)
            if self.contrast > 0:
                c = random.uniform(1 - self.contrast, 1 + self.contrast)
                mean = img.mean()
                img = np.clip(c * img.astype(np.float32) + (1 - c) * mean, 0, 255).astype(np.uint8)
            return img

        # Color: brightness/contrast in BGR space, saturation/hue in HSV space
        result = img.astype(np.float32)

        if self.brightness > 0:
            b = random.uniform(-self.brightness, self.brightness) * 255
            result = np.clip(result + b, 0, 255)

        if self.contrast > 0:
            c = random.uniform(1 - self.contrast, 1 + self.contrast)
            mean = result.mean()
            result = np.clip(c * result + (1 - c) * mean, 0, 255)

        result = result.astype(np.uint8)

        if self.saturation > 0 or self.hue > 0:
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
            if self.saturation > 0:
                s = random.uniform(1 - self.saturation, 1 + self.saturation)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s, 0, 255)
            if self.hue > 0:
                h_shift = random.uniform(-self.hue, self.hue) * 180
                hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return result


# ---------- Noise ----------

class GaussianNoise(Transform):
    """Add Gaussian noise."""

    def __init__(self, mean: float = 0.0, std_limit: float = 25.0, p: float = 0.5):
        super().__init__(p)
        self.mean = mean
        self.std_limit = std_limit

    def apply(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        std = random.uniform(0, self.std_limit)
        noise = np.random.normal(self.mean, std, img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


class SaltAndPepperNoise(Transform):
    """Add salt and pepper noise."""

    def __init__(self, amount: float = 0.01, p: float = 0.5):
        super().__init__(p)
        self.amount = amount

    def apply(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        result = img.copy()
        # Salt
        n_salt = int(self.amount * img.size / (img.ndim))
        coords = [np.random.randint(0, i, n_salt) for i in img.shape]
        result[tuple(coords)] = 255
        # Pepper
        n_pepper = int(self.amount * img.size / (img.ndim))
        coords = [np.random.randint(0, i, n_pepper) for i in img.shape]
        result[tuple(coords)] = 0
        return result


# ---------- Blur ----------

class RandomGaussianBlur(Transform):
    """Apply Gaussian blur with random kernel size."""

    def __init__(self, ksize_limit: tuple[int, int] = (3, 7), p: float = 0.3):
        super().__init__(p)
        self.ksize_limit = ksize_limit

    def apply(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        ksize = random.choice(range(self.ksize_limit[0], self.ksize_limit[1] + 1, 2))
        return cv2.GaussianBlur(img, (ksize, ksize), 0)


class RandomMotionBlur(Transform):
    """Simulate motion blur with a random kernel."""

    def __init__(self, kernel_limit: int = 15, p: float = 0.3):
        super().__init__(p)
        self.kernel_limit = kernel_limit

    def apply(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        ksize = random.randrange(3, self.kernel_limit + 1, 2)
        angle = random.uniform(0, 180)
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        mid = ksize // 2
        cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        for i in range(ksize):
            offset = i - mid
            x = int(mid + offset * cos_a)
            y = int(mid + offset * sin_a)
            if 0 <= x < ksize and 0 <= y < ksize:
                kernel[y, x] = 1.0
        kernel /= kernel.sum()
        return cv2.filter2D(img, -1, kernel)


# ---------- Erasing ----------

class RandomErasing(Transform):
    """Randomly erase a rectangular region."""

    def __init__(self, area_ratio: tuple[float, float] = (0.02, 0.2),
                 fill_value: int = 0, p: float = 0.3):
        super().__init__(p)
        self.area_ratio = area_ratio
        self.fill_value = fill_value

    def apply(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        h, w = img.shape[:2]
        area = h * w
        target_area = random.uniform(*self.area_ratio) * area
        aspect = random.uniform(0.3, 1 / 0.3)
        eh = int(round(np.sqrt(target_area * aspect)))
        ew = int(round(np.sqrt(target_area / aspect)))
        if ew >= w or eh >= h:
            return img
        x = random.randint(0, w - ew)
        y = random.randint(0, h - eh)
        result = img.copy()
        result[y:y + eh, x:x + ew] = self.fill_value
        return result


class Cutout(Transform):
    """Cutout: mask out random square patches."""

    def __init__(self, n_holes: int = 1, hole_size: int = 32,
                 fill_value: int = 0, p: float = 0.5):
        super().__init__(p)
        self.n_holes = n_holes
        self.hole_size = hole_size
        self.fill_value = fill_value

    def apply(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        h, w = img.shape[:2]
        result = img.copy()
        half = self.hole_size // 2
        for _ in range(self.n_holes):
            y = random.randint(0, h)
            x = random.randint(0, w)
            y1, y2 = max(0, y - half), min(h, y + half)
            x1, x2 = max(0, x - half), min(w, x + half)
            result[y1:y2, x1:x2] = self.fill_value
        return result


# ---------- Normalization ----------

class Normalize(Transform):
    """Normalize pixel values: (img - mean) / std. Returns float32."""

    def __init__(self, mean: tuple[float, ...] = (0.485, 0.456, 0.406),
                 std: tuple[float, ...] = (0.229, 0.224, 0.225),
                 max_pixel_value: float = 255.0, p: float = 1.0):
        super().__init__(p)
        self.mean = np.array(mean, dtype=np.float32) * max_pixel_value
        self.std = np.array(std, dtype=np.float32) * max_pixel_value

    def apply(self, img: NDArray[np.uint8]) -> NDArray[np.float32]:
        result = (img.astype(np.float32) - self.mean) / self.std
        return result


# ---------- Multi-image transforms ----------

class LongestMaxSize(Transform):
    """Resize the longest side to max_size while preserving aspect ratio."""

    def __init__(self, max_size: int = 640, interpolation: int = cv2.INTER_LINEAR,
                 p: float = 1.0):
        super().__init__(p)
        self.max_size = max_size
        self.interpolation = interpolation

    def apply(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        h, w = img.shape[:2]
        scale = self.max_size / max(h, w)
        if scale >= 1.0:
            return img
        new_h, new_w = int(h * scale), int(w * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=self.interpolation)


class MixUp(Transform):
    """Blend image with another image using alpha compositing."""

    def __init__(self, second_img: NDArray[np.uint8], alpha: float = 0.4, p: float = 0.5):
        super().__init__(p)
        self.second_img = second_img
        self.alpha = alpha

    def apply(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        h, w = img.shape[:2]
        second = cv2.resize(self.second_img, (w, h))
        if len(img.shape) == 2 and len(second.shape) == 3:
            second = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
        elif len(img.shape) == 3 and len(second.shape) == 2:
            second = cv2.cvtColor(second, cv2.COLOR_GRAY2BGR)
        beta = 1.0 - self.alpha
        return cv2.addWeighted(img, beta, second, self.alpha, 0)


class Mosaic(Transform):
    """Combine 3 additional images into a 2x2 mosaic grid."""

    def __init__(self, images: list[NDArray[np.uint8]], target_size: int = 640,
                 p: float = 0.5):
        super().__init__(p)
        self.images = images
        self.target_size = target_size

    def apply(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        ts = self.target_size
        half = ts // 2
        canvas = np.zeros((ts, ts, 3), dtype=np.uint8) if len(img.shape) == 3 else np.zeros((ts, ts), dtype=np.uint8)

        def resize_to(img_in, w, h):
            return cv2.resize(img_in, (w, h))

        # Top-left: current image
        canvas[:half, :half] = resize_to(img, half, half)

        # Other 3 positions from provided images
        positions = [(half, 0), (0, half), (half, half)]
        for i, (ox, oy) in enumerate(positions):
            if i < len(self.images):
                src = self.images[i]
                canvas[oy:oy + half, ox:ox + half] = resize_to(src, half, half)
            else:
                # Fill with random image if not enough provided
                noise = np.random.randint(0, 256, (half, half) + img.shape[2:3], dtype=np.uint8) if len(img.shape) == 3 else np.random.randint(0, 256, (half, half), dtype=np.uint8)
                canvas[oy:oy + half, ox:ox + half] = noise
        return canvas


# ---------- BBox-aware augmentation ----------

# Bbox format: list of dicts with keys (cls, xc, yc, bw, bh) in normalized YOLO coords
BBox = list[dict[str, Any]]


class BBoxTransform:
    """Base class for transforms that operate on (image, bboxes) pairs."""

    def __init__(self, p: float = 1.0):
        self.p = p

    def __repr__(self) -> str:
        params = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        args = ', '.join(f'{k}={v!r}' for k, v in params.items())
        return f'{self.__class__.__name__}({args})'

    def __call__(self, img: NDArray[np.uint8], bboxes: BBox) -> tuple[NDArray[np.uint8], BBox]:
        if random.random() > self.p:
            return img, bboxes
        return self.apply(img, bboxes)

    def apply(self, img: NDArray[np.uint8], bboxes: BBox) -> tuple[NDArray[np.uint8], BBox]:
        raise NotImplementedError


class BBoxCompose:
    """Chain multiple BBoxTransforms."""

    def __init__(self, transforms: list[BBoxTransform]):
        self.transforms = transforms

    def __call__(self, img: NDArray[np.uint8], bboxes: BBox) -> tuple[NDArray[np.uint8], BBox]:
        for t in self.transforms:
            img, bboxes = t(img, bboxes)
        return img, bboxes

    def __repr__(self) -> str:
        return f"BBoxCompose({self.transforms!r})"


class BBoxHorizontalFlip(BBoxTransform):
    def apply(self, img, bboxes):
        h, w = img.shape[:2]
        img = cv2.flip(img, 1)
        new_boxes = []
        for b in bboxes:
            new_boxes.append({**b, "xc": 1.0 - b["xc"]})
        return img, new_boxes


class BBoxVerticalFlip(BBoxTransform):
    def apply(self, img, bboxes):
        img = cv2.flip(img, 0)
        new_boxes = []
        for b in bboxes:
            new_boxes.append({**b, "yc": 1.0 - b["yc"]})
        return img, new_boxes


class BBoxScale(BBoxTransform):
    def __init__(self, scale_limit: tuple[float, float] = (0.5, 1.5), p: float = 1.0):
        super().__init__(p)
        self.scale_limit = scale_limit

    def apply(self, img, bboxes):
        h, w = img.shape[:2]
        scale = random.uniform(*self.scale_limit)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
        return img, bboxes


class BBoxLetterboxResize(BBoxTransform):
    def __init__(self, target_h: int, target_w: int, p: float = 1.0):
        super().__init__(p)
        self.target_h = target_h
        self.target_w = target_w

    def apply(self, img, bboxes):
        th, tw = self.target_h, self.target_w
        h, w = img.shape[:2]
        scale = min(tw / w, th / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        if len(img.shape) == 2:
            canvas = np.full((th, tw), 114, dtype=np.uint8)
        else:
            canvas = np.full((th, tw, img.shape[2]), 114, dtype=np.uint8)

        pad_x = (tw - new_w) // 2
        pad_y = (th - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # Adjust bboxes: scale coords and shift by padding
        sx = scale * w / tw if tw else 1.0
        new_boxes = []
        for b in bboxes:
            # Normalize: original image → resized image within canvas
            new_xc = (b["xc"] * w * scale + pad_x) / tw
            new_yc = (b["yc"] * h * scale + pad_y) / th
            new_bw = b["bw"] * scale * w / tw
            new_bh = b["bh"] * scale * h / th
            new_boxes.append({**b, "xc": new_xc, "yc": new_yc, "bw": new_bw, "bh": new_bh})
        return canvas, new_boxes


class BBoxColorJitter(BBoxTransform):
    """Color-only transform, bboxes unchanged."""

    def __init__(self, brightness: float = 0.2, contrast: float = 0.2,
                 saturation: float = 0.2, hue: float = 0.1, p: float = 1.0):
        super().__init__(p)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def apply(self, img, bboxes):
        jitter = ColorJitter(brightness=self.brightness, contrast=self.contrast,
                             saturation=self.saturation, hue=self.hue, p=1.0)
        img = jitter.apply(img)
        return img, bboxes


class BBoxRandomCrop(BBoxTransform):
    def __init__(self, crop_h: int, crop_w: int, p: float = 1.0):
        super().__init__(p)
        self.crop_h = crop_h
        self.crop_w = crop_w

    def apply(self, img, bboxes):
        h, w = img.shape[:2]
        ch, cw = self.crop_h, self.crop_w

        # Pad if needed
        if ch > h or cw > w:
            pad_h = max(ch - h, 0)
            pad_w = max(cw - w, 0)
            if len(img.shape) == 2:
                img = np.pad(img, ((0, pad_h), (0, pad_w)), constant_values=114)
            else:
                img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=114)
            h, w = img.shape[:2]

        y = random.randint(0, h - ch)
        x = random.randint(0, w - cw)
        img = img[y:y + ch, x:x + cw]

        new_boxes = []
        for b in bboxes:
            # Convert to absolute coords in original image
            bx1 = (b["xc"] - b["bw"] / 2) * w
            by1 = (b["yc"] - b["bh"] / 2) * h
            bx2 = (b["xc"] + b["bw"] / 2) * w
            by2 = (b["yc"] + b["bh"] / 2) * h

            # Shift by crop offset and clip
            bx1 = max(bx1 - x, 0)
            by1 = max(by1 - y, 0)
            bx2 = min(bx2 - x, cw)
            by2 = min(by2 - y, ch)

            bw_new = bx2 - bx1
            bh_new = by2 - by1
            if bw_new <= 0 or bh_new <= 0:
                continue  # Box fully outside crop
            if bw_new / cw < 0.01 or bh_new / ch < 0.01:
                continue  # Too small

            new_boxes.append({
                **b,
                "xc": ((bx1 + bx2) / 2) / cw,
                "yc": ((by1 + by2) / 2) / ch,
                "bw": bw_new / cw,
                "bh": bh_new / ch,
            })
        return img, new_boxes


BBOX_TRANSFORM_REGISTRY: dict[str, dict[str, Any]] = {
    "BBoxHorizontalFlip": {"class": BBoxHorizontalFlip, "params": {"p": 0.5}},
    "BBoxVerticalFlip": {"class": BBoxVerticalFlip, "params": {"p": 0.5}},
    "BBoxScale": {"class": BBoxScale, "params": {"scale_limit": (0.5, 1.5), "p": 0.5}},
    "BBoxLetterboxResize": {"class": BBoxLetterboxResize, "params": {"target_h": 640, "target_w": 640}},
    "BBoxColorJitter": {"class": BBoxColorJitter, "params": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1, "p": 0.5}},
    "BBoxRandomCrop": {"class": BBoxRandomCrop, "params": {"crop_h": 224, "crop_w": 224, "p": 1.0}},
}


def build_bbox_pipeline(config: list[dict[str, Any]]) -> BBoxCompose:
    """Build a BBoxCompose pipeline from config."""
    transforms = []
    for item in config:
        name = item["name"]
        if name not in BBOX_TRANSFORM_REGISTRY:
            raise ValueError(f"Unknown bbox transform: {name}")
        cls = BBOX_TRANSFORM_REGISTRY[name]["class"]
        params = {**BBOX_TRANSFORM_REGISTRY[name]["params"], **item.get("params", {})}
        transforms.append(cls(**params))
    return BBoxCompose(transforms)


def augment_with_bboxes(img: NDArray[np.uint8], bboxes: BBox,
                        pipeline: BBoxCompose) -> tuple[NDArray[np.uint8], BBox]:
    """Apply bbox-aware augmentation pipeline."""
    return pipeline(img, bboxes)


# ---------- Config serialization ----------

def save_pipeline_config(config: list[dict[str, Any]], path: str) -> None:
    """Save a pipeline config to a JSON file."""
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def load_pipeline_config(path: str) -> list[dict[str, Any]]:
    """Load a pipeline config from a JSON file."""
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- Registry for GUI/CLI ----------

TRANSFORM_REGISTRY: dict[str, dict[str, Any]] = {
    "RandomHorizontalFlip": {"class": RandomHorizontalFlip, "params": {"p": 0.5}},
    "RandomVerticalFlip": {"class": RandomVerticalFlip, "params": {"p": 0.5}},
    "RandomRotate": {"class": RandomRotate, "params": {"limit": 15.0, "p": 0.5}},
    "RandomScale": {"class": RandomScale, "params": {"scale_limit": (0.8, 1.2), "p": 0.5}},
    "RandomCrop": {"class": RandomCrop, "params": {"crop_h": 224, "crop_w": 224, "p": 1.0}},
    "LetterboxResize": {"class": LetterboxResize, "params": {"target_h": 640, "target_w": 640}},
    "ColorJitter": {"class": ColorJitter, "params": {"brightness": 0.2, "contrast": 0.2,
                                                      "saturation": 0.2, "hue": 0.1, "p": 0.5}},
    "GaussianNoise": {"class": GaussianNoise, "params": {"std_limit": 25.0, "p": 0.5}},
    "SaltAndPepperNoise": {"class": SaltAndPepperNoise, "params": {"amount": 0.01, "p": 0.5}},
    "RandomGaussianBlur": {"class": RandomGaussianBlur, "params": {"ksize_limit": (3, 7), "p": 0.3}},
    "RandomMotionBlur": {"class": RandomMotionBlur, "params": {"kernel_limit": 15, "p": 0.3}},
    "RandomErasing": {"class": RandomErasing, "params": {"area_ratio": (0.02, 0.2), "p": 0.3}},
    "Cutout": {"class": Cutout, "params": {"n_holes": 1, "hole_size": 32, "p": 0.5}},
    "Normalize": {"class": Normalize, "params": {"mean": (0.485, 0.456, 0.406),
                                                   "std": (0.229, 0.224, 0.225)}},
    "LongestMaxSize": {"class": LongestMaxSize, "params": {"max_size": 640}},
}


def build_pipeline(config: list[dict[str, Any]]) -> Compose:
    """Build a Compose pipeline from a config list.
    Example: [{"name": "RandomHorizontalFlip", "params": {"p": 0.5}},
              {"name": "ColorJitter", "params": {"brightness": 0.3}}]
    """
    transforms = []
    for item in config:
        name = item["name"]
        if name not in TRANSFORM_REGISTRY:
            raise ValueError(f"Unknown transform: {name}")
        cls = TRANSFORM_REGISTRY[name]["class"]
        params = {**TRANSFORM_REGISTRY[name]["params"], **item.get("params", {})}
        transforms.append(cls(**params))
    return Compose(transforms)


def augment_image(img: NDArray[np.uint8], pipeline: Compose) -> NDArray[np.uint8]:
    """Apply augmentation pipeline to a single image."""
    return pipeline(img)


def augment_dataset(input_dir: str, output_dir: str, config: list[dict[str, Any]],
                    copies: int = 1, seed: int | None = None,
                    max_workers: int = 4, fmt: str = "png", quality: int = 95,
                    progress_callback: Callable[[int, int], None] | None = None,
                    cancel_check: Callable[[], bool] | None = None) -> dict[str, int]:
    """Apply augmentation pipeline to all images in a directory.

    Args:
        input_dir: Directory containing source images
        output_dir: Directory for augmented output
        config: Pipeline config list for build_pipeline()
        copies: Number of augmented copies per input image
        seed: Random seed for reproducibility
        max_workers: Thread pool size
        fmt: Output format (png, jpg, webp)
        quality: Output quality (1-100)
        progress_callback: Called with (current, total)
        cancel_check: Return True to cancel

    Returns:
        dict with 'total_input', 'total_output', 'errors' counts
    """
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from core.image_io import read_image, write_image
    from utils.helpers import get_image_files, ensure_dir

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    ensure_dir(output_dir)
    files = get_image_files(input_dir)
    pipeline = build_pipeline(config)
    total_tasks = len(files) * copies
    stats = {"total_input": len(files), "total_output": 0, "errors": 0}
    base_seed = seed

    def process_one(f: str, copy_idx: int) -> bool:
        if base_seed is not None:
            task_seed = (base_seed + hash(f) + copy_idx) % (2**32)
            random.seed(task_seed)
            np.random.seed(task_seed)
        img = read_image(f)
        if img is None:
            return False
        result = augment_image(img, pipeline)
        # Normalize and similar transforms may return float32; clip back to uint8 for saving
        if result.dtype != np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        base = os.path.splitext(os.path.basename(f))[0]
        suffix = f"_{copy_idx}" if copies > 1 else ""
        out = os.path.join(output_dir, f"{base}{suffix}.{fmt}")
        write_image(out, result, quality)
        return True

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for f in files:
            for c in range(copies):
                fut = pool.submit(process_one, f, c)
                futures[fut] = (f, c)

        for future in as_completed(futures):
            if cancel_check and cancel_check():
                pool.shutdown(wait=False, cancel_futures=True)
                break
            try:
                if future.result():
                    stats["total_output"] += 1
                else:
                    stats["errors"] += 1
            except Exception as e:
                logger.warning("Augmentation task failed: %s", e)
                stats["errors"] += 1
            done += 1
            if progress_callback:
                progress_callback(done, total_tasks)

    return stats
