import cv2
import numpy as np


def crop_image(img, x, y, w, h):
    return img[y:y + h, x:x + w].copy()


def center_crop(img, w, h):
    ih, iw = img.shape[:2]
    x = max(0, (iw - w) // 2)
    y = max(0, (ih - h) // 2)
    return img[y:y + h, x:x + w].copy()


def pad_image(img, top, bottom, left, right, mode="constant", value=(0, 0, 0)):
    """Pad image edges. mode: constant, reflect, replicate."""
    border_map = {"constant": cv2.BORDER_CONSTANT, "reflect": cv2.BORDER_REFLECT,
                  "replicate": cv2.BORDER_REPLICATE}
    return cv2.copyMakeBorder(img, top, bottom, left, right,
                              border_map.get(mode, cv2.BORDER_CONSTANT),
                              value=value if mode == "constant" else None)


def rotate_image(img, angle, center=None, scale=1.0, keep_size=True):
    h, w = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    if keep_size:
        return cv2.warpAffine(img, M, (w, h))
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += new_w / 2 - center[0]
    M[1, 2] += new_h / 2 - center[1]
    return cv2.warpAffine(img, M, (new_w, new_h))


def flip_image(img, direction):
    """direction: 'horizontal' (1), 'vertical' (0), 'both' (-1)"""
    code = {"horizontal": 1, "vertical": 0, "both": -1}
    return cv2.flip(img, code.get(direction, 1))


def adjust_brightness_contrast(img, brightness=0, contrast=1.0):
    """brightness: -255 to 255, contrast: >0 (1.0 = no change)"""
    img = img.astype(np.float32)
    img = contrast * img + brightness
    return np.clip(img, 0, 255).astype(np.uint8)


def adjust_saturation(img, factor=1.0):
    """Adjust saturation. factor: 0=grayscale, 1.0=original, >1=more saturated."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def histogram_equalize(img, adaptive=False, clip_limit=2.0, tile_size=8):
    """Histogram equalization. adaptive=CLAHE for localized."""
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        if adaptive:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
            l = clahe.apply(l)
        else:
            l = cv2.equalizeHist(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if adaptive:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        return clahe.apply(img)
    return cv2.equalizeHist(img)


def apply_filter(img, filter_type, ksize=3):
    """Apply various filters: blur, gaussian, median, bilateral, sharpen."""
    if filter_type == "blur":
        return cv2.blur(img, (ksize, ksize))
    if filter_type == "gaussian":
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    if filter_type == "median":
        return cv2.medianBlur(img, ksize)
    if filter_type == "bilateral":
        return cv2.bilateralFilter(img, ksize, 75, 75)
    if filter_type == "sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)
    return img


def edge_detect(img, method="canny", threshold1=100, threshold2=200):
    """Edge detection: canny, sobel, laplacian."""
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if method == "canny":
        return cv2.Canny(gray, threshold1, threshold2)
    if method == "sobel":
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return cv2.convertScaleAbs(cv2.magnitude(sx, sy))
    if method == "laplacian":
        return cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F))
    return gray


def threshold_image(img, method="otsu", thresh=127, maxval=255, block_size=11, C=2):
    """Threshold: binary, otsu, adaptive_mean, adaptive_gaussian."""
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if method == "binary":
        _, result = cv2.threshold(gray, thresh, maxval, cv2.THRESH_BINARY)
        return result
    if method == "otsu":
        _, result = cv2.threshold(gray, 0, maxval, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return result
    if method == "adaptive_mean":
        return cv2.adaptiveThreshold(gray, maxval, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, block_size | 1, C)
    if method == "adaptive_gaussian":
        return cv2.adaptiveThreshold(gray, maxval, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, block_size | 1, C)
    return gray


def morphology_op(img, op_type="erode", ksize=3, iterations=1):
    """Morphological operations: erode, dilate, open, close."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    ops = {"erode": cv2.MORPH_ERODE, "dilate": cv2.MORPH_DILATE,
           "open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE}
    return cv2.morphologyEx(img, ops.get(op_type, cv2.MORPH_ERODE), kernel, iterations=iterations)


def remove_alpha(img, bg_color=(255, 255, 255)):
    """Flatten alpha channel onto a background color."""
    if len(img.shape) < 3 or img.shape[2] != 4:
        return img
    bgr = img[:, :, :3]
    alpha = img[:, :, 3:4] / 255.0
    bg = np.full_like(bgr, bg_color, dtype=np.uint8)
    return (bgr * alpha + bg * (1 - alpha)).astype(np.uint8)


def add_alpha(img, alpha_value=255):
    """Add an alpha channel to an image."""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        alpha = np.full((img.shape[0], img.shape[1], 1), alpha_value, dtype=img.dtype)
        return np.concatenate([img, alpha], axis=2)
    return img


def overlay_image(background, foreground, x=0, y=0, opacity=1.0):
    """Overlay foreground on background with alpha support."""
    result = background.copy()
    fg_h, fg_w = foreground.shape[:2]
    bg_h, bg_w = result.shape[:2]
    x = max(0, min(x, bg_w)); y = max(0, min(y, bg_h))
    roi_w = min(fg_w, bg_w - x); roi_h = min(fg_h, bg_h - y)

    if foreground.shape[2] == 4:
        fg_bgr = foreground[:roi_h, :roi_w, :3]
        fg_alpha = (foreground[:roi_h, :roi_w, 3:4] / 255.0) * opacity
        bg_roi = result[y:y + roi_h, x:x + roi_w]
        result[y:y + roi_h, x:x + roi_w] = (fg_bgr * fg_alpha + bg_roi * (1 - fg_alpha)).astype(np.uint8)
    else:
        fg_roi = foreground[:roi_h, :roi_w, :3]
        blended = cv2.addWeighted(fg_roi, opacity, result[y:y + roi_h, x:x + roi_w], 1 - opacity, 0)
        result[y:y + roi_h, x:x + roi_w] = blended
    return result
