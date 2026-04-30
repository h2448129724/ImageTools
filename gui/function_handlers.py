"""Function handlers: extracted from MainWindow to separate concerns."""
from __future__ import annotations

import os
import cv2
import numpy as np

from core.image_io import read_image, write_image, resize_image, convert_format
from core.color_conversion import convert_color, extract_channel
from core.basic_processing import (crop_image, center_crop, pad_image, rotate_image, flip_image,
                                    adjust_brightness_contrast, adjust_saturation, histogram_equalize,
                                    apply_filter, edge_detect, threshold_image, morphology_op,
                                    remove_alpha, add_alpha, overlay_image)
from core.tiling import tile_image, grid_tile, tile_image_file
from core.annotation import (draw_yolo_boxes, draw_coco_boxes, validate_yolo_annotations,
                              crop_roi_from_yolo)
from utils.helpers import get_output_path, ensure_dir


# ---- Color conversion key mapping ----
_COLOR_KEY_MAP = {
    "color_bgr2rgb": "BGR → RGB", "color_rgb2bgr": "RGB → BGR",
    "color_bgr2hsv": "BGR → HSV", "color_hsv2bgr": "HSV → BGR",
    "color_bgr2lab": "BGR → LAB", "color_lab2bgr": "LAB → BGR",
    "color_bgr2gray": "BGR → GRAY", "color_gray2bgr": "GRAY → BGR",
    "color_bgr2yuv": "BGR → YUV", "color_yuv2bgr": "YUV → BGR",
    "color_bgr2hls": "BGR → HLS", "color_hls2bgr": "HLS → BGR",
    "color_bgr2ycrcb": "BGR → YCrCb", "color_ycrcb2bgr": "YCrCb → BGR",
}

_FILTER_MAP = {
    "filter_blur": "blur", "filter_gaussian": "gaussian",
    "filter_median": "median", "filter_bilateral": "bilateral",
    "filter_sharpen": "sharpen",
}

_EDGE_MAP = {
    "edge_canny": "canny", "edge_sobel": "sobel", "edge_laplacian": "laplacian",
}

_CH_MAP = {"B (0)": 0, "G (1)": 1, "R (2)": 2, "A (3)": 3}


def _handle_resize(img, params):
    if params.get("scale", 0) > 0:
        return resize_image(img, scale=params["scale"], keep_aspect=True)
    return resize_image(img, params.get("width", 0), params.get("height", 0),
                        keep_aspect=params.get("keep_aspect", True))


def _handle_overlay(img, params):
    path = params.get("overlay_path", "")
    if path and os.path.exists(path):
        fg = read_image(path)
        if fg is not None:
            return overlay_image(img, fg, params.get("x", 0), params.get("y", 0),
                                 params.get("opacity", 1.0))
    return img


def _handle_remove_alpha(img, params):
    bg = (255, 255, 255) if params.get("bg_color", "white") == "white" else (0, 0, 0)
    return remove_alpha(img, bg)


def _handle_border(img, params):
    cmap = {"black": (0, 0, 0), "white": (255, 255, 255), "red": (0, 0, 255),
            "green": (0, 255, 0), "blue": (255, 0, 0)}
    b = params.get("border_size", 10)
    return cv2.copyMakeBorder(img, b, b, b, b, cv2.BORDER_CONSTANT,
                              value=cmap.get(params.get("color", "black"), (0, 0, 0)))


def _handle_channel_extract(img, params):
    return extract_channel(img, _CH_MAP.get(params.get("channel", "B (0)"), 0))


_SIMPLE_DISPATCH: dict[str, Callable] = {}


def _init_simple_dispatch():
    global _SIMPLE_DISPATCH
    _SIMPLE_DISPATCH = {
        "resize": _handle_resize,
        "crop": lambda img, p: crop_image(img, p.get("x", 0), p.get("y", 0), p.get("w", 256), p.get("h", 256)),
        "center_crop": lambda img, p: center_crop(img, p.get("w", 512), p.get("h", 512)),
        "rotate": lambda img, p: rotate_image(img, p.get("angle", 90), keep_size=p.get("keep_size", False)),
        "flip": lambda img, p: flip_image(img, p.get("direction", "horizontal")),
        "brightness_contrast": lambda img, p: adjust_brightness_contrast(img, p.get("brightness", 0), p.get("contrast", 1.0)),
        "saturation": lambda img, p: adjust_saturation(img, p.get("factor", 1.5)),
        "histogram_eq": lambda img, p: histogram_equalize(img, p.get("adaptive", False), p.get("clip_limit", 2.0), p.get("tile_size", 8)),
        "threshold": lambda img, p: threshold_image(img, p.get("method", "otsu"), p.get("thresh", 127), p.get("maxval", 255), p.get("block_size", 11)),
        "morphology": lambda img, p: morphology_op(img, p.get("op_type", "erode"), p.get("ksize", 3), p.get("iterations", 1)),
        "pad": lambda img, p: pad_image(img, p.get("top", 10), p.get("bottom", 10), p.get("left", 10), p.get("right", 10), p.get("mode", "constant")),
        "remove_alpha": _handle_remove_alpha,
        "add_alpha": lambda img, _: add_alpha(img),
        "format_convert": lambda img, _: img,
        "overlay": _handle_overlay,
        "channel_extract": _handle_channel_extract,
        "batch_add_border": _handle_border,
    }
    # Add color conversions
    for key, color_key in _COLOR_KEY_MAP.items():
        _SIMPLE_DISPATCH[key] = lambda img, p, ck=color_key: convert_color(img, ck)
    # Add filters
    for key, ftype in _FILTER_MAP.items():
        _SIMPLE_DISPATCH[key] = lambda img, p, ft=ftype: apply_filter(img, ft, p.get("ksize", 5))
    # Add edge detection
    for key, etype in _EDGE_MAP.items():
        _SIMPLE_DISPATCH[key] = lambda img, p, et=etype: edge_detect(img, et, p.get("threshold1", 100), p.get("threshold2", 200))


_init_simple_dispatch()


def apply_simple(func_key: str, img: np.ndarray, params: dict) -> np.ndarray | None:
    """Apply a simple image→image function. Returns transformed image or None."""
    handler = _SIMPLE_DISPATCH.get(func_key)
    if handler:
        return handler(img, params)
    return img


def _export_onnx(params: dict, output_dir: str | None, log_fn=None) -> np.ndarray | None:
    """Export a YOLO .pt model to ONNX format."""
    model_path = params.get("model_path", "")
    if not model_path or not os.path.exists(model_path):
        if log_fn:
            log_fn("错误: 请指定有效的模型文件路径")
        return None

    try:
        from ultralytics import YOLO
    except ImportError:
        if log_fn:
            log_fn("错误: 未安装ultralytics，请执行: pip install ultralytics")
        return None

    kwargs = {
        "format": "onnx",
        "imgsz": params.get("imgsz", 640),
        "opset": params.get("opset", 11),
        "simplify": params.get("simplify", True),
        "dynamic": params.get("dynamic", False),
        "half": params.get("half", False),
    }
    device = params.get("device", "").strip()
    if device:
        kwargs["device"] = device

    if log_fn:
        log_fn(f"正在导出ONNX: {os.path.basename(model_path)} (imgsz={kwargs['imgsz']}, opset={kwargs['opset']})")

    model = YOLO(model_path)
    result = model.export(**kwargs)

    if output_dir:
        from pathlib import Path
        out_name = Path(model_path).stem + ".onnx"
        out_path = os.path.join(output_dir, out_name)
        Path(result).rename(out_path)
        if log_fn:
            log_fn(f"导出完成: {out_path}")
    else:
        if log_fn:
            log_fn(f"导出完成: {result}")

    return None


def apply_complex(func_key: str, img: np.ndarray, params: dict,
                  filepath: str | None = None, output_dir: str | None = None,
                  batch_mode: bool = False, log_fn=None) -> np.ndarray | None:
    """Apply a complex function that may need filepath/output_dir context."""

    if func_key == "tile_fixed":
        if batch_mode and filepath and output_dir:
            tile_image_file(filepath, output_dir, params.get("tile_w", 512),
                            params.get("tile_h", 512), params.get("overlap", 0),
                            params.get("discard_incomplete", True))
            return None
        tiles = tile_image(img, params.get("tile_w", 512), params.get("tile_h", 512),
                           params.get("overlap", 0), params.get("discard_incomplete", True))
        if log_fn:
            log_fn(f"切块: {len(tiles)} 块")
        if not tiles:
            return img
        # Save all tiles to disk when output_dir is available
        if filepath and output_dir:
            base = os.path.splitext(os.path.basename(filepath))[0]
            tile_dir = os.path.join(output_dir, base + "_tiles")
            ensure_dir(tile_dir)
            for i, (t, x, y, tw, th) in enumerate(tiles):
                write_image(os.path.join(tile_dir, f"{base}_x{x:04d}_y{y:04d}.png"), t)
            if log_fn:
                log_fn(f"已保存 {len(tiles)} 个切块到 {tile_dir}")
        if batch_mode:
            return None
        return tiles[0][0]

    if func_key == "tile_grid":
        tiles = grid_tile(img, params.get("rows", 3), params.get("cols", 3))
        if log_fn:
            log_fn(f"网格切块: {len(tiles)} 块")
        if not tiles:
            return img
        if filepath and output_dir:
            base = os.path.splitext(os.path.basename(filepath))[0]
            tile_dir = os.path.join(output_dir, base + "_tiles")
            ensure_dir(tile_dir)
            rh = max(1, img.shape[0] // params.get("rows", 3))
            rw = max(1, img.shape[1] // params.get("cols", 3))
            for i, (tile, x, y, tw, th) in enumerate(tiles):
                write_image(os.path.join(tile_dir, f"{base}_r{y // rh:02d}_c{x // rw:02d}.png"), tile)
            if log_fn:
                log_fn(f"已保存 {len(tiles)} 个切块到 {tile_dir}")
        if batch_mode:
            return None
        return tiles[0][0]

    if func_key == "annot_draw_yolo":
        txt_dir = params.get("txt_dir", "")
        cats = ([n.strip() for n in params.get("categories", "").split(",")]
                if params.get("categories") else None)
        if filepath and txt_dir:
            base = os.path.splitext(os.path.basename(filepath))[0]
            return draw_yolo_boxes(img, os.path.join(txt_dir, base + ".txt"), cats)
        return img

    if func_key == "annot_draw_coco":
        coco_path = params.get("coco_path", "")
        if coco_path and filepath:
            return draw_coco_boxes(img, coco_path, image_name=os.path.basename(filepath))
        return img

    if func_key == "annot_crop_roi":
        txt_dir = params.get("txt_dir", "")
        cats = ([n.strip() for n in params.get("categories", "").split(",")]
                if params.get("categories") else None)
        if filepath and txt_dir and output_dir:
            base = os.path.splitext(os.path.basename(filepath))[0]
            count = crop_roi_from_yolo(filepath, os.path.join(txt_dir, base + ".txt"),
                                       os.path.join(output_dir, "roi_crops"),
                                       cats, params.get("padding", 0))
            if log_fn:
                log_fn(f"ROI裁剪: {count} 个目标")
        return None

    if func_key == "annot_validate_yolo":
        if batch_mode and filepath:
            txt_dir = params.get("txt_dir", "")
            base = os.path.splitext(os.path.basename(filepath))[0]
            txt_path = os.path.join(txt_dir, base + ".txt")
            for issue in validate_yolo_annotations(txt_path, img.shape[1], img.shape[0]):
                if log_fn:
                    log_fn(f"[{base}] {issue}")
        return None

    if func_key == "batch_roi_crop":
        x, y = params.get("x", 0), params.get("y", 0)
        w, h = params.get("w", 400), params.get("h", 400)
        ih, iw = img.shape[:2]
        x2, y2 = min(x + w, iw), min(y + h, ih)
        if x2 <= x or y2 <= y:
            return None
        return img[y:y2, x:x2].copy()

    if func_key == "batch_add_border":
        return apply_simple(func_key, img, params)

    if func_key == "export_onnx":
        return _export_onnx(params, output_dir, log_fn)

    return img


def is_batch_function(func_key: str) -> bool:
    """Check if a function requires batch processing mode (legacy, used by tests)."""
    return needs_batch_mode(func_key, has_single_image=False)


def needs_batch_mode(func_key: str, has_single_image: bool = False) -> bool:
    """Determine if function should use batch processing pipeline.

    When has_single_image=True and the function supports single-image mode,
    returns False so the single-image path is used instead.
    """
    from core.function_registry import get_function_def, can_single_image
    if has_single_image and can_single_image(func_key):
        return False

    func_def = get_function_def(func_key)
    if func_def:
        if func_def.handler_type == "batch":
            return True
        if func_def.handler_type == "complex" and func_key.startswith("annot_"):
            return True
    # Fallback for keys not in registry
    return (func_key.startswith("batch_") or func_key.startswith("dataset_") or
            func_key.startswith("augment_") or
            func_key in ("seg_tile", "mask_to_polygons", "polygons_to_mask",
                         "export_onnx", "yolo_train"))
