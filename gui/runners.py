"""Batch operation runners extracted from MainWindow.

Each runner builds a task function and delegates execution to the provided
worker launcher. This keeps MainWindow focused on UI concerns.
"""
from __future__ import annotations

import os
from typing import Callable

from core.dataset_split import random_split, stratified_split, kfold_split
from core.format_conversion import (
    yolo_to_coco, coco_to_yolo, voc_to_yolo, yolo_to_voc,
    voc_to_coco, coco_to_voc, create_classification_dataset,
    xanylabeling_to_yolo,
)
from core.batch_processing import (
    batch_rename, batch_resize, batch_convert_format,
    deduplicate_images, batch_add_border,
)
from core.annotation import annotation_statistics
from core.mask_polygon import batch_mask_to_labelme, batch_labelme_to_mask
from core.annotation_augment import augment_batch
from core.segmentation_tiling import tile_segmentation_dataset
from gui.function_handlers import apply_complex

_TRANSFORM_KWARGS: dict[str, Callable[[dict], dict]] = {
    "flip": lambda p: {"direction": p.get("direction", "horizontal")},
    "rotate": lambda p: {"angle": p.get("angle", 90), "keep_size": p.get("keep_size", True)},
    "resize": lambda p: {"new_w": p.get("new_w", 640), "new_h": p.get("new_h", 480)},
    "crop": lambda p: {"x": p.get("crop_x", 0), "y": p.get("crop_y", 0),
                       "w": p.get("crop_w", 256), "h": p.get("crop_h", 256)},
}

_BORDER_COLORS = {
    "black": (0, 0, 0), "white": (255, 255, 255), "red": (0, 0, 255),
    "green": (0, 255, 0), "blue": (255, 0, 0),
}


def _input_dir_from_files(files: list[str]) -> str:
    """Derive an input directory from the first selected file."""
    path = files[0] if files else ""
    return os.path.dirname(path) if os.path.isfile(path) else path


class RunnerController:
    """Encapsulates batch operation logic and delegates worker execution."""

    def __init__(
        self,
        run_worker: Callable,
        log_fn: Callable[[str], None],
    ) -> None:
        self._run_worker = run_worker
        self._log = log_fn

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------
    def run_export_onnx(self, params: dict, output_dir: str | None) -> None:
        def run():
            return apply_complex("export_onnx", None, params, output_dir=output_dir, log_fn=self._log)
        self._run_worker(run, lambda _: self._log("ONNX导出完成"))

    # ------------------------------------------------------------------
    # Dataset split
    # ------------------------------------------------------------------
    def run_dataset_op(self, func_key: str, params: dict, output_dir: str, files: list[str]) -> None:
        input_dir = _input_dir_from_files(files)

        def _random_split():
            r = (params.get("train_ratio", 0.7), params.get("val_ratio", 0.2),
                 params.get("test_ratio", 0.1))
            return random_split(input_dir, output_dir, r, params.get("label_dir") or None,
                                params.get("seed", 42))

        def _stratified_split():
            r = (params.get("train_ratio", 0.7), params.get("val_ratio", 0.2),
                 params.get("test_ratio", 0.1))
            return stratified_split(input_dir, output_dir, r, params.get("seed", 42))

        def _kfold():
            return kfold_split(input_dir, output_dir, params.get("k", 5), params.get("seed", 42))

        _DATASET_RUNNERS: dict[str, Callable] = {
            "dataset_random_split": _random_split,
            "dataset_stratified_split": _stratified_split,
            "dataset_kfold": _kfold,
        }

        def run():
            runner = _DATASET_RUNNERS.get(func_key)
            return runner() if runner else None

        self._run_worker(run, lambda r: self._log(f"数据集划分完成: {r}"))

    # ------------------------------------------------------------------
    # Segmentation tiling
    # ------------------------------------------------------------------
    def run_seg_tile_op(self, func_key: str, params: dict, output_dir: str, files: list[str]) -> None:
        image_dir = _input_dir_from_files(files)
        ann_dir = params.get("ann_dir", "")
        if not ann_dir:
            raise ValueError("请指定标注文件夹(JSON)路径")

        def run():
            return tile_segmentation_dataset(
                image_dir, ann_dir, output_dir,
                params.get("tile_w", 256), params.get("tile_h", 256),
                params.get("overlap", 0), params.get("discard_empty", False),
                params.get("discard_incomplete", True))

        def on_done(r):
            self._log(f"分割切块: {r['total_pairs']}对 → {r['total_tiles']}块 "
                      f"(跳过空:{r['skipped_empty']} 跳过不完整:{r['skipped_incomplete']})")
            self._log(f"图片: {r['output_image_dir']}")
            self._log(f"标注: {r['output_ann_dir']}")

        self._run_worker(run, on_done)

    # ------------------------------------------------------------------
    # Format conversion
    # ------------------------------------------------------------------
    def run_format_op(self, func_key: str, params: dict, output_dir: str, files: list[str]) -> None:
        def _yolo2coco():
            cats = [c.strip() for c in params.get("categories", "class0,class1").split(",")]
            return yolo_to_coco(params.get("yolo_dir", ""), params.get("image_dir", ""),
                                os.path.join(output_dir, "coco.json"), cats)

        def _coco2yolo():
            return coco_to_yolo(params.get("coco_path", ""), output_dir)

        def _voc2yolo():
            cats = [c.strip() for c in params.get("categories", "class0,class1").split(",")]
            return voc_to_yolo(params.get("voc_dir", ""), output_dir, cats)

        def _yolo2voc():
            cats = [c.strip() for c in params.get("categories", "class0,class1").split(",")]
            return yolo_to_voc(params.get("yolo_dir", ""), params.get("image_dir", ""),
                               output_dir, cats)

        def _voc2coco():
            cats = [c.strip() for c in params.get("categories", "class0,class1").split(",")]
            return voc_to_coco(params.get("voc_dir", ""), params.get("image_dir", ""),
                               os.path.join(output_dir, "coco.json"), cats)

        def _coco2voc():
            return coco_to_voc(params.get("coco_path", ""), output_dir)

        def _xanylabeling2yolo():
            src_dir = params.get("src_dir", "")
            if not src_dir and files:
                src_dir = os.path.dirname(files[0])
            cats_str = params.get("categories", "").strip()
            cats = [c.strip() for c in cats_str.split(",") if c.strip()] if cats_str else None
            return xanylabeling_to_yolo(
                src_dir, output_dir,
                val_ratio=params.get("val_ratio", 0.2),
                seed=params.get("seed", 42),
                categories=cats)

        def _classification():
            input_dir = os.path.dirname(files[0]) if files else ""
            return create_classification_dataset(input_dir, output_dir)

        _FORMAT_RUNNERS: dict[str, Callable] = {
            "format_yolo2coco": _yolo2coco,
            "format_coco2yolo": _coco2yolo,
            "format_voc2yolo": _voc2yolo,
            "format_yolo2voc": _yolo2voc,
            "format_voc2coco": _voc2coco,
            "format_coco2voc": _coco2voc,
            "format_xanylabeling2yolo": _xanylabeling2yolo,
            "format_classification": _classification,
        }

        def run():
            runner = _FORMAT_RUNNERS.get(func_key)
            return runner() if runner else None

        self._run_worker(run, lambda r: self._log(f"格式转换完成: {r}"))

    # ------------------------------------------------------------------
    # Mask / polygon conversion
    # ------------------------------------------------------------------
    def run_mask_polygon_op(self, func_key: str, params: dict, output_dir: str, files: list[str]) -> None:
        def _mask_to_polygons():
            mask_dir = params.get("mask_dir", "")
            if not mask_dir and files:
                mask_dir = files[0]
                if os.path.isfile(mask_dir):
                    mask_dir = os.path.dirname(mask_dir)
            return batch_mask_to_labelme(
                mask_dir, output_dir,
                params.get("label", "object"),
                params.get("epsilon_factor", 0.001))

        def _polygons_to_mask():
            ann_dir = params.get("ann_dir", "")
            image_dir = params.get("image_dir") or None
            if image_dir:
                dh, dw = None, None
            else:
                dh = params.get("default_h") or None
                dw = params.get("default_w") or None
            return batch_labelme_to_mask(
                ann_dir, output_dir,
                image_dir, dh, dw)

        _MASK_RUNNERS: dict[str, Callable] = {
            "mask_to_polygons": _mask_to_polygons,
            "polygons_to_mask": _polygons_to_mask,
        }

        def run():
            runner = _MASK_RUNNERS.get(func_key)
            return runner() if runner else None

        self._run_worker(run, lambda r: self._log(f"转换完成: {r}"))

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------
    def run_augment_op(self, func_key: str, params: dict, output_dir: str, files: list[str]) -> None:
        input_dir = _input_dir_from_files(files)
        ann_dir = params.get("ann_dir", "")
        if not ann_dir:
            raise ValueError("请指定标注目录")

        transform_type = params.get("transform_type", "flip")
        ann_format = "yolo" if func_key == "augment_yolo" else "labelme"

        kwargs = _TRANSFORM_KWARGS.get(transform_type, lambda _: {})(params)

        def run():
            return augment_batch(input_dir, ann_dir, output_dir,
                                 transform_type, ann_format, **kwargs)

        self._run_worker(run, lambda r: self._log(f"标注增强完成: {r}"))

    # ------------------------------------------------------------------
    # Simple batch operations
    # ------------------------------------------------------------------
    def run_simple_batch_op(self, func_key: str, params: dict, output_dir: str, files: list[str]) -> None:
        input_dir = _input_dir_from_files(files)

        def _deduplicate():
            return ("deduplicate", deduplicate_images(
                input_dir,
                mode=params.get("mode", "exact"),
                similarity_threshold=params.get("similarity_threshold", 10)))

        def _rename():
            return ("rename", batch_rename(input_dir, output_dir, params.get("prefix", "img_"),
                                           params.get("start_index", 1), params.get("digits", 4),
                                           params.get("keep_ext", True)))

        def _convert():
            fmt = params.get("fmt", "png")
            return ("convert", batch_convert_format(input_dir, output_dir, fmt))

        def _resize():
            return ("resize", batch_resize(input_dir, output_dir,
                                           params.get("width", 0), params.get("height", 0),
                                           params.get("scale", 0), params.get("keep_aspect", True)))

        def _border():
            return ("border", batch_add_border(input_dir, output_dir,
                                               params.get("border_size", 10),
                                               _BORDER_COLORS.get(params.get("color", "black"), (0, 0, 0))))

        def _stats():
            s = annotation_statistics(params.get("ann_dir", ""), params.get("img_dir", ""),
                                      params.get("format_type", "yolo"))
            return ("stats", s)

        _BATCH_RUNNERS: dict[str, Callable] = {
            "batch_deduplicate": _deduplicate,
            "batch_rename": _rename,
            "batch_convert_format": _convert,
            "batch_resize": _resize,
            "batch_add_border": _border,
            "annot_statistics": _stats,
        }

        def run():
            runner = _BATCH_RUNNERS.get(func_key)
            return runner() if runner else ("unknown", None)

        def on_finish(result):
            op, data = result
            if op == "deduplicate":
                dupes = data
                for a, b in dupes[:10]:
                    self._log(f"重复: {os.path.basename(a)} = {os.path.basename(b)}")
                self._log(f"发现 {len(dupes)} 组重复")
            elif op == "stats":
                self._log(f"标注统计: {data}")
            else:
                self._log(f"操作完成: {data}")

        self._run_worker(run, on_finish)
