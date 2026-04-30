"""Main application window with dark theme, keyboard shortcuts, and recent files."""
import os
import json
import cv2
import numpy as np
from PySide6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                                QSplitter, QMessageBox, QStatusBar, QProgressBar,
                                QTextEdit, QLabel, QPushButton, QApplication, QScrollArea,
                                QLineEdit)
from PySide6.QtCore import Qt, QTimer, QStandardPaths
from PySide6.QtGui import QAction, QKeySequence, QShortcut

from gui.input_panel import InputPanel
from gui.function_panel import FunctionPanel
from gui.param_panel import ParamPanel
from gui.preview_widget import PreviewPanel
from gui.output_panel import OutputPanel
from gui.workers import WorkerThread

from core.image_io import read_image, write_image, resize_image, convert_format
from core.color_conversion import convert_color
from core.basic_processing import *
from core.tiling import tile_image, grid_tile, tile_image_file, tile_directory
from core.dataset_split import random_split, stratified_split, kfold_split
from core.format_conversion import *
from core.batch_processing import *
from core.annotation import *
from core.segmentation_tiling import tile_segmentation_dataset, tile_segmentation_single

from utils.helpers import get_image_files, ensure_dir


RECENT_FILE = os.path.join(
    QStandardPaths.writableLocation(QStandardPaths.AppDataLocation), "img_tools_recent.json"
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像处理工具箱")
        self.resize(1480, 900)
        self._current_image = None
        self._current_file = None
        self._worker = None
        self._recent = self._load_recent()
        self._setup_ui()
        self._setup_menu()
        self._setup_shortcuts()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # === Left panel (scrollable) ===
        left_panel = QWidget()
        left_panel.setFixedWidth(320)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left_panel)

        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(6, 8, 6, 8)
        left_layout.setSpacing(6)

        # Input section
        self.input_panel = InputPanel()
        left_layout.addWidget(self.input_panel)

        # Function section
        self.function_panel = FunctionPanel()
        left_layout.addWidget(self.function_panel, 1)  # stretch factor 1 -> takes remaining space

        # Parameter section
        self.param_panel = ParamPanel()
        left_layout.addWidget(self.param_panel)

        # Output section
        self.output_panel = OutputPanel()
        left_layout.addWidget(self.output_panel)

        # === Right panel (preview) ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(0)

        self.preview = PreviewPanel()
        right_layout.addWidget(self.preview)

        # Image info bar
        info_bar = QHBoxLayout()
        info_bar.setContentsMargins(8, 2, 8, 2)
        self.lbl_img_info = QLabel("未加载图片")
        self.lbl_img_info.setObjectName("imgInfo")
        info_bar.addWidget(self.lbl_img_info)
        info_bar.addStretch()
        self.lbl_coord_display = QLabel("")
        self.lbl_coord_display.setObjectName("imgInfo")
        info_bar.addWidget(self.lbl_coord_display)
        right_layout.addLayout(info_bar)

        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 1100])
        main_layout.addWidget(splitter)

        # === Status bar ===
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(260)
        self.progress_bar.setMaximumHeight(20)
        self.progress_bar.hide()
        self.status_bar.addPermanentWidget(self.progress_bar)

        self.lbl_status = QLabel("就绪")
        self.status_bar.addWidget(self.lbl_status)

        # Log shown as tooltip on status click or small widget
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(90)
        self.log_output.setMinimumHeight(90)
        self.status_bar.addPermanentWidget(self.log_output)

        # === Connect signals ===
        self.input_panel.previewRequested.connect(self._on_preview_file)
        self.input_panel.filesChanged.connect(self._on_files_changed)
        self.function_panel.functionSelected.connect(self.param_panel.set_function)
        self.param_panel.btn_preview.clicked.connect(self._on_preview)
        self.param_panel.btn_run.clicked.connect(self._on_run)
        self.preview.original_view.pixelClicked.connect(self._on_preview_pixel_clicked)

    def _setup_menu(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("文件(&F)")
        open_act = file_menu.addAction("打开图片...\tCtrl+O")
        open_act.triggered.connect(lambda: self.input_panel._add_files())
        open_dir_act = file_menu.addAction("打开文件夹...\tCtrl+D")
        open_dir_act.triggered.connect(lambda: self.input_panel._add_dir())
        file_menu.addSeparator()

        # Recent submenu
        self.recent_menu = file_menu.addMenu("最近打开")
        self._update_recent_menu()
        file_menu.addSeparator()

        quit_act = file_menu.addAction("退出\tCtrl+Q")
        quit_act.triggered.connect(self.close)

        # Help menu
        help_menu = menubar.addMenu("帮助(&H)")
        about_act = help_menu.addAction("关于")
        about_act.triggered.connect(self._show_about)

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+O"), self, lambda: self.input_panel._add_files())
        QShortcut(QKeySequence("Ctrl+D"), self, lambda: self.input_panel._add_dir())
        QShortcut(QKeySequence("Ctrl+R"), self, self._on_run)
        QShortcut(QKeySequence("Ctrl+Q"), self, self.close)
        QShortcut(QKeySequence("Escape"), self, self._on_escape)
        QShortcut(QKeySequence("F5"), self, self._on_run)

    def _on_escape(self):
        """Escape: cancel picker mode if active."""
        if hasattr(self.preview, 'btn_picker') and self.preview.btn_picker.isChecked():
            self.preview.btn_picker.setChecked(False)

    def keyPressEvent(self, event):
        """Handle A/D keys for image navigation (skip when typing in text fields)."""
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit)):
            super().keyPressEvent(event)
            return
        key = event.key()
        if key == Qt.Key_A:
            self._nav_prev()
        elif key == Qt.Key_D:
            self._nav_next()
        else:
            super().keyPressEvent(event)

    def _nav_prev(self):
        lst = self.input_panel.file_list
        if lst.count() == 0:
            return
        cur = lst.currentRow()
        idx = cur - 1 if cur > 0 else lst.count() - 1
        lst.setCurrentRow(idx)
        self._on_preview_file(self.input_panel._files[idx])

    def _nav_next(self):
        lst = self.input_panel.file_list
        if lst.count() == 0:
            return
        cur = lst.currentRow()
        idx = cur + 1 if cur < lst.count() - 1 else 0
        lst.setCurrentRow(idx)
        self._on_preview_file(self.input_panel._files[idx])

    def _on_preview(self):
        """Apply current function to the current image and show result (no save)."""
        params_info = self.param_panel.get_params()
        func_key = params_info["function"]
        params = params_info["params"]

        if not func_key:
            QMessageBox.warning(self, "提示", "请先选择处理功能")
            return
        if self._current_image is None:
            QMessageBox.warning(self, "提示", "请先选择图片")
            return

        img = self._current_image.copy()
        try:
            result = self._apply_function(img, func_key, params)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"预览失败:\n{e}")
            return

        if result is not None and isinstance(result, np.ndarray):
            self.preview.set_result(result)
            self.lbl_status.setText("预览结果（未保存）")
            self._log(f"预览: {func_key}")
        else:
            self._log("此功能无图像输出，无法预览")

    # ---- Preview & image info ----
    def _on_preview_file(self, filepath):
        self._current_file = filepath
        img = read_image(filepath)
        if img is not None:
            self._current_image = img
            self.preview.set_original(img)
            h, w = img.shape[:2]
            c = img.shape[2] if len(img.shape) > 2 else 1
            fsize = os.path.getsize(filepath)
            fsize_str = f"{fsize / 1024:.0f}KB" if fsize < 1024 * 1024 else f"{fsize / 1048576:.1f}MB"
            self.lbl_img_info.setText(
                f"{os.path.basename(filepath)}  |  {w}x{h}  {c}通道  |  {fsize_str}"
            )
            self._add_recent(filepath)
            self._log(f"加载: {os.path.basename(filepath)} ({w}x{h})")

    def _on_files_changed(self, files):
        if files and self._current_image is None:
            self._on_preview_file(files[0])

    def _on_preview_pixel_clicked(self, x, y):
        self.lbl_coord_display.setText(f"点击: ({x}, {y}) → 已复制")

    # ---- Recent files ----
    def _get_recent_path(self):
        return RECENT_FILE

    def _load_recent(self):
        try:
            ensure_dir(os.path.dirname(self._get_recent_path()))
            with open(self._get_recent_path(), "r") as f:
                data = json.load(f)
                return [p for p in data.get("paths", []) if os.path.exists(p)]
        except Exception:
            return []

    def _save_recent(self):
        try:
            ensure_dir(os.path.dirname(self._get_recent_path()))
            with open(self._get_recent_path(), "w") as f:
                json.dump({"paths": self._recent[:10]}, f)
        except Exception:
            pass

    def _add_recent(self, path):
        if path in self._recent:
            self._recent.remove(path)
        self._recent.insert(0, path)
        self._recent = self._recent[:10]
        self._save_recent()
        self._update_recent_menu()

    def _update_recent_menu(self):
        self.recent_menu.clear()
        if not self._recent:
            self.recent_menu.addAction("(无)").setEnabled(False)
        else:
            for p in self._recent:
                name = os.path.basename(p) if os.path.isfile(p) else os.path.basename(p.rstrip(os.sep))
                act = self.recent_menu.addAction(name)
                act.setToolTip(p)
                act.triggered.connect(lambda checked, path=p: self._open_recent(path))
            self.recent_menu.addSeparator()
            clear_act = self.recent_menu.addAction("清除记录")
            clear_act.triggered.connect(self._clear_recent)

    def _open_recent(self, path):
        if os.path.isfile(path):
            self._on_preview_file(path)
            self.input_panel._add_paths([path])
        elif os.path.isdir(path):
            self.input_panel._add_paths([path])

    def _clear_recent(self):
        self._recent = []
        self._save_recent()
        self._update_recent_menu()

    # ---- Run / Process ----
    def _on_run(self):
        params_info = self.param_panel.get_params()
        func_key = params_info["function"]
        params = params_info["params"]
        output_dir = self.output_panel.get_output_dir()

        if not func_key:
            QMessageBox.warning(self, "提示", "请先选择处理功能")
            return
        if not output_dir:
            QMessageBox.warning(self, "提示", "请先设置输出目录")
            return

        files = self.input_panel.get_files()
        is_batch = (func_key.startswith("batch_") or func_key.startswith("dataset_") or
                    func_key.startswith("format_") or func_key.startswith("annot_") or
                    func_key == "seg_tile")

        if is_batch and not files:
            QMessageBox.warning(self, "提示", "批处理需要选择图片或文件夹")
            return

        if not is_batch:
            if self._current_image is None:
                QMessageBox.warning(self, "提示", "请先选择图片")
                return
            self._process_single(func_key, params, output_dir)
        else:
            self._process_batch(func_key, params, output_dir, files)

    def _process_single(self, func_key, params, output_dir):
        img = self._current_image.copy()
        try:
            result = self._apply_function(img, func_key, params)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理失败:\n{e}")
            return

        if result is not None:
            self._current_image = result
            self.preview.set_result(result)
            if self._current_file:
                base = os.path.splitext(os.path.basename(self._current_file))[0]
                out_path = os.path.join(output_dir, f"{base}_processed.png")
                ensure_dir(output_dir)
                write_image(out_path, result)
                self._log(f"已保存: {os.path.basename(out_path)}")
                self.lbl_status.setText(f"处理完成 - {os.path.basename(out_path)}")
        else:
            QMessageBox.warning(self, "提示", "此功能不产生图像输出")

    def _process_batch(self, func_key, params, output_dir, files):
        self._log(f"开始批处理: {func_key}, 共 {len(files)} 个文件")

        if func_key in ("dataset_random_split", "dataset_stratified_split", "dataset_kfold"):
            self._run_dataset_op(func_key, params, output_dir, files)
            return
        if func_key.startswith("format_"):
            self._run_format_op(func_key, params, output_dir, files)
            return
        if func_key in ("batch_deduplicate",):
            self._run_simple_batch_op(func_key, params, output_dir, files)
            return
        if func_key in ("annot_statistics", "annot_validate_yolo"):
            self._run_simple_batch_op(func_key, params, output_dir, files)
            return
        if func_key == "seg_tile":
            self._run_seg_tile_op(func_key, params, output_dir, files)
            return

        ensure_dir(output_dir)
        self.progress_bar.setMaximum(len(files))
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.lbl_status.setText(f"处理中... 0/{len(files)}")

        def process():
            for i, f in enumerate(files):
                img = read_image(f)
                if img is not None:
                    try:
                        result = self._apply_function(img, func_key, params, filepath=f,
                                                       output_dir=output_dir, batch_mode=True)
                        if result is not None and isinstance(result, np.ndarray):
                            if func_key == "batch_roi_crop":
                                prefix = params.get("prefix", "crop")
                                out_name = f"{prefix}_{str(i + 1).zfill(4)}.png"
                            else:
                                out_name = os.path.splitext(os.path.basename(f))[0] + "_out.png"
                            write_image(os.path.join(output_dir, out_name), result)
                    except Exception as e:
                        self._log(f"失败 {os.path.basename(f)}: {e}")
                QTimer.singleShot(0, lambda c=i + 1: self._update_progress(c))

        self._run_worker(process, lambda _: self._on_batch_done(len(files)))

    def _update_progress(self, count):
        self.progress_bar.setValue(count)
        self.lbl_status.setText(f"处理中... {count}/{self.progress_bar.maximum()}")

    def _on_batch_done(self, count):
        self.progress_bar.hide()
        self.lbl_status.setText(f"批处理完成 - {count} 个文件")
        self._log(f"批处理完成，处理了 {count} 个文件")

    # ---- Function routing (same as before) ----
    def _apply_function(self, img, func_key, params, filepath=None, output_dir=None, batch_mode=False):
        key_map = {
            "color_bgr2rgb": "BGR → RGB", "color_rgb2bgr": "RGB → BGR",
            "color_bgr2hsv": "BGR → HSV", "color_hsv2bgr": "HSV → BGR",
            "color_bgr2lab": "BGR → LAB", "color_lab2bgr": "LAB → BGR",
            "color_bgr2gray": "BGR → GRAY", "color_gray2bgr": "GRAY → BGR",
            "color_bgr2yuv": "BGR → YUV", "color_bgr2hls": "BGR → HLS",
            "color_bgr2ycrcb": "BGR → YCrCb",
        }
        if func_key in key_map:
            return convert_color(img, key_map[func_key])
        if func_key == "resize":
            scale = params.get("scale", 0)
            if scale > 0:
                return resize_image(img, scale=scale, keep_aspect=True)
            return resize_image(img, params.get("width", 0), params.get("height", 0),
                                keep_aspect=params.get("keep_aspect", True))
        if func_key == "crop":
            return crop_image(img, params.get("x", 0), params.get("y", 0),
                              params.get("w", 256), params.get("h", 256))
        if func_key == "center_crop":
            return center_crop(img, params.get("w", 512), params.get("h", 512))
        if func_key == "rotate":
            return rotate_image(img, params.get("angle", 90), keep_size=params.get("keep_size", False))
        if func_key == "flip":
            return flip_image(img, params.get("direction", "horizontal"))
        if func_key == "brightness_contrast":
            return adjust_brightness_contrast(img, params.get("brightness", 0), params.get("contrast", 1.0))
        if func_key == "saturation":
            return adjust_saturation(img, params.get("factor", 1.5))
        if func_key == "histogram_eq":
            return histogram_equalize(img, params.get("adaptive", False),
                                      params.get("clip_limit", 2.0), params.get("tile_size", 8))
        if func_key == "threshold":
            return threshold_image(img, params.get("method", "otsu"), params.get("thresh", 127),
                                   params.get("maxval", 255), params.get("block_size", 11))
        if func_key == "morphology":
            return morphology_op(img, params.get("op_type", "erode"),
                                 params.get("ksize", 3), params.get("iterations", 1))
        if func_key == "pad":
            return pad_image(img, params.get("top", 10), params.get("bottom", 10),
                             params.get("left", 10), params.get("right", 10), params.get("mode", "constant"))
        if func_key == "remove_alpha":
            bg = (255, 255, 255) if params.get("bg_color", "white") == "white" else (0, 0, 0)
            return remove_alpha(img, bg)
        if func_key == "add_alpha":
            return add_alpha(img)
        if func_key == "overlay":
            overlay_path = params.get("overlay_path", "")
            if overlay_path and os.path.exists(overlay_path):
                fg = read_image(overlay_path)
                if fg is not None:
                    return overlay_image(img, fg, params.get("x", 0), params.get("y", 0),
                                         params.get("opacity", 1.0))
            return img
        if func_key == "channel_extract":
            ch_map = {"B (0)": 0, "G (1)": 1, "R (2)": 2, "A (3)": 3}
            return extract_channel(img, ch_map.get(params.get("channel", "B (0)"), 0))
        filter_map = {"filter_blur": "blur", "filter_gaussian": "gaussian",
                      "filter_median": "median", "filter_bilateral": "bilateral", "filter_sharpen": "sharpen"}
        if func_key in filter_map:
            return apply_filter(img, filter_map[func_key], params.get("ksize", 5))
        edge_map = {"edge_canny": "canny", "edge_sobel": "sobel", "edge_laplacian": "laplacian"}
        if func_key in edge_map:
            return edge_detect(img, edge_map[func_key], params.get("threshold1", 100),
                               params.get("threshold2", 200))
        if func_key == "tile_fixed":
            if batch_mode and filepath and output_dir:
                tile_image_file(filepath, output_dir, params.get("tile_w", 512),
                                params.get("tile_h", 512), params.get("overlap", 0),
                                params.get("discard_incomplete", True))
                return None
            tiles = tile_image(img, params.get("tile_w", 512), params.get("tile_h", 512),
                               params.get("overlap", 0), params.get("discard_incomplete", True))
            self._log(f"切块: {len(tiles)} 块")
            return img
        if func_key == "tile_grid":
            tiles = grid_tile(img, params.get("rows", 3), params.get("cols", 3))
            self._log(f"网格切块: {len(tiles)} 块")
            if batch_mode and filepath and output_dir:
                base = os.path.splitext(os.path.basename(filepath))[0]
                tile_dir = os.path.join(output_dir, base + "_tiles"); ensure_dir(tile_dir)
                rh = img.shape[0] // params.get("rows", 3); rw = img.shape[1] // params.get("cols", 3)
                for i, (tile, x, y, tw, th) in enumerate(tiles):
                    write_image(os.path.join(tile_dir, f"{base}_r{y//rh:02d}_c{x//rw:02d}.png"), tile)
                return None
            return img
        if func_key == "annot_draw_yolo":
            txt_dir = params.get("txt_dir", "")
            cats = [n.strip() for n in params.get("categories", "").split(",")] if params.get("categories") else None
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
            cats = [n.strip() for n in params.get("categories", "").split(",")] if params.get("categories") else None
            if filepath and txt_dir and output_dir:
                base = os.path.splitext(os.path.basename(filepath))[0]
                count = crop_roi_from_yolo(filepath, os.path.join(txt_dir, base + ".txt"),
                                           os.path.join(output_dir, "roi_crops"),
                                           cats, params.get("padding", 0))
                self._log(f"ROI裁剪: {count} 个目标")
            return None
        if func_key == "annot_validate_yolo":
            if batch_mode and filepath:
                txt_dir = params.get("txt_dir", "")
                base = os.path.splitext(os.path.basename(filepath))[0]
                txt_path = os.path.join(txt_dir, base + ".txt")
                for issue in validate_yolo_annotations(txt_path, img.shape[1], img.shape[0]):
                    self._log(f"[{base}] {issue}")
            return None
        if func_key == "batch_roi_crop":
            x, y, w, h = params.get("x", 0), params.get("y", 0), params.get("w", 400), params.get("h", 400)
            ih, iw = img.shape[:2]
            x2, y2 = min(x + w, iw), min(y + h, ih)
            if x2 <= x or y2 <= y:
                return None
            return img[y:y2, x:x2].copy()
        if func_key == "batch_add_border":
            cmap = {"black": (0, 0, 0), "white": (255, 255, 255), "red": (0, 0, 255),
                    "green": (0, 255, 0), "blue": (255, 0, 0)}
            b = params.get("border_size", 10)
            return cv2.copyMakeBorder(img, b, b, b, b, cv2.BORDER_CONSTANT,
                                      value=cmap.get(params.get("color", "black"), (0, 0, 0)))
        return img

    # ---- Dataset / Format / Segment ops ----
    def _run_dataset_op(self, func_key, params, output_dir, files):
        input_dir = files[0] if files else ""
        if os.path.isfile(input_dir):
            input_dir = os.path.dirname(input_dir)

        def run():
            if func_key == "dataset_random_split":
                r = (params.get("train_ratio", 0.7), params.get("val_ratio", 0.2),
                     params.get("test_ratio", 0.1))
                return random_split(input_dir, output_dir, r, params.get("label_dir") or None,
                                    params.get("seed", 42))
            elif func_key == "dataset_stratified_split":
                r = (params.get("train_ratio", 0.7), params.get("val_ratio", 0.2),
                     params.get("test_ratio", 0.1))
                return stratified_split(input_dir, output_dir, r, params.get("seed", 42))
            elif func_key == "dataset_kfold":
                return kfold_split(input_dir, output_dir, params.get("k", 5), params.get("seed", 42))

        self._run_worker(run, lambda r: self._log(f"数据集划分完成: {r}"))

    def _run_seg_tile_op(self, func_key, params, output_dir, files):
        image_dir = files[0] if files else ""
        if os.path.isfile(image_dir):
            image_dir = os.path.dirname(image_dir)
        ann_dir = params.get("ann_dir", "")
        if not ann_dir:
            QMessageBox.warning(self, "提示", "请指定标注文件夹(JSON)路径")
            return

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

    def _run_format_op(self, func_key, params, output_dir, files):
        def run():
            if func_key == "format_yolo2coco":
                cats = [c.strip() for c in params.get("categories", "class0,class1").split(",")]
                return yolo_to_coco(params.get("yolo_dir", ""), params.get("image_dir", ""),
                                    os.path.join(output_dir, "coco.json"), cats)
            elif func_key == "format_coco2yolo":
                return coco_to_yolo(params.get("coco_path", ""), output_dir)
            elif func_key == "format_voc2yolo":
                cats = [c.strip() for c in params.get("categories", "class0,class1").split(",")]
                return voc_to_yolo(params.get("voc_dir", ""), output_dir, cats)
            elif func_key == "format_voc2coco":
                cats = [c.strip() for c in params.get("categories", "class0,class1").split(",")]
                return voc_to_coco(params.get("voc_dir", ""), params.get("image_dir", ""),
                                   os.path.join(output_dir, "coco.json"), cats)
            elif func_key == "format_coco2voc":
                return coco_to_voc(params.get("coco_path", ""), output_dir)
            elif func_key == "format_classification":
                return create_classification_dataset(files[0] if files else "", output_dir)

        self._run_worker(run, lambda r: self._log(f"格式转换完成: {r}"))

    def _run_simple_batch_op(self, func_key, params, output_dir, files):
        input_dir = files[0] if files else ""
        if os.path.isfile(input_dir):
            input_dir = os.path.dirname(input_dir)

        def run():
            if func_key == "batch_deduplicate":
                dupes = deduplicate_images(input_dir)
                for a, b in dupes[:10]:
                    self._log(f"重复: {os.path.basename(a)} = {os.path.basename(b)}")
                return f"发现 {len(dupes)} 组重复"
            if func_key == "batch_rename":
                return batch_rename(input_dir, output_dir, params.get("prefix", "img_"),
                                    params.get("start_index", 1), params.get("digits", 4),
                                    params.get("keep_ext", True))
            if func_key == "annot_statistics":
                s = annotation_statistics(params.get("ann_dir", ""), params.get("img_dir", ""),
                                          params.get("format_type", "yolo"))
                self._log(f"标注统计: {s}")
                return s

        self._run_worker(run, lambda r: self._log(f"操作完成: {r}"))

    # ---- Worker helpers ----
    def _run_worker(self, func, on_finish=None):
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        self.lbl_status.setText("处理中...")
        self.param_panel.btn_run.setEnabled(False)
        self._worker = WorkerThread(func)
        self._worker.finished.connect(lambda r: self._on_worker_done(r, on_finish))
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()

    def _on_worker_done(self, result, callback=None):
        self.progress_bar.hide()
        self.progress_bar.setRange(0, 100)
        self.param_panel.btn_run.setEnabled(True)
        self.lbl_status.setText("就绪")
        if callback:
            callback(result)

    def _on_worker_error(self, msg):
        self.progress_bar.hide()
        self.param_panel.btn_run.setEnabled(True)
        self.lbl_status.setText("处理出错")
        self._log(f"错误: {msg}")
        QMessageBox.critical(self, "错误", msg)

    def _log(self, msg):
        self.log_output.append(msg)

    def _show_about(self):
        QMessageBox.about(self, "关于",
                          "图像处理工具箱 v1.1\n\n"
                          "面向深度学习的图像处理工具\n"
                          "支持: 颜色转换 · 图像处理 · 大图切块 ·\n"
                          "       标注切块 · 数据集划分 · 格式转换\n\n"
                          "快捷键:\n"
                          "  Ctrl+O  打开图片    Ctrl+D  打开文件夹\n"
                          "  Ctrl+R  执行处理    F5      执行处理\n"
                          "  Esc     退出坐标拾取")
