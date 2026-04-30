"""Parameter configuration panel that adapts to selected function."""
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QSpinBox,
                                QDoubleSpinBox, QComboBox, QCheckBox, QLabel,
                                QHBoxLayout, QPushButton, QLineEdit, QGroupBox,
                                QSlider, QFileDialog)
from PySide6.QtCore import Signal, Qt
import os


class ParamPanel(QWidget):
    """Dynamic parameter panel that changes based on selected function."""
    paramsChanged = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._widgets = {}
        self._current_key = None
        self._layout = None
        self._setup_ui()

    def _setup_ui(self):
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(4, 4, 4, 4)
        self._title_label = QLabel("参数设置")
        f = self._title_label.font(); f.setBold(True); self._title_label.setFont(f)
        self._main_layout.addWidget(self._title_label)

        self._scroll_area = QWidget()
        self._layout = QFormLayout(self._scroll_area)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.addWidget(self._scroll_area)

        self.btn_preview = QPushButton("预览 (仅当前图片)")
        self.btn_preview.setToolTip("对当前选中的图片进行预览，结果不保存到磁盘")
        self._main_layout.addWidget(self.btn_preview)

        self.btn_run = QPushButton("▶ 执行处理 (Ctrl+R)")
        self.btn_run.setObjectName("btnRun")
        self._main_layout.addWidget(self.btn_run)

        self._main_layout.addStretch()
        self._current_params = {}
        self._container_refs = []  # keep alive container widgets for dir/file pickers

    def set_function(self, key, name):
        """Show parameters for the selected function."""
        self._current_key = key
        # Clear existing widgets
        while self._layout.rowCount() > 0:
            self._layout.removeRow(0)
        self._widgets.clear()
        self._container_refs.clear()
        self._current_params = {}

        self._title_label.setText(f"参数设置 — {name}")
        param_specs = PARAM_SPECS.get(key, [])
        for spec in param_specs:
            widget = self._create_widget(spec)
            self._widgets[spec["name"]] = widget
            if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox) or isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(self._on_param_changed) if isinstance(widget, QComboBox) else \
                widget.valueChanged.connect(self._on_param_changed)
            elif isinstance(widget, QCheckBox):
                widget.toggled.connect(self._on_param_changed)
            elif isinstance(widget, QLineEdit):
                widget.textChanged.connect(self._on_param_changed)
            self._layout.addRow(spec.get("label", spec["name"]), widget)

        self._collect_params()
        self.paramsChanged.emit(self._current_params)

    def _create_widget(self, spec):
        wtype = spec.get("type", "int")
        if wtype == "int":
            w = QSpinBox()
            w.setRange(spec.get("min", 0), spec.get("max", 9999))
            w.setValue(spec.get("default", 0))
            return w
        if wtype == "float":
            w = QDoubleSpinBox()
            w.setRange(spec.get("min", 0.0), spec.get("max", 100.0))
            w.setDecimals(2)
            w.setSingleStep(spec.get("step", 0.1))
            w.setValue(spec.get("default", 1.0))
            return w
        if wtype == "combo":
            w = QComboBox()
            w.addItems(spec.get("options", []))
            default = spec.get("default", "")
            if default:
                w.setCurrentText(default)
            return w
        if wtype == "bool":
            w = QCheckBox()
            w.setChecked(spec.get("default", False))
            return w
        if wtype == "text":
            w = QLineEdit()
            w.setText(spec.get("default", ""))
            return w
        if wtype == "file":
            container = QWidget()
            h = QHBoxLayout(container)
            h.setContentsMargins(0, 0, 0, 0)
            le = QLineEdit()
            le.setText(spec.get("default", ""))
            btn = QPushButton("...")
            btn.setMaximumWidth(30)
            btn.clicked.connect(lambda checked, s=spec, l=le: self._browse_file(l, s))
            h.addWidget(le)
            h.addWidget(btn)
            self._container_refs.append(container)
            return le
        if wtype == "dir":
            container = QWidget()
            h = QHBoxLayout(container)
            h.setContentsMargins(0, 0, 0, 0)
            le = QLineEdit()
            le.setText(spec.get("default", ""))
            btn = QPushButton("...")
            btn.setMaximumWidth(30)
            btn.clicked.connect(lambda checked, s=spec, l=le: self._browse_dir(l, s))
            h.addWidget(le)
            h.addWidget(btn)
            self._container_refs.append(container)
            return le
        return QLabel("")

    def _browse_file(self, line_edit, spec):
        path, _ = QFileDialog.getOpenFileName(self, spec.get("label", "选择文件"))
        if path:
            line_edit.setText(path)

    def _browse_dir(self, line_edit, spec):
        path = QFileDialog.getExistingDirectory(self, spec.get("label", "选择文件夹"))
        if path:
            line_edit.setText(path)

    def _on_param_changed(self, *_):
        self._collect_params()
        self.paramsChanged.emit(self._current_params)

    def _collect_params(self):
        self._current_params = {}
        for name, widget in self._widgets.items():
            if isinstance(widget, QSpinBox):
                self._current_params[name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                self._current_params[name] = widget.value()
            elif isinstance(widget, QComboBox):
                self._current_params[name] = widget.currentText()
            elif isinstance(widget, QCheckBox):
                self._current_params[name] = widget.isChecked()
            elif isinstance(widget, QLineEdit):
                self._current_params[name] = widget.text()

    def get_params(self):
        self._collect_params()
        return {"function": self._current_key, "params": self._current_params}


# Parameter specifications for each function
PARAM_SPECS = {
    "resize": [
        {"name": "width", "label": "宽度", "type": "int", "default": 640, "min": 1, "max": 10000},
        {"name": "height", "label": "高度", "type": "int", "default": 480, "min": 1, "max": 10000},
        {"name": "keep_aspect", "label": "保持比例", "type": "bool", "default": True},
        {"name": "scale", "label": "缩放比例(0=使用宽高)", "type": "float", "default": 0, "min": 0, "max": 10, "step": 0.01},
    ],
    "crop": [
        {"name": "x", "label": "X坐标", "type": "int", "default": 0, "min": 0, "max": 10000},
        {"name": "y", "label": "Y坐标", "type": "int", "default": 0, "min": 0, "max": 10000},
        {"name": "w", "label": "宽度", "type": "int", "default": 256, "min": 1, "max": 10000},
        {"name": "h", "label": "高度", "type": "int", "default": 256, "min": 1, "max": 10000},
    ],
    "center_crop": [
        {"name": "w", "label": "宽度", "type": "int", "default": 512, "min": 1, "max": 10000},
        {"name": "h", "label": "高度", "type": "int", "default": 512, "min": 1, "max": 10000},
    ],
    "rotate": [
        {"name": "angle", "label": "旋转角度", "type": "int", "default": 90, "min": -360, "max": 360},
        {"name": "keep_size", "label": "保持原尺寸", "type": "bool", "default": False},
    ],
    "flip": [
        {"name": "direction", "label": "翻转方向", "type": "combo",
         "options": ["horizontal", "vertical", "both"], "default": "horizontal"},
    ],
    "brightness_contrast": [
        {"name": "brightness", "label": "亮度(-255~255)", "type": "int", "default": 0, "min": -255, "max": 255},
        {"name": "contrast", "label": "对比度", "type": "float", "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05},
    ],
    "saturation": [
        {"name": "factor", "label": "饱和度因子", "type": "float", "default": 1.5, "min": 0.0, "max": 5.0, "step": 0.1},
    ],
    "histogram_eq": [
        {"name": "adaptive", "label": "自适应(CLAHE)", "type": "bool", "default": False},
        {"name": "clip_limit", "label": "对比度阈值", "type": "float", "default": 2.0, "min": 0.5, "max": 10.0, "step": 0.5},
        {"name": "tile_size", "label": "网格大小", "type": "int", "default": 8, "min": 2, "max": 32},
    ],
    "threshold": [
        {"name": "method", "label": "方法", "type": "combo",
         "options": ["otsu", "binary", "adaptive_mean", "adaptive_gaussian"], "default": "otsu"},
        {"name": "thresh", "label": "阈值", "type": "int", "default": 127, "min": 0, "max": 255},
        {"name": "maxval", "label": "最大值", "type": "int", "default": 255, "min": 0, "max": 255},
        {"name": "block_size", "label": "邻域大小(自适应)", "type": "int", "default": 11, "min": 3, "max": 99},
    ],
    "morphology": [
        {"name": "op_type", "label": "操作", "type": "combo",
         "options": ["erode", "dilate", "open", "close"], "default": "erode"},
        {"name": "ksize", "label": "核大小", "type": "int", "default": 3, "min": 1, "max": 31},
        {"name": "iterations", "label": "迭代次数", "type": "int", "default": 1, "min": 1, "max": 10},
    ],
    "pad": [
        {"name": "top", "label": "上边距", "type": "int", "default": 10, "min": 0, "max": 500},
        {"name": "bottom", "label": "下边距", "type": "int", "default": 10, "min": 0, "max": 500},
        {"name": "left", "label": "左边距", "type": "int", "default": 10, "min": 0, "max": 500},
        {"name": "right", "label": "右边距", "type": "int", "default": 10, "min": 0, "max": 500},
        {"name": "mode", "label": "填充模式", "type": "combo",
         "options": ["constant", "reflect", "replicate"], "default": "constant"},
    ],
    "overlay": [
        {"name": "overlay_path", "label": "叠加图片", "type": "file", "default": ""},
        {"name": "x", "label": "X坐标", "type": "int", "default": 0, "min": -5000, "max": 10000},
        {"name": "y", "label": "Y坐标", "type": "int", "default": 0, "min": -5000, "max": 10000},
        {"name": "opacity", "label": "透明度", "type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
    ],
    "channel_extract": [
        {"name": "channel", "label": "通道", "type": "combo",
         "options": ["B (0)", "G (1)", "R (2)", "A (3)"], "default": "B (0)"},
    ],
    "filter_blur": [{"name": "ksize", "label": "核大小", "type": "int", "default": 5, "min": 1, "max": 31}],
    "filter_gaussian": [{"name": "ksize", "label": "核大小(奇数)", "type": "int", "default": 5, "min": 1, "max": 31}],
    "filter_median": [{"name": "ksize", "label": "核大小(奇数)", "type": "int", "default": 5, "min": 1, "max": 31}],
    "filter_bilateral": [{"name": "ksize", "label": "直径", "type": "int", "default": 9, "min": 1, "max": 31}],
    "filter_sharpen": [],
    "edge_canny": [
        {"name": "threshold1", "label": "低阈值", "type": "int", "default": 100, "min": 0, "max": 500},
        {"name": "threshold2", "label": "高阈值", "type": "int", "default": 200, "min": 0, "max": 500},
    ],
    "edge_sobel": [],
    "edge_laplacian": [],
    "remove_alpha": [
        {"name": "bg_color", "label": "背景色", "type": "combo",
         "options": ["white", "black"], "default": "white"},
    ],
    "add_alpha": [],
    "tile_fixed": [
        {"name": "tile_w", "label": "切块宽度", "type": "int", "default": 512, "min": 32, "max": 4096},
        {"name": "tile_h", "label": "切块高度", "type": "int", "default": 512, "min": 32, "max": 4096},
        {"name": "overlap", "label": "重叠像素", "type": "int", "default": 0, "min": 0, "max": 2048},
        {"name": "discard_incomplete", "label": "丢弃不完整块", "type": "bool", "default": True},
    ],
    "tile_grid": [
        {"name": "rows", "label": "行数", "type": "int", "default": 3, "min": 1, "max": 100},
        {"name": "cols", "label": "列数", "type": "int", "default": 3, "min": 1, "max": 100},
    ],
    "seg_tile": [
        {"name": "tile_w", "label": "切块宽度", "type": "int", "default": 256, "min": 32, "max": 4096},
        {"name": "tile_h", "label": "切块高度", "type": "int", "default": 256, "min": 32, "max": 4096},
        {"name": "overlap", "label": "重叠像素", "type": "int", "default": 0, "min": 0, "max": 2048},
        {"name": "discard_empty", "label": "丢弃无标注块", "type": "bool", "default": False},
        {"name": "discard_incomplete", "label": "丢弃不完整块", "type": "bool", "default": True},
        {"name": "ann_dir", "label": "标注文件夹(JSON)", "type": "dir", "default": ""},
    ],
    "dataset_random_split": [
        {"name": "train_ratio", "label": "训练集比例", "type": "float", "default": 0.7, "min": 0.1, "max": 0.9, "step": 0.05},
        {"name": "val_ratio", "label": "验证集比例", "type": "float", "default": 0.2, "min": 0.05, "max": 0.5, "step": 0.05},
        {"name": "test_ratio", "label": "测试集比例", "type": "float", "default": 0.1, "min": 0.0, "max": 0.5, "step": 0.05},
        {"name": "seed", "label": "随机种子", "type": "int", "default": 42, "min": 0, "max": 9999},
        {"name": "label_dir", "label": "标注文件夹(可选)", "type": "dir", "default": ""},
    ],
    "dataset_stratified_split": [
        {"name": "train_ratio", "label": "训练集比例", "type": "float", "default": 0.7, "min": 0.1, "max": 0.9, "step": 0.05},
        {"name": "val_ratio", "label": "验证集比例", "type": "float", "default": 0.2, "min": 0.05, "max": 0.5, "step": 0.05},
        {"name": "test_ratio", "label": "测试集比例", "type": "float", "default": 0.1, "min": 0.0, "max": 0.5, "step": 0.05},
        {"name": "seed", "label": "随机种子", "type": "int", "default": 42, "min": 0, "max": 9999},
    ],
    "dataset_kfold": [
        {"name": "k", "label": "折数", "type": "int", "default": 5, "min": 2, "max": 10},
        {"name": "seed", "label": "随机种子", "type": "int", "default": 42, "min": 0, "max": 9999},
    ],
    "format_yolo2coco": [
        {"name": "yolo_dir", "label": "YOLO标注目录", "type": "dir", "default": ""},
        {"name": "image_dir", "label": "图片目录", "type": "dir", "default": ""},
        {"name": "categories", "label": "类别名(逗号分隔)", "type": "text", "default": "class0,class1,class2"},
    ],
    "format_coco2yolo": [
        {"name": "coco_path", "label": "COCO JSON文件", "type": "file", "default": ""},
    ],
    "format_voc2yolo": [
        {"name": "voc_dir", "label": "VOC标注目录", "type": "dir", "default": ""},
        {"name": "categories", "label": "类别名(逗号分隔)", "type": "text", "default": "class0,class1,class2"},
    ],
    "format_voc2coco": [
        {"name": "voc_dir", "label": "VOC标注目录", "type": "dir", "default": ""},
        {"name": "image_dir", "label": "图片目录", "type": "dir", "default": ""},
        {"name": "categories", "label": "类别名(逗号分隔)", "type": "text", "default": "class0,class1,class2"},
    ],
    "format_coco2voc": [
        {"name": "coco_path", "label": "COCO JSON文件", "type": "file", "default": ""},
    ],
    "format_classification": [],
    "annot_draw_yolo": [
        {"name": "txt_dir", "label": "YOLO标注目录", "type": "dir", "default": ""},
        {"name": "categories", "label": "类别名(逗号分隔)", "type": "text", "default": "class0,class1"},
    ],
    "annot_draw_coco": [
        {"name": "coco_path", "label": "COCO JSON文件", "type": "file", "default": ""},
    ],
    "annot_validate_yolo": [
        {"name": "txt_dir", "label": "YOLO标注目录", "type": "dir", "default": ""},
        {"name": "img_w", "label": "图片宽度", "type": "int", "default": 1920, "min": 1, "max": 100000},
        {"name": "img_h", "label": "图片高度", "type": "int", "default": 1080, "min": 1, "max": 100000},
    ],
    "annot_statistics": [
        {"name": "ann_dir", "label": "标注目录", "type": "dir", "default": ""},
        {"name": "img_dir", "label": "图片目录", "type": "dir", "default": ""},
        {"name": "format_type", "label": "标注格式", "type": "combo", "options": ["yolo", "coco"], "default": "yolo"},
    ],
    "annot_crop_roi": [
        {"name": "txt_dir", "label": "YOLO标注目录", "type": "dir", "default": ""},
        {"name": "categories", "label": "类别名(逗号分隔)", "type": "text", "default": "class0,class1"},
        {"name": "padding", "label": "额外边距(px)", "type": "int", "default": 0, "min": 0, "max": 200},
    ],
    "batch_rename": [
        {"name": "prefix", "label": "文件名前缀", "type": "text", "default": "img_"},
        {"name": "start_index", "label": "起始编号", "type": "int", "default": 1, "min": 0, "max": 99999},
        {"name": "digits", "label": "编号位数", "type": "int", "default": 4, "min": 1, "max": 10},
        {"name": "keep_ext", "label": "保持原格式", "type": "bool", "default": True},
    ],
    "batch_resize": [
        {"name": "width", "label": "目标宽度", "type": "int", "default": 640, "min": 1, "max": 10000},
        {"name": "height", "label": "目标高度", "type": "int", "default": 480, "min": 1, "max": 10000},
        {"name": "keep_aspect", "label": "保持比例", "type": "bool", "default": True},
    ],
    "batch_convert_format": [
        {"name": "fmt", "label": "目标格式", "type": "combo",
         "options": ["png", "jpg", "bmp", "tiff", "webp"], "default": "png"},
        {"name": "quality", "label": "质量", "type": "int", "default": 95, "min": 1, "max": 100},
    ],
    "batch_add_border": [
        {"name": "border_size", "label": "边框大小", "type": "int", "default": 10, "min": 0, "max": 200},
        {"name": "color", "label": "颜色", "type": "combo",
         "options": ["black", "white", "red", "green", "blue"], "default": "black"},
    ],
    "batch_roi_crop": [
        {"name": "x", "label": "起始X", "type": "int", "default": 100, "min": 0, "max": 10000},
        {"name": "y", "label": "起始Y", "type": "int", "default": 100, "min": 0, "max": 10000},
        {"name": "w", "label": "裁剪宽度", "type": "int", "default": 400, "min": 1, "max": 10000},
        {"name": "h", "label": "裁剪高度", "type": "int", "default": 400, "min": 1, "max": 10000},
        {"name": "prefix", "label": "输出前缀", "type": "text", "default": "crop"},
    ],
    "batch_deduplicate": [],
}

# Color conversion functions don't need params
for cc_key in ["color_bgr2rgb", "color_rgb2bgr", "color_bgr2hsv", "color_hsv2bgr",
               "color_bgr2lab", "color_lab2bgr", "color_bgr2gray", "color_gray2bgr",
               "color_bgr2yuv", "color_bgr2hls", "color_bgr2ycrcb"]:
    PARAM_SPECS.setdefault(cc_key, [])
