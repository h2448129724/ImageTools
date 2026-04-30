"""CAB-F stitch point editor dialog integrated into img_tools."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QAction, QColor, QImage, QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QDoubleSpinBox,
    QSizePolicy,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)


DEFAULT_COSMOS_ROOT = Path(r"D:\project\changrui\cosmos")
DEFAULT_DETECTOR_SCRIPT = DEFAULT_COSMOS_ROOT / "algo" / "cab_f" / "sew_point_detector.py"
DEFAULT_MODEL_PATH = DEFAULT_COSMOS_ROOT / "assets" / "weights" / "cab_f" / "sew_point_detector.onnx"


def read_image(image_path) -> np.ndarray:
    image_path = Path(image_path)
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    return image


def save_annotation_json(path, annotation: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(annotation, f, ensure_ascii=False, indent=2)


def make_empty_annotation(image_path: str, width: int, height: int, sample_id: str):
    return {
        "schema_version": "1.0",
        "sample_id": sample_id,
        "image_path": image_path,
        "image_size": {"width": int(width), "height": int(height)},
        "roi": None,
        "spacing_hint": None,
        "points": [],
        "segments": [],
        "metadata": {},
    }


class DetectorRunner:
    """Lazy loader for the external CAB-F stitch detector."""

    def __init__(self):
        self._detector_class = None
        self._detector = None
        self._cache_key = None

    def _load_detector_class(self, detector_script_path: str):
        detector_path = Path(detector_script_path)
        spec = importlib.util.spec_from_file_location("cab_f_sew_point_detector_in_img_tools", detector_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载检测器文件: {detector_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.SewPointDetector

    def detect(self, image_bgr: np.ndarray, detector_script_path: str, model_path: str, conf: float):
        cache_key = (str(detector_script_path), str(model_path), float(conf))
        if self._detector is None or self._cache_key != cache_key:
            self._detector_class = self._load_detector_class(detector_script_path)
            self._detector = self._detector_class({"path": str(model_path), "conf": float(conf)})
            self._cache_key = cache_key

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self._detector.evaluate(image_rgb)
        raw_points = result.get("points", [])
        return [
            {
                "id": idx,
                "x": float(point[0]),
                "y": float(point[1]),
                "score": float(point[2]) if len(point) >= 3 else 1.0,
                "source": "model",
            }
            for idx, point in enumerate(raw_points)
        ]


class PointCanvas(QWidget):
    """Interactive image canvas for point add/move/delete."""

    pointSelectionChanged = Signal(object)
    pointCountChanged = Signal(int)
    statusMessage = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.image_bgr: Optional[np.ndarray] = None
        self.image_rgb: Optional[np.ndarray] = None
        self.image_qimage: Optional[QImage] = None
        self.image_path: str = ""
        self.points: list[dict] = []
        self.selected_point_id: Optional[int] = None
        self.mode = "select"

        self.scale = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        self._drag_point_id: Optional[int] = None
        self._pan_anchor: Optional[QPoint] = None

    def set_mode(self, mode: str):
        self.mode = mode
        self.update()

    def set_image(self, image_bgr: np.ndarray, image_path: str = ""):
        self.image_bgr = image_bgr
        self.image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = self.image_rgb.shape[:2]
        bytes_per_line = width * 3
        self.image_qimage = QImage(self.image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
        self.image_path = image_path
        self.points = []
        self.selected_point_id = None
        self.fit_view()
        self.pointCountChanged.emit(len(self.points))

    def set_points(self, points: list[dict]):
        self.points = [
            {
                "id": int(point["id"]),
                "x": float(point["x"]),
                "y": float(point["y"]),
                "score": float(point.get("score", 1.0)),
                "source": point.get("source", "manual"),
            }
            for point in points
        ]
        self.selected_point_id = None
        self.pointCountChanged.emit(len(self.points))
        self.update()

    def fit_view(self):
        if self.image_qimage is None or self.width() <= 0 or self.height() <= 0:
            return
        image_w = self.image_qimage.width()
        image_h = self.image_qimage.height()
        self.scale = min(self.width() / max(image_w, 1), self.height() / max(image_h, 1))
        self.scale = max(self.scale, 0.05)
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()

    def _clamp_pan(self):
        if self.image_qimage is None:
            return
        view_w = self.width() / max(self.scale, 1e-6)
        view_h = self.height() / max(self.scale, 1e-6)
        max_pan_x = max(self.image_qimage.width() - view_w, 0.0)
        max_pan_y = max(self.image_qimage.height() - view_h, 0.0)
        self.pan_x = float(np.clip(self.pan_x, 0.0, max_pan_x))
        self.pan_y = float(np.clip(self.pan_y, 0.0, max_pan_y))

    def _canvas_to_image(self, pos: QPoint):
        return (
            self.pan_x + pos.x() / max(self.scale, 1e-6),
            self.pan_y + pos.y() / max(self.scale, 1e-6),
        )

    def _image_to_canvas(self, x: float, y: float):
        return (
            int(round((x - self.pan_x) * self.scale)),
            int(round((y - self.pan_y) * self.scale)),
        )

    def _nearest_point(self, image_x: float, image_y: float, max_screen_dist: float = 14.0):
        if not self.points:
            return None
        pts = np.asarray([[point["x"], point["y"]] for point in self.points], dtype=np.float32)
        click = np.asarray([image_x, image_y], dtype=np.float32)
        distances = np.linalg.norm(pts - click[None, :], axis=1)
        idx = int(np.argmin(distances))
        max_dist = max_screen_dist / max(self.scale, 1e-6)
        if float(distances[idx]) > max_dist:
            return None
        return self.points[idx]

    def _next_point_id(self) -> int:
        if not self.points:
            return 0
        return max(int(point["id"]) for point in self.points) + 1

    def delete_selected_point(self):
        if self.selected_point_id is None:
            return
        self.points = [point for point in self.points if int(point["id"]) != int(self.selected_point_id)]
        self.statusMessage.emit(f"已删除点 {self.selected_point_id}")
        self.selected_point_id = None
        self.pointSelectionChanged.emit(None)
        self.pointCountChanged.emit(len(self.points))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#111111"))

        if self.image_qimage is None:
            painter.setPen(QColor("#ffffff"))
            painter.drawText(self.rect(), Qt.AlignCenter, "请先选择图片")
            return

        self._clamp_pan()
        src_rect = QRect(
            int(self.pan_x),
            int(self.pan_y),
            int(np.ceil(self.width() / max(self.scale, 1e-6))),
            int(np.ceil(self.height() / max(self.scale, 1e-6))),
        )
        painter.drawImage(self.rect(), self.image_qimage, src_rect)

        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)

        for point in self.points:
            cx, cy = self._image_to_canvas(point["x"], point["y"])
            is_selected = int(point["id"]) == self.selected_point_id
            radius = 6 if is_selected else 4
            fill = QColor(0, 220, 255) if is_selected else QColor(80, 255, 80)
            painter.setPen(QPen(QColor(20, 20, 20), 1))
            painter.setBrush(fill)
            painter.drawEllipse(QPoint(cx, cy), radius, radius)
            painter.setPen(QColor("#ffffff"))
            painter.drawText(cx + 6, cy - 6, str(point["id"]))

        header = [
            f"mode={self.mode} points={len(self.points)} selected={self.selected_point_id}",
            f"zoom={self.scale:.2f}",
        ]
        painter.setPen(QColor(80, 255, 80))
        y = 22
        for line in header:
            painter.drawText(10, y, line)
            y += 20

    def mousePressEvent(self, event):
        if self.image_qimage is None:
            return
        if event.button() == Qt.RightButton:
            self._pan_anchor = event.pos()
            return
        if event.button() != Qt.LeftButton:
            return

        image_x, image_y = self._canvas_to_image(event.position().toPoint())
        if self.mode == "add":
            new_point = {
                "id": self._next_point_id(),
                "x": float(image_x),
                "y": float(image_y),
                "score": 1.0,
                "source": "manual",
            }
            self.points.append(new_point)
            self.selected_point_id = int(new_point["id"])
            self.pointSelectionChanged.emit(new_point)
            self.pointCountChanged.emit(len(self.points))
            self.statusMessage.emit(f"已新增点 {new_point['id']}")
            self.update()
            return

        nearest = self._nearest_point(image_x, image_y)
        if nearest is None:
            self.selected_point_id = None
            self.pointSelectionChanged.emit(None)
            self.update()
            return

        self.selected_point_id = int(nearest["id"])
        self.pointSelectionChanged.emit(nearest)
        if self.mode == "delete":
            self.delete_selected_point()
            return
        if self.mode == "move":
            self._drag_point_id = int(nearest["id"])
        self.update()

    def mouseMoveEvent(self, event):
        if self.image_qimage is None:
            return
        if self._pan_anchor is not None:
            delta = event.pos() - self._pan_anchor
            self.pan_x -= delta.x() / max(self.scale, 1e-6)
            self.pan_y -= delta.y() / max(self.scale, 1e-6)
            self._pan_anchor = event.pos()
            self.update()
            return
        if self._drag_point_id is None or self.mode != "move":
            return
        image_x, image_y = self._canvas_to_image(event.position().toPoint())
        for point in self.points:
            if int(point["id"]) == int(self._drag_point_id):
                point["x"] = float(image_x)
                point["y"] = float(image_y)
                point["source"] = "manual"
                self.pointSelectionChanged.emit(point)
                break
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self._pan_anchor = None
        if event.button() == Qt.LeftButton:
            self._drag_point_id = None

    def wheelEvent(self, event: QWheelEvent):
        if self.image_qimage is None:
            return
        old_scale = self.scale
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        new_scale = float(np.clip(old_scale * factor, 0.05, 20.0))
        if abs(new_scale - old_scale) < 1e-6:
            return
        image_x, image_y = self._canvas_to_image(event.position().toPoint())
        self.scale = new_scale
        self.pan_x = image_x - event.position().x() / max(self.scale, 1e-6)
        self.pan_y = image_y - event.position().y() / max(self.scale, 1e-6)
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.image_qimage is not None:
            self._clamp_pan()


class StitchPointEditorDialog(QDialog):
    """Integrated CAB-F stitch point editor dialog."""

    def __init__(self, parent=None, image: Optional[np.ndarray] = None, image_path: str = ""):
        super().__init__(parent)
        self.setWindowTitle("CAB-F 针点编辑器")
        self.resize(1500, 920)
        self.detector_runner = DetectorRunner()
        self.image_bgr: Optional[np.ndarray] = None
        self.image_path = image_path

        self._build_ui()
        self._connect_signals()

        if image is not None:
            self.set_image(image, image_path)

    def _build_ui(self):
        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        left = QWidget()
        left.setMinimumWidth(360)
        left.setMaximumWidth(460)
        left_layout = QVBoxLayout(left)

        form_box = QFrame()
        form_layout = QFormLayout(form_box)

        self.edit_image = QLineEdit()
        self.edit_detector_script = QLineEdit(str(DEFAULT_DETECTOR_SCRIPT))
        self.edit_model = QLineEdit(str(DEFAULT_MODEL_PATH))
        self.edit_output = QLineEdit("")
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.01, 1.0)
        self.spin_conf.setDecimals(3)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(0.5)

        form_layout.addRow("图片", self.edit_image)
        form_layout.addRow("检测脚本", self.edit_detector_script)
        form_layout.addRow("模型", self.edit_model)
        form_layout.addRow("置信度", self.spin_conf)
        form_layout.addRow("输出JSON", self.edit_output)
        left_layout.addWidget(form_box)

        row1 = QHBoxLayout()
        self.btn_choose_image = QPushButton("选择图片")
        self.btn_use_current = QPushButton("使用当前图")
        self.btn_choose_script = QPushButton("选择脚本")
        row1.addWidget(self.btn_choose_image)
        row1.addWidget(self.btn_use_current)
        row1.addWidget(self.btn_choose_script)
        left_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.btn_choose_model = QPushButton("选择模型")
        self.btn_detect = QPushButton("模型出点")
        self.btn_clear = QPushButton("清空点")
        row2.addWidget(self.btn_choose_model)
        row2.addWidget(self.btn_detect)
        row2.addWidget(self.btn_clear)
        left_layout.addLayout(row2)

        row3 = QHBoxLayout()
        self.btn_choose_output = QPushButton("选择输出")
        self.btn_save = QPushButton("保存JSON")
        row3.addWidget(self.btn_choose_output)
        row3.addWidget(self.btn_save)
        left_layout.addLayout(row3)

        self.toolbar = QToolBar()
        self.toolbar.setMovable(False)
        self.action_group = []
        for text, mode in [
            ("选择", "select"),
            ("新增", "add"),
            ("移动", "move"),
            ("删除", "delete"),
        ]:
            action = QAction(text, self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked=False, m=mode: self._set_mode(m))
            self.toolbar.addAction(action)
            self.action_group.append((action, mode))
        self.action_group[0][0].setChecked(True)
        left_layout.addWidget(self.toolbar)

        info_box = QFrame()
        info_layout = QFormLayout(info_box)
        self.lbl_point_count = QLabel("0")
        self.lbl_selected = QLabel("-")
        self.lbl_xy = QLabel("-")
        info_layout.addRow("点数", self.lbl_point_count)
        info_layout.addRow("选中点", self.lbl_selected)
        info_layout.addRow("坐标", self.lbl_xy)
        left_layout.addWidget(info_box)

        self.lbl_help = QLabel(
            "操作说明\n"
            "- 选择图片后点“模型出点”\n"
            "- 模式切到“新增”可补点\n"
            "- 模式切到“移动”后拖拽点\n"
            "- 模式切到“删除”后点击误检点\n"
            "- 鼠标滚轮缩放，右键拖拽平移\n"
            "- 保存 JSON 后可继续接 GNN 标注流程"
        )
        self.lbl_help.setWordWrap(True)
        left_layout.addWidget(self.lbl_help)
        left_layout.addStretch(1)

        self.status_label = QLabel("请选择图片，然后点击“模型出点”。")
        left_layout.addWidget(self.status_label)

        self.canvas = PointCanvas()
        splitter.addWidget(left)
        splitter.addWidget(self.canvas)
        splitter.setSizes([400, 1100])

    def _connect_signals(self):
        self.btn_choose_image.clicked.connect(self.choose_image)
        self.btn_use_current.clicked.connect(self.use_current_image)
        self.btn_choose_script.clicked.connect(self.choose_detector_script)
        self.btn_choose_model.clicked.connect(self.choose_model)
        self.btn_choose_output.clicked.connect(self.choose_output)
        self.btn_detect.clicked.connect(self.detect_points)
        self.btn_clear.clicked.connect(self.clear_points)
        self.btn_save.clicked.connect(self.save_json)
        self.canvas.pointSelectionChanged.connect(self._on_point_selection_changed)
        self.canvas.pointCountChanged.connect(lambda count: self.lbl_point_count.setText(str(count)))
        self.canvas.statusMessage.connect(self.status_label.setText)

    def _set_mode(self, mode: str):
        for action, action_mode in self.action_group:
            action.setChecked(action_mode == mode)
        self.canvas.set_mode(mode)
        self.status_label.setText(f"当前模式: {mode}")

    def set_image(self, image_bgr: np.ndarray, image_path: str = ""):
        self.image_bgr = image_bgr.copy()
        self.image_path = image_path
        self.canvas.set_image(self.image_bgr, image_path=image_path)
        if image_path:
            self.edit_image.setText(image_path)
        if not self.edit_output.text().strip() and image_path:
            image_path_obj = Path(image_path)
            self.edit_output.setText(str(image_path_obj.with_name(f"{image_path_obj.stem}_points_anno.json")))
        self.status_label.setText("图片已加载。")

    def use_current_image(self):
        parent = self.parent()
        if parent is None or getattr(parent, "_current_image", None) is None:
            QMessageBox.information(self, "提示", "主窗口当前没有已加载图片。")
            return
        self.set_image(parent._current_image, getattr(parent, "_current_file", "") or "")

    def choose_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            str(Path.cwd()),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)",
        )
        if not path:
            return
        try:
            self.set_image(read_image(path), path)
        except Exception as exc:
            QMessageBox.critical(self, "打开图片失败", str(exc))

    def choose_detector_script(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 sew_point_detector.py", str(DEFAULT_DETECTOR_SCRIPT.parent), "Python (*.py)")
        if path:
            self.edit_detector_script.setText(path)

    def choose_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 ONNX 模型", str(DEFAULT_MODEL_PATH.parent), "ONNX (*.onnx);;All Files (*)")
        if path:
            self.edit_model.setText(path)

    def choose_output(self):
        init = self.edit_output.text().strip()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存 JSON",
            init or str(Path.cwd() / "stitch_points_anno.json"),
            "JSON (*.json)",
        )
        if path:
            self.edit_output.setText(path)

    def clear_points(self):
        self.canvas.set_points([])
        self.status_label.setText("已清空当前点。")

    def detect_points(self):
        if self.image_bgr is None:
            QMessageBox.warning(self, "提示", "请先选择图片。")
            return
        detector_script = self.edit_detector_script.text().strip()
        model_path = self.edit_model.text().strip()
        if not detector_script or not Path(detector_script).exists():
            QMessageBox.warning(self, "提示", "请先选择有效的 sew_point_detector.py。")
            return
        if not model_path or not Path(model_path).exists():
            QMessageBox.warning(self, "提示", "请先选择有效的 ONNX 模型。")
            return

        self.status_label.setText("模型推理中，请稍候...")
        self.btn_detect.setEnabled(False)
        try:
            points = self.detector_runner.detect(
                self.image_bgr,
                detector_script_path=detector_script,
                model_path=model_path,
                conf=self.spin_conf.value(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "模型出点失败", str(exc))
            self.status_label.setText("模型出点失败。")
            self.btn_detect.setEnabled(True)
            return

        self.canvas.set_points(points)
        self.status_label.setText(f"模型出点完成，共 {len(points)} 个点。")
        self.btn_detect.setEnabled(True)

    def _on_point_selection_changed(self, point):
        if point is None:
            self.lbl_selected.setText("-")
            self.lbl_xy.setText("-")
            return
        self.lbl_selected.setText(str(point["id"]))
        self.lbl_xy.setText(f"({point['x']:.1f}, {point['y']:.1f})")

    def save_json(self):
        if self.image_bgr is None:
            QMessageBox.warning(self, "提示", "没有可保存的图片上下文。")
            return
        output_path = self.edit_output.text().strip()
        if not output_path:
            QMessageBox.warning(self, "提示", "请先设置输出 JSON 路径。")
            return

        sample_id = Path(output_path).stem
        annotation = make_empty_annotation(
            image_path=self.edit_image.text().strip(),
            width=int(self.image_bgr.shape[1]),
            height=int(self.image_bgr.shape[0]),
            sample_id=sample_id,
        )
        annotation["points"] = [
            {
                "id": int(point["id"]),
                "x": float(point["x"]),
                "y": float(point["y"]),
                "score": float(point.get("score", 1.0)),
                "source": point.get("source", "manual"),
            }
            for point in sorted(self.canvas.points, key=lambda item: int(item["id"]))
        ]
        annotation["segments"] = []
        annotation["metadata"] = {
            "source": "img_tools_stitch_point_editor",
            "detector_script": self.edit_detector_script.text().strip(),
            "model_path": self.edit_model.text().strip(),
            "conf": float(self.spin_conf.value()),
            "point_count": len(self.canvas.points),
        }
        save_annotation_json(output_path, annotation)
        self.status_label.setText(f"已保存: {output_path}")
