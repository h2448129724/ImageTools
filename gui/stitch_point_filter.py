"""CAB-F stitch point dataset filtering dialog."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
POINT_COLOR = QColor(80, 255, 80)
DEFAULT_IMAGE_DIR = r"D:\project\changrui\CAB-F\sew_point\images"
DEFAULT_LABEL_DIR = r"D:\project\changrui\CAB-F\sew_point\images"


def read_image(image_path: str | Path) -> np.ndarray:
    image_path = Path(image_path)
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    return image


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_points(points: list[dict]) -> list[dict]:
    normalized = []
    for idx, point in enumerate(points):
        if "x" not in point or "y" not in point:
            continue
        normalized.append(
            {
                "id": int(point.get("id", idx)),
                "x": float(point["x"]),
                "y": float(point["y"]),
                "score": float(point.get("score", 1.0)),
                "source": point.get("source", "manual"),
            }
        )
    return normalized


def load_labelme_points(json_path: Path) -> list[dict]:
    data = load_json(json_path)
    shapes = data.get("shapes", []) if isinstance(data, dict) else []
    points = []
    next_id = 0
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        if shape.get("shape_type") != "point":
            continue
        label = str(shape.get("label", "")).strip().lower()
        if label and label not in {"sew", "keypoint"}:
            continue
        raw_points = shape.get("points", [])
        if not raw_points or len(raw_points[0]) < 2:
            continue
        xy = raw_points[0]
        points.append(
            {
                "id": next_id,
                "x": float(xy[0]),
                "y": float(xy[1]),
                "score": float(shape["score"]) if shape.get("score") is not None else 1.0,
                "source": "labelme_point",
            }
        )
        next_id += 1
    return points


def load_points_from_json(json_path: Path) -> list[dict]:
    data = load_json(json_path)
    if isinstance(data, dict) and isinstance(data.get("points"), list):
        return normalize_points(data.get("points", []))
    return load_labelme_points(json_path)


@dataclass
class FilterItem:
    image_path: Path
    label_path: Path

    @property
    def stem(self) -> str:
        return self.image_path.stem


def collect_filter_items(image_dir: Path, label_dir: Optional[Path] = None) -> list[FilterItem]:
    label_root = label_dir or image_dir
    items: list[FilterItem] = []
    for path in sorted(image_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = label_root / f"{path.stem}.json"
        if label_path.exists() and label_path.is_file():
            items.append(FilterItem(image_path=path, label_path=label_path))
    return items


def make_trash_dir(project_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    trash_dir = project_root / ".trash" / timestamp / "cab_f_stitch_point_filter"
    trash_dir.mkdir(parents=True, exist_ok=True)
    return trash_dir


def _resolve_unique_path(dest_dir: Path, source_name: str) -> Path:
    candidate = dest_dir / source_name
    if not candidate.exists():
        return candidate
    stem = Path(source_name).stem
    suffix = Path(source_name).suffix
    index = 1
    while True:
        candidate = dest_dir / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def move_file_safe(source: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    source_resolved = source.resolve()
    target = _resolve_unique_path(dest_dir, source.name)
    if target.resolve() == source_resolved:
        target = _resolve_unique_path(dest_dir, f"{source.stem}_moved{source.suffix}")
    shutil.move(str(source), str(target))
    return target


def move_item_pair(item: FilterItem, dest_dir: Path) -> tuple[Path, Path]:
    moved_image = move_file_safe(item.image_path, dest_dir)
    moved_label = move_file_safe(item.label_path, dest_dir)
    return moved_image, moved_label


class PointFilterCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(840, 620)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.image_bgr: Optional[np.ndarray] = None
        self.image_rgb: Optional[np.ndarray] = None
        self.image_qimage: Optional[QImage] = None
        self.points: list[dict] = []

        self.scale = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._pan_anchor: Optional[QPoint] = None

    def set_data(self, image_bgr: np.ndarray, points: list[dict]):
        self.image_bgr = image_bgr
        self.image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = self.image_rgb.shape[:2]
        bytes_per_line = width * 3
        self.image_qimage = QImage(self.image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
        self.points = normalize_points(points)
        self.fit_view()

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

    def change_zoom(self, delta_steps: float):
        self.scale = float(np.clip(self.scale * (1.15 ** delta_steps), 0.05, 20.0))
        self._clamp_pan()
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

    def _canvas_to_image(self, pos: QPoint) -> tuple[float, float]:
        return (
            self.pan_x + pos.x() / max(self.scale, 1e-6),
            self.pan_y + pos.y() / max(self.scale, 1e-6),
        )

    def _image_to_canvas(self, x: float, y: float) -> tuple[int, int]:
        return (
            int(round((x - self.pan_x) * self.scale)),
            int(round((y - self.pan_y) * self.scale)),
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(24, 24, 24))
        if self.image_qimage is None:
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "请先加载数据")
            return

        target = self.rect()
        source = QRect(
            int(round(self.pan_x)),
            int(round(self.pan_y)),
            int(round(target.width() / max(self.scale, 1e-6))),
            int(round(target.height() / max(self.scale, 1e-6))),
        )
        painter.drawImage(target, self.image_qimage, source)

        pen = QPen(POINT_COLOR, 2)
        painter.setPen(pen)
        painter.setBrush(POINT_COLOR)
        radius = max(3, int(round(4 * self.scale)))
        for point in self.points:
            cx, cy = self._image_to_canvas(point["x"], point["y"])
            painter.drawEllipse(QPoint(cx, cy), radius, radius)

    def mousePressEvent(self, event):
        if self.image_qimage is None:
            return
        if event.button() == Qt.MiddleButton:
            self._pan_anchor = event.pos()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.image_qimage is None:
            return
        if self._pan_anchor is not None:
            dx = event.pos().x() - self._pan_anchor.x()
            dy = event.pos().y() - self._pan_anchor.y()
            self.pan_x -= dx / max(self.scale, 1e-6)
            self.pan_y -= dy / max(self.scale, 1e-6)
            self._pan_anchor = event.pos()
            self._clamp_pan()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._pan_anchor = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

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
        self._clamp_pan()
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.image_qimage is not None:
            self._clamp_pan()


class StitchPointFilterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CAB-F 缝纫点数据筛选")
        self.resize(1600, 960)

        self.items: list[FilterItem] = []
        self.current_index: int = -1
        self.current_item: Optional[FilterItem] = None
        self.trash_dir: Optional[Path] = None
        self.saved_count = 0
        self.trash_count = 0

        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        left = QWidget()
        left.setMinimumWidth(360)
        left.setMaximumWidth(480)
        left_layout = QVBoxLayout(left)

        folder_box = QFrame()
        folder_layout = QFormLayout(folder_box)
        self.edit_image_dir = QLineEdit(DEFAULT_IMAGE_DIR)
        self.edit_label_dir = QLineEdit(DEFAULT_LABEL_DIR)
        self.edit_save_dir = QLineEdit("")
        folder_layout.addRow("图片输入路径", self.edit_image_dir)
        folder_layout.addRow("标签输入路径", self.edit_label_dir)
        folder_layout.addRow("保存路径", self.edit_save_dir)
        left_layout.addWidget(folder_box)

        row_folder1 = QHBoxLayout()
        self.btn_choose_image_dir = QPushButton("选择图片路径")
        self.btn_choose_label_dir = QPushButton("选择标签路径")
        row_folder1.addWidget(self.btn_choose_image_dir)
        row_folder1.addWidget(self.btn_choose_label_dir)
        left_layout.addLayout(row_folder1)

        row_folder2 = QHBoxLayout()
        self.btn_choose_save_dir = QPushButton("选择保存路径")
        self.btn_load = QPushButton("加载数据")
        row_folder2.addWidget(self.btn_choose_save_dir)
        row_folder2.addWidget(self.btn_load)
        left_layout.addLayout(row_folder2)

        info_box = QFrame()
        info_layout = QFormLayout(info_box)
        self.lbl_current_name = QLabel("-")
        self.lbl_index = QLabel("0 / 0")
        self.lbl_point_count = QLabel("0")
        self.lbl_saved_count = QLabel("0")
        self.lbl_trash_count = QLabel("0")
        info_layout.addRow("当前图片", self.lbl_current_name)
        info_layout.addRow("进度", self.lbl_index)
        info_layout.addRow("点数", self.lbl_point_count)
        info_layout.addRow("已保存", self.lbl_saved_count)
        info_layout.addRow("已移垃圾桶", self.lbl_trash_count)
        left_layout.addWidget(info_box)

        row_nav = QHBoxLayout()
        self.btn_prev = QPushButton("上一张(A)")
        self.btn_next = QPushButton("下一张(D)")
        left_layout.addLayout(row_nav)
        row_nav.addWidget(self.btn_prev)
        row_nav.addWidget(self.btn_next)

        row_action = QHBoxLayout()
        self.btn_save = QPushButton("保存当前(S)")
        self.btn_trash = QPushButton("移到垃圾桶(W)")
        row_action.addWidget(self.btn_save)
        row_action.addWidget(self.btn_trash)
        left_layout.addLayout(row_action)

        self.file_list = QListWidget()
        left_layout.addWidget(self.file_list, 1)

        self.lbl_help = QLabel(
            "使用方式\n"
            "1. 选择图片输入路径、标签输入路径和保存路径\n"
            "2. 若图片和 json 混放，标签路径可与图片路径相同\n"
            "3. 程序按同名主文件名自动配对，只加载成对成功的数据\n"
            "4. A/D 切图，S 移动到保存路径并切换下一张\n"
            "5. W 移动当前图片和 json 到项目 .trash 后切换下一张"
        )
        self.lbl_help.setWordWrap(True)
        left_layout.addWidget(self.lbl_help)

        self.status_label = QLabel("请先加载数据。")
        left_layout.addWidget(self.status_label)

        self.canvas = PointFilterCanvas()
        splitter.addWidget(left)
        splitter.addWidget(self.canvas)
        splitter.setSizes([430, 1170])

    def _connect_signals(self):
        self.btn_choose_image_dir.clicked.connect(self.choose_image_dir)
        self.btn_choose_label_dir.clicked.connect(self.choose_label_dir)
        self.btn_choose_save_dir.clicked.connect(self.choose_save_dir)
        self.btn_load.clicked.connect(self.open_dataset)
        self.btn_prev.clicked.connect(lambda: self.jump_to_index(self.current_index - 1))
        self.btn_next.clicked.connect(lambda: self.jump_to_index(self.current_index + 1))
        self.btn_save.clicked.connect(self.move_current_to_save)
        self.btn_trash.clicked.connect(self.move_current_to_trash)
        self.file_list.currentRowChanged.connect(self.jump_to_index)

    def choose_image_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择图片输入路径", self.edit_image_dir.text().strip())
        if path:
            self.edit_image_dir.setText(path)

    def choose_label_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择标签输入路径", self.edit_label_dir.text().strip())
        if path:
            self.edit_label_dir.setText(path)

    def choose_save_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择保存路径", self.edit_save_dir.text().strip())
        if path:
            self.edit_save_dir.setText(path)

    def open_dataset(self):
        image_dir_text = self.edit_image_dir.text().strip()
        label_dir_text = self.edit_label_dir.text().strip() or image_dir_text
        if not image_dir_text:
            QMessageBox.warning(self, "提示", "请先选择图片输入路径。")
            return
        image_dir = Path(image_dir_text)
        label_dir = Path(label_dir_text)
        if not image_dir.exists():
            QMessageBox.warning(self, "提示", "图片输入路径不存在。")
            return
        if not label_dir.exists():
            QMessageBox.warning(self, "提示", "标签输入路径不存在。")
            return

        self.items = collect_filter_items(image_dir, label_dir)
        self.current_index = -1
        self.current_item = None
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for item in self.items:
            QListWidgetItem(item.image_path.name, self.file_list)
        self.file_list.blockSignals(False)

        if not self.items:
            self.lbl_current_name.setText("-")
            self.lbl_index.setText("0 / 0")
            self.lbl_point_count.setText("0")
            self.status_label.setText("没有找到成对的图片和 json。")
            return

        self.status_label.setText(f"已加载 {len(self.items)} 对数据，未配对项已自动跳过。")
        self.jump_to_index(0)

    def jump_to_index(self, index: int):
        if not self.items:
            return
        index = int(np.clip(index, 0, len(self.items) - 1))
        if index == self.current_index and self.current_item is not None:
            return

        item = self.items[index]
        try:
            image = read_image(item.image_path)
            points = load_points_from_json(item.label_path)
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", str(exc))
            return

        self.current_index = index
        self.current_item = item
        self.file_list.blockSignals(True)
        self.file_list.setCurrentRow(index)
        self.file_list.blockSignals(False)
        self.canvas.set_data(image, points)
        self.lbl_current_name.setText(item.image_path.name)
        self.lbl_index.setText(f"{index + 1} / {len(self.items)}")
        self.lbl_point_count.setText(str(len(points)))
        self.status_label.setText(f"已加载 {item.image_path.name}")

    def _refresh_counts(self):
        self.lbl_saved_count.setText(str(self.saved_count))
        self.lbl_trash_count.setText(str(self.trash_count))

    def _take_current_item(self) -> tuple[Optional[FilterItem], int]:
        if self.current_index < 0 or self.current_index >= len(self.items):
            return None, -1
        item = self.items.pop(self.current_index)
        self.file_list.blockSignals(True)
        self.file_list.takeItem(self.current_index)
        self.file_list.blockSignals(False)
        removed_index = self.current_index
        self.current_index = -1
        self.current_item = None
        return item, removed_index

    def _show_after_removal(self, removed_index: int, status_message: str):
        if not self.items:
            self.lbl_current_name.setText("-")
            self.lbl_index.setText("0 / 0")
            self.lbl_point_count.setText("0")
            self.canvas.points = []
            self.canvas.image_bgr = None
            self.canvas.image_rgb = None
            self.canvas.image_qimage = None
            self.canvas.update()
            self.status_label.setText(status_message + " 当前没有剩余数据。")
            return
        next_index = min(removed_index, len(self.items) - 1)
        self.status_label.setText(status_message)
        self.jump_to_index(next_index)

    def _validate_save_dir(self) -> Optional[Path]:
        save_dir_text = self.edit_save_dir.text().strip()
        if not save_dir_text:
            QMessageBox.warning(self, "提示", "请先选择保存路径。")
            return None
        return Path(save_dir_text)

    def move_current_to_save(self):
        dest_dir = self._validate_save_dir()
        if dest_dir is None:
            return
        item, removed_index = self._take_current_item()
        if item is None:
            return
        try:
            moved_image, moved_label = move_item_pair(item, dest_dir)
        except Exception as exc:
            self.items.insert(removed_index, item)
            self.file_list.blockSignals(True)
            self.file_list.insertItem(removed_index, item.image_path.name)
            self.file_list.blockSignals(False)
            self.current_index = -1
            QMessageBox.critical(self, "保存失败", str(exc))
            self.jump_to_index(removed_index)
            return
        self.saved_count += 1
        self._refresh_counts()
        self._show_after_removal(removed_index, f"已移动到保存路径: {moved_image.name}, {moved_label.name}")

    def _get_trash_dir(self) -> Path:
        if self.trash_dir is None:
            self.trash_dir = make_trash_dir(Path.cwd())
        return self.trash_dir

    def move_current_to_trash(self):
        item, removed_index = self._take_current_item()
        if item is None:
            return
        trash_dir = self._get_trash_dir()
        try:
            moved_image, moved_label = move_item_pair(item, trash_dir)
        except Exception as exc:
            self.items.insert(removed_index, item)
            self.file_list.blockSignals(True)
            self.file_list.insertItem(removed_index, item.image_path.name)
            self.file_list.blockSignals(False)
            self.current_index = -1
            QMessageBox.critical(self, "移动失败", str(exc))
            self.jump_to_index(removed_index)
            return
        self.trash_count += 1
        self._refresh_counts()
        self._show_after_removal(removed_index, f"已移到垃圾桶: {moved_image.name}, {moved_label.name}")

    def keyPressEvent(self, event):
        focused = QApplication.focusWidget()
        if isinstance(focused, QLineEdit):
            super().keyPressEvent(event)
            return
        if event.key() == Qt.Key_A:
            self.jump_to_index(self.current_index - 1)
            event.accept()
            return
        if event.key() == Qt.Key_D:
            self.jump_to_index(self.current_index + 1)
            event.accept()
            return
        if event.key() == Qt.Key_S:
            self.move_current_to_save()
            event.accept()
            return
        if event.key() == Qt.Key_W:
            self.move_current_to_trash()
            event.accept()
            return
        if event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self.canvas.change_zoom(+0.5)
            self.status_label.setText(f"当前缩放: {self.canvas.scale:.2f}x")
            event.accept()
            return
        if event.key() in (Qt.Key_Minus, Qt.Key_Underscore):
            self.canvas.change_zoom(-0.5)
            self.status_label.setText(f"当前缩放: {self.canvas.scale:.2f}x")
            event.accept()
            return
        super().keyPressEvent(event)
