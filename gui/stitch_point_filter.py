"""CAB-F stitch point dataset filtering dialog."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
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
from gui.tools.base import make_card, make_page_header, set_primary
from core.cabf_shared import (
    IMAGE_SUFFIXES,
    load_labelme_points,
    normalize_edges_for_editor,
    normalize_master_annotation,
    normalize_points_for_editor,
    read_json as load_json,
)

POINT_COLOR = QColor(80, 255, 80)
EDGE_COLOR = QColor(74, 144, 226)
DEFAULT_IMAGE_DIR = ""
DEFAULT_LABEL_DIR = ""


def read_image(image_path: str | Path) -> np.ndarray:
    image_path = Path(image_path)
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    return image
def normalize_points(points: list[dict]) -> list[dict]:
    return [point for point in normalize_points_for_editor(points) if "x" in point and "y" in point]


def load_annotation_from_json(json_path: Path) -> tuple[list[dict], list[dict]]:
    data = load_json(json_path)
    if isinstance(data, dict) and (isinstance(data.get("points"), list) or isinstance(data.get("edges"), list)):
        normalized, _ = normalize_master_annotation(data, sample_id=json_path.stem, image_path=f"{json_path.stem}.png")
        points = normalize_points(normalized.get("points", []))
        edges = normalize_edges_for_editor(normalized.get("edges", []))
        return points, edges
    return normalize_points(load_labelme_points(data)), []


@dataclass
class FilterItem:
    image_path: Path
    label_path: Optional[Path] = None

    @property
    def stem(self) -> str:
        return self.image_path.stem

    @property
    def has_label(self) -> bool:
        return self.label_path is not None


def collect_filter_items(
    image_dir: Path,
    label_dir: Optional[Path] = None,
    *,
    require_label: bool = True,
) -> list[FilterItem]:
    label_root = label_dir or image_dir
    items: list[FilterItem] = []
    for path in sorted(image_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = label_root / f"{path.stem}.json"
        if label_path.exists() and label_path.is_file():
            items.append(FilterItem(image_path=path, label_path=label_path))
        elif not require_label:
            items.append(FilterItem(image_path=path, label_path=None))
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


def move_item_files(item: FilterItem, dest_dir: Path) -> tuple[Path, Optional[Path]]:
    moved_image = move_file_safe(item.image_path, dest_dir)
    moved_label = move_file_safe(item.label_path, dest_dir) if item.label_path else None
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
        self.edges: list[dict] = []

        self.scale = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._pan_anchor: Optional[QPoint] = None

    def set_data(self, image_bgr: np.ndarray, points: list[dict], edges: Optional[list[dict]] = None):
        self.image_bgr = image_bgr
        self.image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = self.image_rgb.shape[:2]
        bytes_per_line = width * 3
        self.image_qimage = QImage(self.image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
        self.points = normalize_points(points)
        self.edges = normalize_edges_for_editor(edges or [])
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

        point_lookup = {int(point["id"]): point for point in self.points}
        edge_pen = QPen(EDGE_COLOR, max(1, int(round(2 * self.scale))))
        painter.setPen(edge_pen)
        for edge in self.edges:
            src = point_lookup.get(int(edge.get("src", -1)))
            dst = point_lookup.get(int(edge.get("dst", -1)))
            if src is None or dst is None:
                continue
            x1, y1 = self._image_to_canvas(src["x"], src["y"])
            x2, y2 = self._image_to_canvas(dst["x"], dst["y"])
            painter.drawLine(x1, y1, x2, y2)

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
    applyRequested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CAB-F 缝纫点数据筛选")
        self.resize(1480, 900)

        self.items: list[FilterItem] = []
        self.current_index: int = -1
        self.current_item: Optional[FilterItem] = None
        self.trash_dir: Optional[Path] = None
        self.saved_count = 0
        self.trash_count = 0

        self._build_ui()
        self._connect_signals()

    def configure_paths(
        self,
        *,
        mode: str = "unlabeled",
        image_dir: str = "",
        label_dir: str = "",
        save_dir: str = "",
        auto_load: bool = False,
    ) -> None:
        mode_value = "labeled" if str(mode).lower() == "labeled" else "unlabeled"
        idx = 1 if mode_value == "labeled" else 0
        self.combo_mode.setCurrentIndex(idx)
        if image_dir:
            self.edit_image_dir.setText(image_dir)
        if label_dir:
            self.edit_label_dir.setText(label_dir)
        if save_dir:
            self.edit_save_dir.setText(save_dir)
        if auto_load and image_dir:
            self.open_dataset()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)
        root.addWidget(make_page_header("缝纫点数据筛选", "快速浏览样本，保留有效数据，剔除不需要的图片与标签。"))

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        left = QWidget()
        left.setMinimumWidth(360)
        left.setMaximumWidth(460)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        folder_box = make_card()
        folder_layout = QFormLayout(folder_box)
        folder_layout.setContentsMargins(14, 12, 14, 12)
        folder_layout.setSpacing(8)
        self.combo_mode = QComboBox()
        self.combo_mode.addItem("无标签模式", "unlabeled")
        self.combo_mode.addItem("有标签模式", "labeled")
        self.edit_image_dir = QLineEdit(DEFAULT_IMAGE_DIR)
        self.edit_label_dir = QLineEdit(DEFAULT_LABEL_DIR)
        self.edit_save_dir = QLineEdit("")
        folder_layout.addRow("筛选模式", self.combo_mode)
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

        info_box = make_card()
        info_layout = QFormLayout(info_box)
        info_layout.setContentsMargins(14, 12, 14, 12)
        info_layout.setSpacing(8)
        self.lbl_current_name = QLabel("-")
        self.lbl_index = QLabel("0 / 0")
        self.lbl_point_count = QLabel("0")
        self.lbl_edge_count = QLabel("0")
        self.lbl_label_state = QLabel("无")
        self.lbl_saved_count = QLabel("0")
        self.lbl_trash_count = QLabel("0")
        info_layout.addRow("当前图片", self.lbl_current_name)
        info_layout.addRow("进度", self.lbl_index)
        info_layout.addRow("标签状态", self.lbl_label_state)
        info_layout.addRow("点数", self.lbl_point_count)
        info_layout.addRow("边数", self.lbl_edge_count)
        info_layout.addRow("已保存", self.lbl_saved_count)
        info_layout.addRow("已移垃圾桶", self.lbl_trash_count)
        left_layout.addWidget(info_box)

        action_box = make_card()
        action_lay = QVBoxLayout(action_box)
        action_lay.setContentsMargins(14, 12, 14, 12)
        action_lay.setSpacing(8)

        row_nav = QHBoxLayout()
        row_nav.setSpacing(8)
        self.btn_prev = QPushButton("上一张(A)")
        self.btn_next = QPushButton("下一张(D)")
        row_nav.addWidget(self.btn_prev)
        row_nav.addWidget(self.btn_next)
        action_lay.addLayout(row_nav)

        row_action = QHBoxLayout()
        row_action.setSpacing(8)
        self.btn_save = QPushButton("保存当前(S)")
        set_primary(self.btn_save)
        self.btn_trash = QPushButton("移到垃圾桶(W)")
        row_action.addWidget(self.btn_save)
        row_action.addWidget(self.btn_trash)
        action_lay.addLayout(row_action)

        self.btn_apply_flow = QPushButton("应用筛选结果到后续流程")
        self.btn_apply_flow.clicked.connect(self.apply_to_workflow)
        action_lay.addWidget(self.btn_apply_flow)
        left_layout.addWidget(action_box)

        list_box = make_card()
        list_lay = QVBoxLayout(list_box)
        list_lay.setContentsMargins(14, 12, 14, 12)
        list_lay.setSpacing(8)
        list_title = QLabel("图片列表")
        list_title.setStyleSheet("color:#111827;font-size:14px;font-weight:700;")
        list_lay.addWidget(list_title)
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(220)
        list_lay.addWidget(self.file_list)
        left_layout.addWidget(list_box, 1)

        self.lbl_help = QLabel(
            "使用方式\n"
            "1. 先选择筛选模式，再设置图片路径、标签路径和保存路径\n"
            "2. 无标签模式会加载全部图片；如果存在同名 json，会同时显示点和边\n"
            "3. 有标签模式只加载能配对成功的数据，并按母标签显示点和边\n"
            "4. A/D 切图，S 移动到保存路径并切换下一张\n"
            "5. W 移动当前图片与同名 json 到项目 .trash 后切换下一张"
        )
        self.lbl_help.setWordWrap(True)
        self.lbl_help.setStyleSheet("color:#6B7280;font-size:12px;")
        left_layout.addWidget(self.lbl_help)

        self.status_label = QLabel("请先加载数据。")
        left_layout.addWidget(self.status_label)

        canvas_card = make_card()
        canvas_lay = QVBoxLayout(canvas_card)
        canvas_lay.setContentsMargins(14, 12, 14, 12)
        canvas_lay.setSpacing(8)
        canvas_title = QLabel("样本预览")
        canvas_title.setStyleSheet("color:#111827;font-size:14px;font-weight:700;")
        canvas_lay.addWidget(canvas_title)
        self.canvas = PointFilterCanvas()
        canvas_lay.addWidget(self.canvas, 1)
        splitter.addWidget(left)
        splitter.addWidget(canvas_card)
        splitter.setSizes([400, 1180])

    def _connect_signals(self):
        self.combo_mode.currentIndexChanged.connect(self._sync_mode_ui)
        self.btn_choose_image_dir.clicked.connect(self.choose_image_dir)
        self.btn_choose_label_dir.clicked.connect(self.choose_label_dir)
        self.btn_choose_save_dir.clicked.connect(self.choose_save_dir)
        self.btn_load.clicked.connect(self.open_dataset)
        self.btn_prev.clicked.connect(lambda: self.jump_to_index(self.current_index - 1))
        self.btn_next.clicked.connect(lambda: self.jump_to_index(self.current_index + 1))
        self.btn_save.clicked.connect(self.move_current_to_save)
        self.btn_trash.clicked.connect(self.move_current_to_trash)
        self.file_list.currentRowChanged.connect(self.jump_to_index)
        self._sync_mode_ui()

    def _current_mode(self) -> str:
        return str(self.combo_mode.currentData() or "unlabeled")

    def _sync_mode_ui(self):
        labeled_mode = self._current_mode() == "labeled"
        self.edit_label_dir.setEnabled(True)
        self.btn_choose_label_dir.setEnabled(True)
        if labeled_mode:
            self.status_label.setText("有标签模式: 仅加载图片与同名 json 配对成功的数据。")
        else:
            self.status_label.setText("无标签模式: 加载全部图片；若存在同名 json，会额外显示点和边。")

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
        labeled_mode = self._current_mode() == "labeled"
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
        if labeled_mode and not label_dir.exists():
            QMessageBox.warning(self, "提示", "标签输入路径不存在。")
            return
        if not labeled_mode and not label_dir.exists():
            label_dir = None

        self.items = collect_filter_items(image_dir, label_dir, require_label=labeled_mode)
        self.current_index = -1
        self.current_item = None
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for item in self.items:
            label_suffix = "" if item.has_label else "  [无标签]"
            QListWidgetItem(f"{item.image_path.name}{label_suffix}", self.file_list)
        self.file_list.blockSignals(False)

        if not self.items:
            self.lbl_current_name.setText("-")
            self.lbl_index.setText("0 / 0")
            self.lbl_point_count.setText("0")
            self.lbl_edge_count.setText("0")
            self.lbl_label_state.setText("无")
            self.status_label.setText("没有找到符合当前模式的数据。")
            return

        labeled_count = sum(1 for item in self.items if item.has_label)
        unlabeled_count = len(self.items) - labeled_count
        if labeled_mode:
            self.status_label.setText(f"已加载 {len(self.items)} 对有标签数据，未配对项已自动跳过。")
        else:
            self.status_label.setText(f"已加载 {len(self.items)} 张图片，其中 {labeled_count} 张带标签，{unlabeled_count} 张无标签。")
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
            if item.label_path is not None:
                points, edges = load_annotation_from_json(item.label_path)
            else:
                points, edges = [], []
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", str(exc))
            return

        self.current_index = index
        self.current_item = item
        self.file_list.blockSignals(True)
        self.file_list.setCurrentRow(index)
        self.file_list.blockSignals(False)
        self.canvas.set_data(image, points, edges)
        self.lbl_current_name.setText(item.image_path.name)
        self.lbl_index.setText(f"{index + 1} / {len(self.items)}")
        self.lbl_label_state.setText("有" if item.has_label else "无")
        self.lbl_point_count.setText(str(len(points)))
        self.lbl_edge_count.setText(str(len(edges)))
        self.status_label.setText(
            f"已加载 {item.image_path.name}"
            + (f" ，点 {len(points)} / 边 {len(edges)}" if item.has_label else " ，当前无标签")
        )

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
            self.lbl_label_state.setText("无")
            self.lbl_point_count.setText("0")
            self.lbl_edge_count.setText("0")
            self.canvas.points = []
            self.canvas.edges = []
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
            moved_image, moved_label = move_item_files(item, dest_dir)
        except Exception as exc:
            self.items.insert(removed_index, item)
            self.file_list.blockSignals(True)
            label_suffix = "" if item.has_label else "  [无标签]"
            self.file_list.insertItem(removed_index, f"{item.image_path.name}{label_suffix}")
            self.file_list.blockSignals(False)
            self.current_index = -1
            QMessageBox.critical(self, "保存失败", str(exc))
            self.jump_to_index(removed_index)
            return
        self.saved_count += 1
        self._refresh_counts()
        moved_parts = [moved_image.name]
        if moved_label is not None:
            moved_parts.append(moved_label.name)
        self._show_after_removal(removed_index, f"已移动到保存路径: {', '.join(moved_parts)}")

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
            moved_image, moved_label = move_item_files(item, trash_dir)
        except Exception as exc:
            self.items.insert(removed_index, item)
            self.file_list.blockSignals(True)
            label_suffix = "" if item.has_label else "  [无标签]"
            self.file_list.insertItem(removed_index, f"{item.image_path.name}{label_suffix}")
            self.file_list.blockSignals(False)
            self.current_index = -1
            QMessageBox.critical(self, "移动失败", str(exc))
            self.jump_to_index(removed_index)
            return
        self.trash_count += 1
        self._refresh_counts()
        moved_parts = [moved_image.name]
        if moved_label is not None:
            moved_parts.append(moved_label.name)
        self._show_after_removal(removed_index, f"已移到垃圾桶: {', '.join(moved_parts)}")

    def apply_to_workflow(self):
        dest_dir = self._validate_save_dir()
        if dest_dir is None:
            return
        dest_dir.mkdir(parents=True, exist_ok=True)
        self.applyRequested.emit(str(dest_dir))
        self.status_label.setText(f"已应用到后续流程：{dest_dir}")

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
