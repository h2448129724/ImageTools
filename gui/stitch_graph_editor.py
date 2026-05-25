"""Simplified CAB-F edge annotation dialog for img_tools."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QImage, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import (
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
SELECTED_POINT_COLOR = QColor(0, 220, 255)
PENDING_POINT_COLOR = QColor(255, 255, 255)
EDGE_COLOR = QColor(255, 180, 0)
DEFAULT_IMAGE_DIR = r"D:\project\changrui\CAB-F\sew_point\images"
DEFAULT_LABEL_DIR = r"D:\project\changrui\CAB-F\sew_point\train_edge_labeled"
DEFAULT_OUTPUT_DIR = r"D:\project\changrui\CAB-F\sew_point\train_edge_labeled"
POINT_LABEL_ALIASES = {"sew", "keypoint"}
MODE_STATUS_STYLES = {
    "edge": ("当前模式：连边", "background:#fff3cd;color:#856404;border:1px solid #ffe08a;padding:6px;border-radius:4px;"),
    "add": ("当前模式：新增点", "background:#d1f7d6;color:#176b2c;border:1px solid #8bd8a0;padding:6px;border-radius:4px;"),
    "move": ("当前模式：移动点", "background:#d8ebff;color:#155a9c;border:1px solid #8ec2ff;padding:6px;border-radius:4px;"),
    "delete": ("当前模式：删点", "background:#ffe1e1;color:#a12626;border:1px solid #ff9e9e;padding:6px;border-radius:4px;"),
}


def read_image(image_path) -> np.ndarray:
    image_path = Path(image_path)
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    return image


def load_json(path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def make_empty_annotation(image_path: str, width: int, height: int, sample_id: str):
    return {
        "schema_version": "1.1",
        "sample_id": sample_id,
        "image_path": image_path,
        "image_size": {"width": int(width), "height": int(height)},
        "points": [],
        "edges": [],
        "segments": [],
        "metadata": {},
    }


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
        if str(shape.get("label", "")).strip().lower() not in POINT_LABEL_ALIASES:
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


def convert_labelme_json_to_base(json_path: Path, image_path: Path) -> dict:
    data = load_json(json_path)
    width = int(data.get("imageWidth", 0) or 256)
    height = int(data.get("imageHeight", 0) or 256)
    annotation = make_empty_annotation(str(image_path), width, height, image_path.stem)
    annotation["points"] = load_labelme_points(json_path)
    annotation["metadata"] = {
        "source": "labelme_point_folder",
        "origin_json": str(json_path),
        "has_point_json": True,
    }
    return annotation


def normalize_points(points: list[dict]) -> list[dict]:
    normalized = []
    for idx, point in enumerate(points):
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


def normalize_edges(edges: list[dict]) -> list[dict]:
    normalized = []
    seen = set()
    for edge in edges:
        src = int(edge.get("src", -1))
        dst = int(edge.get("dst", -1))
        if src < 0 or dst < 0 or src == dst:
            continue
        key = tuple(sorted((src, dst)))
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "edge_id": str(edge.get("edge_id") or f"edge_{len(normalized) + 1:04d}"),
                "src": key[0],
                "dst": key[1],
                "label": int(edge.get("label", 1)),
                "source": edge.get("source", "manual"),
            }
        )
    return normalized


@dataclass
class FolderItem:
    image_path: Path
    source_json_path: Optional[Path]

    @property
    def stem(self) -> str:
        return self.image_path.stem


def collect_folder_items(image_folder: Path, label_folder: Optional[Path] = None) -> list[FolderItem]:
    label_map: dict[str, Path] = {}
    if label_folder is not None and label_folder.exists():
        label_map = {path.stem: path for path in sorted(label_folder.glob("*.json"))}

    items: list[FolderItem] = []
    for path in sorted(image_folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            source_json = label_map.get(path.stem)
            items.append(FolderItem(image_path=path, source_json_path=source_json))
    return items


class EdgeAnnotationCanvas(QWidget):
    pointSelectionChanged = Signal(object)
    pointCountChanged = Signal(int)
    edgeCountChanged = Signal(int)
    pendingChanged = Signal(object)
    statusMessage = Signal(str)
    annotationModified = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(840, 620)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.image_bgr: Optional[np.ndarray] = None
        self.image_rgb: Optional[np.ndarray] = None
        self.image_qimage: Optional[QImage] = None
        self.image_path: str = ""
        self.points: list[dict] = []
        self.edges: list[dict] = []
        self.selected_point_id: Optional[int] = None
        self.pending_start_id: Optional[int] = None
        self.mode: str = "edge"

        self.scale = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._pan_anchor: Optional[QPoint] = None
        self._drag_point_id: Optional[int] = None

    def set_image(self, image_bgr: np.ndarray, image_path: str = ""):
        self.image_bgr = image_bgr
        self.image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = self.image_rgb.shape[:2]
        bytes_per_line = width * 3
        self.image_qimage = QImage(self.image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
        self.image_path = image_path
        self.points = []
        self.edges = []
        self.selected_point_id = None
        self.pending_start_id = None
        self.fit_view()
        self.pointCountChanged.emit(0)
        self.edgeCountChanged.emit(0)
        self.pendingChanged.emit(None)

    def set_points(self, points: list[dict]):
        self.points = normalize_points(points)
        self.selected_point_id = None
        self.pending_start_id = None
        self.pointSelectionChanged.emit(None)
        self.pointCountChanged.emit(len(self.points))
        self.pendingChanged.emit(None)
        self.update()

    def set_edges(self, edges: list[dict]):
        self.edges = normalize_edges(edges)
        self.edgeCountChanged.emit(len(self.edges))
        self.update()

    def set_mode(self, mode: str):
        self.mode = mode
        if mode != "edge":
            self.pending_start_id = None
            self.pendingChanged.emit(None)
        cursor_map = {
            "edge": Qt.PointingHandCursor,
            "add": Qt.CrossCursor,
            "move": Qt.OpenHandCursor,
            "delete": Qt.ForbiddenCursor,
        }
        self.setCursor(QCursor(cursor_map.get(mode, Qt.ArrowCursor)))
        self.update()

    def clear_edges(self):
        self.edges = []
        self.pending_start_id = None
        self.edgeCountChanged.emit(0)
        self.pendingChanged.emit(None)
        self.annotationModified.emit()
        self.statusMessage.emit("已清空所有连边。")
        self.update()

    def undo_last_edge(self):
        if not self.edges:
            return
        removed = self.edges.pop()
        self.edgeCountChanged.emit(len(self.edges))
        self.annotationModified.emit()
        self.statusMessage.emit(f"已撤销连边 {removed['src']} - {removed['dst']}")
        self.update()

    def _next_point_id(self) -> int:
        if not self.points:
            return 0
        return max(int(point["id"]) for point in self.points) + 1

    def _remove_point_and_edges(self, point_id: int):
        point_id = int(point_id)
        before_points = len(self.points)
        before_edges = len(self.edges)
        self.points = [point for point in self.points if int(point["id"]) != point_id]
        self.edges = [
            edge for edge in self.edges
            if int(edge["src"]) != point_id and int(edge["dst"]) != point_id
        ]
        self.selected_point_id = None if self.selected_point_id == point_id else self.selected_point_id
        self.pending_start_id = None if self.pending_start_id == point_id else self.pending_start_id
        self.pointSelectionChanged.emit(None)
        self.pointCountChanged.emit(len(self.points))
        self.edgeCountChanged.emit(len(self.edges))
        self.pendingChanged.emit(self.pending_start_id)
        self.annotationModified.emit()
        self.statusMessage.emit(
            f"已删除点 {point_id}，移除 {before_points - len(self.points)} 个点和 {before_edges - len(self.edges)} 条边。"
        )
        self.update()

    def _add_point(self, image_x: float, image_y: float):
        point_id = self._next_point_id()
        point = {
            "id": point_id,
            "x": float(image_x),
            "y": float(image_y),
            "score": 1.0,
            "source": "manual",
        }
        self.points.append(point)
        self.selected_point_id = point_id
        self.pointSelectionChanged.emit(point)
        self.pointCountChanged.emit(len(self.points))
        self.annotationModified.emit()
        self.statusMessage.emit(f"已新增点 {point_id} ({image_x:.1f}, {image_y:.1f})")
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

    def change_zoom(self, delta: float):
        new_scale = float(np.clip(self.scale + delta, 0.05, 20.0))
        if abs(new_scale - self.scale) < 1e-6:
            return
        self.scale = new_scale
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

    def _find_point(self, point_id: int) -> Optional[dict]:
        return next((p for p in self.points if int(p["id"]) == int(point_id)), None)

    def _has_edge(self, src: int, dst: int) -> bool:
        key = tuple(sorted((int(src), int(dst))))
        return any(tuple(sorted((int(edge["src"]), int(edge["dst"])))) == key for edge in self.edges)

    def _toggle_edge(self, src: int, dst: int):
        key = tuple(sorted((int(src), int(dst))))
        for idx, edge in enumerate(self.edges):
            if tuple(sorted((int(edge["src"]), int(edge["dst"])))) == key:
                self.edges.pop(idx)
                self.edgeCountChanged.emit(len(self.edges))
                self.annotationModified.emit()
                self.statusMessage.emit(f"已删除连边 {key[0]} - {key[1]}")
                self.update()
                return
        self.edges.append(
            {
                "edge_id": f"edge_{len(self.edges) + 1:04d}",
                "src": key[0],
                "dst": key[1],
                "label": 1,
                "source": "manual",
            }
        )
        self.edgeCountChanged.emit(len(self.edges))
        self.annotationModified.emit()
        self.statusMessage.emit(f"已新增连边 {key[0]} - {key[1]}")
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#111111"))

        if self.image_qimage is None:
            painter.setPen(QColor("#ffffff"))
            painter.drawText(self.rect(), Qt.AlignCenter, "请先选择数据文件夹")
            return

        self._clamp_pan()
        src_rect = QRect(
            int(self.pan_x),
            int(self.pan_y),
            int(np.ceil(self.width() / max(self.scale, 1e-6))),
            int(np.ceil(self.height() / max(self.scale, 1e-6))),
        )
        painter.drawImage(self.rect(), self.image_qimage, src_rect)

        painter.setPen(QPen(EDGE_COLOR, 2))
        for edge in self.edges:
            pa = self._find_point(int(edge["src"]))
            pb = self._find_point(int(edge["dst"]))
            if pa is None or pb is None:
                continue
            ax, ay = self._image_to_canvas(pa["x"], pa["y"])
            bx, by = self._image_to_canvas(pb["x"], pb["y"])
            painter.drawLine(ax, ay, bx, by)

        if self.pending_start_id is not None:
            point = self._find_point(int(self.pending_start_id))
            if point is not None:
                px, py = self._image_to_canvas(point["x"], point["y"])
                painter.setPen(QPen(PENDING_POINT_COLOR, 2, Qt.DashLine))
                painter.drawEllipse(QPoint(px, py), 10, 10)

        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        for point in self.points:
            cx, cy = self._image_to_canvas(point["x"], point["y"])
            point_id = int(point["id"])
            if point_id == self.selected_point_id:
                fill = SELECTED_POINT_COLOR
                radius = 6
            elif point_id == self.pending_start_id:
                fill = PENDING_POINT_COLOR
                radius = 5
            else:
                fill = POINT_COLOR
                radius = 4
            painter.setPen(QPen(QColor(20, 20, 20), 1))
            painter.setBrush(fill)
            painter.drawEllipse(QPoint(cx, cy), radius, radius)
            painter.setPen(QColor("#ffffff"))
            painter.drawText(cx + 6, cy - 6, str(point_id))

        painter.setPen(QColor(80, 255, 80))
        painter.drawText(10, 22, f"mode={self.mode} points={len(self.points)} edges={len(self.edges)} zoom={self.scale:.2f}")

    def mousePressEvent(self, event):
        if self.image_qimage is None:
            return
        if event.button() == Qt.RightButton:
            if self.pending_start_id is not None:
                self.pending_start_id = None
                self.selected_point_id = None
                self.pointSelectionChanged.emit(None)
                self.pendingChanged.emit(None)
                self.statusMessage.emit("已取消当前待连边起点。")
                self.update()
                return
            self._pan_anchor = event.pos()
            return
        if event.button() != Qt.LeftButton:
            return

        image_x, image_y = self._canvas_to_image(event.position().toPoint())
        nearest = self._nearest_point(image_x, image_y)

        if self.mode == "add":
            self._add_point(image_x, image_y)
            return

        if nearest is None:
            self.selected_point_id = None
            self.pointSelectionChanged.emit(None)
            self.update()
            return

        point_id = int(nearest["id"])
        self.selected_point_id = point_id
        self.pointSelectionChanged.emit(nearest)

        if self.mode == "delete":
            self._remove_point_and_edges(point_id)
            return

        if self.mode == "move":
            self._drag_point_id = point_id
            self.statusMessage.emit(f"已选择点 {point_id}，拖动可修改位置。")
            self.update()
            return

        if self.pending_start_id is None:
            self.pending_start_id = point_id
            self.pendingChanged.emit(point_id)
            self.statusMessage.emit(f"已选择起点 {point_id}，请再点一个点连边。")
            self.update()
            return

        if point_id == self.pending_start_id:
            self.pending_start_id = None
            self.pendingChanged.emit(None)
            self.statusMessage.emit("已取消当前待连边起点。")
            self.update()
            return

        self._toggle_edge(self.pending_start_id, point_id)
        keep_chaining = bool(event.modifiers() & Qt.ShiftModifier)
        self.pending_start_id = point_id if keep_chaining else None
        self.pendingChanged.emit(self.pending_start_id)
        if keep_chaining:
            self.statusMessage.emit(f"已选择起点 {point_id}，可继续连下一条边。")
        else:
            self.statusMessage.emit("已完成当前连边。")
        self.update()

    def mouseMoveEvent(self, event):
        if self.image_qimage is None:
            return
        if self._drag_point_id is not None and self.mode == "move":
            image_x, image_y = self._canvas_to_image(event.position().toPoint())
            point = self._find_point(self._drag_point_id)
            if point is not None:
                point["x"] = float(image_x)
                point["y"] = float(image_y)
                self.pointSelectionChanged.emit(point)
                self.annotationModified.emit()
                self.update()
            return
        if self._pan_anchor is not None:
            delta = event.pos() - self._pan_anchor
            self.pan_x -= delta.x() / max(self.scale, 1e-6)
            self.pan_y -= delta.y() / max(self.scale, 1e-6)
            self._pan_anchor = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._drag_point_id is not None:
            point = self._find_point(self._drag_point_id)
            if point is not None:
                self.statusMessage.emit(f"已移动点 {self._drag_point_id} 到 ({point['x']:.1f}, {point['y']:.1f})")
            self._drag_point_id = None
        if event.button() == Qt.RightButton:
            self._pan_anchor = None

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


class StitchGraphEditorDialog(QDialog):
    def __init__(self, parent=None, image: Optional[np.ndarray] = None, image_path: str = ""):
        super().__init__(parent)
        self.setWindowTitle("CAB-F 点边一体标注器")
        self.resize(1600, 960)

        self.folder_items: list[FolderItem] = []
        self.current_index: int = -1
        self.current_annotation: Optional[dict] = None
        self.current_output_path: Optional[Path] = None

        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        left = QWidget()
        left.setMinimumWidth(360)
        left.setMaximumWidth(460)
        left_layout = QVBoxLayout(left)

        folder_box = QFrame()
        folder_layout = QFormLayout(folder_box)
        self.edit_src_dir = QLineEdit(DEFAULT_IMAGE_DIR)
        self.edit_label_dir = QLineEdit(DEFAULT_LABEL_DIR)
        self.edit_output_dir = QLineEdit(DEFAULT_OUTPUT_DIR)
        folder_layout.addRow("图片文件夹", self.edit_src_dir)
        folder_layout.addRow("标签文件夹", self.edit_label_dir)
        folder_layout.addRow("输出文件夹", self.edit_output_dir)
        self.check_overwrite_source = QPushButton("直接覆盖原标签：关")
        self.check_overwrite_source.setCheckable(True)
        self.check_overwrite_source.toggled.connect(self._on_toggle_overwrite_source)
        folder_layout.addRow("保存方式", self.check_overwrite_source)
        left_layout.addWidget(folder_box)

        row_folder = QHBoxLayout()
        self.btn_choose_src = QPushButton("选择图片文件夹")
        self.btn_choose_label = QPushButton("选择标签文件夹")
        self.btn_choose_out = QPushButton("选择输出文件夹")
        row_folder.addWidget(self.btn_choose_src)
        row_folder.addWidget(self.btn_choose_label)
        row_folder.addWidget(self.btn_choose_out)
        left_layout.addLayout(row_folder)

        row_open = QHBoxLayout()
        self.btn_open_folder = QPushButton("加载文件夹")
        self.btn_reload_current = QPushButton("重新读取当前图")
        row_open.addWidget(self.btn_open_folder)
        row_open.addWidget(self.btn_reload_current)
        left_layout.addLayout(row_open)

        nav_box = QFrame()
        nav_layout = QFormLayout(nav_box)
        self.lbl_current_name = QLabel("-")
        self.lbl_index = QLabel("0 / 0")
        self.lbl_point_count = QLabel("0")
        self.lbl_edge_count = QLabel("0")
        self.lbl_selected = QLabel("-")
        self.lbl_pending = QLabel("-")
        nav_layout.addRow("当前图片", self.lbl_current_name)
        nav_layout.addRow("进度", self.lbl_index)
        nav_layout.addRow("点数", self.lbl_point_count)
        nav_layout.addRow("连边数", self.lbl_edge_count)
        nav_layout.addRow("选中点", self.lbl_selected)
        nav_layout.addRow("待连起点", self.lbl_pending)
        left_layout.addWidget(nav_box)

        row_nav = QHBoxLayout()
        self.btn_prev = QPushButton("上一张")
        self.btn_next = QPushButton("下一张")
        self.btn_save = QPushButton("保存当前")
        row_nav.addWidget(self.btn_prev)
        row_nav.addWidget(self.btn_next)
        row_nav.addWidget(self.btn_save)
        left_layout.addLayout(row_nav)

        row_edge = QHBoxLayout()
        self.btn_undo_edge = QPushButton("撤销上一条边")
        self.btn_clear_edges = QPushButton("清空所有边")
        row_edge.addWidget(self.btn_undo_edge)
        row_edge.addWidget(self.btn_clear_edges)
        left_layout.addLayout(row_edge)

        row_mode = QHBoxLayout()
        self.btn_mode_edge = QPushButton("连边模式")
        self.btn_mode_add = QPushButton("新增点")
        self.btn_mode_move = QPushButton("移动点")
        self.btn_mode_delete = QPushButton("删点")
        for btn in (self.btn_mode_edge, self.btn_mode_add, self.btn_mode_move, self.btn_mode_delete):
            btn.setCheckable(True)
            row_mode.addWidget(btn)
        left_layout.addLayout(row_mode)

        self.file_list = QListWidget()
        left_layout.addWidget(self.file_list, 1)

        self.lbl_help = QLabel(
            "使用方式\n"
            "1. 分别选择图片文件夹和标签文件夹\n"
            "2. 程序会按同名文件匹配图片和 json，并把点显示到图上\n"
            "3. 连边模式下，左键点两个点即可添加或取消一条边\n"
            "4. 新增点/移动点/删点模式用于修正缝纫点\n"
            "5. Shift+左键可连续串边，右键取消当前起点或平移\n"
            "6. A/D 切图，S 保存，+/- 缩放"
        )
        self.lbl_help.setWordWrap(True)
        left_layout.addWidget(self.lbl_help)

        self.status_label = QLabel("请先加载数据文件夹。")
        left_layout.addWidget(self.status_label)

        self.canvas = EdgeAnnotationCanvas()
        splitter.addWidget(left)
        splitter.addWidget(self.canvas)
        splitter.setSizes([420, 1180])
        self._set_mode("edge")

    def _connect_signals(self):
        self.btn_choose_src.clicked.connect(self.choose_src_dir)
        self.btn_choose_label.clicked.connect(self.choose_label_dir)
        self.btn_choose_out.clicked.connect(self.choose_out_dir)
        self.btn_open_folder.clicked.connect(self.open_folder)
        self.btn_reload_current.clicked.connect(self.reload_current_item)
        self.btn_prev.clicked.connect(lambda: self.jump_to_index(self.current_index - 1))
        self.btn_next.clicked.connect(lambda: self.jump_to_index(self.current_index + 1))
        self.btn_save.clicked.connect(lambda: self.save_current_annotation(silent=False))
        self.btn_undo_edge.clicked.connect(self.canvas.undo_last_edge)
        self.btn_clear_edges.clicked.connect(self.canvas.clear_edges)
        self.btn_mode_edge.clicked.connect(lambda: self._set_mode("edge"))
        self.btn_mode_add.clicked.connect(lambda: self._set_mode("add"))
        self.btn_mode_move.clicked.connect(lambda: self._set_mode("move"))
        self.btn_mode_delete.clicked.connect(lambda: self._set_mode("delete"))
        self.file_list.currentRowChanged.connect(self.jump_to_index)
        self.canvas.pointSelectionChanged.connect(self._on_point_selection_changed)
        self.canvas.pointCountChanged.connect(lambda count: self.lbl_point_count.setText(str(count)))
        self.canvas.edgeCountChanged.connect(lambda count: self.lbl_edge_count.setText(str(count)))
        self.canvas.pendingChanged.connect(self._on_pending_changed)
        self.canvas.statusMessage.connect(self.status_label.setText)
        self.canvas.annotationModified.connect(self._auto_save_current)
        self._apply_mode_status_style("edge")

    def _set_mode(self, mode: str):
        mapping = {
            "edge": self.btn_mode_edge,
            "add": self.btn_mode_add,
            "move": self.btn_mode_move,
            "delete": self.btn_mode_delete,
        }
        for key, button in mapping.items():
            button.blockSignals(True)
            button.setChecked(key == mode)
            button.blockSignals(False)
        self.canvas.set_mode(mode)
        self._apply_mode_status_style(mode)

    def _apply_mode_status_style(self, mode: str):
        text, style = MODE_STATUS_STYLES.get(mode, ("当前模式已切换", ""))
        self.status_label.setText(text)
        self.status_label.setStyleSheet(style)

    def _set_status_message(self, text: str):
        self.status_label.setText(text)

    def _on_toggle_overwrite_source(self, checked: bool):
        self.check_overwrite_source.setText(f"直接覆盖原标签：{'开' if checked else '关'}")
        if checked:
            self.edit_output_dir.setEnabled(False)
            self.btn_choose_out.setEnabled(False)
            self._set_status_message("已开启直接覆盖原标签。保存时会写回原始 json 或同目录 json。")
        else:
            self.edit_output_dir.setEnabled(True)
            self.btn_choose_out.setEnabled(True)
            self._apply_mode_status_style(self.canvas.mode)

    def choose_src_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择图片文件夹", self.edit_src_dir.text().strip())
        if path:
            self.edit_src_dir.setText(path)

    def choose_label_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择标签文件夹", self.edit_label_dir.text().strip())
        if path:
            self.edit_label_dir.setText(path)

    def choose_out_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出文件夹", self.edit_output_dir.text().strip())
        if path:
            self.edit_output_dir.setText(path)
            self._auto_save_current()

    def open_folder(self):
        image_folder = Path(self.edit_src_dir.text().strip())
        label_folder_text = self.edit_label_dir.text().strip()
        label_folder = Path(label_folder_text) if label_folder_text else None
        if not image_folder.exists():
            QMessageBox.warning(self, "提示", "图片文件夹不存在。")
            return
        if label_folder is None or not label_folder.exists():
            QMessageBox.warning(self, "提示", "标签文件夹不存在。")
            return
        self.folder_items = collect_folder_items(image_folder, label_folder)
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for item in self.folder_items:
            QListWidgetItem(item.image_path.name, self.file_list)
        self.file_list.blockSignals(False)
        if not self.folder_items:
            self.current_index = -1
            self.lbl_index.setText("0 / 0")
            self.lbl_current_name.setText("-")
            self.status_label.setText("该文件夹下没有图片。")
            return
        self.status_label.setText(f"已加载文件夹，共 {len(self.folder_items)} 张图。")
        self.jump_to_index(0)

    def _build_output_path(self, item: FolderItem) -> Optional[Path]:
        if self.check_overwrite_source.isChecked():
            if item.source_json_path is not None:
                return item.source_json_path
            return item.image_path.with_suffix(".json")
        out_dir_text = self.edit_output_dir.text().strip()
        if not out_dir_text:
            return None
        return Path(out_dir_text) / f"{item.stem}.json"

    def _load_annotation_for_item(self, item: FolderItem) -> dict:
        output_path = self._build_output_path(item)
        if output_path is not None and output_path.exists():
            data = load_json(output_path)
            annotation = make_empty_annotation(str(item.image_path), 256, 256, item.stem)
            annotation.update(data if isinstance(data, dict) else {})
            annotation["points"] = normalize_points(annotation.get("points", []))
            annotation["edges"] = normalize_edges(annotation.get("edges", []))
            return annotation

        if item.source_json_path is not None and item.source_json_path.exists():
            data = load_json(item.source_json_path)
            if isinstance(data, dict) and "points" in data:
                annotation = make_empty_annotation(str(item.image_path), 256, 256, item.stem)
                annotation.update(data)
                annotation["points"] = normalize_points(annotation.get("points", []))
                annotation["edges"] = normalize_edges(annotation.get("edges", []))
                annotation["image_path"] = str(item.image_path)
                return annotation
            return convert_labelme_json_to_base(item.source_json_path, item.image_path)

        image = read_image(item.image_path)
        return make_empty_annotation(str(item.image_path), image.shape[1], image.shape[0], item.stem)

    def jump_to_index(self, index: int):
        if not self.folder_items:
            return
        index = int(np.clip(index, 0, len(self.folder_items) - 1))
        if self.current_index == index and self.current_annotation is not None:
            return
        self.save_current_annotation(silent=True)
        self.current_index = index
        self.file_list.blockSignals(True)
        self.file_list.setCurrentRow(index)
        self.file_list.blockSignals(False)
        item = self.folder_items[index]
        try:
            image = read_image(item.image_path)
            annotation = self._load_annotation_for_item(item)
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", str(exc))
            return

        self.current_annotation = annotation
        self.current_output_path = self._build_output_path(item)
        self.canvas.set_image(image, image_path=str(item.image_path))
        self.canvas.set_points(annotation.get("points", []))
        self.canvas.set_edges(annotation.get("edges", []))
        self.lbl_current_name.setText(item.image_path.name)
        self.lbl_index.setText(f"{index + 1} / {len(self.folder_items)}")
        self._apply_mode_status_style(self.canvas.mode)

    def reload_current_item(self):
        if self.current_index < 0 or not self.folder_items:
            return
        current = self.current_index
        self.current_annotation = None
        self.current_output_path = None
        self.jump_to_index(current)

    def _on_point_selection_changed(self, point):
        if point is None:
            self.lbl_selected.setText("-")
            return
        self.lbl_selected.setText(f"{point['id']} ({point['x']:.1f}, {point['y']:.1f})")

    def _on_pending_changed(self, point_id):
        self.lbl_pending.setText("-" if point_id is None else str(point_id))

    def _build_current_annotation_payload(self) -> Optional[dict]:
        if self.current_index < 0 or self.current_index >= len(self.folder_items):
            return None
        item = self.folder_items[self.current_index]
        image = self.canvas.image_bgr
        if image is None:
            return None
        annotation = make_empty_annotation(
            image_path=str(item.image_path),
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            sample_id=item.stem,
        )
        annotation["points"] = normalize_points(self.canvas.points)
        annotation["edges"] = normalize_edges(self.canvas.edges)
        annotation["segments"] = []
        annotation["metadata"] = {
            "source": "img_tools_point_edge_editor",
            "origin_json": str(item.source_json_path) if item.source_json_path is not None else None,
            "point_count": len(annotation["points"]),
            "edge_count": len(annotation["edges"]),
        }
        return annotation

    def save_current_annotation(self, silent: bool = True):
        if self.current_index < 0 or not self.folder_items:
            return False
        output_path = self._build_output_path(self.folder_items[self.current_index])
        if output_path is None:
            if not silent:
                QMessageBox.warning(self, "提示", "请先选择输出文件夹。")
            return False
        payload = self._build_current_annotation_payload()
        if payload is None:
            return False
        try:
            save_json(output_path, payload)
        except Exception as exc:
            if not silent:
                QMessageBox.critical(self, "保存失败", str(exc))
            return False
        self.current_annotation = payload
        self.current_output_path = output_path
        if not silent:
            self._set_status_message(f"已保存: {output_path}")
        return True

    def _auto_save_current(self):
        if self.check_overwrite_source.isChecked() or self.edit_output_dir.text().strip():
            self.save_current_annotation(silent=True)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_A:
            self.jump_to_index(self.current_index - 1)
            event.accept()
            return
        if event.key() == Qt.Key_D:
            self.jump_to_index(self.current_index + 1)
            event.accept()
            return
        if event.key() == Qt.Key_S:
            self.save_current_annotation(silent=False)
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
