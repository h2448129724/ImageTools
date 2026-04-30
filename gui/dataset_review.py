"""Dataset review dialog for browsing and curating image+annotation pairs."""
import os
import json
import send2trash
from pathlib import Path
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QListWidget,
                                QListWidgetItem, QPushButton, QLabel, QComboBox,
                                QSplitter, QMessageBox, QWidget)
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from gui.preview_widget import ZoomableLabel, cv2_to_qpixmap
from core.image_io import read_image
from core.annotation import (draw_labelme_shapes, draw_yolo_boxes,
                              parse_labelme, parse_yolo_file)

_ANN_EXTS = (".json", ".txt")
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


class DatasetReviewDialog(QDialog):
    def __init__(self, dataset_dir, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"数据集审查 — {os.path.basename(dataset_dir)}")
        self.resize(1400, 800)
        self.setMinimumSize(900, 500)
        self._dataset_dir = dataset_dir
        self._items = []        # list of dicts: {img, labels, base, ann_count, label_summary}
        self._deleted = 0
        self._current_idx = -1
        self._setup_ui()
        self._load_dataset()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # ---- Toolbar ----
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        self.btn_prev = QPushButton("◄ 上一张")
        self.btn_prev.setFixedHeight(32)
        self.btn_prev.clicked.connect(self._go_prev)

        self.btn_next = QPushButton("下一张 ►")
        self.btn_next.setFixedHeight(32)
        self.btn_next.clicked.connect(self._go_next)

        self.btn_delete = QPushButton("🗑 删除 (Del)")
        self.btn_delete.setFixedHeight(32)
        self.btn_delete.setStyleSheet(
            "QPushButton { color: white; background-color: #c0392b; font-weight: bold; "
            "border-radius: 4px; padding: 0 12px; }"
            "QPushButton:hover { background-color: #e74c3c; }")
        self.btn_delete.clicked.connect(self._delete_current)

        toolbar.addWidget(self.btn_prev)
        toolbar.addWidget(self.btn_next)
        toolbar.addWidget(self.btn_delete)

        toolbar.addWidget(QLabel("|"))
        toolbar.addWidget(QLabel("筛选:"))
        self.cmb_filter = QComboBox()
        self.cmb_filter.addItems(["全部", "有标注", "无标注"])
        self.cmb_filter.currentIndexChanged.connect(self._apply_filter)
        toolbar.addWidget(self.cmb_filter)

        toolbar.addStretch()

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("font-weight: bold;")
        toolbar.addWidget(self.lbl_status)
        layout.addLayout(toolbar)

        # ---- Splitter ----
        splitter = QSplitter(Qt.Horizontal)

        left = QVBoxLayout()
        self.file_list = QListWidget()
        self.file_list.setFocusPolicy(Qt.NoFocus)
        self.file_list.currentRowChanged.connect(self._on_list_selection)
        left.addWidget(self.file_list)
        self.lbl_ann_info = QLabel("")
        self.lbl_ann_info.setWordWrap(True)
        self.lbl_ann_info.setStyleSheet("font-size: 11px; color: #555;")
        self.lbl_ann_info.setMaximumHeight(60)
        left.addWidget(self.lbl_ann_info)

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setMaximumWidth(400)
        splitter.addWidget(left_widget)

        self.preview = ZoomableLabel()
        self.preview.setFocusPolicy(Qt.NoFocus)
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

    # ---- Keyboard ----
    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key_A, Qt.Key_Left):
            self._go_prev()
        elif key in (Qt.Key_D, Qt.Key_Right):
            self._go_next()
        elif key == Qt.Key_Delete:
            self._delete_current()
        else:
            super().keyPressEvent(event)

    # ---- Loading (one-time scan) ----
    def _load_dataset(self):
        self._items = []
        seen = set()
        for f in sorted(os.listdir(self._dataset_dir)):
            base, ext = os.path.splitext(f)
            if ext.lower() not in _IMG_EXTS or base in seen:
                continue
            seen.add(base)
            img_path = os.path.join(self._dataset_dir, f)
            label_paths = []
            for lext in _ANN_EXTS:
                lp = os.path.join(self._dataset_dir, base + lext)
                if os.path.exists(lp):
                    label_paths.append(lp)

            # Cache annotation count and label summary at load time
            ann_count, label_summary = 0, {}
            for lp in label_paths:
                if lp.endswith(".json"):
                    data = parse_labelme(lp)
                    if data:
                        for s in data.get("shapes", []):
                            lbl = s.get("label", "?")
                            label_summary[lbl] = label_summary.get(lbl, 0) + 1
                        ann_count += len(data.get("shapes", []))
                elif lp.endswith(".txt"):
                    boxes = parse_yolo_file(lp)
                    for b in boxes:
                        c = str(b["cls"])
                        label_summary[c] = label_summary.get(c, 0) + 1
                    ann_count += len(boxes)

            self._items.append({
                "img": img_path,
                "labels": label_paths,
                "base": base,
                "ann_count": ann_count,
                "label_summary": label_summary,
            })
        self._apply_filter()

    # ---- Filter ----
    def _apply_filter(self):
        mode = self.cmb_filter.currentIndex()
        self.file_list.clear()
        self._filtered_indices = []
        for i, item in enumerate(self._items):
            if item["img"] is None:
                continue  # skip deleted
            ann_count = item["ann_count"]
            if mode == 1 and ann_count == 0:
                continue
            if mode == 2 and ann_count > 0:
                continue
            self._filtered_indices.append(i)
            tag = f" ({ann_count})" if ann_count > 0 else " (空)"
            list_item = QListWidgetItem(item["base"] + tag)
            if ann_count == 0:
                list_item.setForeground(Qt.gray)
            self.file_list.addItem(list_item)
        self._update_status()
        if self._filtered_indices:
            self.file_list.setCurrentRow(0)

    # ---- Navigation ----
    def _on_list_selection(self, row):
        if row < 0 or row >= len(self._filtered_indices):
            self._current_idx = -1
            return
        self._current_idx = self._filtered_indices[row]
        self._show_current()

    def _show_current(self):
        if self._current_idx < 0 or self._current_idx >= len(self._items):
            return
        item = self._items[self._current_idx]
        img_path = item["img"]
        if img_path is None:
            return
        img = read_image(img_path)
        if img is None:
            return
        # Draw annotations
        for lp in item["labels"]:
            if lp.endswith(".json"):
                img = draw_labelme_shapes(img, lp)
                break
            elif lp.endswith(".txt"):
                img = draw_yolo_boxes(img, lp)
                break
        self.preview.set_pixmap(cv2_to_qpixmap(img))

        # Show cached label info (no file re-read)
        ann_count = item["ann_count"]
        summary = item["label_summary"]
        if ann_count > 0:
            parts = [f"{k}: {v}" for k, v in sorted(summary.items())]
            lbl_files = [os.path.basename(p) for p in item["labels"]]
            self.lbl_ann_info.setText(
                f"标注 {ann_count} 个 | " + " | ".join(parts) +
                f"\n标签: {', '.join(lbl_files)}")
        elif item["labels"]:
            self.lbl_ann_info.setText(f"标签为空 | 文件: {os.path.basename(item['labels'][0])}")
        else:
            self.lbl_ann_info.setText("无标签文件")

        self._update_status()

    def _update_status(self):
        total = sum(1 for it in self._items if it["img"] is not None)
        filtered = len(self._filtered_indices)
        with_ann = sum(1 for it in self._items if it["img"] is not None and it["ann_count"] > 0)
        pos = self.file_list.currentRow() + 1 if self.file_list.currentRow() >= 0 else 0
        self.lbl_status.setText(
            f"{pos}/{filtered}  |  总计 {total}, 有标注 {with_ann}, "
            f"无标注 {total - with_ann}, 已删 {self._deleted}")

    def _go_prev(self):
        row = self.file_list.currentRow()
        if row > 0:
            self.file_list.setCurrentRow(row - 1)

    def _go_next(self):
        row = self.file_list.currentRow()
        if row < self.file_list.count() - 1:
            self.file_list.setCurrentRow(row + 1)

    @staticmethod
    def _safe_trash(path):
        p = str(Path(path).resolve())
        if p.startswith('\\\\?\\'):
            p = p[4:]
        send2trash.send2trash(p)

    def _delete_current(self):
        if self._current_idx < 0 or self._current_idx >= len(self._items):
            return
        item = self._items[self._current_idx]
        if item["img"] is None:
            return
        try:
            self._safe_trash(item["img"])
            for lp in item["labels"]:
                if os.path.exists(lp):
                    self._safe_trash(lp)
        except Exception as e:
            QMessageBox.warning(self, "删除失败", str(e))
            return

        self._deleted += 1
        item["img"] = None
        item["labels"] = []
        row = self.file_list.currentRow()
        self._filtered_indices.pop(row)
        self.file_list.takeItem(row)
        if row < self.file_list.count():
            self.file_list.setCurrentRow(row)
        elif self.file_list.count() > 0:
            self.file_list.setCurrentRow(self.file_list.count() - 1)
        self._update_status()
