"""Input panel: file/folder selection with file list."""
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                                QListWidget, QListWidgetItem, QFileDialog, QLabel, QCheckBox)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QPixmap, QIcon
import os
from utils.helpers import get_image_files
from core.image_io import read_image
from gui.preview_widget import cv2_to_qpixmap


class InputPanel(QWidget):
    """Panel for selecting input images or folders."""
    filesChanged = Signal(list)
    previewRequested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._files = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Title
        title = QLabel("输入源")
        f = title.font(); f.setBold(True); title.setFont(f)
        layout.addWidget(title)

        # Buttons row
        btn_row = QHBoxLayout()
        self.btn_file = QPushButton("选择文件")
        self.btn_file.clicked.connect(self._add_files)
        self.btn_dir = QPushButton("选择文件夹")
        self.btn_dir.clicked.connect(self._add_dir)
        self.btn_clear = QPushButton("清空")
        self.btn_clear.clicked.connect(self._clear)
        btn_row.addWidget(self.btn_file)
        btn_row.addWidget(self.btn_dir)
        btn_row.addWidget(self.btn_clear)
        layout.addLayout(btn_row)

        # Recursive checkbox
        self.chk_recursive = QCheckBox("包含子文件夹")
        self.chk_recursive.setChecked(True)
        layout.addWidget(self.chk_recursive)

        # File list
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SingleSelection)
        self.file_list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.file_list)

        # Count label
        self.lbl_count = QLabel("共 0 张图片")
        layout.addWidget(self.lbl_count)

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp);;All Files (*)")
        if paths:
            self._add_paths(paths)

    def _add_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if path:
            self._add_paths([path])

    def _add_paths(self, paths):
        new_files = []
        for p in paths:
            if os.path.isfile(p):
                new_files.append(p)
            elif os.path.isdir(p):
                new_files.extend(get_image_files(p))

        existing = set(self._files)
        for f in new_files:
            if f not in existing:
                self._files.append(f)
                existing.add(f)
                item = QListWidgetItem(os.path.basename(f))
                item.setToolTip(f)
                self.file_list.addItem(item)

        self._update_count()
        if self._files:
            self.filesChanged.emit(self._files)

    def _clear(self):
        self._files.clear()
        self.file_list.clear()
        self._update_count()
        self.filesChanged.emit([])

    def _on_item_clicked(self, item):
        idx = self.file_list.row(item)
        if 0 <= idx < len(self._files):
            self.previewRequested.emit(self._files[idx])

    def _update_count(self):
        self.lbl_count.setText(f"共 {len(self._files)} 张图片")

    def get_files(self):
        return self._files

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        paths = []
        for url in event.mimeData().urls():
            p = url.toLocalFile()
            if os.path.exists(p):
                paths.append(p)
        if paths:
            self._add_paths(paths)
