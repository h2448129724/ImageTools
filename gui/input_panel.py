"""Input panel: file/folder selection with file list."""
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                                QListWidget, QListWidgetItem, QFileDialog, QLabel,
                                QCheckBox, QComboBox, QLineEdit)
from PySide6.QtCore import Signal, Qt
import fnmatch
import os
from utils.helpers import get_image_files
from gui.preview_widget import cv2_to_qpixmap

_ALL_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


class InputPanel(QWidget):
    """Panel for selecting input images or folders."""
    filesChanged = Signal(list)
    previewRequested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._all_files = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Title
        title = QLabel("输入源")
        f = title.font()
        f.setBold(True)
        title.setFont(f)
        layout.addWidget(title)

        # Buttons row
        btn_row = QHBoxLayout()
        self.btn_file = QPushButton("选择文件")
        self.btn_file.clicked.connect(self.add_files)
        self.btn_dir = QPushButton("选择文件夹")
        self.btn_dir.clicked.connect(self.add_dir)
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

        # File filter
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("文件筛选:"))
        self.txt_filter = QLineEdit()
        self.txt_filter.setPlaceholderText("如: *_bottom.jpg 或 *.png，留空显示全部")
        self.txt_filter.textChanged.connect(self._on_filter_changed)
        filter_row.addWidget(self.txt_filter)
        layout.addLayout(filter_row)

        # File list
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SingleSelection)
        self.file_list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.file_list)

        # Count label
        self.lbl_count = QLabel("共 0 张图片")
        layout.addWidget(self.lbl_count)

    def add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp);;All Files (*)")
        if paths:
            self._add_paths(paths)

    def add_dir(self):
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

        existing = set(self._all_files)
        for f in new_files:
            if f not in existing:
                self._all_files.append(f)
                existing.add(f)

        self._refresh_list()

    def _clear(self):
        self._all_files.clear()
        self.file_list.clear()
        self._update_count()
        self.filesChanged.emit([])

    def _on_item_clicked(self, item):
        idx = self.file_list.row(item)
        filtered = self._get_filtered_files()
        if 0 <= idx < len(filtered):
            self.previewRequested.emit(filtered[idx])

    def _update_count(self):
        filtered = self._get_filtered_files()
        total = len(self._all_files)
        if total == len(filtered):
            self.lbl_count.setText(f"共 {total} 张图片")
        else:
            self.lbl_count.setText(f"共 {len(filtered)} / {total} 张图片")

    def _get_filtered_files(self):
        pattern = self.txt_filter.text().strip()
        if not pattern:
            return list(self._all_files)
        return [f for f in self._all_files if fnmatch.fnmatch(os.path.basename(f), pattern)]

    def _refresh_list(self):
        filtered = self._get_filtered_files()
        self.file_list.clear()
        for f in filtered:
            item = QListWidgetItem(os.path.basename(f))
            item.setToolTip(f)
            self.file_list.addItem(item)
        self._update_count()
        self.filesChanged.emit(filtered)

    def _on_filter_changed(self):
        self._refresh_list()

    def get_files(self):
        return self._get_filtered_files()

    @property
    def all_files(self):
        return self._all_files

    def get_file(self, index: int) -> str | None:
        if 0 <= index < len(self._all_files):
            return self._all_files[index]
        return None

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
