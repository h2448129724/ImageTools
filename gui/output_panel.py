"""Output settings panel."""
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                                QLabel, QLineEdit, QFileDialog, QComboBox)
from PySide6.QtCore import Signal
import os
import sys
import subprocess


class OutputPanel(QWidget):
    """Panel for configuring output directory, format, and naming."""
    outputChanged = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        title = QLabel("输出设置")
        f = title.font()
        f.setBold(True)
        title.setFont(f)
        layout.addWidget(title)

        # Output directory
        dir_row = QHBoxLayout()
        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText("选择输出目录...")
        btn_browse = QPushButton("...")
        btn_browse.setMaximumWidth(30)
        btn_browse.clicked.connect(self._browse_output)
        dir_row.addWidget(self.output_dir)
        dir_row.addWidget(btn_browse)
        layout.addLayout(dir_row)

        # Open output button
        self.btn_open = QPushButton("打开输出文件夹")
        self.btn_open.clicked.connect(self._open_output)
        layout.addWidget(self.btn_open)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.output_dir.setText(path)

    def _open_output(self):
        path = self.output_dir.text()
        if path and os.path.isdir(path):
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])

    def get_output_dir(self):
        return self.output_dir.text()
