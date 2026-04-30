"""Recent functions widget for quick access to recently used features."""
import logging
import os
import json

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QListWidget,
                                QListWidgetItem, QHBoxLayout, QPushButton)
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QApplication

from core.function_registry import get_function_def

logger = logging.getLogger(__name__)


def _get_config_path() -> str:
    app = QApplication.instance()
    if app:
        from PySide6.QtCore import QStandardPaths
        data_dir = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
    else:
        data_dir = os.path.join(os.path.expanduser("~"), ".config", "img_tools")
    return os.path.join(data_dir, "img_tools_config.json")


class RecentFunctionsWidget(QWidget):
    functionClicked = Signal(str, str)  # key, name

    def __init__(self, parent=None):
        super().__init__(parent)
        self._max_items = 10
        self._setup_ui()
        self._load()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        header = QHBoxLayout()
        title = QLabel("最近使用")
        f = title.font()
        f.setBold(True)
        title.setFont(f)
        header.addWidget(title)

        self.btn_clear = QPushButton("清空")
        self.btn_clear.setFlat(True)
        self.btn_clear.setMaximumWidth(40)
        self.btn_clear.clicked.connect(self.clear_recent)
        header.addWidget(self.btn_clear)
        layout.addLayout(header)

        self.list_widget = QListWidget()
        self.list_widget.setMaximumHeight(120)
        self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.list_widget)

    def _on_item_clicked(self, item):
        data = item.data(Qt.UserRole)
        if data:
            self.functionClicked.emit(data["key"], data["name"])

    def add_function(self, key: str, name: str):
        """Add a function to the recent list."""
        # Remove if already exists
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            data = item.data(Qt.UserRole)
            if data and data["key"] == key:
                self.list_widget.takeItem(i)
                break

        # Insert at top
        item = QListWidgetItem(name)
        item.setData(Qt.UserRole, {"key": key, "name": name})
        item.setToolTip(name)
        self.list_widget.insertItem(0, item)

        # Trim
        while self.list_widget.count() > self._max_items:
            self.list_widget.takeItem(self.list_widget.count() - 1)

        self._save()

    def clear_recent(self):
        self.list_widget.clear()
        self._save()

    def _load(self):
        try:
            with open(_get_config_path(), "r", encoding="utf-8") as f:
                config = json.load(f)
            recent = config.get("recent_functions", [])
            for entry in reversed(recent):
                key = entry.get("key", "")
                name = entry.get("name", "")
                if key and name:
                    item = QListWidgetItem(name)
                    item.setData(Qt.UserRole, {"key": key, "name": name})
                    item.setToolTip(name)
                    self.list_widget.addItem(item)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to load recent functions: %s", e)

    def _save(self):
        recent = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            data = item.data(Qt.UserRole)
            if data:
                recent.append(data)
        config_path = _get_config_path()
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            config = {}
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.debug("Config file not readable (will create new): %s", e)
            config["recent_functions"] = recent
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except (OSError, TypeError) as e:
            logger.warning("Failed to save recent functions: %s", e)
