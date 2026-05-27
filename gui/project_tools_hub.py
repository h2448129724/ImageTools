"""Project-specific tool hub dialogs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class ProjectToolEntry:
    key: str
    title: str
    description: str
    launch: Callable[[], None]


class ProjectToolsHubDialog(QDialog):
    """Lightweight navigation hub for project-specific tools."""

    def __init__(self, project_name: str, tools: Sequence[ProjectToolEntry], parent=None):
        super().__init__(parent)
        self._project_name = project_name
        self._tools = list(tools)
        self._setup_ui()
        self._populate_tools()

    def _setup_ui(self) -> None:
        self.setWindowTitle(f"{self._project_name} 工具中心")
        self.resize(860, 520)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        header = QLabel(f"{self._project_name} 工具中心")
        font = header.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 2)
        header.setFont(font)
        root.addWidget(header)

        intro = QLabel("左侧选择具体功能，右侧查看说明并打开对应工具。这个入口也可以继续扩展到其他项目。")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #666;")
        root.addWidget(intro)

        content = QHBoxLayout()
        content.setSpacing(12)
        root.addLayout(content, 1)

        self.tool_list = QListWidget()
        self.tool_list.setMinimumWidth(240)
        self.tool_list.currentRowChanged.connect(self._on_tool_changed)
        content.addWidget(self.tool_list, 0)

        detail_panel = QWidget()
        detail_layout = QVBoxLayout(detail_panel)
        detail_layout.setContentsMargins(12, 12, 12, 12)
        detail_layout.setSpacing(10)
        detail_panel.setStyleSheet("background: rgba(0, 0, 0, 0.03); border-radius: 8px;")
        content.addWidget(detail_panel, 1)

        self.title_label = QLabel("")
        title_font = self.title_label.font()
        title_font.setBold(True)
        title_font.setPointSize(title_font.pointSize() + 1)
        self.title_label.setFont(title_font)
        detail_layout.addWidget(self.title_label)

        self.desc_label = QLabel("")
        self.desc_label.setWordWrap(True)
        self.desc_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        detail_layout.addWidget(self.desc_label, 1)

        detail_layout.addStretch()

        button_row = QHBoxLayout()
        button_row.addStretch()
        self.open_btn = QPushButton("打开工具")
        self.open_btn.clicked.connect(self._open_current_tool)
        button_row.addWidget(self.open_btn)
        detail_layout.addLayout(button_row)

    def _populate_tools(self) -> None:
        self.tool_list.clear()
        for tool in self._tools:
            item = QListWidgetItem(tool.title)
            item.setData(Qt.UserRole, tool.key)
            self.tool_list.addItem(item)
        if self._tools:
            self.tool_list.setCurrentRow(0)
        else:
            self.title_label.setText("暂无工具")
            self.desc_label.setText("这个项目目前还没有注册可用工具。")
            self.open_btn.setEnabled(False)

    def _on_tool_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._tools):
            self.title_label.setText("")
            self.desc_label.setText("")
            self.open_btn.setEnabled(False)
            return
        tool = self._tools[row]
        self.title_label.setText(tool.title)
        self.desc_label.setText(tool.description)
        self.open_btn.setEnabled(True)

    def _open_current_tool(self) -> None:
        row = self.tool_list.currentRow()
        if row < 0 or row >= len(self._tools):
            return
        self._tools[row].launch()
