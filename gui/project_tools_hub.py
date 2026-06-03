"""Project-specific tool hub dialogs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from gui.tools.base import make_page_header


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
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        hero = make_page_header(
            f"{self._project_name} 工具中心",
            "集中放置当前项目的扩展工具入口，保持和主工具一致的轻量风格。",
        )
        self.meta_label = QLabel("")
        self.meta_label.setStyleSheet("color:#334155;font-size:12px;")
        hero.layout().addWidget(self.meta_label)
        root.addWidget(hero)

        content = QHBoxLayout()
        content.setSpacing(10)
        root.addLayout(content, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        grid_widget = QWidget()
        self._grid_layout = QGridLayout(grid_widget)
        self._grid_layout.setSpacing(10)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(grid_widget)
        content.addWidget(scroll, 1)

    def _populate_tools(self) -> None:
        cols = 2
        for i, tool in enumerate(self._tools):
            card = QFrame()
            card.setStyleSheet(
                "QFrame{background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;}"
                "QFrame:hover{border-color:#2563EB;background:#f8fbff;}"
            )
            card.setCursor(Qt.PointingHandCursor)
            lay = QVBoxLayout(card)
            lay.setContentsMargins(14, 12, 14, 12)
            lay.setSpacing(4)

            title = QLabel(tool.title)
            title.setStyleSheet("color:#0f172a;font-size:16px;font-weight:700;")
            lay.addWidget(title)

            desc = QLabel(tool.description)
            desc.setWordWrap(True)
            desc.setStyleSheet("color:#64748b;font-size:13px;line-height:1.5;")
            lay.addWidget(desc, 1)

            btn = QPushButton("打开")
            btn.setFixedWidth(72)
            btn.clicked.connect(lambda checked=False, t=tool: t.launch())
            lay.addWidget(btn, 0, Qt.AlignRight)

            row, col = divmod(i, cols)
            self._grid_layout.addWidget(card, row, col)

        if self._tools:
            self.meta_label.setText(f"已注册 {len(self._tools)} 个可用工具。")
        else:
            self.meta_label.setText("当前项目暂无已注册工具。")
