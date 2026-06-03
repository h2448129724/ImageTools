"""CAB-F工具 — 主窗口（工具导航中心）。"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton,
    QStackedWidget, QStatusBar, QToolBar, QVBoxLayout, QWidget,
)

from project_modules.cabf_pipeline.config_model import load_config
from project_modules.cabf_pipeline.flow import CONFIG_PATH

from gui.project_tools_registry import get_registered_projects
from gui.tools.stitch_workflow import StitchWorkflowPage
from gui.tools.keyword_split import KeywordSplitPage
from gui.tools.batch_crop import BatchCropPage
from gui.tools.auto_tile_crop import AutoTileCropPage
from gui.tools.roi_config_editor import RoiConfigEditorPage
from gui.theme import APP_STYLESHEET, TOKENS
STYLESHEET = APP_STYLESHEET


class MainWindow(QMainWindow):
    def __init__(self, config_path: Path | None = None):
        super().__init__()
        self.config_path = Path(config_path or CONFIG_PATH)
        self.config_data = load_config(self.config_path)
        self._project_tool_registry = get_registered_projects()
        self._sidebar_visible = True
        self._sidebar_expanded_width = 204
        self._sidebar_collapsed_width = 52

        self.setWindowTitle("CAB-F工具")
        self.resize(1320, 820)

        self._tool_pages = [
            StitchWorkflowPage(self),
            KeywordSplitPage(self),
            BatchCropPage(self),
            AutoTileCropPage(self),
            RoiConfigEditorPage(self),
        ]

        self._build_ui()
        self._build_menu()
        self._tool_list.setCurrentRow(0)

    # ------------------------------------------------------------------- ui
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        sidebar = self._build_sidebar()
        self._sidebar = sidebar
        sidebar.setFixedWidth(self._sidebar_expanded_width)
        root.addWidget(sidebar)

        content_shell = QWidget()
        content_shell.setObjectName("contentShell")
        content_layout = QVBoxLayout(content_shell)
        content_layout.setContentsMargins(12, 12, 12, 12)
        content_layout.setSpacing(6)

        topbar = QHBoxLayout()
        self._btn_toggle_sidebar = QPushButton("收起工具栏")
        self._btn_toggle_sidebar.setMinimumHeight(34)
        self._btn_toggle_sidebar.clicked.connect(lambda checked=False: self._toggle_sidebar())
        topbar.addWidget(self._btn_toggle_sidebar)
        topbar.addStretch(1)
        content_layout.addLayout(topbar)

        page_surface = QFrame()
        page_surface.setObjectName("surfaceCard")
        page_surface_layout = QVBoxLayout(page_surface)
        page_surface_layout.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedWidget()
        for page in self._tool_pages:
            self.stack.addWidget(page)
        page_surface_layout.addWidget(self.stack)
        content_layout.addWidget(page_surface, 1)
        root.addWidget(content_shell, 1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.lbl_status = QLabel("就绪")
        self.lbl_status.setStyleSheet("color:#64748b;")
        self.status_bar.addWidget(self.lbl_status)

    # -------------------------------------------------------------- sidebar
    def _build_sidebar(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("sidebar")
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(12, 14, 12, 10)
        lay.setSpacing(6)

        # app title row with collapse button
        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        self._sidebar_title = QLabel("CAB-F")
        self._sidebar_title.setStyleSheet(
            f"color:{TOKENS['text_main']}; font-size:22px; font-weight:700; letter-spacing:0.3px;"
        )
        title_row.addWidget(self._sidebar_title)
        title_row.addStretch()

        self._btn_collapse = QPushButton("◁")
        self._btn_collapse.setFixedSize(28, 28)
        self._btn_collapse.setToolTip("收起侧边栏")
        self._btn_collapse.setStyleSheet(
            f"QPushButton{{border:none;background:transparent;color:{TOKENS['text_secondary']};font-size:16px;}}"
            f"QPushButton:hover{{background:{TOKENS['sidebar_hover']};border-radius:6px;}}"
        )
        self._btn_collapse.clicked.connect(lambda checked=False: self._toggle_sidebar())
        title_row.addWidget(self._btn_collapse)
        lay.addLayout(title_row)

        self._sidebar_subtitle = QLabel("工具集")
        self._sidebar_subtitle.setStyleSheet(
            f"color:{TOKENS['text_secondary']}; font-size:11px; margin-bottom:6px; letter-spacing:0;"
        )
        lay.addWidget(self._sidebar_subtitle)

        # divider
        div = QFrame()
        div.setFixedHeight(1)
        div.setStyleSheet(f"background: {TOKENS['border_main']}; margin: 0 0 6px 0;")
        lay.addWidget(div)

        hdr = QLabel("工具")
        hdr.setObjectName("sidebarHeader")
        lay.addWidget(hdr)

        self._tool_list = QListWidget()
        self._tool_list.setObjectName("toolList")
        self._tool_list.currentRowChanged.connect(self._on_tool_selected)
        for page in self._tool_pages:
            nav_title = getattr(page, "tool_nav_title", page.tool_title)
            icon = getattr(page, "tool_icon", "")
            item = QListWidgetItem(nav_title)
            item.setData(Qt.UserRole, page.tool_key)
            item.setData(Qt.UserRole + 1, icon)
            self._tool_list.addItem(item)
        lay.addWidget(self._tool_list)

        self._sidebar_footnote = QLabel("")
        self._sidebar_footnote.setObjectName("sidebarFootnote")
        self._sidebar_footnote.setWordWrap(True)
        lay.addStretch(1)
        return frame

    def _toggle_sidebar(self, expanded: bool | None = None):
        if expanded is None:
            expanded = not self._sidebar_visible
        self._sidebar_visible = expanded
        if expanded:
            self._sidebar.setFixedWidth(self._sidebar_expanded_width)
            self._sidebar_title.setText("CAB-F")
            self._sidebar_subtitle.show()
            self._btn_collapse.setText("◁")
            self._btn_collapse.setToolTip("收起侧边栏")
            self._btn_toggle_sidebar.setText("收起工具栏")
            for i in range(self._tool_list.count()):
                item = self._tool_list.item(i)
                item.setText(getattr(self._tool_pages[i], "tool_nav_title", self._tool_pages[i].tool_title))
            self._tool_list.setStyleSheet("")
        else:
            self._sidebar.setFixedWidth(self._sidebar_collapsed_width)
            self._sidebar_title.setText("C")
            self._sidebar_subtitle.hide()
            self._btn_collapse.setText("▷")
            self._btn_collapse.setToolTip("展开侧边栏")
            self._btn_toggle_sidebar.setText("展开工具栏")
            for i in range(self._tool_list.count()):
                item = self._tool_list.item(i)
                icon = item.data(Qt.UserRole + 1) or "•"
                item.setText(icon)
            self._tool_list.setStyleSheet(
                "#toolList::item{padding:10px 0;text-align:center;font-size:18px;}"
            )

    def _on_tool_selected(self, idx):
        if 0 <= idx < len(self._tool_pages):
            old = self.stack.currentIndex()
            if old != idx and 0 <= old < len(self._tool_pages):
                self._tool_pages[old].on_deactivated()
            self._tool_pages[idx].on_activated()
            self.stack.setCurrentIndex(idx)
            self._refresh_overview(idx)

    def _refresh_overview(self, idx: int) -> None:
        if not 0 <= idx < len(self._tool_pages):
            return
        page = self._tool_pages[idx]
        self.setWindowTitle(f"CAB-F工具 - {page.tool_title}")

    # ----------------------------------------------------------------- menu
    def _build_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("文件(&F)")
        open_cfg_act = file_menu.addAction("打开配置...")
        open_cfg_act.triggered.connect(self._open_config)
        file_menu.addSeparator()
        quit_act = file_menu.addAction("退出\tCtrl+Q")
        quit_act.triggered.connect(self.close)

        tools_menu = menubar.addMenu("工具(&T)")
        act_editor = tools_menu.addAction("点边一体标注器")
        act_editor.triggered.connect(self._show_stitch_graph_editor)
        act_filter = tools_menu.addAction("缝纫点数据筛选")
        act_filter.triggered.connect(self._show_stitch_point_filter)
        act_dataset = tools_menu.addAction("数据集校验与导出")
        act_dataset.triggered.connect(self._show_cabf_dataset_tool)

        if self._project_tool_registry:
            tools_menu.addSeparator()
            for project in self._project_tool_registry:
                if project.key == "cabf":
                    continue
                from gui.project_tools_hub import ProjectToolsHubDialog, ProjectToolEntry
                project_act = tools_menu.addAction(project.menu_title)
                project_act.triggered.connect(
                    lambda checked=False, pk=project.key: self._show_project_tools_hub(pk)
                )

        help_menu = menubar.addMenu("帮助(&H)")
        about_act = help_menu.addAction("关于")
        about_act.triggered.connect(self._show_about)

        # Global keyboard shortcuts
        for i, page in enumerate(self._tool_pages):
            shortcut = QKeySequence(f"Ctrl+{i + 1}")
            act = QAction(f"切换到 {page.tool_nav_title}", self)
            act.setShortcut(shortcut)
            act.triggered.connect(lambda checked=False, idx=i: self._tool_list.setCurrentRow(idx))
            self.addAction(act)

        save_act = QAction("保存配置", self)
        save_act.setShortcut(QKeySequence.Save)
        save_act.triggered.connect(self._save_current_tool)
        self.addAction(save_act)

    # ---------------------------------------------------------- shared helpers
    def show_status(self, msg: str):
        self.lbl_status.setText(msg)

    def refresh_tool_overview(self) -> None:
        if not hasattr(self, "stack"):
            return
        idx = self.stack.currentIndex()
        if idx >= 0:
            self._refresh_overview(idx)

    def _save_current_tool(self):
        idx = self.stack.currentIndex()
        if 0 <= idx < len(self._tool_pages):
            page = self._tool_pages[idx]
            if hasattr(page, "_sync_form"):
                page._sync_form()
                self.show_status("配置已保存")

    def _open_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开配置", str(self.config_path.parent), "JSON (*.json)"
        )
        if path:
            self.config_path = Path(path)
            self.config_data = load_config(self.config_path)

    # ----------------------------------------------------------- tool dialogs
    def _show_stitch_graph_editor(self):
        from gui.stitch_graph_editor import StitchGraphEditorDialog
        StitchGraphEditorDialog(self).exec()

    def _show_stitch_point_editor(self):
        self._show_stitch_graph_editor()

    def _show_stitch_point_filter(self):
        from gui.stitch_point_filter import StitchPointFilterDialog
        StitchPointFilterDialog(self).exec()

    def _show_cabf_dataset_tool(self):
        from gui.cabf_dataset_tool import CabfDatasetToolDialog
        CabfDatasetToolDialog(self).exec()

    def _show_project_tools_hub(self, project_key: str):
        project = next((item for item in self._project_tool_registry if item.key == project_key), None)
        if project is None:
            QMessageBox.warning(self, "提示", f"未找到项目工具配置: {project_key}")
            return
        from gui.project_tools_hub import ProjectToolsHubDialog, ProjectToolEntry
        entries = []
        for tool in project.tools:
            launch = getattr(self, tool.launcher_name, None)
            if callable(launch):
                entries.append(ProjectToolEntry(key=tool.key, title=tool.title,
                                                description=tool.description, launch=launch))
        dlg = ProjectToolsHubDialog(project.display_name, entries, self)
        dlg.exec()

    def _show_about(self):
        QMessageBox.about(self, "关于", "CAB-F工具 v3.0\n\n缝纫点/边标注、数据处理与流程编排工具集")


def launch_standalone(config_path: Path | None = None) -> int:
    app = QApplication.instance()
    owns = app is None
    if app is None:
        app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    window = MainWindow(config_path=config_path)
    window.show()
    if owns:
        return int(app.exec())
    window.raise_()
    window.activateWindow()
    return 0
