"""Unified QSS styles for the application."""

_LIGHT = {
    "window_bg": "#f5f5f5",
    "text": "#333333",
    "text_secondary": "#e0e0e0",
    "btn_primary": "#0078d7",
    "btn_hover": "#005a9e",
    "btn_pressed": "#004578",
    "btn_disabled_bg": "#cccccc",
    "btn_disabled_fg": "#888888",
    "btn_run": "#27ae60",
    "btn_run_hover": "#219a52",
    "btn_run_pressed": "#1e8449",
    "input_border": "#cccccc",
    "input_focus_border": "#0078d7",
    "input_bg": "white",
    "input_fg": "#333333",
    "combo_popup_border": "#cccccc",
    "combo_popup_bg": "white",
    "list_border": "#dddddd",
    "list_bg": "white",
    "list_hover": "#e6f2ff",
    "edit_border": "#dddddd",
    "edit_bg": "#fafafa",
    "splitter": "#dddddd",
    "status_bg": "#e8e8e8",
    "status_border": "#cccccc",
    "status_fg": "#333333",
    "progress_border": "#cccccc",
    "progress_bg": "white",
    "progress_fg": "#333333",
    "menubar_bg": "#f0f0f0",
    "menubar_border": "#cccccc",
    "menu_bg": "white",
    "menu_border": "#cccccc",
    "tab_border": "#dddddd",
    "tab_bg": "#e8e8e8",
    "tab_selected_bg": "white",
    "group_border": "#dddddd",
}

_DARK = {
    "window_bg": "#2d2d2d",
    "text": "#e0e0e0",
    "text_secondary": "#e0e0e0",
    "btn_primary": "#0078d7",
    "btn_hover": "#005a9e",
    "btn_pressed": "#004578",
    "btn_disabled_bg": "#555555",
    "btn_disabled_fg": "#888888",
    "btn_run": "#27ae60",
    "btn_run_hover": "#219a52",
    "btn_run_pressed": "#1e8449",
    "input_border": "#555555",
    "input_focus_border": "#0078d7",
    "input_bg": "#3d3d3d",
    "input_fg": "#e0e0e0",
    "combo_popup_border": "#555555",
    "combo_popup_bg": "#3d3d3d",
    "list_border": "#444444",
    "list_bg": "#3d3d3d",
    "list_hover": "#4a4a4a",
    "edit_border": "#444444",
    "edit_bg": "#2d2d2d",
    "splitter": "#444444",
    "status_bg": "#333333",
    "status_border": "#444444",
    "status_fg": "#e0e0e0",
    "progress_border": "#555555",
    "progress_bg": "#3d3d3d",
    "progress_fg": "#e0e0e0",
    "menubar_bg": "#333333",
    "menubar_border": "#444444",
    "menu_bg": "#3d3d3d",
    "menu_border": "#555555",
    "tab_border": "#444444",
    "tab_bg": "#3d3d3d",
    "tab_selected_bg": "#4a4a4a",
    "group_border": "#444444",
}

_QSS_TEMPLATE = """
QMainWindow {{
    background-color: {window_bg};
}}
QWidget {{
    font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
    font-size: 13px;
    color: {text_secondary};
}}
QLabel {{
    color: {text};
}}
QPushButton {{
    background-color: {btn_primary};
    color: white;
    border: none;
    border-radius: 4px;
    padding: 6px 14px;
    font-size: 13px;
}}
QPushButton:hover {{
    background-color: {btn_hover};
}}
QPushButton:pressed {{
    background-color: {btn_pressed};
}}
QPushButton:disabled {{
    background-color: {btn_disabled_bg};
    color: {btn_disabled_fg};
}}
QPushButton#btnRun {{
    background-color: {btn_run};
    font-weight: bold;
    padding: 8px 16px;
}}
QPushButton#btnRun:hover {{
    background-color: {btn_run_hover};
}}
QPushButton#btnRun:pressed {{
    background-color: {btn_run_pressed};
}}
QPushButton#btnRun:disabled {{
    background-color: {btn_disabled_bg};
}}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    border: 1px solid {input_border};
    border-radius: 4px;
    padding: 4px 6px;
    background-color: {input_bg};
    color: {input_fg};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1px solid {input_focus_border};
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox QAbstractItemView {{
    border: 1px solid {combo_popup_border};
    background-color: {combo_popup_bg};
    color: {input_fg};
    selection-background-color: {btn_primary};
}}
QCheckBox {{
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
}}
QListWidget, QTreeWidget {{
    border: 1px solid {list_border};
    border-radius: 4px;
    background-color: {list_bg};
    color: {input_fg};
    outline: none;
}}
QListWidget::item:selected, QTreeWidget::item:selected {{
    background-color: {btn_primary};
    color: white;
}}
QListWidget::item:hover, QTreeWidget::item:hover {{
    background-color: {list_hover};
}}
QTextEdit {{
    border: 1px solid {edit_border};
    border-radius: 4px;
    background-color: {edit_bg};
    color: {input_fg};
    font-family: Consolas, "Courier New", monospace;
    font-size: 12px;
}}
QScrollArea {{
    border: none;
}}
QSplitter::handle {{
    background-color: {splitter};
}}
QSplitter::handle:horizontal {{
    width: 2px;
}}
QSplitter::handle:vertical {{
    height: 2px;
}}
QStatusBar {{
    background-color: {status_bg};
    border-top: 1px solid {status_border};
    color: {status_fg};
}}
QProgressBar {{
    border: 1px solid {progress_border};
    border-radius: 3px;
    text-align: center;
    background-color: {progress_bg};
    color: {progress_fg};
}}
QProgressBar::chunk {{
    background-color: {btn_primary};
    border-radius: 3px;
}}
QMenuBar {{
    background-color: {menubar_bg};
    border-bottom: 1px solid {menubar_border};
    color: {status_fg};
}}
QMenuBar::item:selected {{
    background-color: {btn_primary};
    color: white;
}}
QMenu {{
    background-color: {menu_bg};
    border: 1px solid {menu_border};
    color: {status_fg};
}}
QMenu::item:selected {{
    background-color: {btn_primary};
    color: white;
}}
QTabWidget::pane {{
    border: 1px solid {tab_border};
    background-color: {list_bg};
}}
QTabBar::tab {{
    background-color: {tab_bg};
    color: {input_fg};
    padding: 6px 14px;
    border: 1px solid {tab_border};
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}
QTabBar::tab:selected {{
    background-color: {tab_selected_bg};
    border-bottom: 2px solid {btn_primary};
}}
QGroupBox {{
    font-weight: bold;
    border: 1px solid {group_border};
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 8px;
    color: {text};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}}
"""

LIGHT_THEME = _QSS_TEMPLATE.format(**_LIGHT)
DARK_THEME = _QSS_TEMPLATE.format(**_DARK)


def get_stylesheet(theme: str = "light") -> str:
    if theme == "dark":
        return DARK_THEME
    return LIGHT_THEME
