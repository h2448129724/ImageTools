"""Shared theme tokens and QSS for the CAB-F desktop UI."""
from __future__ import annotations


TOKENS = {
    "bg_page": "#F8FAFC",
    "bg_card": "#FFFFFF",
    "border_main": "#E5E7EB",
    "border_weak": "#EEF0F3",
    "primary": "#2563EB",
    "text_main": "#111827",
    "text_secondary": "#6B7280",
    "text_weak": "#9CA3AF",
    "success": "#16A34A",
    "warning": "#D97706",
    "danger": "#DC2626",
    "success_bg": "#ECFDF3",
    "warning_bg": "#FFFBEB",
    "danger_bg": "#FEF2F2",
    "neutral_bg": "#F3F4F6",
    "neutral_text": "#6B7280",
    "sidebar_selected": "#EFF6FF",
    "sidebar_hover": "#F3F4F6",
    "overview_bg": "#F8FBFF",
}


APP_STYLESHEET = f"""
QMainWindow {{
    background: {TOKENS["bg_page"]};
}}

QWidget {{
    font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
    font-size: 14px;
    color: {TOKENS["text_main"]};
}}

QMenuBar {{
    background: {TOKENS["bg_card"]};
    border-bottom: 1px solid {TOKENS["border_weak"]};
    padding: 3px 8px;
}}
QMenuBar::item {{
    padding: 5px 9px;
    border-radius: 8px;
}}
QMenuBar::item:selected {{
    background: {TOKENS["sidebar_hover"]};
}}
QMenu {{
    background: {TOKENS["bg_card"]};
    border: 1px solid {TOKENS["border_main"]};
    border-radius: 10px;
    padding: 6px;
}}
QMenu::item {{
    padding: 8px 14px;
    border-radius: 8px;
}}
QMenu::item:selected {{
    background: {TOKENS["sidebar_selected"]};
    color: {TOKENS["primary"]};
}}

#sidebar {{
    background: {TOKENS["bg_card"]};
    border-right: 1px solid {TOKENS["border_main"]};
}}
#sidebarHeader,
#sidebar QLabel {{
    color: {TOKENS["text_secondary"]};
}}
#sidebarFootnote {{
    color: {TOKENS["text_secondary"]};
    font-size: 13px;
    line-height: 1.45;
}}

#toolList {{
    border: none;
    background: transparent;
    outline: none;
    padding: 2px;
}}
#toolList::item {{
    background: transparent;
    color: {TOKENS["text_main"]};
    padding: 8px 10px;
    margin: 1px 0;
    border-radius: 9px;
}}
#toolList::item:selected {{
    background: {TOKENS["sidebar_selected"]};
    color: {TOKENS["primary"]};
    border-left: 3px solid {TOKENS["primary"]};
    padding-left: 11px;
    font-weight: 600;
}}
#toolList::item:hover:!selected {{
    background: {TOKENS["sidebar_hover"]};
}}

#contentShell {{
    background: {TOKENS["bg_page"]};
}}

#overviewCard {{
    background: {TOKENS["bg_card"]};
    border: 1px solid {TOKENS["border_main"]};
    border-radius: 12px;
}}
#overviewBadge {{
    background: {TOKENS["sidebar_selected"]};
    color: {TOKENS["primary"]};
    border-radius: 999px;
    padding: 3px 9px;
    font-size: 11px;
    font-weight: 700;
}}
#overviewTitle {{
    color: {TOKENS["text_main"]};
    font-size: 16px;
    font-weight: 700;
}}
#overviewText {{
    color: {TOKENS["text_secondary"]};
    font-size: 11px;
}}
#overviewMeta {{
    color: {TOKENS["text_secondary"]};
    font-size: 11px;
}}

#miniStat {{
    background: {TOKENS["bg_page"]};
    border: 1px solid {TOKENS["border_weak"]};
    border-radius: 8px;
}}
#miniStatValue {{
    color: {TOKENS["text_main"]};
    font-size: 15px;
    font-weight: 700;
}}
#miniStatLabel {{
    color: {TOKENS["text_secondary"]};
    font-size: 11px;
    font-weight: 600;
}}

#surfaceCard {{
    background: transparent;
    border: none;
}}

#card {{
    background: {TOKENS["bg_card"]};
    border: 1px solid {TOKENS["border_main"]};
    border-radius: 12px;
}}

#hintPanel {{
    background: {TOKENS["overview_bg"]};
    border: 1px solid {TOKENS["border_weak"]};
    border-radius: 10px;
}}

#stepSidebar {{
    background: transparent;
    border: none;
    border-radius: 0;
}}

#stepContentCard,
#configTableCard,
#logCard {{
    background: {TOKENS["bg_card"]};
    border: 1px solid {TOKENS["border_main"]};
    border-radius: 12px;
}}

#configRow {{
    background: transparent;
    border-bottom: 1px solid {TOKENS["border_weak"]};
}}

#configLabel {{
    color: {TOKENS["text_main"]};
    font-size: 14px;
    font-weight: 600;
}}

#fieldHint {{
    color: {TOKENS["text_secondary"]};
    font-size: 13px;
}}

QPushButton {{
    min-height: 34px;
    max-height: 34px;
    padding: 0 12px;
    background: {TOKENS["bg_card"]};
    border: 1px solid {TOKENS["border_main"]};
    border-radius: 8px;
    color: {TOKENS["text_main"]};
    font-size: 13px;
    font-weight: 500;
}}
QPushButton:hover {{
    background: {TOKENS["sidebar_hover"]};
    border-color: #D1D5DB;
}}
QPushButton:pressed {{
    background: #E5E7EB;
}}
QPushButton:disabled {{
    background: {TOKENS["bg_page"]};
    color: {TOKENS["text_weak"]};
}}
QPushButton[primary="true"] {{
    background: {TOKENS["primary"]};
    border: 1px solid {TOKENS["primary"]};
    color: #FFFFFF;
}}
QPushButton[primary="true"]:hover {{
    background: #1D4ED8;
    border-color: #1D4ED8;
}}
QPushButton[primary="true"]:pressed {{
    background: #1E40AF;
    border-color: #1E40AF;
}}

QLineEdit,
QComboBox,
QSpinBox,
QDoubleSpinBox {{
    min-height: 34px;
    max-height: 34px;
    padding: 0 9px;
    background: {TOKENS["bg_card"]};
    border: 1px solid {TOKENS["border_main"]};
    border-radius: 8px;
    selection-background-color: {TOKENS["primary"]};
}}
QLineEdit:focus,
QComboBox:focus,
QSpinBox:focus,
QDoubleSpinBox:focus {{
    border-color: {TOKENS["primary"]};
}}

QPlainTextEdit {{
    background: {TOKENS["bg_page"]};
    border: 1px solid {TOKENS["border_weak"]};
    border-radius: 10px;
    padding: 6px;
    color: {TOKENS["text_secondary"]};
    font-family: "Cascadia Code", "Consolas", monospace;
    font-size: 11px;
}}

QListWidget {{
    background: {TOKENS["bg_card"]};
    border: 1px solid {TOKENS["border_main"]};
    border-radius: 10px;
    outline: none;
}}

QScrollArea {{
    border: none;
    background: transparent;
}}
QScrollBar:horizontal,
QScrollBar:vertical {{
    background: transparent;
    border: none;
}}
QScrollBar::handle:horizontal,
QScrollBar::handle:vertical {{
    background: #D1D5DB;
    border-radius: 5px;
    min-width: 24px;
    min-height: 24px;
}}
QScrollBar::add-line,
QScrollBar::sub-line,
QScrollBar::add-page,
QScrollBar::sub-page {{
    background: transparent;
    border: none;
}}

QCheckBox,
QRadioButton {{
    color: {TOKENS["text_main"]};
    spacing: 8px;
}}

QGroupBox {{
    background: {TOKENS["bg_card"]};
    border: 1px solid {TOKENS["border_main"]};
    border-radius: 12px;
    margin-top: 12px;
    padding: 12px;
    color: {TOKENS["text_main"]};
    font-weight: 600;
}}
QGroupBox::title {{
    left: 12px;
    padding: 0 6px;
    color: {TOKENS["text_secondary"]};
}}

QStatusBar {{
    background: {TOKENS["bg_card"]};
    border-top: 1px solid {TOKENS["border_weak"]};
    color: {TOKENS["text_secondary"]};
}}

QMessageBox {{
    background: {TOKENS["bg_card"]};
}}
"""


def badge_style(kind: str) -> str:
    styles = {
        "success": (TOKENS["success_bg"], TOKENS["success"]),
        "warning": (TOKENS["warning_bg"], TOKENS["warning"]),
        "danger": (TOKENS["danger_bg"], TOKENS["danger"]),
        "neutral": (TOKENS["neutral_bg"], TOKENS["neutral_text"]),
        "info": (TOKENS["sidebar_selected"], TOKENS["primary"]),
    }
    bg, fg = styles.get(kind, styles["neutral"])
    return (
        f"background:{bg};color:{fg};border:1px solid transparent;"
        "border-radius:999px;padding:4px 10px;font-size:12px;font-weight:600;"
    )
