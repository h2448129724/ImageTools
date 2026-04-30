"""Parameter configuration panel that adapts to selected function."""
from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QSpinBox,
                                QDoubleSpinBox, QComboBox, QCheckBox, QLabel,
                                QHBoxLayout, QPushButton, QLineEdit,
                                QScrollArea, QFileDialog)
from PySide6.QtCore import Signal

from core.function_registry import get_param_specs, get_function_description


class ParamPanel(QWidget):
    """Dynamic parameter panel that changes based on selected function."""
    paramsChanged = Signal(dict)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._widgets: dict[str, QWidget] = {}
        self._current_key: str | None = None
        self._layout: QFormLayout | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(4, 4, 4, 4)
        self._title_label = QLabel("参数设置")
        f = self._title_label.font()
        f.setBold(True)
        self._title_label.setFont(f)
        self._main_layout.addWidget(self._title_label)

        self._desc_label = QLabel("")
        self._desc_label.setWordWrap(True)
        self._desc_label.setStyleSheet("color: #666666; font-size: 12px; padding: 4px;")
        self._main_layout.addWidget(self._desc_label)

        self._form_container = QWidget()
        self._layout = QFormLayout(self._form_container)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.addWidget(self._form_container)

        self.btn_preview = QPushButton("预览 (仅当前图片)")
        self.btn_preview.setToolTip("对当前选中的图片进行预览，结果不保存到磁盘")
        self._main_layout.addWidget(self.btn_preview)

        self.btn_save_result = QPushButton("保存当前结果 (Ctrl+S)")
        self.btn_save_result.setToolTip("保存当前预览的处理结果到输出目录")
        self.btn_save_result.setEnabled(False)
        self._main_layout.addWidget(self.btn_save_result)

        self.btn_run = QPushButton("▶ 执行处理 (Ctrl+R)")
        self.btn_run.setObjectName("btnRun")
        self._main_layout.addWidget(self.btn_run)
        self._current_params = {}
        self._container_refs = []  # keep alive container widgets for dir/file pickers

    def set_function(self, key: str, name: str) -> None:
        """Show parameters for the selected function."""
        self._current_key = key
        # Clear existing widgets
        while self._layout.rowCount() > 0:
            self._layout.removeRow(0)
        self._widgets.clear()
        self._container_refs.clear()
        self._current_params = {}

        self._title_label.setText(f"参数设置 — {name}")
        desc = get_function_description(key)
        self._desc_label.setText(desc)
        self._desc_label.setVisible(bool(desc))
        param_specs = get_param_specs(key)
        for spec in param_specs:
            widget = self._create_widget(spec)
            self._widgets[spec["name"]] = widget
            inner = getattr(widget, "_inner_line_edit", None)
            if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                widget.valueChanged.connect(self._on_param_changed)
            elif isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(self._on_param_changed)
            elif isinstance(widget, QCheckBox):
                widget.toggled.connect(self._on_param_changed)
            elif isinstance(widget, QLineEdit):
                widget.textChanged.connect(self._on_param_changed)
            elif inner is not None:
                inner.textChanged.connect(self._on_param_changed)
            self._layout.addRow(spec.get("label", spec["name"]), widget)

        self._collect_params()
        self.paramsChanged.emit(self._current_params)

    def _create_widget(self, spec: dict) -> QWidget:
        wtype = spec.get("type", "int")
        if wtype == "int":
            w = QSpinBox()
            w.setRange(spec.get("min", 0), spec.get("max", 9999))
            w.setValue(spec.get("default", 0))
            return w
        if wtype == "float":
            w = QDoubleSpinBox()
            w.setRange(spec.get("min", 0.0), spec.get("max", 100.0))
            w.setDecimals(2)
            w.setSingleStep(spec.get("step", 0.1))
            w.setValue(spec.get("default", 1.0))
            return w
        if wtype == "combo":
            w = QComboBox()
            w.addItems(spec.get("options", []))
            default = spec.get("default", "")
            if default:
                w.setCurrentText(default)
            return w
        if wtype == "choice":
            w = QComboBox()
            labels = spec.get("choice_labels", spec.get("choices", []))
            values = spec.get("choices", [])
            for label, value in zip(labels, values):
                w.addItem(label, value)
            default = spec.get("default", "")
            if default:
                idx = values.index(default) if default in values else 0
                w.setCurrentIndex(idx)
            return w
        if wtype == "bool":
            w = QCheckBox()
            w.setChecked(spec.get("default", False))
            return w
        if wtype == "text":
            w = QLineEdit()
            w.setText(spec.get("default", ""))
            return w
        if wtype == "file":
            container = QWidget()
            h = QHBoxLayout(container)
            h.setContentsMargins(0, 0, 0, 0)
            le = QLineEdit()
            le.setText(spec.get("default", ""))
            btn = QPushButton("...")
            btn.setMaximumWidth(30)
            btn.clicked.connect(lambda checked, s=spec, l=le: self._browse_file(l, s))
            h.addWidget(le)
            h.addWidget(btn)
            container._inner_line_edit = le
            return container
        if wtype == "dir":
            container = QWidget()
            h = QHBoxLayout(container)
            h.setContentsMargins(0, 0, 0, 0)
            le = QLineEdit()
            le.setText(spec.get("default", ""))
            btn = QPushButton("...")
            btn.setMaximumWidth(30)
            btn.clicked.connect(lambda checked, s=spec, l=le: self._browse_dir(l, s))
            h.addWidget(le)
            h.addWidget(btn)
            container._inner_line_edit = le
            return container
        return QLabel("")

    def _browse_file(self, line_edit: QLineEdit, spec: dict) -> None:
        path, _ = QFileDialog.getOpenFileName(self, spec.get("label", "选择文件"))
        if path:
            line_edit.setText(path)

    def _browse_dir(self, line_edit: QLineEdit, spec: dict) -> None:
        path = QFileDialog.getExistingDirectory(self, spec.get("label", "选择文件夹"))
        if path:
            line_edit.setText(path)

    def _on_param_changed(self, *_) -> None:
        self._collect_params()
        self.paramsChanged.emit(self._current_params)

    def _collect_params(self) -> None:
        self._current_params = {}
        for name, widget in self._widgets.items():
            if isinstance(widget, QSpinBox):
                self._current_params[name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                self._current_params[name] = widget.value()
            elif isinstance(widget, QComboBox):
                data = widget.currentData()
                self._current_params[name] = data if data is not None else widget.currentText()
            elif isinstance(widget, QCheckBox):
                self._current_params[name] = widget.isChecked()
            elif isinstance(widget, QLineEdit):
                self._current_params[name] = widget.text()
            else:
                inner = getattr(widget, "_inner_line_edit", None)
                if inner is not None:
                    self._current_params[name] = inner.text()

    def get_params(self) -> dict[str, Any]:
        self._collect_params()
        return {"function": self._current_key, "params": self._current_params}
