"""Main application window with workflow guidance, dark theme, and keyboard shortcuts."""
import logging
import os
import json
import numpy as np
from PySide6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                                QSplitter, QMessageBox, QStatusBar, QProgressBar,
                                QTextEdit, QLabel, QPushButton, QApplication, QScrollArea,
                                QLineEdit)
from PySide6.QtCore import Qt, QStandardPaths, Signal
from PySide6.QtGui import QKeySequence, QShortcut

from gui.input_panel import InputPanel
from gui.function_panel import FunctionPanel
from gui.param_panel import ParamPanel
from gui.preview_widget import PreviewPanel
from gui.output_panel import OutputPanel
from gui.workers import WorkerThread
from gui.function_handlers import apply_simple, apply_complex, is_batch_function, needs_batch_mode
from gui.stepper_bar import StepperBar
from gui.recent_functions import RecentFunctionsWidget
from gui.styles import get_stylesheet
from gui.runners import RunnerController
from core.function_registry import get_all_functions_flat, get_function_def

from core.image_io import read_image, write_image
from gui.dataset_review import DatasetReviewDialog
from gui.training_monitor import TrainingMonitorDialog
from gui.training_results import TrainingResultsDialog
from gui.stitch_graph_editor import StitchGraphEditorDialog
from gui.stitch_point_filter import StitchPointFilterDialog

from utils.helpers import get_image_files, ensure_dir, get_output_path

logger = logging.getLogger(__name__)

CONFIG_VERSION = 1

RECENT_FILE = os.path.join(
    QStandardPaths.writableLocation(QStandardPaths.AppDataLocation), "img_tools_recent.json"
)
CONFIG_FILE = os.path.join(
    QStandardPaths.writableLocation(QStandardPaths.AppDataLocation), "img_tools_config.json"
)


class MainWindow(QMainWindow):
    _log_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像处理工具箱")
        self.resize(1480, 900)
        self._current_image = None
        self._current_file = None
        self._worker = None
        self._undo_stack: list[np.ndarray] = []
        self._redo_stack: list[np.ndarray] = []
        self._MAX_UNDO = 20
        self._recent = self._load_recent()
        self._last_func_key = ""
        self._preview_result = None
        self._theme = "light"
        self._runner = RunnerController(self._run_worker, self._log)
        self._setup_ui()
        self._setup_menu()
        self._setup_shortcuts()
        self._log_signal.connect(self._append_log)
        self._restore_config()
        self._update_stepper()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        left_panel = self._build_left_panel()
        right_splitter = self._build_right_panel()
        splitter = self._build_main_splitter(left_panel, right_splitter)
        main_layout.addWidget(splitter)

        self._setup_status_bar()
        self._connect_panel_signals()
        self._apply_theme()

    def _build_left_panel(self) -> QWidget:
        """Construct the left panel as a clean vertical stack without nested scrollbars."""
        left_panel = QWidget()
        left_panel.setMinimumWidth(380)
        left_panel.setMaximumWidth(480)

        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 10, 8, 10)
        left_layout.setSpacing(8)

        self.stepper = StepperBar()
        left_layout.addWidget(self.stepper)

        self.input_panel = InputPanel()
        left_layout.addWidget(self.input_panel)

        self.recent_panel = RecentFunctionsWidget()
        self.recent_panel.functionClicked.connect(self._on_recent_function_clicked)
        left_layout.addWidget(self.recent_panel)

        self.function_panel = FunctionPanel()
        left_layout.addWidget(self.function_panel)

        self.param_panel = ParamPanel()
        left_layout.addWidget(self.param_panel)

        self.batch_hint = QLabel("")
        self.batch_hint.setWordWrap(True)
        self.batch_hint.setStyleSheet(
            "background-color: #fff3cd; color: #856404; border: 1px solid #ffeaa7; "
            "border-radius: 4px; padding: 6px; font-size: 12px;"
        )
        self.batch_hint.setVisible(False)
        left_layout.addWidget(self.batch_hint)

        self.output_panel = OutputPanel()
        left_layout.addWidget(self.output_panel)

        return left_panel

    def _build_preview_area(self) -> QWidget:
        """Construct the preview widget with an info bar below it."""
        preview_area = QWidget()
        preview_layout = QVBoxLayout(preview_area)
        preview_layout.setContentsMargins(4, 4, 4, 4)
        preview_layout.setSpacing(0)

        self.preview = PreviewPanel()
        preview_layout.addWidget(self.preview)

        info_bar = QHBoxLayout()
        info_bar.setContentsMargins(8, 2, 8, 2)
        self.lbl_img_info = QLabel("未加载图片")
        self.lbl_img_info.setObjectName("imgInfo")
        info_bar.addWidget(self.lbl_img_info)
        info_bar.addStretch()
        self.lbl_coord_display = QLabel("")
        self.lbl_coord_display.setObjectName("imgInfo")
        info_bar.addWidget(self.lbl_coord_display)
        preview_layout.addLayout(info_bar)

        return preview_area

    def _build_log_widget(self) -> QWidget:
        """Construct the log widget with header and text area."""
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(4, 4, 4, 4)
        log_layout.setSpacing(4)

        log_header = QHBoxLayout()
        log_title = QLabel("处理日志")
        f = log_title.font()
        f.setBold(True)
        log_title.setFont(f)
        log_header.addWidget(log_title)
        log_header.addStretch()

        self.btn_clear_log = QPushButton("清空")
        self.btn_clear_log.setMaximumWidth(50)
        self.btn_clear_log.clicked.connect(self._clear_log)
        log_header.addWidget(self.btn_clear_log)

        self.btn_copy_log = QPushButton("复制")
        self.btn_copy_log.setMaximumWidth(50)
        self.btn_copy_log.clicked.connect(self._copy_log)
        log_header.addWidget(self.btn_copy_log)

        log_layout.addLayout(log_header)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(80)
        log_layout.addWidget(self.log_output)

        return log_widget

    def _build_right_panel(self) -> QSplitter:
        """Construct the right vertical splitter containing preview and log areas."""
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.addWidget(self._build_preview_area())
        right_splitter.addWidget(self._build_log_widget())
        right_splitter.setSizes([700, 120])
        return right_splitter

    def _build_main_splitter(self, left_widget: QWidget, right_widget: QWidget) -> QSplitter:
        """Construct the main horizontal splitter joining left and right panels."""
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([400, 1080])
        return splitter

    def _setup_status_bar(self) -> None:
        """Configure the status bar with progress bar and status label."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(260)
        self.progress_bar.setMaximumHeight(20)
        self.progress_bar.hide()
        self.status_bar.addPermanentWidget(self.progress_bar)

        self.lbl_status = QLabel("就绪")
        self.status_bar.addWidget(self.lbl_status)

    def _connect_panel_signals(self) -> None:
        """Wire up widget signals to their handlers."""
        self.input_panel.previewRequested.connect(self._on_preview_file)
        self.input_panel.filesChanged.connect(self._on_files_changed)
        self.function_panel.functionSelected.connect(self._on_function_selected)
        self.function_panel.functionSelected.connect(self.param_panel.set_function)
        self.param_panel.btn_preview.clicked.connect(self._on_preview)
        self.param_panel.btn_save_result.clicked.connect(self._on_save_result)
        self.param_panel.btn_run.clicked.connect(self._on_run)
        self.param_panel.paramsChanged.connect(self._on_params_changed)
        self.preview.original_view.pixelClicked.connect(self._on_preview_pixel_clicked)
        self.preview.polygonCreated.connect(self._on_polygon_created)
        self.preview.roiSelected.connect(self._on_roi_selected)
        self.preview.saveAllRois.connect(self._on_save_all_rois)

    def _apply_theme(self):
        self.setStyleSheet(get_stylesheet(self._theme))

    def _setup_menu(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("文件(&F)")
        open_act = file_menu.addAction("打开图片...\tCtrl+O")
        open_act.triggered.connect(lambda: self.input_panel.add_files())
        open_dir_act = file_menu.addAction("打开文件夹...\tCtrl+D")
        open_dir_act.triggered.connect(lambda: self.input_panel.add_dir())
        file_menu.addSeparator()

        # Recent submenu
        self.recent_menu = file_menu.addMenu("最近打开")
        self._update_recent_menu()
        file_menu.addSeparator()

        quit_act = file_menu.addAction("退出\tCtrl+Q")
        quit_act.triggered.connect(self.close)

        # Edit menu
        edit_menu = menubar.addMenu("编辑(&E)")
        undo_act = edit_menu.addAction("撤销\tCtrl+Z")
        undo_act.triggered.connect(self._on_undo)
        redo_act = edit_menu.addAction("重做\tCtrl+Shift+Z")
        redo_act.triggered.connect(self._on_redo)

        # View menu
        view_menu = menubar.addMenu("视图(&V)")
        self.theme_act = view_menu.addAction("深色主题")
        self.theme_act.setCheckable(True)
        self.theme_act.setChecked(self._theme == "dark")
        self.theme_act.triggered.connect(self._toggle_theme)

        # Tools menu
        tools_menu = menubar.addMenu("工具(&T)")
        train_results_act = tools_menu.addAction("YOLO 训练结果管理")
        train_results_act.triggered.connect(self._show_training_results)
        stitch_editor_act = tools_menu.addAction("CAB-F 连边标注器")
        stitch_editor_act.triggered.connect(self._show_stitch_point_editor)
        stitch_filter_act = tools_menu.addAction("CAB-F 缝纫点数据筛选")
        stitch_filter_act.triggered.connect(self._show_stitch_point_filter)

        # Help menu
        help_menu = menubar.addMenu("帮助(&H)")
        about_act = help_menu.addAction("关于")
        about_act.triggered.connect(self._show_about)

    def _toggle_theme(self):
        self._theme = "dark" if self._theme == "light" else "light"
        self.theme_act.setChecked(self._theme == "dark")
        self._apply_theme()
        self._save_config()

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+O"), self, lambda: self.input_panel.add_files())
        QShortcut(QKeySequence("Ctrl+D"), self, lambda: self.input_panel.add_dir())
        QShortcut(QKeySequence("Ctrl+R"), self, self._on_run)
        QShortcut(QKeySequence("F5"), self, self._on_run)
        QShortcut(QKeySequence("Ctrl+S"), self, self._on_save_result)
        QShortcut(QKeySequence("Ctrl+Q"), self, self.close)
        QShortcut(QKeySequence("Escape"), self, self._on_escape)
        QShortcut(QKeySequence("Ctrl+Z"), self, self._on_undo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self._on_redo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self._on_redo)

    def _on_escape(self):
        if hasattr(self.preview, 'btn_picker') and self.preview.btn_picker.isChecked():
            self.preview.btn_picker.setChecked(False)
        if hasattr(self.preview, 'btn_polygon') and self.preview.btn_polygon.isChecked():
            self.preview.btn_polygon.setChecked(False)
        if hasattr(self.preview, 'btn_rect_select') and self.preview.btn_rect_select.isChecked():
            self.preview.btn_rect_select.setChecked(False)

    def _push_undo(self, img: np.ndarray | None):
        if img is not None:
            self._undo_stack.append(img.copy())
            if len(self._undo_stack) > self._MAX_UNDO:
                self._undo_stack.pop(0)

    def _on_function_selected(self, key: str, name: str):
        self._last_func_key = key
        self.recent_panel.add_function(key, name)
        self.stepper.mark_complete(1)
        self.stepper.set_step(2)
        self._update_batch_hint(key)
        self._update_stepper()

    def _on_recent_function_clicked(self, key: str, name: str):
        self.function_panel.select_function(key)

    def _on_files_changed(self, files):
        if files:
            self.stepper.mark_complete(0)
            self.stepper.set_step(1)
            self._on_preview_file(files[0])
        else:
            self.stepper.reset()
        self._update_stepper()

    def _on_params_changed(self, params):
        self.stepper.mark_complete(2)
        self.stepper.set_step(3)
        self._update_stepper()

    def _on_preview(self):
        params_info = self.param_panel.get_params()
        func_key = params_info["function"]
        params = params_info["params"]

        if not func_key:
            QMessageBox.warning(self, "提示", "请先选择处理功能")
            return
        if self._current_image is None:
            QMessageBox.warning(self, "提示", "请先选择并加载图片")
            return

        self.stepper.mark_complete(3)
        self.stepper.set_step(4)

        output_dir = self.output_panel.get_output_dir() or ""
        self._process_single(func_key, params, output_dir, save=False)

    def _update_stepper(self):
        # Determine current step based on state
        files = self.input_panel.get_files()
        func_key = self.param_panel.get_params().get("function", "")
        has_params = bool(self.param_panel.get_params().get("params"))

        if not files:
            self.stepper.set_step(0)
            return
        if not func_key:
            self.stepper.set_step(1)
            return
        if not has_params:
            self.stepper.set_step(2)
            return

    def _update_batch_hint(self, key: str):
        from core.function_registry import can_single_image
        if can_single_image(key):
            self.batch_hint.setText(
                "此功能支持单图和批量处理：预览查看效果后保存，或执行直接保存到输出目录")
            self.batch_hint.setStyleSheet(
                "background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; "
                "border-radius: 4px; padding: 6px; font-size: 12px;")
            self.batch_hint.setVisible(True)
        elif is_batch_function(key):
            self.batch_hint.setText("此功能为批量处理模式，将对输入文件夹中的所有图片进行处理")
            self.batch_hint.setVisible(True)
        else:
            self.batch_hint.setVisible(False)

    def _save_config(self):
        try:
            ensure_dir(os.path.dirname(CONFIG_FILE))
            config = {
                "version": CONFIG_VERSION,
                "geometry": self.saveGeometry().toBase64().data().decode(),
                "window_state": self.saveState().toBase64().data().decode(),
                "last_func_key": self._last_func_key,
                "output_dir": self.output_panel.get_output_dir() if hasattr(self.output_panel, 'get_output_dir') else "",
                "theme": self._theme,
            }
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f)
        except (OSError, TypeError) as e:
            logger.warning("Failed to save config: %s", e)

    def _restore_config(self):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError, TypeError) as e:
            logger.warning("Failed to restore config: %s", e)
            return

        # Version check and migration
        cfg_version = config.get("version", 0)
        if cfg_version > CONFIG_VERSION:
            logger.warning("Config version %d is newer than app version %d; some settings may be ignored", cfg_version, CONFIG_VERSION)
        if cfg_version < CONFIG_VERSION:
            config = self._migrate_config(config, cfg_version)

        if "geometry" in config:
            from PySide6.QtCore import QByteArray
            self.restoreGeometry(QByteArray.fromBase64(config["geometry"].encode()))
        if "window_state" in config:
            from PySide6.QtCore import QByteArray
            self.restoreState(QByteArray.fromBase64(config["window_state"].encode()))
        if "last_func_key" in config and config["last_func_key"]:
            key = config["last_func_key"]
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, lambda: self._restore_function(key))
        if "theme" in config:
            self._theme = config["theme"]
            self._apply_theme()
            if hasattr(self, 'theme_act'):
                self.theme_act.setChecked(self._theme == "dark")

    def _migrate_config(self, config: dict, from_version: int) -> dict:
        """Migrate config from an older version to the current version."""
        if from_version < 1:
            # Pre-versioned configs had no migrations needed
            pass
        # Future migrations go here:
        # if from_version < 2:
        #     config = self._migrate_v1_to_v2(config)
        config["version"] = CONFIG_VERSION
        return config

    def _restore_function(self, key: str):
        for fkey, fname, cat_name in get_all_functions_flat():
            if fkey == key:
                self.function_panel.select_function(key)
                return

    def closeEvent(self, event):
        self._save_config()
        super().closeEvent(event)

    def _on_undo(self):
        if not self._undo_stack:
            self._log("无法撤销")
            return
        if self._current_image is not None:
            self._redo_stack.append(self._current_image.copy())
        self._current_image = self._undo_stack.pop()
        self._preview_result = self._current_image
        self.preview.set_result(self._current_image)
        self.param_panel.btn_save_result.setEnabled(True)
        self._log("撤销")

    def _on_redo(self):
        if not self._redo_stack:
            self._log("无法重做")
            return
        if self._current_image is not None:
            self._undo_stack.append(self._current_image.copy())
        self._current_image = self._redo_stack.pop()
        self._preview_result = self._current_image
        self.preview.set_result(self._current_image)
        self.param_panel.btn_save_result.setEnabled(True)
        self._log("重做")

    def keyPressEvent(self, event):
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit)):
            super().keyPressEvent(event)
            return
        key = event.key()
        if key == Qt.Key_A:
            self._nav_prev()
        elif key == Qt.Key_D:
            self._nav_next()
        else:
            super().keyPressEvent(event)

    def _nav_prev(self):
        lst = self.input_panel.file_list
        if lst.count() == 0:
            return
        cur = lst.currentRow()
        idx = cur - 1 if cur > 0 else lst.count() - 1
        lst.setCurrentRow(idx)
        self._on_preview_file(self.input_panel.all_files[idx])

    def _nav_next(self):
        lst = self.input_panel.file_list
        if lst.count() == 0:
            return
        cur = lst.currentRow()
        idx = cur + 1 if cur < lst.count() - 1 else 0
        lst.setCurrentRow(idx)
        self._on_preview_file(self.input_panel.all_files[idx])

    def _on_preview_file(self, path: str):
        img = read_image(path)
        if img is None:
            self._log(f"无法读取图片: {path}")
            return
        self._current_file = path
        self._current_image = img
        self.preview.set_original(img)
        h, w = img.shape[:2]
        info = f"{w}×{h}  {os.path.basename(path)}"
        self.lbl_img_info.setText(info)
        self._add_recent(path)
        # Reset result tab
        self.preview.set_result(None)
        self._preview_result = None
        self.param_panel.btn_save_result.setEnabled(False)
        self._log(f"已加载: {os.path.basename(path)}")

    def _on_preview_pixel_clicked(self, x, y):
        self.lbl_coord_display.setText(f"坐标: ({x}, {y})")

    def _on_polygon_created(self, points):
        self._log(f"多边形: {points}")

    def _on_roi_selected(self, x1, y1, x2, y2):
        params = self.param_panel.get_params()["params"]
        w, h = x2 - x1, y2 - y1
        for name, value in [("x", x1), ("y", y1), ("w", w), ("h", h)]:
            widget = self.param_panel._widgets.get(name)
            if widget:
                widget.setValue(value)
        self._log(f"框选ROI: ({x1}, {y1}) -> ({x2}, {y2})")

    def _on_save_all_rois(self):
        rects = self.preview.get_roi_rects()
        if not rects:
            QMessageBox.warning(self, "提示", "请先框选至少一个ROI区域")
            return
        if self._current_image is None:
            QMessageBox.warning(self, "提示", "请先加载图片")
            return

        output_dir = self.output_panel.get_output_dir()
        if not output_dir:
            QMessageBox.warning(self, "提示", "请先设置输出目录")
            return

        ensure_dir(output_dir)
        img = self._current_image
        ih, iw = img.shape[:2]
        base_name = os.path.splitext(os.path.basename(self._current_file))[0] if self._current_file else "img"

        saved = 0
        for i, (x1, y1, x2, y2) in enumerate(rects):
            cx2, cy2 = min(x2, iw), min(y2, ih)
            if cx2 <= x1 or cy2 <= y1:
                continue
            crop = img[y1:cy2, x1:cx2].copy()
            out_path = os.path.join(output_dir, f"{base_name}_roi{i + 1}.png")
            write_image(out_path, crop)
            saved += 1
            self._log(f"保存ROI {i + 1}: ({x1}, {y1}) -> ({cx2}, {cy2}) -> {os.path.basename(out_path)}")

        self.lbl_status.setText(f"已保存 {saved} 个ROI到 {output_dir}")
        self._log(f"共保存 {saved} 个ROI到 {output_dir}")

    # ---- Recent files ----
    def _get_recent_path(self):
        return RECENT_FILE

    def _load_recent(self):
        try:
            ensure_dir(os.path.dirname(self._get_recent_path()))
            with open(self._get_recent_path(), "r") as f:
                data = json.load(f)
                return [p for p in data.get("paths", []) if os.path.exists(p)]
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to load recent files: %s", e)
            return []

    def _save_recent(self):
        try:
            ensure_dir(os.path.dirname(self._get_recent_path()))
            with open(self._get_recent_path(), "w") as f:
                json.dump({"paths": self._recent[:10]}, f)
        except (OSError, TypeError) as e:
            logger.warning("Failed to save recent files: %s", e)

    def _add_recent(self, path):
        if path in self._recent:
            self._recent.remove(path)
        self._recent.insert(0, path)
        self._recent = self._recent[:10]
        self._save_recent()
        self._update_recent_menu()

    def _update_recent_menu(self):
        self.recent_menu.clear()
        if not self._recent:
            self.recent_menu.addAction("(无)").setEnabled(False)
        else:
            for p in self._recent:
                name = os.path.basename(p) if os.path.isfile(p) else os.path.basename(p.rstrip(os.sep))
                act = self.recent_menu.addAction(name)
                act.setToolTip(p)
                act.triggered.connect(lambda checked, path=p: self._open_recent(path))
            self.recent_menu.addSeparator()
            clear_act = self.recent_menu.addAction("清除记录")
            clear_act.triggered.connect(self._clear_recent)

    def _open_recent(self, path):
        if os.path.isfile(path):
            self._on_preview_file(path)
            self.input_panel._add_paths([path])
        elif os.path.isdir(path):
            self.input_panel._add_paths([path])

    def _clear_recent(self):
        self._recent = []
        self._save_recent()
        self._update_recent_menu()

    # ---- Log helpers ----
    def _log(self, msg):
        self._log_signal.emit(msg)

    def _append_log(self, msg):
        self.log_output.append(msg)

    def _clear_log(self):
        self.log_output.clear()

    def _copy_log(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.log_output.toPlainText())

    # ---- Run / Process ----
    def _on_run(self):
        params_info = self.param_panel.get_params()
        func_key = params_info["function"]
        params = params_info["params"]
        output_dir = self.output_panel.get_output_dir()

        # Validate steps
        files = self.input_panel.get_files()
        if not files:
            self.stepper.mark_error(0)
            QMessageBox.warning(self, "提示", "请先选择输入图片或文件夹")
            return
        if not func_key:
            self.stepper.mark_error(1)
            QMessageBox.warning(self, "提示", "请先选择处理功能")
            return
        if not output_dir:
            QMessageBox.warning(self, "提示", "请先设置输出目录")
            return

        # Clear step errors
        for i in range(5):
            self.stepper.clear_error(i)

        self.stepper.mark_complete(4)
        has_single = (self._current_image is not None
                      and self._current_file is not None
                      and len(files) <= 1)
        is_batch = needs_batch_mode(func_key, has_single)

        if is_batch and not files:
            QMessageBox.warning(self, "提示", "批处理需要选择图片或文件夹")
            return

        if not is_batch:
            if self._current_image is None:
                QMessageBox.warning(self, "提示", "请先选择图片")
                return
            self._process_single(func_key, params, output_dir)
        else:
            self._process_batch(func_key, params, output_dir, files)

    def _process_single(self, func_key, params, output_dir, save=True):
        img = self._current_image.copy()
        try:
            result = self._dispatch_function(img, func_key, params,
                                             filepath=self._current_file,
                                             output_dir=output_dir,
                                             batch_mode=False)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理失败:\n{e}")
            return

        if result is None:
            if func_key in ("tile_fixed", "tile_grid", "annot_crop_roi"):
                self.lbl_status.setText("处理完成，结果已保存到输出目录")
                self._log("处理完成，结果已保存到输出目录")
            else:
                QMessageBox.warning(self, "提示", "此功能不产生图像输出")
            return

        self._push_undo(self._current_image)
        self._redo_stack.clear()
        self._current_image = result
        self._preview_result = result
        self.preview.set_result(result)
        self.param_panel.btn_save_result.setEnabled(True)

        if save:
            self._save_preview_result(func_key, params, output_dir)
        else:
            self._log("预览完成，点击「保存当前结果」或 Ctrl+S 保存")
            self.lbl_status.setText("预览完成（未保存）")

    def _save_preview_result(self, func_key=None, params=None, output_dir=None):
        if self._preview_result is None or self._current_file is None:
            return
        func_key = func_key or self._last_func_key
        params = params or self.param_panel.get_params()["params"]
        output_dir = output_dir or self.output_panel.get_output_dir()
        if not output_dir:
            QMessageBox.warning(self, "提示", "请先设置输出目录")
            return
        ensure_dir(output_dir)
        if func_key == "format_convert":
            ext = params.get("fmt", "png")
            out_path = get_output_path(self._current_file, output_dir,
                                       suffix="_converted", ext=f".{ext}")
            write_image(out_path, self._preview_result, quality=params.get("quality", 95))
        else:
            out_path = get_output_path(self._current_file, output_dir,
                                       suffix="_processed", ext=".png")
            write_image(out_path, self._preview_result)
        self.param_panel.btn_save_result.setEnabled(False)
        self._log(f"已保存: {os.path.basename(out_path)}")
        self.lbl_status.setText(f"已保存 - {os.path.basename(out_path)}")

    def _on_save_result(self):
        if self._preview_result is None:
            QMessageBox.warning(self, "提示", "没有待保存的处理结果")
            return
        self._save_preview_result()

    def _process_batch(self, func_key, params, output_dir, files):
        self._log(f"开始批处理: {func_key}, 共 {len(files)} 个文件")

        if func_key == "dataset_review":
            if not files:
                QMessageBox.warning(self, "提示", "请选择数据集文件夹")
                return
            dataset_dir = files[0] if os.path.isdir(files[0]) else os.path.dirname(files[0])
            dlg = DatasetReviewDialog(dataset_dir, self)
            dlg.exec()
            return

        if func_key == "export_onnx":
            self._runner.run_export_onnx(params, output_dir)
            return

        if func_key == "yolo_train":
            self._run_yolo_train(params)
            return

        if func_key in ("dataset_random_split", "dataset_stratified_split", "dataset_kfold"):
            self._runner.run_dataset_op(func_key, params, output_dir, files)
            return
        if func_key.startswith("format_"):
            self._runner.run_format_op(func_key, params, output_dir, files)
            return
        if func_key in ("batch_deduplicate", "batch_rename", "batch_convert_format",
                        "batch_resize", "batch_add_border"):
            self._runner.run_simple_batch_op(func_key, params, output_dir, files)
            return
        if func_key in ("annot_statistics", "annot_validate_yolo"):
            self._runner.run_simple_batch_op(func_key, params, output_dir, files)
            return
        if func_key == "seg_tile":
            try:
                self._runner.run_seg_tile_op(func_key, params, output_dir, files)
            except ValueError as e:
                QMessageBox.warning(self, "提示", str(e))
            return
        if func_key in ("mask_to_polygons", "polygons_to_mask"):
            self._runner.run_mask_polygon_op(func_key, params, output_dir, files)
            return
        if func_key.startswith("augment_"):
            try:
                self._runner.run_augment_op(func_key, params, output_dir, files)
            except ValueError as e:
                QMessageBox.warning(self, "提示", str(e))
            return

        ensure_dir(output_dir)
        if func_key == "batch_roi_crop":
            x, y = params.get("x", 0), params.get("y", 0)
            w, h = params.get("w", 512), params.get("h", 512)
            self._log(f"批量定点裁剪: 起点({x}, {y}), 裁剪大小({w}×{h})")
        self.progress_bar.setMaximum(len(files))
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.lbl_status.setText(f"处理中... 0/{len(files)}")

        def process(progress_callback=None, cancel_check=None):
            for i, f in enumerate(files):
                if cancel_check and cancel_check():
                    break
                img = read_image(f)
                if img is not None:
                    try:
                        result = self._dispatch_function(img, func_key, params, filepath=f,
                                                         output_dir=output_dir, batch_mode=True)
                        if result is not None and isinstance(result, np.ndarray):
                            if func_key == "batch_roi_crop":
                                prefix = params.get("prefix", "crop")
                                out_path = os.path.join(output_dir, f"{prefix}_{str(i + 1).zfill(4)}.png")
                            else:
                                out_path = get_output_path(f, output_dir, suffix="_out")
                            write_image(out_path, result)
                    except Exception as e:
                        logger.warning("Batch processing failed for %s: %s", f, e)
                if progress_callback:
                    progress_callback(i + 1, len(files))

        worker = WorkerThread(process)
        worker.progress.connect(self._update_progress)
        worker.finished.connect(lambda _: self._on_batch_done(len(files)))
        worker.error.connect(self._on_worker_error)
        worker.log.connect(self._log)
        self.param_panel.btn_run.setEnabled(False)
        self._worker = worker
        worker.start()

    def _update_progress(self, count):
        self.progress_bar.setValue(count)
        self.lbl_status.setText(f"处理中... {count}/{self.progress_bar.maximum()}")

    def _on_batch_done(self, count):
        self.progress_bar.hide()
        self.param_panel.btn_run.setEnabled(True)
        self.lbl_status.setText(f"批处理完成 - {count} 个文件")
        self._log(f"批处理完成，处理了 {count} 个文件")

    # ---- Function dispatch ----
    def _dispatch_function(self, img, func_key, params, filepath=None, output_dir=None, batch_mode=False):
        fdef = get_function_def(func_key)
        handler_type = fdef.handler_type if fdef else "simple"

        if handler_type == "simple":
            return apply_simple(func_key, img, params)
        return apply_complex(func_key, img, params,
                             filepath=filepath, output_dir=output_dir,
                             batch_mode=batch_mode, log_fn=self._log)

    def _run_yolo_train(self, params):
        data_path = params.get("data", "")
        if not data_path or not os.path.exists(data_path):
            QMessageBox.warning(self, "提示", "请指定有效的数据集 YAML 文件路径")
            return

        dlg = TrainingMonitorDialog(params, self)
        dlg.exec()

    def _show_training_results(self):
        dlg = TrainingResultsDialog("runs", self)
        dlg.exec()

    def _show_stitch_point_editor(self):
        image = self._current_image if self._current_image is not None else None
        image_path = self._current_file or ""
        dlg = StitchGraphEditorDialog(self, image=image, image_path=image_path)
        dlg.exec()

    def _show_stitch_point_filter(self):
        dlg = StitchPointFilterDialog(self)
        dlg.exec()

    # ---- Worker helpers ----
    def _run_worker(self, func, on_finish=None):
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        self.lbl_status.setText("处理中...")
        self.param_panel.btn_run.setEnabled(False)
        self._worker = WorkerThread(func)
        self._worker.finished.connect(lambda r: self._on_worker_done(r, on_finish))
        self._worker.error.connect(self._on_worker_error)
        self._worker.log.connect(self._log)
        self._worker.start()

    def _on_worker_done(self, result, callback=None):
        self.progress_bar.hide()
        self.progress_bar.setRange(0, 100)
        self.param_panel.btn_run.setEnabled(True)
        self.lbl_status.setText("就绪")
        if callback:
            callback(result)

    def _on_worker_error(self, user_msg: str, full_traceback: str = ""):
        self.progress_bar.hide()
        self.param_panel.btn_run.setEnabled(True)
        self.lbl_status.setText("处理出错")
        self._log(f"错误: {user_msg}")
        if full_traceback:
            logger.error("Worker error: %s\n%s", user_msg, full_traceback)
        QMessageBox.critical(self, "错误", user_msg)

    def _show_about(self):
        QMessageBox.about(self, "关于",
                          "图像处理工具箱 v1.2\n\n"
                          "面向深度学习的图像处理工具\n"
                          "支持: 颜色转换 · 图像处理 · 大图切块 ·\n"
                          "       标注切块 · 数据集划分 · 格式转换 · YOLO训练\n\n"
                          "快捷键:\n"
                          "  Ctrl+O  打开图片    Ctrl+D  打开文件夹\n"
                          "  Ctrl+R  执行处理    F5      执行处理\n"
                          "  Esc     退出坐标拾取")
