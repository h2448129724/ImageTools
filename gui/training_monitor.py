"""Training monitor dialog with real-time log and optional matplotlib charts."""
import os
import time
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QPushButton, QProgressBar, QSplitter, QWidget, QMessageBox,
    QGridLayout, QGroupBox,
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread

from core.yolo_training import run_yolo_training, parse_training_logs

# Optional matplotlib support
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


class TrainingWorker(QThread):
    log = Signal(str)
    finished = Signal(dict)
    progress = Signal(int, int)  # current_epoch, total_epochs

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._cancelled = False

    def run(self):
        def log_fn(msg):
            self.log.emit(msg)

        def cancel_check():
            return self._cancelled

        result = run_yolo_training(
            self.config,
            log_fn=log_fn,
            cancel_check=cancel_check,
        )
        self.finished.emit(result)

    def cancel(self):
        self._cancelled = True


class _ExportWorker(QThread):
    log = Signal(str)
    finished = Signal(dict)

    def __init__(self, model_path, formats, imgsz, log_fn):
        super().__init__()
        self._model_path = model_path
        self._formats = formats
        self._imgsz = imgsz
        self._log_fn = log_fn

    def run(self):
        from core.yolo_training import export_after_training
        result = export_after_training(
            self._model_path, self._formats,
            imgsz=self._imgsz,
            log_fn=lambda msg: self.log.emit(msg),
        )
        self.finished.emit(result)


class TrainingMonitorDialog(QDialog):
    """Dialog that shows training progress in real-time."""

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self.run_dir = os.path.join(
            config.get("project", "runs"),
            config.get("name", "detect")
        )
        self.epochs = config.get("epochs", 100)
        self._worker = None
        self._timer = None
        self._logs = []

        self.setWindowTitle(f"YOLO 训练监控 — {config.get('model', 'yolov8n.pt')}")
        self.resize(900, 700)
        self._setup_ui()
        self._start_training()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Info bar
        info = QGroupBox("训练信息")
        info_grid = QGridLayout(info)
        info_grid.addWidget(QLabel("模型:"), 0, 0)
        self.lbl_model = QLabel(self.config.get("model", ""))
        info_grid.addWidget(self.lbl_model, 0, 1)
        info_grid.addWidget(QLabel("数据集:"), 0, 2)
        self.lbl_data = QLabel(self.config.get("data", ""))
        info_grid.addWidget(self.lbl_data, 0, 3)
        info_grid.addWidget(QLabel("Epochs:"), 1, 0)
        self.lbl_epochs = QLabel(str(self.epochs))
        info_grid.addWidget(self.lbl_epochs, 1, 1)
        info_grid.addWidget(QLabel("尺寸:"), 1, 2)
        self.lbl_imgsz = QLabel(str(self.config.get("imgsz", 640)))
        info_grid.addWidget(self.lbl_imgsz, 1, 3)
        layout.addWidget(info)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(self.epochs)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Epoch %v / %m")
        layout.addWidget(self.progress_bar)

        # Charts or placeholder
        if _HAS_MPL:
            self._setup_charts(layout)
        else:
            self.lbl_no_chart = QLabel(
                "未安装 matplotlib，无法显示实时图表。\n"
                "可执行: pip install matplotlib 以启用图表功能。"
            )
            self.lbl_no_chart.setAlignment(Qt.AlignCenter)
            self.lbl_no_chart.setStyleSheet("color: #888; padding: 20px;")
            layout.addWidget(self.lbl_no_chart, 1)

        # Log output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(180)
        layout.addWidget(self.log_output)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_stop = QPushButton("停止训练")
        self.btn_stop.setStyleSheet(
            "QPushButton { background-color: #e74c3c; color: white; font-weight: bold; }"
            "QPushButton:hover { background-color: #c0392b; }"
        )
        self.btn_stop.clicked.connect(self._stop_training)
        btn_row.addWidget(self.btn_stop)

        btn_row.addStretch()

        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.close)
        self.btn_close.setEnabled(False)
        btn_row.addWidget(self.btn_close)

        layout.addLayout(btn_row)

    def _setup_charts(self, layout):
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, 1)

        # Two subplots: losses and metrics
        self.ax_loss = self.figure.add_subplot(121)
        self.ax_metric = self.figure.add_subplot(122)
        self.ax_loss.set_title("训练损失")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_metric.set_title("验证 mAP")
        self.ax_metric.set_xlabel("Epoch")
        self.figure.tight_layout()

    def _start_training(self):
        self._worker = TrainingWorker(self.config)
        self._worker.log.connect(self._on_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

        # Timer to poll results.csv for chart updates
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_charts)
        self._timer.start(2000)  # every 2 seconds

    def _on_log(self, msg: str):
        self.log_output.append(msg)
        # Auto-scroll
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _update_charts(self):
        if not _HAS_MPL:
            return
        if not os.path.exists(self.run_dir):
            return

        logs = parse_training_logs(self.run_dir)
        if not logs:
            return

        epochs = [int(r.get("epoch", i + 1)) for i, r in enumerate(logs)]

        self.ax_loss.clear()
        self.ax_loss.set_title("训练损失")
        self.ax_loss.set_xlabel("Epoch")

        for key, label in [
            ("train/box_loss", "box"),
            ("train/cls_loss", "cls"),
            ("train/dfl_loss", "dfl"),
        ]:
            vals = [r.get(key) for r in logs]
            if any(v is not None for v in vals):
                self.ax_loss.plot(epochs, vals, label=label)
        self.ax_loss.legend()
        self.ax_loss.grid(True, alpha=0.3)

        self.ax_metric.clear()
        self.ax_metric.set_title("验证 mAP")
        self.ax_metric.set_xlabel("Epoch")

        for key, label in [
            ("metrics/precision(B)", "Precision"),
            ("metrics/recall(B)", "Recall"),
            ("metrics/mAP50(B)", "mAP@50"),
            ("metrics/mAP50-95(B)", "mAP@50-95"),
        ]:
            vals = [r.get(key) for r in logs]
            if any(v is not None for v in vals):
                self.ax_metric.plot(epochs, vals, label=label)
        self.ax_metric.legend()
        self.ax_metric.grid(True, alpha=0.3)

        self.canvas.draw()

        # Update progress bar from last epoch
        if epochs:
            self.progress_bar.setValue(epochs[-1])

    def _on_finished(self, result: dict):
        if self._timer:
            self._timer.stop()
        self._update_charts()

        if result.get("success"):
            self.progress_bar.setValue(self.epochs)
            self.log_output.append("\n=== 训练完成 ===")
            metrics = result.get("final_metrics", {})
            for k, v in metrics.items():
                self.log_output.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

            best = result.get("best_model")
            if best:
                self.log_output.append(f"\n最佳模型: {best}")

            # Ask to export
            reply = QMessageBox.question(
                self, "训练完成",
                "训练已完成！是否导出 ONNX 模型？",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes and best:
                self._export_worker = _ExportWorker(
                    best, ["onnx"],
                    self.config.get("imgsz", 640),
                    lambda msg: self._on_log(msg),
                )
                self._export_worker.log.connect(self._on_log)
                self._export_worker.finished.connect(self._on_export_done)
                self._on_log("正在导出 ONNX 模型...")
                self._export_worker.start()
        else:
            self.log_output.append(f"\n=== 训练失败 ===")
            self.log_output.append(result.get("error", "未知错误"))

        self.btn_stop.setEnabled(False)
        self.btn_stop.setText("已结束")
        self.btn_close.setEnabled(True)

    def _on_export_done(self, exported):
        for fmt, path in exported.items():
            self._on_log(f"导出 {fmt}: {path}")
        if not exported:
            self._on_log("导出失败或无输出文件")

    def _stop_training(self):
        if self._worker:
            self._worker.cancel()
            self._on_log("正在停止训练...")
            self.btn_stop.setEnabled(False)

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            reply = QMessageBox.question(
                self, "确认",
                "训练正在进行中，确定要关闭吗？",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._worker.cancel()
                self._timer.stop()
                self._worker.wait(3000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
