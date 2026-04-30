"""Training results manager: browse history, compare runs, export models."""
import os
from datetime import datetime
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QTableWidget, QTableWidgetItem, QMessageBox, QSplitter,
    QWidget, QHeaderView, QGroupBox, QGridLayout,
)
from PySide6.QtCore import Qt, QThread, Signal

from core.yolo_training import get_training_history, export_after_training, delete_training_run

# Optional matplotlib
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


class TrainingResultsDialog(QDialog):
    """Dialog to browse, compare, and export training runs."""

    def __init__(self, project_dir: str = "runs", parent=None):
        super().__init__(parent)
        self.project_dir = project_dir
        self.setWindowTitle("YOLO 训练结果管理")
        self.resize(1100, 700)
        self._setup_ui()
        self._load_history()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Top info
        info = QLabel("浏览历史训练结果，选择多个进行指标对比，或导出模型。")
        info.setStyleSheet("color: #666;")
        layout.addWidget(info)

        # Splitter: list on left, detail on right
        splitter = QSplitter(Qt.Horizontal)

        # Left: run list
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_layout.addWidget(QLabel("训练记录 (按时间倒序):"))
        self.run_list = QListWidget()
        self.run_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.run_list.itemSelectionChanged.connect(self._on_selection_changed)
        left_layout.addWidget(self.run_list)

        btn_refresh = QPushButton("刷新")
        btn_refresh.clicked.connect(self._load_history)
        left_layout.addWidget(btn_refresh)

        splitter.addWidget(left)

        # Right: detail panel
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Metrics table
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(5)
        self.metrics_table.setHorizontalHeaderLabels([
            "指标", "数值", "", "", ""
        ])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        right_layout.addWidget(self.metrics_table)

        # Chart area (optional)
        if _HAS_MPL:
            self.figure = Figure(figsize=(6, 3), dpi=100)
            self.canvas = FigureCanvas(self.figure)
            self.ax = self.figure.add_subplot(111)
            self.ax.set_title("mAP 对比")
            self.ax.set_xlabel("Epoch")
            self.figure.tight_layout()
            right_layout.addWidget(self.canvas)
        else:
            right_layout.addWidget(QLabel("安装 matplotlib 可启用图表对比"))

        # Action buttons
        btn_row = QHBoxLayout()
        self.btn_export = QPushButton("导出 ONNX")
        self.btn_export.clicked.connect(self._export_onnx)
        self.btn_export.setEnabled(False)
        btn_row.addWidget(self.btn_export)

        self.btn_delete = QPushButton("删除选中")
        self.btn_delete.setStyleSheet("QPushButton { color: #e74c3c; }")
        self.btn_delete.clicked.connect(self._delete_selected)
        btn_row.addWidget(self.btn_delete)

        btn_row.addStretch()
        right_layout.addLayout(btn_row)

        splitter.addWidget(right)
        splitter.setSizes([300, 700])
        layout.addWidget(splitter)

    def _load_history(self):
        self.run_list.clear()
        history = get_training_history(self.project_dir)

        for run in history:
            name = run["name"]
            model = run.get("model", "未知")
            epochs = run.get("epochs_completed", 0)
            map50 = run.get("final_metrics", {}).get("metrics/mAP50(B)", 0)
            map50_95 = run.get("final_metrics", {}).get("metrics/mAP50-95(B)", 0)
            created = datetime.fromtimestamp(run["created_time"]).strftime("%Y-%m-%d %H:%M")

            text = f"{name}  |  {model}  |  {epochs}ep  |  mAP50={map50:.3f}  |  {created}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, run)
            self.run_list.addItem(item)

    def _on_selection_changed(self):
        items = self.run_list.selectedItems()
        self.btn_export.setEnabled(len(items) == 1)

        if not items:
            self.metrics_table.setRowCount(0)
            if _HAS_MPL:
                self.ax.clear()
                self.canvas.draw()
            return

        # Show metrics for the first selected
        run = items[0].data(Qt.UserRole)
        metrics = run.get("final_metrics", {})

        key_labels = [
            ("epoch", "Epoch"),
            ("metrics/precision(B)", "Precision"),
            ("metrics/recall(B)", "Recall"),
            ("metrics/mAP50(B)", "mAP@50"),
            ("metrics/mAP50-95(B)", "mAP@50-95"),
            ("train/box_loss", "Box Loss"),
            ("train/cls_loss", "Cls Loss"),
            ("val/box_loss", "Val Box Loss"),
            ("val/cls_loss", "Val Cls Loss"),
        ]

        self.metrics_table.setRowCount(len(key_labels))
        for i, (key, label) in enumerate(key_labels):
            val = metrics.get(key, "—")
            if isinstance(val, float):
                val = f"{val:.4f}"
            self.metrics_table.setItem(i, 0, QTableWidgetItem(label))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(str(val)))

        # Chart comparison
        if _HAS_MPL:
            self._update_comparison_chart([item.data(Qt.UserRole) for item in items])

    def _update_comparison_chart(self, runs: list):
        self.ax.clear()
        self.ax.set_title("mAP@50 对比")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("mAP@50")

        from core.yolo_training import parse_training_logs

        for run in runs:
            logs = parse_training_logs(run["run_dir"])
            if not logs:
                continue
            epochs = [int(r.get("epoch", i + 1)) for i, r in enumerate(logs)]
            vals = [r.get("metrics/mAP50(B)") for r in logs]
            self.ax.plot(epochs, vals, label=run["name"])

        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def _export_onnx(self):
        items = self.run_list.selectedItems()
        if len(items) != 1:
            return
        run = items[0].data(Qt.UserRole)
        best_model = os.path.join(run["run_dir"], "weights", "best.pt")
        if not os.path.exists(best_model):
            QMessageBox.warning(self, "提示", "未找到 best.pt 模型文件")
            return

        self.btn_export.setEnabled(False)
        self.btn_export.setText("导出中...")

        class _ExportWorker(QThread):
            done = Signal(dict)
            def __init__(self, model_path):
                super().__init__()
                self._model_path = model_path
            def run(self):
                result = export_after_training(self._model_path, ["onnx"])
                self.done.emit(result)

        self._export_worker = _ExportWorker(best_model)
        self._export_worker.done.connect(self._on_export_done)
        self._export_worker.start()

    def _on_export_done(self, exported):
        self.btn_export.setEnabled(True)
        self.btn_export.setText("导出 ONNX")
        if exported:
            QMessageBox.information(self, "导出成功", f"ONNX 模型已导出到:\n{exported.get('onnx', '')}")
        else:
            QMessageBox.critical(self, "导出失败", "请检查 ultralytics 是否已安装")

    def _delete_selected(self):
        items = self.run_list.selectedItems()
        if not items:
            return
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要将选中的 {len(items)} 项训练记录移入回收站吗？",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        for item in items:
            run = item.data(Qt.UserRole)
            if delete_training_run(run["run_dir"]):
                self.run_list.takeItem(self.run_list.row(item))
