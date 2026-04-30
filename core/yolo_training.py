"""YOLO training wrapper and utilities for the image tools toolbox.

Soft dependency on `ultralytics` — import at runtime with graceful fallback.
"""
from __future__ import annotations

import csv
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


def _check_ultralytics():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError:
        raise RuntimeError(
            "未安装 ultralytics，请执行: pip install ultralytics"
        )


def run_yolo_training(
    config: dict,
    log_fn: Callable[[str], None] | None = None,
    progress_callback: Callable | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict:
    """Run YOLO training with the given configuration.

    Returns a dict with keys:
        - success: bool
        - project: str
        - name: str
        - best_model: str | None
        - epochs_completed: int
        - final_metrics: dict
        - error: str | None
    """
    YOLO = _check_ultralytics()

    data = config.get("data", "")
    model_name = config.get("model", "yolov8n.pt")
    epochs = config.get("epochs", 100)
    imgsz = config.get("imgsz", 640)
    batch = config.get("batch", 16)
    workers = config.get("workers", 8)
    lr0 = config.get("lr0", 0.01)
    lrf = config.get("lrf", 0.01)
    optimizer = config.get("optimizer", "SGD")
    device = config.get("device", "")
    patience = config.get("patience", 50)
    seed = config.get("seed", 0)
    project = config.get("project", "runs")
    name = config.get("name", "detect")
    augment = config.get("augment", False)

    if not data or not os.path.exists(data):
        return {"success": False, "error": f"数据集 YAML 文件不存在: {data}"}

    if log_fn:
        log_fn(f"开始训练: model={model_name}, data={data}, epochs={epochs}, imgsz={imgsz}")

    try:
        model = YOLO(model_name)
        kwargs = {
            "data": data,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "workers": workers,
            "lr0": lr0,
            "lrf": lrf,
            "optimizer": optimizer,
            "patience": patience,
            "seed": seed,
            "project": project,
            "name": name,
            "augment": augment,
        }
        if device:
            kwargs["device"] = device

        # Train
        results = model.train(**kwargs)

        # Determine best model path
        run_dir = os.path.join(project, name)
        best_model = os.path.join(run_dir, "weights", "best.pt")
        if not os.path.exists(best_model):
            best_model = None

        # Parse final metrics from results.csv
        final_metrics = _parse_latest_metrics(run_dir)

        if log_fn:
            log_fn(f"训练完成: {run_dir}")
            if best_model:
                log_fn(f"最佳模型: {best_model}")

        return {
            "success": True,
            "project": project,
            "name": name,
            "best_model": best_model,
            "epochs_completed": final_metrics.get("epoch", epochs),
            "final_metrics": final_metrics,
            "error": None,
        }

    except KeyboardInterrupt:
        raise
    except Exception as e:
        if log_fn:
            log_fn(f"训练出错: {e}")
        return {"success": False, "error": str(e)}


def _parse_latest_metrics(run_dir: str) -> dict:
    """Read the last row of results.csv to get final metrics."""
    csv_path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(csv_path):
        return {}

    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                return {k.strip(): float(v) for k, v in rows[-1].items()}
    except (OSError, ValueError) as e:
        logger.warning("Failed to parse metrics from %s: %s", csv_path, e)
    return {}


def parse_training_logs(run_dir: str) -> list[dict]:
    """Parse all rows from results.csv as a list of metric dicts."""
    csv_path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(csv_path):
        return []

    logs = []
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                logs.append({k.strip(): float(v) for k, v in row.items()})
    except (OSError, ValueError) as e:
        logger.warning("Failed to parse training logs from %s: %s", csv_path, e)
    return logs


def get_training_history(project_dir: str = "runs") -> list[dict]:
    """Scan project directory for completed training runs.

    Returns a list of dicts with keys:
        - run_dir: str
        - name: str
        - model: str | None
        - epochs_completed: int
        - final_metrics: dict
        - created_time: float
    """
    history = []
    if not os.path.isdir(project_dir):
        return history

    for subdir in os.listdir(project_dir):
        run_path = os.path.join(project_dir, subdir)
        if not os.path.isdir(run_path):
            continue

        results_csv = os.path.join(run_path, "results.csv")
        if not os.path.exists(results_csv):
            continue

        metrics = _parse_latest_metrics(run_path)
        args_yaml = os.path.join(run_path, "args.yaml")
        model_name = None
        if os.path.exists(args_yaml):
            try:
                import yaml
                with open(args_yaml, "r", encoding="utf-8") as f:
                    args = yaml.safe_load(f)
                model_name = args.get("model", None)
            except (ImportError, yaml.YAMLError, OSError) as e:
                logger.warning("Failed to parse args.yaml in %s: %s", run_path, e)

        history.append({
            "run_dir": run_path,
            "name": subdir,
            "model": model_name,
            "epochs_completed": int(metrics.get("epoch", 0)),
            "final_metrics": metrics,
            "created_time": os.path.getctime(run_path),
        })

    history.sort(key=lambda x: x["created_time"], reverse=True)
    return history


def export_after_training(
    model_path: str,
    formats: list[str],
    imgsz: int = 640,
    simplify: bool = True,
    dynamic: bool = False,
    half: bool = False,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, str]:
    """Export a trained model to specified formats.

    Supported formats: onnx, tensorrt, engine, openvino, coreml, tflite.
    Returns a dict mapping format to exported file path.
    """
    YOLO = _check_ultralytics()

    if not os.path.exists(model_path):
        if log_fn:
            log_fn(f"模型文件不存在: {model_path}")
        return {}

    exported = {}
    model = YOLO(model_path)

    fmt_map = {
        "onnx": "onnx",
        "tensorrt": "engine",
        "engine": "engine",
        "openvino": "openvino",
        "coreml": "coreml",
        "tflite": "tflite",
    }

    for fmt in formats:
        ultralytics_fmt = fmt_map.get(fmt.lower())
        if not ultralytics_fmt:
            if log_fn:
                log_fn(f"不支持的导出格式: {fmt}")
            continue

        try:
            if log_fn:
                log_fn(f"正在导出 {fmt}...")
            result = model.export(
                format=ultralytics_fmt,
                imgsz=imgsz,
                simplify=simplify,
                dynamic=dynamic,
                half=half,
            )
            exported[fmt] = result
            if log_fn:
                log_fn(f"导出完成: {result}")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if log_fn:
                log_fn(f"导出 {fmt} 失败: {e}")

    return exported


def delete_training_run(run_dir: str) -> bool:
    """Move a training run directory to trash/recycle bin."""
    try:
        import send2trash
        send2trash.send2trash(run_dir)
        return True
    except ImportError:
        shutil.rmtree(run_dir, ignore_errors=True)
        return True
    except OSError as e:
        logger.error("Failed to delete training run %s: %s", run_dir, e)
        return False
