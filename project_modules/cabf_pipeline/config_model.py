from __future__ import annotations

import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_train_modules_root() -> str:
    return str(_repo_root() / "project_modules" / "train_model_modules")


FIELDS = [
    ("img_tools_root", "img_tools 根目录", "dir"),
    ("train_model_modules_root", "train_model/modules 目录", "dir"),
    ("dataset_root", "数据集根目录", "dir"),
    ("pending_filter_dir", "待筛选目录", "dir"),
    ("filtered_keep_dir", "筛选保留目录", "dir"),
    ("master_images_dir", "母图目录", "dir"),
    ("master_annotations_dir", "母标注目录", "dir"),
    ("point_predictions_dir", "点预测目录", "dir"),
    ("edge_predictions_dir", "边预测目录", "dir"),
    ("model_a_export_root", "模型A导出根目录", "dir"),
    ("model_b_export_root", "模型B导出根目录", "dir"),
    ("weights.sew_point_onnx", "点模型 ONNX", "file"),
    ("weights.sew_point_connector_pth", "连边模型 PTH", "file"),
    ("outputs.sew_point_train_out", "点模型训练输出", "dir"),
    ("outputs.sew_point_conntect_train_out", "连边模型训练输出", "dir"),
]

DEFAULT_CONFIG = {
    "img_tools_root": str(_repo_root()),
    "train_model_modules_root": default_train_modules_root(),
    "dataset_root": "",
    "pending_filter_dir": "",
    "filtered_keep_dir": "",
    "master_images_dir": "",
    "master_annotations_dir": "",
    "point_predictions_dir": "",
    "edge_predictions_dir": "",
    "model_a_export_root": "",
    "model_b_export_root": "",
    "weights": {
        "sew_point_onnx": "",
        "sew_point_connector_pth": "",
    },
    "predict": {
        "point_threshold": "0.3",
        "point_distance_threshold": "3.0",
    },
    "outputs": {
        "sew_point_train_out": "",
        "sew_point_conntect_train_out": "",
    },
}


def _deep_merge(defaults: dict, overrides: dict) -> dict:
    result = dict(defaults)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def apply_defaults(data: dict) -> dict:
    return _deep_merge(DEFAULT_CONFIG, data)


def load_config(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    return apply_defaults(data)


def save_config(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_nested(data: dict, dotted_key: str) -> str:
    current = data
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    value = current.get(parts[-1], "")
    return str(value)


def set_nested(data: dict, dotted_key: str, value: str) -> None:
    current = data
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value

