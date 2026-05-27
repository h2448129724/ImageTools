"""CAB-F master annotation schema, validation, and export helpers."""
from __future__ import annotations

import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
MASTER_SCHEMA_VERSION = "1.2"
POINT_LABEL_ALIASES = {"sew", "keypoint"}


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_image_size(image_path: str | Path) -> tuple[int, int]:
    image_path = Path(image_path)
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    h, w = image.shape[:2]
    return int(w), int(h)


def iter_image_files(image_dir: str | Path) -> list[Path]:
    folder = Path(image_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"图片目录不存在: {folder}")
    files = []
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            files.append(path)
    return files


def iter_json_files(annotation_dir: str | Path) -> list[Path]:
    folder = Path(annotation_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"标注目录不存在: {folder}")
    return sorted([path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".json"])


def make_empty_master_annotation(image_path: str, width: int, height: int, sample_id: str) -> dict:
    return {
        "schema_version": MASTER_SCHEMA_VERSION,
        "sample_id": sample_id,
        "image_path": image_path,
        "image_size": {"width": int(width), "height": int(height)},
        "roi": None,
        "spacing_hint": None,
        "points": [],
        "edges": [],
        "segments": [],
        "metadata": {},
    }


def is_labelme_point_annotation(data: dict) -> bool:
    if not isinstance(data, dict):
        return False
    if "shapes" not in data:
        return False
    shapes = data.get("shapes")
    return isinstance(shapes, list)


def load_labelme_points(data: dict) -> list[dict]:
    shapes = data.get("shapes", [])
    points = []
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        if shape.get("shape_type") != "point":
            continue
        if str(shape.get("label", "")).strip().lower() not in POINT_LABEL_ALIASES:
            continue
        raw_points = shape.get("points", [])
        if not raw_points or len(raw_points[0]) < 2:
            continue
        xy = raw_points[0]
        points.append(
            {
                "x": float(xy[0]),
                "y": float(xy[1]),
                "score": float(shape.get("score", 1.0) or 1.0),
                "source": "labelme_point",
            }
        )
    return points


def convert_labelme_to_master(data: dict, image_path: str, sample_id: str) -> dict:
    width = int(data.get("imageWidth", 0) or 256)
    height = int(data.get("imageHeight", 0) or 256)
    annotation = make_empty_master_annotation(image_path=image_path, width=width, height=height, sample_id=sample_id)
    points = load_labelme_points(data)
    annotation["points"] = [
        {
            "id": idx,
            "x": float(point["x"]),
            "y": float(point["y"]),
            "score": float(point.get("score", 1.0)),
            "source": point.get("source", "labelme_point"),
        }
        for idx, point in enumerate(points)
    ]
    annotation["metadata"] = {
        "source": "labelme_point",
        "origin_format": "labelme",
    }
    return annotation


def _normalize_points(raw_points: list[dict]) -> tuple[list[dict], dict[int, int], list[str]]:
    issues: list[str] = []
    normalized = []
    used_ids: set[int] = set()
    old_to_new: dict[int, int] = {}
    next_id = 0

    for idx, point in enumerate(raw_points):
        if not isinstance(point, dict):
            issues.append(f"point[{idx}] 不是对象，已跳过")
            continue
        try:
            old_id = int(point.get("id", idx))
            x = float(point["x"])
            y = float(point["y"])
        except Exception:
            issues.append(f"point[{idx}] 缺少有效的 id/x/y，已跳过")
            continue
        if old_id in used_ids:
            issues.append(f"point id {old_id} 重复，已重排")
            old_id = next_id
        while old_id in used_ids:
            old_id += 1
        used_ids.add(old_id)
        old_to_new[int(point.get("id", idx))] = old_id
        normalized.append(
            {
                "id": old_id,
                "x": x,
                "y": y,
                "score": float(point.get("score", 1.0)),
                "source": point.get("source", "manual"),
            }
        )
        next_id = max(next_id, old_id + 1)

    normalized.sort(key=lambda item: int(item["id"]))
    id_remap = {int(point["id"]): idx for idx, point in enumerate(normalized)}
    final_points = []
    for idx, point in enumerate(normalized):
        final_points.append({**point, "id": idx})
        old_to_new[int(point["id"])] = idx
        id_remap[int(point["id"])] = idx
    return final_points, old_to_new, issues


def _normalize_edges(raw_edges: list[dict], id_remap: dict[int, int]) -> tuple[list[dict], list[str]]:
    issues: list[str] = []
    normalized = []
    seen: set[tuple[int, int]] = set()

    for idx, edge in enumerate(raw_edges):
        if not isinstance(edge, dict):
            issues.append(f"edge[{idx}] 不是对象，已跳过")
            continue
        try:
            src = int(edge["src"])
            dst = int(edge["dst"])
        except Exception:
            issues.append(f"edge[{idx}] 缺少有效的 src/dst，已跳过")
            continue
        if src not in id_remap or dst not in id_remap:
            issues.append(f"edge[{idx}] 引用了不存在的点 {src}/{dst}，已跳过")
            continue
        src = id_remap[src]
        dst = id_remap[dst]
        if src == dst:
            issues.append(f"edge[{idx}] 是自环，已跳过")
            continue
        key = tuple(sorted((src, dst)))
        if key in seen:
            issues.append(f"edge[{idx}] 与已有边重复，已去重")
            continue
        seen.add(key)
        normalized.append(
            {
                "edge_id": str(edge.get("edge_id") or f"edge_{len(normalized) + 1:04d}"),
                "src": int(key[0]),
                "dst": int(key[1]),
                "label": int(edge.get("label", 1)),
                "source": edge.get("source", "manual"),
            }
        )
    return normalized, issues


def normalize_master_annotation(data: dict, *, sample_id: str | None = None, image_path: str | None = None) -> tuple[dict, list[str]]:
    issues: list[str] = []
    if is_labelme_point_annotation(data):
        sample_id = sample_id or Path(str(image_path or data.get("imagePath", ""))).stem or "sample"
        data = convert_labelme_to_master(data, image_path=image_path or str(data.get("imagePath", "")), sample_id=sample_id)

    if not isinstance(data, dict):
        raise ValueError("标注内容不是有效 JSON 对象")

    width = int(data.get("image_size", {}).get("width", 0) or 256)
    height = int(data.get("image_size", {}).get("height", 0) or 256)
    normalized = make_empty_master_annotation(
        image_path=str(image_path or data.get("image_path", "")),
        width=width,
        height=height,
        sample_id=str(sample_id or data.get("sample_id", Path(str(image_path or "sample")).stem)),
    )
    normalized["schema_version"] = str(data.get("schema_version", MASTER_SCHEMA_VERSION) or MASTER_SCHEMA_VERSION)
    normalized["roi"] = data.get("roi")
    normalized["spacing_hint"] = data.get("spacing_hint")
    normalized["segments"] = data.get("segments", []) if isinstance(data.get("segments", []), list) else []
    normalized["metadata"] = data.get("metadata", {}) if isinstance(data.get("metadata", {}), dict) else {}

    points, id_remap, point_issues = _normalize_points(data.get("points", []) if isinstance(data.get("points", []), list) else [])
    edges, edge_issues = _normalize_edges(data.get("edges", []) if isinstance(data.get("edges", []), list) else [], id_remap)
    issues.extend(point_issues)
    issues.extend(edge_issues)
    normalized["points"] = points
    normalized["edges"] = edges
    normalized["schema_version"] = MASTER_SCHEMA_VERSION
    if not normalized["image_path"]:
        normalized["image_path"] = f"{normalized['sample_id']}.png"
        issues.append("image_path 缺失，已回填为 sample_id.png")
    return normalized, issues


def master_to_labelme(master_annotation: dict) -> dict:
    image_path = str(master_annotation.get("image_path", ""))
    width = int(master_annotation.get("image_size", {}).get("width", 256) or 256)
    height = int(master_annotation.get("image_size", {}).get("height", 256) or 256)
    shapes = []
    for point in master_annotation.get("points", []):
        shapes.append(
            {
                "label": "sew",
                "points": [[float(point["x"]), float(point["y"])]],
                "group_id": None,
                "description": "",
                "shape_type": "point",
                "flags": {},
                "score": float(point.get("score", 1.0)),
            }
        )
    return {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def collect_stem_maps(image_dir: str | Path, annotation_dir: str | Path) -> tuple[dict[str, Path], dict[str, Path]]:
    image_map = {path.stem: path for path in iter_image_files(image_dir)}
    json_map = {path.stem: path for path in iter_json_files(annotation_dir)}
    return image_map, json_map


def _inspect_master_sample(stem: str, image_path: Path, json_path: Path | None) -> tuple[dict, dict | None]:
    if json_path is None or not json_path.exists():
        return (
            {
                "sample_id": stem,
                "image_path": str(image_path),
                "json_path": str(json_path) if json_path else "",
                "point_count": 0,
                "edge_count": 0,
                "errors": ["缺少标注文件"],
                "warnings": [],
            },
            None,
        )

    try:
        raw = read_json(json_path)
    except Exception as exc:
        return (
            {
                "sample_id": stem,
                "image_path": str(image_path),
                "json_path": str(json_path),
                "point_count": 0,
                "edge_count": 0,
                "errors": [f"JSON 读取失败: {exc}"],
                "warnings": [],
            },
            None,
        )

    try:
        image_w, image_h = read_image_size(image_path)
    except Exception as exc:
        return (
            {
                "sample_id": stem,
                "image_path": str(image_path),
                "json_path": str(json_path),
                "point_count": 0,
                "edge_count": 0,
                "errors": [f"图片读取失败: {exc}"],
                "warnings": [],
            },
            None,
        )

    issues: list[str] = []
    warnings: list[str] = []
    normalized, normalize_issues = normalize_master_annotation(raw, sample_id=stem, image_path=image_path.name)
    issues.extend(normalize_issues)

    ann_w = int(normalized.get("image_size", {}).get("width", 0) or 0)
    ann_h = int(normalized.get("image_size", {}).get("height", 0) or 0)
    if ann_w != image_w or ann_h != image_h:
        warnings.append(f"image_size 与实际图片不一致: 标注=({ann_w},{ann_h}) 实际=({image_w},{image_h})")

    points = normalized["points"]
    edges = normalized["edges"]
    degree = Counter()
    for edge in edges:
        degree[int(edge["src"])] += 1
        degree[int(edge["dst"])] += 1
    overflow_points = sorted([point_id for point_id, deg in degree.items() if deg > 2])
    if overflow_points:
        warnings.append(f"{len(overflow_points)} 个点的度数超过 2: {overflow_points[:10]}")

    return (
        {
            "sample_id": stem,
            "image_path": str(image_path),
            "json_path": str(json_path),
            "point_count": len(points),
            "edge_count": len(edges),
            "errors": issues,
            "warnings": warnings,
        },
        normalized,
    )


def validate_master_dataset(image_dir: str | Path, annotation_dir: str | Path) -> dict:
    image_map, json_map = collect_stem_maps(image_dir, annotation_dir)
    missing_annotations = sorted(set(image_map) - set(json_map))
    orphan_annotations = sorted(set(json_map) - set(image_map))
    sample_reports = []
    stats = Counter()
    point_count_hist = Counter()
    edge_count_hist = Counter()

    for stem in sorted(set(image_map) & set(json_map)):
        image_path = image_map[stem]
        json_path = json_map[stem]
        sample_report, _ = _inspect_master_sample(stem, image_path, json_path)
        if any(msg.startswith("JSON 读取失败:") for msg in sample_report.get("errors", [])):
            stats["invalid_json"] += 1
        if any(msg.startswith("图片读取失败:") for msg in sample_report.get("errors", [])):
            stats["invalid_image"] += 1
        if sample_report.get("warnings") and any("度数超过 2" in msg for msg in sample_report["warnings"]):
            stats["degree_overflow_samples"] += 1

        point_count = int(sample_report.get("point_count", 0) or 0)
        edge_count = int(sample_report.get("edge_count", 0) or 0)
        point_count_hist[min(point_count, 20)] += 1
        edge_count_hist[min(edge_count, 20)] += 1
        if point_count == 0:
            stats["zero_point_samples"] += 1
        if point_count == 1:
            stats["one_point_samples"] += 1
        if edge_count == 0:
            stats["zero_edge_samples"] += 1
        if edge_count > 0:
            stats["samples_with_edges"] += 1
        if point_count > 0:
            stats["samples_with_points"] += 1

        sample_reports.append(sample_report)

    stats["num_images"] = len(image_map)
    stats["num_annotations"] = len(json_map)
    stats["paired_samples"] = len(set(image_map) & set(json_map))
    stats["missing_annotations"] = len(missing_annotations)
    stats["orphan_annotations"] = len(orphan_annotations)
    stats["samples_with_errors"] = sum(1 for item in sample_reports if item.get("errors"))
    stats["samples_with_warnings"] = sum(1 for item in sample_reports if item.get("warnings"))
    zero_point_samples = sorted(item["sample_id"] for item in sample_reports if item.get("point_count") == 0)
    one_point_samples = sorted(item["sample_id"] for item in sample_reports if item.get("point_count") == 1)
    zero_edge_samples = sorted(item["sample_id"] for item in sample_reports if item.get("edge_count") == 0)
    error_samples = sorted(item["sample_id"] for item in sample_reports if item.get("errors"))
    warning_samples = sorted(item["sample_id"] for item in sample_reports if item.get("warnings"))

    return {
        "schema_version": MASTER_SCHEMA_VERSION,
        "image_dir": str(image_dir),
        "annotation_dir": str(annotation_dir),
        "summary": dict(stats),
        "missing_annotations": missing_annotations,
        "orphan_annotations": orphan_annotations,
        "zero_point_samples": zero_point_samples,
        "one_point_samples": one_point_samples,
        "zero_edge_samples": zero_edge_samples,
        "error_samples": error_samples,
        "warning_samples": warning_samples,
        "point_count_histogram": dict(point_count_hist),
        "edge_count_histogram": dict(edge_count_hist),
        "samples": sample_reports,
    }


def _copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _resolve_export_dirs(output_dir: str | Path) -> tuple[Path, Path, Path]:
    output_root = Path(output_dir)
    return output_root / "images", output_root / "annotations", output_root / "error"


def _route_sample_to_error(image_path: Path, json_path: Path | None, error_dir: Path) -> None:
    _copy_image(image_path, error_dir / image_path.name)
    if json_path and json_path.exists():
        _copy_image(json_path, error_dir / json_path.name)


def _load_master_from_pair(image_path: Path, json_path: Path | None) -> dict:
    width, height = read_image_size(image_path)
    if json_path is None or not json_path.exists():
        return make_empty_master_annotation(image_path=image_path.name, width=width, height=height, sample_id=image_path.stem)
    raw = read_json(json_path)
    normalized, _ = normalize_master_annotation(raw, sample_id=image_path.stem, image_path=image_path.name)
    normalized["image_size"] = {"width": width, "height": height}
    normalized["image_path"] = image_path.name
    return normalized


def export_master_to_model_a(
    image_dir: str | Path,
    annotation_dir: str | Path,
    output_dir: str | Path,
    *,
    include_empty: bool = True,
) -> dict:
    image_map, json_map = collect_stem_maps(image_dir, annotation_dir)
    output_image_dir, output_annotation_dir, error_dir = _resolve_export_dirs(output_dir)
    report = Counter()

    for stem, image_path in sorted(image_map.items()):
        json_path = json_map.get(stem)
        sample_report, master = _inspect_master_sample(stem, image_path, json_path)
        if sample_report.get("errors") or sample_report.get("warnings") or sample_report.get("point_count", 0) <= 1 or sample_report.get("edge_count", 0) == 0:
            _route_sample_to_error(image_path, json_path, error_dir)
            report["samples_routed_to_error"] += 1
            continue
        assert master is not None
        if not include_empty and not master.get("points"):
            report["skipped_empty_annotations"] += 1
            continue
        _copy_image(image_path, output_image_dir / image_path.name)
        report["images_exported"] += 1
        labelme = master_to_labelme(master)
        write_json(output_annotation_dir / f"{stem}.json", labelme)
        report["annotations_exported"] += 1

    result = dict(report)
    result.update(
        {
            "output_dir": str(Path(output_dir)),
            "images_dir": str(output_image_dir),
            "annotations_dir": str(output_annotation_dir),
            "error_dir": str(error_dir),
        }
    )
    return result


def export_master_to_model_b(
    image_dir: str | Path,
    annotation_dir: str | Path,
    output_dir: str | Path,
    *,
    include_empty: bool = True,
) -> dict:
    image_map, json_map = collect_stem_maps(image_dir, annotation_dir)
    output_image_dir, output_annotation_dir, error_dir = _resolve_export_dirs(output_dir)
    report = Counter()

    for stem, image_path in sorted(image_map.items()):
        json_path = json_map.get(stem)
        sample_report, master = _inspect_master_sample(stem, image_path, json_path)
        if sample_report.get("errors") or sample_report.get("warnings") or sample_report.get("point_count", 0) <= 1 or sample_report.get("edge_count", 0) == 0:
            _route_sample_to_error(image_path, json_path, error_dir)
            report["samples_routed_to_error"] += 1
            continue
        assert master is not None
        if not include_empty and not master.get("points"):
            report["skipped_empty_annotations"] += 1
            continue
        _copy_image(image_path, output_image_dir / image_path.name)
        report["images_exported"] += 1
        write_json(output_annotation_dir / f"{stem}.json", master)
        report["annotations_exported"] += 1

    result = dict(report)
    result.update(
        {
            "output_dir": str(Path(output_dir)),
            "images_dir": str(output_image_dir),
            "annotations_dir": str(output_annotation_dir),
            "error_dir": str(error_dir),
        }
    )
    return result


def summarize_validation(report: dict) -> str:
    summary = report.get("summary", {})
    lines = [
        f"images={summary.get('num_images', 0)} annotations={summary.get('num_annotations', 0)} paired={summary.get('paired_samples', 0)}",
        f"missing_annotations={summary.get('missing_annotations', 0)} orphan_annotations={summary.get('orphan_annotations', 0)}",
        f"zero_point={summary.get('zero_point_samples', 0)} one_point={summary.get('one_point_samples', 0)} zero_edge={summary.get('zero_edge_samples', 0)}",
        f"samples_with_errors={summary.get('samples_with_errors', 0)} samples_with_warnings={summary.get('samples_with_warnings', 0)}",
    ]
    return "\n".join(lines)


def summarize_validation_findings(report: dict, *, include_details: bool = False) -> str:
    sample_map = {item.get("sample_id"): item for item in report.get("samples", []) if item.get("sample_id")}
    sample_types: defaultdict[str, list[str]] = defaultdict(list)
    category_labels = [
        ("missing_annotations", "missing_annotation"),
        ("orphan_annotations", "orphan_annotation"),
        ("error_samples", "error"),
        ("warning_samples", "warning"),
        ("zero_point_samples", "zero_point"),
        ("one_point_samples", "one_point"),
        ("zero_edge_samples", "zero_edge"),
    ]
    for key, label in category_labels:
        for sample_id in report.get(key, []):
            sample_types[sample_id].append(label)

    lines: list[str] = []
    for sample_id in sorted(sample_types):
        labels = ", ".join(dict.fromkeys(sample_types[sample_id]))
        lines.append(f"[{sample_id}] {labels}")
        if not include_details:
            continue
        sample = sample_map.get(sample_id, {})
        if sample:
            lines.append(f"  counts: points={sample.get('point_count', 0)} edges={sample.get('edge_count', 0)}")
            for issue in sample.get("errors", []):
                lines.append(f"  error: {issue}")
            for warning in sample.get("warnings", []):
                lines.append(f"  warning: {warning}")
        elif sample_id in report.get("missing_annotations", []):
            lines.append("  error: 缺少标注文件")
        elif sample_id in report.get("orphan_annotations", []):
            lines.append("  error: 缺少对应图片文件")
    return "\n".join(lines)
