from __future__ import annotations

import argparse
from datetime import datetime
import json
import shlex
import subprocess
import sys
from pathlib import Path

from project_modules.cabf_pipeline.config_model import load_config


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config" / "default_paths.json"
LOG_DIR = SCRIPT_DIR / "logs"


def quote_args(args: list[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def append_log(line: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{datetime.now():%Y%m%d}.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_command(args: list[str], cwd: str, dry_run: bool) -> int:
    timestamp = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    cwd_line = f"[{timestamp}] [cwd] {cwd}"
    cmd_line = f"[{timestamp}] [cmd] {quote_args(args)}"
    print(cwd_line)
    print(cmd_line)
    append_log(cwd_line)
    append_log(cmd_line)
    if dry_run:
        append_log(f"[{timestamp}] [result] dry-run")
        return 0
    completed = subprocess.run(args, cwd=cwd, check=False)
    append_log(f"[{timestamp}] [result] exit={completed.returncode}")
    return int(completed.returncode)


def _count_labelme_points(data: dict) -> int:
    shapes = data.get("shapes", []) if isinstance(data, dict) else []
    count = 0
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        if shape.get("shape_type") != "point":
            continue
        if str(shape.get("label", "")).strip() != "sew":
            continue
        raw_points = shape.get("points", [])
        if raw_points and len(raw_points[0]) >= 2:
            count += 1
    return count


def _has_usable_point_json(path_text: str) -> bool:
    path = Path(path_text)
    if not path.is_dir():
        return False
    for json_path in path.glob("*.json"):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            if len(data.get("points", []) or []) >= 2:
                return True
            if _count_labelme_points(data) >= 2:
                return True
    return False


def _resolve_edge_annotation_dir(cfg: dict, requested: str) -> tuple[str, str | None]:
    if requested and _has_usable_point_json(requested):
        return requested, None
    if requested and Path(requested).is_dir():
        fallback = str(cfg.get("point_predictions_dir", "") or "")
        if fallback and fallback != requested and _has_usable_point_json(fallback):
            return fallback, f"annotation_dir fallback: {requested} -> {fallback}"
        return requested, None

    master_dir = str(cfg.get("master_annotations_dir", "") or "")
    if _has_usable_point_json(master_dir):
        return master_dir, None

    fallback = str(cfg.get("point_predictions_dir", "") or "")
    if _has_usable_point_json(fallback):
        return fallback, f"annotation_dir fallback: {master_dir or requested} -> {fallback}"
    return requested or master_dir or fallback, None


def cmd_predict_points(cfg: dict, args: argparse.Namespace) -> int:
    model = args.model or cfg["weights"]["sew_point_onnx"]
    point_distance = float(args.distance_threshold)
    cmd = [
        sys.executable,
        "-m",
        "sew_point.tools.batch_infer",
        "--input_dir",
        args.image_dir or cfg["master_images_dir"],
        "--output_dir",
        args.output_dir or cfg["point_predictions_dir"],
        "--threshold",
        str(args.threshold),
        "--output_format",
        "master",
    ]
    if point_distance > 0:
        cmd.extend(["--cluster_dist", str(point_distance)])
    if str(model).strip():
        cmd.extend(["--model", model])
    return run_command(cmd, cwd=cfg["train_model_modules_root"], dry_run=args.dry_run)


def cmd_predict_edges(cfg: dict, args: argparse.Namespace) -> int:
    model = args.model or cfg["weights"]["sew_point_connector_pth"]
    if not args.dry_run:
        ensure_value(model, "weights.sew_point_connector_pth")
    annotation_dir, note = _resolve_edge_annotation_dir(cfg, args.annotation_dir or cfg["master_annotations_dir"])
    cmd = [
        sys.executable,
        "-m",
        "sew_point_conntect.batch_predict",
        "--image_dir",
        args.image_dir or cfg["master_images_dir"],
        "--annotation_dir",
        annotation_dir,
        "--output_annotation_dir",
        args.output_dir or cfg["edge_predictions_dir"],
        "--postprocess_preset",
        args.postprocess_preset,
    ]
    if str(model).strip():
        cmd.extend(["--model_path", model])
    if args.no_compare_gt:
        cmd.append("--no_compare_gt")
    if note:
        timestamp = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
        print(f"[{timestamp}] [info] {note}")
        append_log(f"[{timestamp}] [info] {note}")
    return run_command(cmd, cwd=cfg["train_model_modules_root"], dry_run=args.dry_run)


def cmd_validate(cfg: dict, args: argparse.Namespace) -> int:
    image_dir = args.image_dir or cfg["master_images_dir"]
    annotation_dir = args.annotation_dir or cfg["master_annotations_dir"]
    timestamp = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    print(f"[{timestamp}] validate --image-dir {image_dir} --annotation-dir {annotation_dir}")
    append_log(f"[{timestamp}] validate --image-dir {image_dir} --annotation-dir {annotation_dir}")
    if args.dry_run:
        append_log(f"[{timestamp}] [result] dry-run")
        return 0
    from core.cabf_shared import (
        summarize_validation,
        summarize_validation_findings,
        validate_master_dataset,
        write_json,
    )
    report = validate_master_dataset(image_dir, annotation_dir)
    print(summarize_validation(report))
    findings = summarize_validation_findings(report, include_details=args.show_samples)
    if findings:
        print(findings)
    if args.report_path:
        write_json(args.report_path, report)
        print(f"saved_report: {args.report_path}")
    append_log(f"[{timestamp}] [result] exit=0")
    return 0


def cmd_export(cfg: dict, args: argparse.Namespace) -> int:
    image_dir = args.image_dir or cfg["master_images_dir"]
    annotation_dir = args.annotation_dir or cfg["master_annotations_dir"]
    timestamp = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    print(f"[{timestamp}] export --image-dir {image_dir} --annotation-dir {annotation_dir}")
    append_log(f"[{timestamp}] export --image-dir {image_dir} --annotation-dir {annotation_dir}")
    if args.dry_run:
        append_log(f"[{timestamp}] [result] dry-run")
        return 0
    import json
    from core.cabf_shared import export_master_to_model_a, export_master_to_model_b

    model_a_output = args.model_a_output or cfg["model_a_export_root"]
    result_a = export_master_to_model_a(
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        output_dir=model_a_output,
    )
    print(json.dumps(result_a, ensure_ascii=False, indent=2))

    model_b_output = args.model_b_output or cfg["model_b_export_root"]
    result_b = export_master_to_model_b(
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        output_dir=model_b_output,
    )
    print(json.dumps(result_b, ensure_ascii=False, indent=2))
    append_log(f"[{timestamp}] [result] exit=0")
    return 0


def cmd_train(cfg: dict, args: argparse.Namespace) -> int:
    cmds = [
        [
            sys.executable,
            "-m",
            "sew_point.train",
            "--img_dir",
            args.model_a_images or str(Path(cfg["model_a_export_root"]) / "images"),
            "--ann_dir",
            args.model_a_annotations or str(Path(cfg["model_a_export_root"]) / "annotations"),
            "--save_dir",
            args.model_a_out or cfg["outputs"]["sew_point_train_out"],
        ],
        [
            sys.executable,
            "-m",
            "sew_point_conntect.train",
            "--image_dir",
            args.model_b_images or str(Path(cfg["model_b_export_root"]) / "images"),
            "--annotation_dir",
            args.model_b_annotations or str(Path(cfg["model_b_export_root"]) / "annotations"),
            "--save_dir",
            args.model_b_out or cfg["outputs"]["sew_point_conntect_train_out"],
        ],
    ]
    for cmd in cmds:
        code = run_command(cmd, cwd=cfg["train_model_modules_root"], dry_run=args.dry_run)
        if code != 0:
            return code
    return 0


def cmd_show_config(cfg: dict, _: argparse.Namespace) -> int:
    print(json.dumps(cfg, ensure_ascii=False, indent=2))
    return 0


def ensure_value(value: str, field_name: str) -> None:
    if not str(value).strip():
        raise ValueError(f"配置缺失: {field_name}")


def cmd_doctor(cfg: dict, _: argparse.Namespace) -> int:
    checks = [
        ("img_tools_root", cfg["img_tools_root"]),
        ("train_model_modules_root", cfg["train_model_modules_root"]),
        ("dataset_root", cfg["dataset_root"]),
        ("master_images_dir", cfg["master_images_dir"]),
        ("master_annotations_dir", cfg["master_annotations_dir"]),
    ]
    for name, value in checks:
        path = Path(value)
        print(f"[{'OK' if path.exists() else 'MISSING'}] {name}: {value}")
    for name, value in cfg.get("weights", {}).items():
        label = "OK" if str(value).strip() and Path(value).exists() else "EMPTY_OR_MISSING"
        print(f"[{label}] weights.{name}: {value}")
    return 0


def cmd_settings(_: dict, __: argparse.Namespace) -> int:
    from gui.main_window import launch_standalone

    launch_standalone(CONFIG_PATH)
    return 0


def cmd_pipeline(cfg: dict, args: argparse.Namespace) -> int:
    if not args.dry_run:
        ensure_value(cfg["weights"]["sew_point_onnx"], "weights.sew_point_onnx")
        ensure_value(cfg["weights"]["sew_point_connector_pth"], "weights.sew_point_connector_pth")

    steps = [
        ("predict-points", cmd_predict_points, argparse.Namespace(
            image_dir=args.image_dir,
            output_dir=args.point_output_dir,
            model=args.point_model or cfg["weights"]["sew_point_onnx"],
            threshold=args.point_threshold,
            distance_threshold=args.point_distance_threshold,
            dry_run=args.dry_run,
        )),
        ("predict-edges", cmd_predict_edges, argparse.Namespace(
            image_dir=args.image_dir,
            annotation_dir=args.edge_annotation_dir or cfg["master_annotations_dir"],
            output_dir=args.edge_output_dir,
            model=args.edge_model or cfg["weights"]["sew_point_connector_pth"],
            postprocess_preset=args.postprocess_preset,
            no_compare_gt=args.no_compare_gt,
            dry_run=args.dry_run,
        )),
        ("validate", cmd_validate, argparse.Namespace(
            image_dir=args.image_dir,
            annotation_dir=args.validate_annotation_dir or cfg["master_annotations_dir"],
            report_path=args.report_path,
            show_samples=args.show_samples,
            dry_run=args.dry_run,
        )),
        ("export", cmd_export, argparse.Namespace(
            image_dir=args.image_dir,
            annotation_dir=args.export_annotation_dir or cfg["master_annotations_dir"],
            model_a_output=args.model_a_output,
            model_b_output=args.model_b_output,
            dry_run=args.dry_run,
        )),
    ]
    if args.include_train:
        steps.append(
            ("train", cmd_train, argparse.Namespace(
                model_a_images=args.model_a_images,
                model_a_annotations=args.model_a_annotations,
                model_a_out=args.model_a_out,
                model_b_images=args.model_b_images,
                model_b_annotations=args.model_b_annotations,
                model_b_out=args.model_b_out,
                dry_run=args.dry_run,
            ))
        )

    for step_name, func, step_args in steps:
        print(f"\n=== {step_name} ===")
        code = func(cfg, step_args)
        if code != 0:
            return code
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CAB-F workflow runner hosted inside img_tools.")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to workflow config JSON.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_points = sub.add_parser("predict-points", help="Run sew_point batch inference in master format.")
    p_points.add_argument("--image-dir", default="")
    p_points.add_argument("--output-dir", default="")
    p_points.add_argument("--model", default="")
    p_points.add_argument("--threshold", type=float, default=0.3)
    p_points.add_argument("--distance-threshold", "--cluster_dist", dest="distance_threshold", type=float, default=0.0)
    p_points.add_argument("--dry-run", action="store_true")
    p_points.set_defaults(func=cmd_predict_points)

    p_edges = sub.add_parser("predict-edges", help="Run sew_point_conntect batch edge prediction.")
    p_edges.add_argument("--image-dir", default="")
    p_edges.add_argument("--annotation-dir", default="")
    p_edges.add_argument("--output-dir", default="")
    p_edges.add_argument("--model", default="")
    p_edges.add_argument("--postprocess-preset", default="balanced", choices=("aggressive", "balanced", "conservative"))
    p_edges.add_argument("--no-compare-gt", action="store_true")
    p_edges.add_argument("--dry-run", action="store_true")
    p_edges.set_defaults(func=cmd_predict_edges)

    p_validate = sub.add_parser("validate", help="Validate CAB-F master annotations.")
    p_validate.add_argument("--image-dir", default="")
    p_validate.add_argument("--annotation-dir", default="")
    p_validate.add_argument("--report-path", default="")
    p_validate.add_argument("--show-samples", action="store_true")
    p_validate.add_argument("--dry-run", action="store_true")
    p_validate.set_defaults(func=cmd_validate)

    p_export = sub.add_parser("export", help="Export model A and model B datasets.")
    p_export.add_argument("--image-dir", default="")
    p_export.add_argument("--annotation-dir", default="")
    p_export.add_argument("--model-a-output", default="")
    p_export.add_argument("--model-b-output", default="")
    p_export.add_argument("--dry-run", action="store_true")
    p_export.set_defaults(func=cmd_export)

    p_train = sub.add_parser("train", help="Launch model A and model B training.")
    p_train.add_argument("--model-a-images", default="")
    p_train.add_argument("--model-a-annotations", default="")
    p_train.add_argument("--model-a-out", default="")
    p_train.add_argument("--model-b-images", default="")
    p_train.add_argument("--model-b-annotations", default="")
    p_train.add_argument("--model-b-out", default="")
    p_train.add_argument("--dry-run", action="store_true")
    p_train.set_defaults(func=cmd_train)

    p_pipeline = sub.add_parser("pipeline", help="Run the standard CAB-F workflow sequence.")
    p_pipeline.add_argument("--image-dir", default="")
    p_pipeline.add_argument("--point-output-dir", default="")
    p_pipeline.add_argument("--point-model", default="")
    p_pipeline.add_argument("--point-threshold", type=float, default=0.3)
    p_pipeline.add_argument("--point-distance-threshold", type=float, default=0.0)
    p_pipeline.add_argument("--edge-annotation-dir", default="")
    p_pipeline.add_argument("--edge-output-dir", default="")
    p_pipeline.add_argument("--edge-model", default="")
    p_pipeline.add_argument("--postprocess-preset", default="balanced", choices=("aggressive", "balanced", "conservative"))
    p_pipeline.add_argument("--no-compare-gt", action="store_true")
    p_pipeline.add_argument("--validate-annotation-dir", default="")
    p_pipeline.add_argument("--export-annotation-dir", default="")
    p_pipeline.add_argument("--report-path", default="")
    p_pipeline.add_argument("--show-samples", action="store_true")
    p_pipeline.add_argument("--model-a-output", default="")
    p_pipeline.add_argument("--model-b-output", default="")
    p_pipeline.add_argument("--include-train", action="store_true")
    p_pipeline.add_argument("--model-a-images", default="")
    p_pipeline.add_argument("--model-a-annotations", default="")
    p_pipeline.add_argument("--model-a-out", default="")
    p_pipeline.add_argument("--model-b-images", default="")
    p_pipeline.add_argument("--model-b-annotations", default="")
    p_pipeline.add_argument("--model-b-out", default="")
    p_pipeline.add_argument("--dry-run", action="store_true")
    p_pipeline.set_defaults(func=cmd_pipeline)

    p_doctor = sub.add_parser("doctor", help="Check key roots and configured weights.")
    p_doctor.set_defaults(func=cmd_doctor)

    p_settings = sub.add_parser("settings", help="Open the visual settings editor.")
    p_settings.set_defaults(func=cmd_settings)

    p_cfg = sub.add_parser("show-config", help="Print resolved workflow config.")
    p_cfg.set_defaults(func=cmd_show_config)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(Path(args.config))
    return int(args.func(cfg, args))


if __name__ == "__main__":
    raise SystemExit(main())
