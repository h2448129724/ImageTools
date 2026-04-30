"""YOLO training commands."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime

from core.commands.base import BaseCommand


class TrainCommand(BaseCommand):
    name = "train"
    help = "Train a YOLO model"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--data", required=True, help="Dataset YAML file path")
        parser.add_argument("--model", default="yolov8n.pt", help="Pretrained model name")
        parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
        parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
        parser.add_argument("--batch", type=int, default=16, help="Batch size")
        parser.add_argument("--workers", type=int, default=8, help="Data loader workers")
        parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
        parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate factor")
        parser.add_argument("--optimizer", default="SGD",
                            choices=["SGD", "Adam", "AdamW", "Lion"],
                            help="Optimizer")
        parser.add_argument("--device", default="", help="Device (empty for auto)")
        parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
        parser.add_argument("--seed", type=int, default=0, help="Random seed (0=random)")
        parser.add_argument("--project", default="runs", help="Output project directory")
        parser.add_argument("--name", default="detect", help="Training run name")
        parser.add_argument("--export-onnx", action="store_true",
                            help="Auto-export ONNX after training")

    @classmethod
    def execute(cls, args: argparse.Namespace) -> int:
        from core.yolo_training import run_yolo_training, export_after_training
        config = {
            "data": args.data,
            "model": args.model,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "lr0": args.lr0,
            "lrf": args.lrf,
            "optimizer": args.optimizer,
            "device": args.device,
            "patience": args.patience,
            "seed": args.seed,
            "project": args.project,
            "name": args.name,
        }
        result = run_yolo_training(config, log_fn=print)
        if result.get("success"):
            print("Training completed successfully.")
            if args.export_onnx and result.get("best_model"):
                exported = export_after_training(
                    result["best_model"], ["onnx"], imgsz=args.imgsz, log_fn=print
                )
                for fmt, path in exported.items():
                    print(f"Exported {fmt}: {path}")
            return 0
        print(f"Training failed: {result.get('error', 'Unknown error')}", file=sys.stderr)
        return 1


class TrainListCommand(BaseCommand):
    name = "train-list"
    help = "List training history"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--project-dir", default="runs",
                            help="Project directory to scan")

    @classmethod
    def execute(cls, args: argparse.Namespace) -> int:
        from core.yolo_training import get_training_history
        history = get_training_history(args.project_dir)
        if not history:
            print("No training history found.")
            return 0
        print(f"{'Name':<20} {'Model':<15} {'Epochs':<8} {'mAP50':<10} {'mAP50-95':<10} {'Date'}")
        print("-" * 80)
        for run in history:
            m = run.get("final_metrics", {})
            map50 = m.get("metrics/mAP50(B)", 0)
            map50_95 = m.get("metrics/mAP50-95(B)", 0)
            dt = datetime.fromtimestamp(run["created_time"]).strftime("%Y-%m-%d %H:%M")
            print(f"{run['name']:<20} {str(run.get('model','')):<15} "
                  f"{run['epochs_completed']:<8} {map50:<10.4f} {map50_95:<10.4f} {dt}")
        return 0


class TrainExportCommand(BaseCommand):
    name = "train-export"
    help = "Export a trained model"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--model", required=True, help="Path to best.pt model")
        parser.add_argument("--format", default="onnx",
                            help="Export format: onnx,engine,openvino,coreml,tflite")
        parser.add_argument("--imgsz", type=int, default=640, help="Export image size")

    @classmethod
    def execute(cls, args: argparse.Namespace) -> int:
        from core.yolo_training import export_after_training
        formats = [f.strip() for f in args.format.split(",")]
        exported = export_after_training(args.model, formats, imgsz=args.imgsz, log_fn=print)
        if exported:
            for fmt, path in exported.items():
                print(f"Exported {fmt}: {path}")
            return 0
        print("Export failed.", file=sys.stderr)
        return 1
