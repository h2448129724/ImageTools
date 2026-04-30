"""Annotation utility commands."""
from __future__ import annotations

import argparse
import json

from core.commands.base import BaseCommand


class AnnotCommand(BaseCommand):
    name = "annot"
    help = "Annotation utilities"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        sub = parser.add_subparsers(dest="annot_command")

        p_validate = sub.add_parser("validate", help="Validate YOLO annotations")
        p_validate.add_argument("--ann-dir", required=True)
        p_validate.add_argument("--image-dir", required=True)

        p_stats = sub.add_parser("stats", help="Annotation statistics")
        p_stats.add_argument("--ann-dir", required=True)
        p_stats.add_argument("--image-dir", default="")
        p_stats.add_argument("--format", default="yolo", choices=["yolo"])

    @classmethod
    def execute(cls, args: argparse.Namespace) -> int:
        if args.annot_command == "validate":
            from core.annotation import validate_yolo_annotations
            from core.image_io import get_image_info
            from utils.helpers import get_image_files
            files = get_image_files(args.image_dir)
            total_issues = 0
            for f in files:
                import os
                base = os.path.splitext(os.path.basename(f))[0]
                txt_path = os.path.join(args.ann_dir, base + ".txt")
                if not os.path.exists(txt_path):
                    continue
                info = get_image_info(f)
                if info is None:
                    continue
                w, h = info[0], info[1]
                issues = validate_yolo_annotations(txt_path, w, h)
                for issue in issues:
                    print(f"[{base}] {issue}")
                    total_issues += 1
            print(f"Done. Found {total_issues} issues.")
            return 0

        if args.annot_command == "stats":
            from core.annotation import annotation_statistics
            result = annotation_statistics(args.ann_dir, args.image_dir, args.format)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0

        print("Error: specify an annotation subcommand (validate, stats)")
        return 1
