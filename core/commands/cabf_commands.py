"""CAB-F dataset validation and export commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.commands.base import BaseCommand
from core.cabf_dataset import (
    export_master_to_model_a,
    export_master_to_model_b,
    summarize_validation,
    validate_master_dataset,
    write_json,
)


class CabfCommand(BaseCommand):
    name = "cabf"
    help = "CAB-F master dataset validation and export utilities"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        sub = parser.add_subparsers(dest="cabf_command")

        p_validate = sub.add_parser("validate", help="Validate CAB-F master annotations")
        p_validate.add_argument("--image-dir", required=True, help="Master dataset image folder")
        p_validate.add_argument("--annotation-dir", required=True, help="Master dataset annotation folder")
        p_validate.add_argument("--report-path", default="", help="Optional JSON report output path")
        p_validate.add_argument("--show-samples", action="store_true", help="Print sample-level issues")

        p_a = sub.add_parser("export-model-a", help="Export master dataset to point detector training format")
        p_a.add_argument("--image-dir", required=True, help="Master dataset image folder")
        p_a.add_argument("--annotation-dir", required=True, help="Master dataset annotation folder")
        p_a.add_argument("--output-image-dir", required=True, help="Output image folder")
        p_a.add_argument("--output-annotation-dir", required=True, help="Output LabelMe point annotation folder")
        p_a.add_argument("--skip-empty", action="store_true", help="Do not export empty annotations")

        p_b = sub.add_parser("export-model-b", help="Export master dataset to edge model training format")
        p_b.add_argument("--image-dir", required=True, help="Master dataset image folder")
        p_b.add_argument("--annotation-dir", required=True, help="Master dataset annotation folder")
        p_b.add_argument("--output-image-dir", required=True, help="Output image folder")
        p_b.add_argument("--output-annotation-dir", required=True, help="Output normalized master annotation folder")
        p_b.add_argument("--skip-empty", action="store_true", help="Do not export empty annotations")

    @classmethod
    def execute(cls, args: argparse.Namespace) -> int:
        if args.cabf_command == "validate":
            report = validate_master_dataset(args.image_dir, args.annotation_dir)
            print(summarize_validation(report))
            if report.get("missing_annotations"):
                print("missing_annotations:")
                for stem in report["missing_annotations"]:
                    print(f"  - {stem}")
            if report.get("orphan_annotations"):
                print("orphan_annotations:")
                for stem in report["orphan_annotations"]:
                    print(f"  - {stem}")
            if args.show_samples:
                for sample in report.get("samples", []):
                    if not sample.get("errors") and not sample.get("warnings"):
                        continue
                    print(f"[{sample['sample_id']}]")
                    for issue in sample.get("errors", []):
                        print(f"  error: {issue}")
                    for warning in sample.get("warnings", []):
                        print(f"  warning: {warning}")
            if args.report_path:
                write_json(args.report_path, report)
                print(f"saved_report: {args.report_path}")
            return 0

        if args.cabf_command == "export-model-a":
            result = export_master_to_model_a(
                image_dir=args.image_dir,
                annotation_dir=args.annotation_dir,
                output_image_dir=args.output_image_dir,
                output_annotation_dir=args.output_annotation_dir,
                include_empty=not args.skip_empty,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

        if args.cabf_command == "export-model-b":
            result = export_master_to_model_b(
                image_dir=args.image_dir,
                annotation_dir=args.annotation_dir,
                output_image_dir=args.output_image_dir,
                output_annotation_dir=args.output_annotation_dir,
                include_empty=not args.skip_empty,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

        print("Error: specify a cabf subcommand (validate, export-model-a, export-model-b)")
        return 1
