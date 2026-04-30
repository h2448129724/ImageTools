"""Annotation format conversion commands."""
from __future__ import annotations

import argparse

from core.commands.base import BaseCommand


class FormatCommand(BaseCommand):
    name = "format"
    help = "Convert annotation formats (YOLO/COCO/VOC)"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        sub = parser.add_subparsers(dest="fmt_command")

        # yolo2coco
        p = sub.add_parser("yolo2coco", help="YOLO → COCO")
        p.add_argument("--yolo-dir", required=True)
        p.add_argument("--image-dir", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--categories", required=True)

        # coco2yolo
        p = sub.add_parser("coco2yolo", help="COCO → YOLO")
        p.add_argument("--coco-path", required=True)
        p.add_argument("--output", required=True)

        # voc2yolo
        p = sub.add_parser("voc2yolo", help="VOC → YOLO")
        p.add_argument("--voc-dir", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--categories", required=True)

        # yolo2voc
        p = sub.add_parser("yolo2voc", help="YOLO → VOC")
        p.add_argument("--yolo-dir", required=True)
        p.add_argument("--image-dir", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--categories", required=True)

        # voc2coco
        p = sub.add_parser("voc2coco", help="VOC → COCO")
        p.add_argument("--voc-dir", required=True)
        p.add_argument("--image-dir", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--categories", required=True)

        # coco2voc
        p = sub.add_parser("coco2voc", help="COCO → VOC")
        p.add_argument("--coco-path", required=True)
        p.add_argument("--output", required=True)

        # xanylabeling2yolo
        p = sub.add_parser("xanylabeling2yolo", help="X-AnyLabeling → YOLO")
        p.add_argument("--src-dir", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--val-ratio", type=float, default=0.2)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--categories", default="")

    @classmethod
    def execute(cls, args: argparse.Namespace) -> int:
        cats = [c.strip() for c in args.categories.split(",")] if hasattr(args, "categories") else []

        if args.fmt_command == "yolo2coco":
            from core.format_conversion import yolo_to_coco
            result = yolo_to_coco(args.yolo_dir, args.image_dir, args.output, cats)
            print(f"Done. {len(result['images'])} images, {len(result['annotations'])} annotations.")
            return 0

        if args.fmt_command == "coco2yolo":
            from core.format_conversion import coco_to_yolo
            coco_to_yolo(args.coco_path, args.output)
            print(f"Done. YOLO files saved to {args.output}")
            return 0

        if args.fmt_command == "voc2yolo":
            from core.format_conversion import voc_to_yolo
            voc_to_yolo(args.voc_dir, args.output, cats)
            print(f"Done. YOLO files saved to {args.output}")
            return 0

        if args.fmt_command == "yolo2voc":
            from core.format_conversion import yolo_to_voc
            yolo_to_voc(args.yolo_dir, args.image_dir, args.output, cats)
            print(f"Done. VOC files saved to {args.output}")
            return 0

        if args.fmt_command == "voc2coco":
            from core.format_conversion import voc_to_coco
            result = voc_to_coco(args.voc_dir, args.image_dir, args.output, cats)
            print(f"Done. {len(result['images'])} images, {len(result['annotations'])} annotations.")
            return 0

        if args.fmt_command == "coco2voc":
            from core.format_conversion import coco_to_voc
            coco_to_voc(args.coco_path, args.output)
            print(f"Done. VOC files saved to {args.output}")
            return 0

        if args.fmt_command == "xanylabeling2yolo":
            from core.format_conversion import xanylabeling_to_yolo
            src_cats = [c.strip() for c in args.categories.split(",") if c.strip()] if args.categories else None
            result = xanylabeling_to_yolo(
                args.src_dir, args.output,
                val_ratio=args.val_ratio, seed=args.seed, categories=src_cats
            )
            print(result)
            return 0

        print("Error: specify a format subcommand")
        return 1
