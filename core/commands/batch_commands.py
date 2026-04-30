"""Batch processing commands: resize, convert, rename, dedup, border."""
from __future__ import annotations

import argparse
import sys

from core.commands.base import BaseCommand


class ResizeCommand(BaseCommand):
    name = "resize"
    help = "Batch resize images"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("input", help="Input directory")
        parser.add_argument("output", help="Output directory")
        parser.add_argument("--width", type=int, help="Target width")
        parser.add_argument("--height", type=int, help="Target height")
        parser.add_argument("--scale", type=float, help="Scale factor")
        parser.add_argument("--no-keep-aspect", action="store_true",
                            help="Don't preserve aspect ratio")

    @classmethod
    def execute(cls, args: argparse.Namespace) -> int:
        from core.batch_processing import batch_resize
        if not args.width and not args.height and not args.scale:
            print("Error: specify at least one of --width, --height, --scale", file=sys.stderr)
            return 1
        count = batch_resize(
            args.input, args.output, width=args.width, height=args.height,
            scale=args.scale, keep_aspect=not args.no_keep_aspect,
            progress_callback=lambda c, t: print(f"\rResizing: {c}/{t}", end="", flush=True)
        )
        print(f"\nDone. Resized {count} images.")
        return 0


class ConvertCommand(BaseCommand):
    name = "convert"
    help = "Batch convert image format"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("input", help="Input directory")
        parser.add_argument("output", help="Output directory")
        parser.add_argument("--format", default="png",
                            choices=["png", "jpg", "webp", "bmp"],
                            help="Target format")
        parser.add_argument("--quality", type=int, default=95,
                            help="Output quality (1-100)")

    @classmethod
    def execute(cls, args: argparse.Namespace) -> int:
        from core.batch_processing import batch_convert_format
        count = batch_convert_format(
            args.input, args.output, fmt=args.format, quality=args.quality,
            progress_callback=lambda c, t: print(f"\rConverting: {c}/{t}", end="", flush=True)
        )
        print(f"\nDone. Converted {count} images.")
        return 0


class RenameCommand(BaseCommand):
    name = "rename"
    help = "Batch rename images"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("input", help="Input directory")
        parser.add_argument("output", help="Output directory")
        parser.add_argument("--prefix", default="img_", help="Filename prefix")
        parser.add_argument("--start", type=int, default=1, help="Start index")
        parser.add_argument("--digits", type=int, default=4,
                            help="Zero-padding digits")

    @classmethod
    def execute(cls, args: argparse.Namespace) -> int:
        from core.batch_processing import batch_rename
        results = batch_rename(
            args.input, args.output, prefix=args.prefix,
            start_index=args.start, digits=args.digits
        )
        print(f"Done. Renamed {len(results)} images.")
        return 0


class DedupCommand(BaseCommand):
    name = "dedup"
    help = "Find duplicate images"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("input", help="Input directory")
        parser.add_argument("--mode", choices=["exact", "perceptual"],
                            default="exact",
                            help="exact=byte-identical, perceptual=visually similar")
        parser.add_argument("--threshold", type=int, default=10,
                            help="Hamming distance threshold for perceptual mode (0-64)")

    @classmethod
    def execute(cls, args: argparse.Namespace) -> int:
        from core.batch_processing import deduplicate_images
        dupes = deduplicate_images(
            args.input, mode=args.mode, similarity_threshold=args.threshold,
            progress_callback=lambda c, t: print(f"\rScanning: {c}/{t}", end="", flush=True)
        )
        print(f"\nFound {len(dupes)} duplicate pairs.")
        for dup, orig in dupes:
            print(f"  {dup} == {orig}")
        return 0


class BorderCommand(BaseCommand):
    name = "border"
    help = "Batch add border to images"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("input", help="Input directory")
        parser.add_argument("output", help="Output directory")
        parser.add_argument("--size", type=int, default=10,
                            help="Border size in pixels")
        parser.add_argument("--color", default="black",
                            choices=["black", "white", "red", "green", "blue"])

    @classmethod
    def execute(cls, args: argparse.Namespace) -> int:
        from core.batch_processing import batch_add_border
        cmap = {
            "black": (0, 0, 0), "white": (255, 255, 255), "red": (0, 0, 255),
            "green": (0, 255, 0), "blue": (255, 0, 0)
        }
        count = batch_add_border(
            args.input, args.output, args.size, cmap[args.color],
            progress_callback=lambda c, t: print(f"\rAdding border: {c}/{t}", end="", flush=True)
        )
        print(f"\nDone. Added border to {count} images.")
        return 0
