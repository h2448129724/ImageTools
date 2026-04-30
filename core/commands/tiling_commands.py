"""Tiling and image info commands."""
from __future__ import annotations

import argparse
import os
import sys

from core.commands.base import BaseCommand


class TileCommand(BaseCommand):
    name = "tile"
    help = "Tile large images"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("input", help="Input file or directory")
        parser.add_argument("output", help="Output directory")
        parser.add_argument("--tile-w", type=int, default=256, help="Tile width")
        parser.add_argument("--tile-h", type=int, default=256, help="Tile height")
        parser.add_argument("--overlap", type=int, default=0, help="Overlap pixels")

    @classmethod
    def execute(cls, args: argparse.Namespace) -> int:
        from core.tiling import tile_image_file, tile_directory
        if os.path.isfile(args.input):
            result = tile_image_file(args.input, args.output, args.tile_w, args.tile_h, args.overlap)
            print(f"Done. {result['tiles']} tiles saved to {result['output_dir']}")
        else:
            result = tile_directory(
                args.input, args.output, args.tile_w, args.tile_h, args.overlap,
                progress_callback=lambda c, t: print(f"\rTiling: {c}/{t}", end="", flush=True)
            )
            print(f"\nDone. {result['total_tiles']} tiles from {result['total_files']} images.")
        return 0


class InfoCommand(BaseCommand):
    name = "info"
    help = "Show image info"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("input", help="Image file path")

    @classmethod
    def execute(cls, args: argparse.Namespace) -> int:
        from core.image_io import get_image_info
        info = get_image_info(args.input)
        if info:
            w, h, c = info
            print(f"Size: {w}x{h}, Channels: {c}")
            return 0
        print("Failed to read image.", file=sys.stderr)
        return 1
