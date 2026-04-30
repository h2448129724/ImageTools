"""Command-line interface for batch image processing."""
from __future__ import annotations

import argparse
import logging
import sys

from core.commands import create_default_registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="image-tools",
        description="面向深度学习的图像处理和数据集准备工具",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    registry = create_default_registry()
    registry.setup_parsers(sub)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return registry.dispatch(args)


if __name__ == "__main__":
    sys.exit(main())
