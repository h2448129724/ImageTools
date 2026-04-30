"""Dataset split and augmentation commands."""
from __future__ import annotations

import argparse
import json

from core.commands.base import BaseCommand


class SplitCommand(BaseCommand):
    name = "split"
    help = "Split dataset into train/val/test"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("input", help="Input directory")
        parser.add_argument("output", help="Output directory")
        parser.add_argument("--train", type=float, default=0.7, help="Train ratio")
        parser.add_argument("--val", type=float, default=0.2, help="Val ratio")
        parser.add_argument("--test", type=float, default=0.1, help="Test ratio")
        parser.add_argument("--seed", type=int, default=42, help="Random seed")

    @classmethod
    def execute(cls, args: argparse.Namespace) -> int:
        from core.dataset_split import random_split
        result = random_split(
            args.input, args.output,
            ratios=(args.train, args.val, args.test), seed=args.seed
        )
        train_n = len(result.get("train", []))
        val_n = len(result.get("val", []))
        test_n = len(result.get("test", []))
        print(f"Done. Train: {train_n}, Val: {val_n}, Test: {test_n}")
        return 0


class StratifiedSplitCommand(BaseCommand):
    name = "stratified-split"
    help = "Stratified split by subfolder (class)"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("input", help="Input directory with class subfolders")
        parser.add_argument("output", help="Output directory")
        parser.add_argument("--train", type=float, default=0.7, help="Train ratio")
        parser.add_argument("--val", type=float, default=0.2, help="Val ratio")
        parser.add_argument("--test", type=float, default=0.1, help="Test ratio")
        parser.add_argument("--seed", type=int, default=42, help="Random seed")

    @classmethod
    def execute(cls, args: argparse.Namespace) -> int:
        from core.dataset_split import stratified_split
        result = stratified_split(
            args.input, args.output,
            ratios=(args.train, args.val, args.test), seed=args.seed
        )
        train_n = sum(len(files) for _, files in result.get("train", []))
        val_n = sum(len(files) for _, files in result.get("val", []))
        test_n = sum(len(files) for _, files in result.get("test", []))
        print(f"Done. Train: {train_n}, Val: {val_n}, Test: {test_n}")
        return 0


class KFoldCommand(BaseCommand):
    name = "kfold"
    help = "K-fold cross-validation split"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("input", help="Input directory")
        parser.add_argument("output", help="Output directory")
        parser.add_argument("--k", type=int, default=5, help="Number of folds")
        parser.add_argument("--seed", type=int, default=42, help="Random seed")

    @classmethod
    def execute(cls, args: argparse.Namespace) -> int:
        from core.dataset_split import kfold_split
        result = kfold_split(args.input, args.output, k=args.k, seed=args.seed)
        for fold in result:
            print(f"Fold {fold['fold']}: Train {fold['train']}, Val {fold['val']}")
        return 0


class AugmentCommand(BaseCommand):
    name = "augment"
    help = "Augment dataset with configurable pipeline"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("input", help="Input directory")
        parser.add_argument("output", help="Output directory")
        parser.add_argument("--config", required=True,
                            help="Pipeline config JSON file")
        parser.add_argument("--copies", type=int, default=1,
                            help="Augmented copies per image")
        parser.add_argument("--seed", type=int, default=None,
                            help="Random seed for reproducibility")
        parser.add_argument("--format", default="png",
                            choices=["png", "jpg", "webp"],
                            help="Output format")
        parser.add_argument("--quality", type=int, default=95,
                            help="Output quality (1-100)")
        parser.add_argument("--workers", type=int, default=4,
                            help="Number of parallel workers")

    @classmethod
    def execute(cls, args: argparse.Namespace) -> int:
        from core.augmentation import augment_dataset
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        stats = augment_dataset(
            args.input, args.output, config,
            copies=args.copies, seed=args.seed,
            max_workers=args.workers, fmt=args.format, quality=args.quality,
            progress_callback=lambda c, t: print(f"\rAugmenting: {c}/{t}", end="", flush=True)
        )
        print(f"\nDone. Input: {stats['total_input']}, Output: {stats['total_output']}, Errors: {stats['errors']}")
        return 0
