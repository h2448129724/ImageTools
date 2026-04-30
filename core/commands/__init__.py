"""CLI command implementations."""
from __future__ import annotations

from core.commands.base import BaseCommand
from core.commands.registry import CommandRegistry

from core.commands.batch_commands import (
    ResizeCommand, ConvertCommand, RenameCommand,
    DedupCommand, BorderCommand,
)
from core.commands.dataset_commands import (
    SplitCommand, StratifiedSplitCommand, KFoldCommand, AugmentCommand,
)
from core.commands.tiling_commands import TileCommand, InfoCommand
from core.commands.training_commands import (
    TrainCommand, TrainListCommand, TrainExportCommand,
)
from core.commands.format_commands import FormatCommand
from core.commands.annot_commands import AnnotCommand

__all__ = ["BaseCommand", "CommandRegistry"]

# Default registry with all built-in commands
DEFAULT_COMMANDS = [
    ResizeCommand, ConvertCommand, RenameCommand, DedupCommand, BorderCommand,
    SplitCommand, StratifiedSplitCommand, KFoldCommand, AugmentCommand,
    TileCommand, InfoCommand,
    TrainCommand, TrainListCommand, TrainExportCommand,
    FormatCommand, AnnotCommand,
]


def create_default_registry() -> CommandRegistry:
    """Create a CommandRegistry pre-populated with all built-in commands."""
    registry = CommandRegistry()
    for cmd in DEFAULT_COMMANDS:
        registry.register(cmd)
    return registry
