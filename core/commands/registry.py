"""Command registry for dispatching CLI commands."""
from __future__ import annotations

import argparse
from typing import Type

from core.commands.base import BaseCommand


class CommandRegistry:
    """Registry for CLI commands."""

    def __init__(self) -> None:
        self._commands: dict[str, Type[BaseCommand]] = {}

    def register(self, cmd_class: Type[BaseCommand]) -> None:
        """Register a command class."""
        if not cmd_class.name:
            raise ValueError(f"Command class {cmd_class.__name__} missing 'name' attribute")
        self._commands[cmd_class.name] = cmd_class

    def get(self, name: str) -> Type[BaseCommand] | None:
        """Get a command class by name."""
        return self._commands.get(name)

    def names(self) -> list[str]:
        """Return all registered command names."""
        return list(self._commands.keys())

    def setup_parsers(self, subparsers: argparse._SubParsersAction) -> None:
        """Create argparse subparsers for all registered commands."""
        for cmd in self._commands.values():
            parser = subparsers.add_parser(cmd.name, help=cmd.help)
            cmd.add_arguments(parser)

    def dispatch(self, args: argparse.Namespace) -> int:
        """Dispatch to the command specified in args.command."""
        cmd_class = self._commands.get(args.command)
        if cmd_class is None:
            return 1
        return cmd_class.run(args)
