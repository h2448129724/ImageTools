"""Base command class for CLI commands."""
from __future__ import annotations

import argparse
import logging
import sys
from abc import ABC, abstractmethod

from utils.exceptions import ImageToolsError

logger = logging.getLogger(__name__)


class BaseCommand(ABC):
    """Abstract base class for all CLI commands.

    Each command subclass must implement:
      - name: str attribute (command name used in CLI)
      - help: str attribute (short help text)
      - add_arguments(parser): add command-specific arguments
      - execute(args): run the command and return exit code (0 for success)
    """

    name: str = ""
    help: str = ""

    @classmethod
    @abstractmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add command-specific arguments to the parser."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def execute(cls, args: argparse.Namespace) -> int:
        """Execute the command. Return 0 on success, non-zero on error."""
        raise NotImplementedError

    @classmethod
    def run(cls, args: argparse.Namespace) -> int:
        """Wrapper that handles common error patterns."""
        try:
            return cls.execute(args)
        except ImageToolsError as e:
            logger.error("%s", e)
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            logger.exception("Unexpected error in %s", cls.name)
            print(f"Unexpected error: {e}", file=sys.stderr)
            return 2
