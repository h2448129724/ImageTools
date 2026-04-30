"""Unified exception hierarchy for image-tools.

All custom exceptions inherit from ImageToolsError so the GUI and CLI
can catch them uniformly while still preserving original tracebacks.
"""
from __future__ import annotations


class ImageToolsError(Exception):
    """Base exception for all domain errors in image-tools."""
    pass


class ImageReadError(ImageToolsError):
    """Raised when an image cannot be read or decoded."""
    pass


class ImageWriteError(ImageToolsError):
    """Raised when an image cannot be written to disk."""
    pass


class AnnotationError(ImageToolsError):
    """Raised when annotation parsing or validation fails."""
    pass


class DatasetError(ImageToolsError):
    """Raised when a dataset operation (split, conversion, etc.) fails."""
    pass


class TrainingError(ImageToolsError):
    """Raised when YOLO training or export fails."""
    pass


class ConfigError(ImageToolsError):
    """Raised when configuration loading/saving fails."""
    pass
