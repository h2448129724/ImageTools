"""Compatibility wrapper for CAB-F dataset helpers.

This module keeps the historical `core.cabf_dataset` import path stable while
delegating the actual implementation to the shared CAB-F layer.
"""
from __future__ import annotations

from core.cabf_shared import (
    IMAGE_SUFFIXES,
    MASTER_SCHEMA_VERSION,
    POINT_LABEL_ALIASES,
    export_master_to_model_a,
    export_master_to_model_b,
    iter_image_files,
    iter_json_files,
    normalize_master_annotation,
    read_image_size,
    read_json,
    summarize_validation,
    summarize_validation_findings,
    validate_master_dataset,
    write_json,
)

__all__ = [
    "IMAGE_SUFFIXES",
    "MASTER_SCHEMA_VERSION",
    "POINT_LABEL_ALIASES",
    "export_master_to_model_a",
    "export_master_to_model_b",
    "iter_image_files",
    "iter_json_files",
    "normalize_master_annotation",
    "read_image_size",
    "read_json",
    "summarize_validation",
    "summarize_validation_findings",
    "validate_master_dataset",
    "write_json",
]
