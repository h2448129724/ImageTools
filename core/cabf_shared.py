"""Bridge module that exposes the new shared CAB-F layer to img_tools.

Tries a normal `from cabf import ...` first (works after `pip install -e .`
inside `project_modules/cabf_pipeline/shared/`).  Falls back to adding the
shared directory to `sys.path` so the import still works in development
without an explicit install step.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

try:
    import cabf as _cabf  # noqa: F401 – test whether the package is importable
    del _cabf
except ImportError:
    _shared_root = str(Path(__file__).resolve().parents[1] / "project_modules" / "cabf_pipeline" / "shared")
    if _shared_root not in sys.path:
        sys.path.append(_shared_root)

from cabf import (
    IMAGE_SUFFIXES,
    MASTER_SCHEMA_VERSION,
    POINT_LABEL_ALIASES,
    export_master_to_model_a,
    export_master_to_model_b,
    iter_image_files,
    iter_json_files,
    load_labelme_points,
    make_empty_master_annotation,
    normalize_edges_for_editor,
    normalize_master_annotation,
    normalize_points_for_editor,
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
    "load_labelme_points",
    "make_empty_master_annotation",
    "normalize_edges_for_editor",
    "normalize_master_annotation",
    "normalize_points_for_editor",
    "read_image_size",
    "read_json",
    "summarize_validation",
    "summarize_validation_findings",
    "validate_master_dataset",
    "write_json",
]
