"""Bridge the in-repo CAB-F shared helpers for migrated train_model modules."""
from __future__ import annotations

import sys
from pathlib import Path


SHARED_ROOT = Path(__file__).resolve().parents[1] / "cabf_pipeline" / "shared"
if str(SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_ROOT))

from cabf import (  # noqa: E402
    MASTER_SCHEMA_VERSION,
    make_empty_master_annotation,
    normalize_master_annotation,
    read_json,
    write_json,
)

__all__ = [
    "MASTER_SCHEMA_VERSION",
    "make_empty_master_annotation",
    "normalize_master_annotation",
    "read_json",
    "write_json",
]
