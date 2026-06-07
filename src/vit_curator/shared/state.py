"""Resumable state primitives for pipeline crash recovery.

Tracks schema version, file modification detection, and stage completion
to allow resuming pipeline runs after interruption.
"""

from __future__ import annotations

from pathlib import Path

from .db import DB, get_meta, set_meta

# ---------------------------------------------------------------------------
# Schema version tracking
# ---------------------------------------------------------------------------

SCHEMA_VERSION_KEY = b"vit_curator_schema_version"


def get_schema_version(db: DB) -> int:
    """Read the stored schema version from meta table.

    Returns:
        Schema version as integer, or 0 if not set.
    """
    raw = get_meta(db.con, "schema_version")
    return int.from_bytes(raw, "little") if raw else 0


def set_schema_version(db: DB, version: int) -> None:
    """Write the schema version to meta table."""
    set_meta(db.con, "schema_version", version.to_bytes(8, "little"))


# ---------------------------------------------------------------------------
# Stage completion tracking
# ---------------------------------------------------------------------------


def is_stage_complete(db: DB, stage: str) -> bool:
    """Check whether a pipeline stage has completed successfully."""
    raw = get_meta(db.con, f"stage_complete:{stage}")
    return raw == b"1"


def mark_stage_complete(db: DB, stage: str) -> None:
    """Mark a pipeline stage as completed."""
    set_meta(db.con, f"stage_complete:{stage}", b"1")


def clear_stage_complete(db: DB, stage: str) -> None:
    """Clear a pipeline stage completion marker (for re-running a stage)."""
    set_meta(db.con, f"stage_complete:{stage}", b"0")


# ---------------------------------------------------------------------------
# File modification detection
# ---------------------------------------------------------------------------


def file_mtime_ns(path: Path) -> int | None:
    """Get file modification time in nanoseconds, or None if file missing."""
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def file_size(path: Path) -> int | None:
    """Get file size in bytes, or None if file missing."""
    try:
        return path.stat().st_size
    except OSError:
        return None
