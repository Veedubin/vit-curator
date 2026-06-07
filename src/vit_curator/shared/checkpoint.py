"""JSON checkpoint/resume utilities for pipeline state persistence.

Salvaged from fastai-preprocessor-uvpkg and adapted for vit_curator.
Provides atomic write (write to tmp, rename) for crash safety.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_checkpoint(path: Path) -> dict[str, Any] | None:
    """Load a JSON checkpoint file, returning None if it doesn't exist or is corrupt.

    Args:
        path: Path to the checkpoint JSON file.

    Returns:
        Parsed checkpoint dict, or None if file missing or invalid.
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Atomically save a checkpoint dict to a JSON file.

    Writes to a temporary file first, then renames to ensure crash safety.

    Args:
        path: Target checkpoint file path.
        state: Checkpoint data to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
