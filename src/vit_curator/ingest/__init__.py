"""Stage 0: Download, extract, and sort archives."""

from __future__ import annotations

from .archive import extract_archive, is_archive
from .fsops import ensure_dir, link_or_copy, move_into, safe_relpath, same_filesystem
from .pipeline import FileTask, IngestMetrics, WorkLayout, run_ingest
from .sorters import BucketLayout, choose_bucket, collision_safe_target
from .state import IngestState
from .urls import UrlItem, iter_urls

__all__ = [
    "BucketLayout",
    "FileTask",
    "IngestMetrics",
    "IngestState",
    "UrlItem",
    "WorkLayout",
    "choose_bucket",
    "collision_safe_target",
    "ensure_dir",
    "extract_archive",
    "is_archive",
    "iter_urls",
    "link_or_copy",
    "move_into",
    "run_ingest",
    "safe_relpath",
    "same_filesystem",
]
