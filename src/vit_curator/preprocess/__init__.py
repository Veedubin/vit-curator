"""Stage 1-2: Scan, hash, dedupe, decode, transform, derivatives, bucket assignment."""

from __future__ import annotations

from vit_curator.preprocess.bucket import BucketAssignment, iter_bucket_assignments
from vit_curator.preprocess.dedupe import DedupeStats, hash_and_mark_dupes
from vit_curator.preprocess.scan import ScanStats, scan_into_duckdb

__all__ = [
    "BucketAssignment",
    "DedupeStats",
    "PerceptualDedupeConfig",
    "PerceptualDedupeResult",
    "ScanStats",
    "TransformResult",
    "TransformSettings",
    "analyze_transform",
    "apply_transform",
    "hash_and_mark_dupes",
    "iter_bucket_assignments",
    "run_perceptual_dedupe",
    "run_pipeline",
    "scan_into_duckdb",
]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    """Lazy imports for modules requiring optional dependencies."""
    if name == "run_pipeline":
        from vit_curator.preprocess.derivatives import run_pipeline  # noqa: PLC0415

        return run_pipeline
    if name in ("TransformSettings", "TransformResult", "analyze_transform", "apply_transform"):
        from vit_curator.preprocess import transform as _mod  # noqa: PLC0415

        return getattr(_mod, name)
    if name in ("PerceptualDedupeConfig", "PerceptualDedupeResult", "run_perceptual_dedupe"):
        from vit_curator.preprocess import perceptual_dedupe as _mod  # noqa: PLC0415

        return getattr(_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
