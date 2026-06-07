"""Tests for vit_curator.preprocess.dedupe — hash and mark duplicates."""

from __future__ import annotations

from pathlib import Path

import duckdb

from tests.conftest import make_rgb_image


def test_hash_dedupe_marks_dupe(db: duckdb.DuckDBPyConnection, src_dir: Path) -> None:
    """Test that exact duplicates are properly detected and marked."""
    from vit_curator.preprocess.dedupe import hash_and_mark_dupes
    from vit_curator.preprocess.scan import scan_into_duckdb

    make_rgb_image(src_dir / "a.jpg", (255, 0, 0))
    (src_dir / "b.jpg").write_bytes((src_dir / "a.jpg").read_bytes())  # exact copy
    make_rgb_image(src_dir / "c.jpg", (0, 255, 0))  # different

    scan_into_duckdb(db, src_dir, start_file_pk=1, allow_exts=None, max_files=None)
    stats = hash_and_mark_dupes(db, src_dir, num_workers=2, metrics_every_s=0)

    assert stats.dupes == 1
    # Verify database state
    dupe_count = db.execute(
        "SELECT COUNT(*) FROM files WHERE dupe_of_file_pk IS NOT NULL"
    ).fetchone()
    assert dupe_count is not None and dupe_count[0] == 1

    claims_count = db.execute("SELECT COUNT(*) FROM content_claims").fetchone()
    assert claims_count is not None and claims_count[0] == 2  # Both files share a claim


def test_hash_dedupe_no_dupes(db: duckdb.DuckDBPyConnection, src_dir: Path) -> None:
    """Test that unique files are not marked as duplicates."""
    from vit_curator.preprocess.dedupe import hash_and_mark_dupes
    from vit_curator.preprocess.scan import scan_into_duckdb

    make_rgb_image(src_dir / "a.jpg", (255, 0, 0))
    make_rgb_image(src_dir / "b.jpg", (0, 255, 0))

    scan_into_duckdb(db, src_dir, start_file_pk=1, allow_exts=None, max_files=None)
    stats = hash_and_mark_dupes(db, src_dir, num_workers=1, metrics_every_s=0)

    assert stats.dupes == 0
    assert stats.hashed_ok == 2


def test_dedupe_stats_type() -> None:
    """Test that DedupeStats dataclass is constructable."""
    from vit_curator.preprocess.dedupe import DedupeStats

    stats = DedupeStats(
        total_candidates=100,
        hashed_ok=95,
        hash_err=5,
        uniques=90,
        dupes=10,
    )
    assert stats.total_candidates == 100
    assert stats.hashed_ok == 95
    assert stats.hash_err == 5
    assert stats.uniques == 90
    assert stats.dupes == 10
