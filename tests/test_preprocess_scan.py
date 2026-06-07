"""Tests for vit_curator.preprocess.scan — file discovery and DuckDB insertion."""

from __future__ import annotations

from pathlib import Path

import duckdb

from tests.conftest import make_rgb_image


def test_scan_insert_files(db: duckdb.DuckDBPyConnection, src_dir: Path) -> None:
    """Test that scan_into_duckdb discovers and inserts files."""
    from vit_curator.preprocess.scan import scan_into_duckdb

    make_rgb_image(src_dir / "a.jpg", (255, 0, 0))
    make_rgb_image(src_dir / "b.png", (0, 255, 0))

    stats = scan_into_duckdb(db, src_dir, start_file_pk=100, allow_exts=None, max_files=None)
    assert stats.seen == 2
    assert stats.inserted == 2
    assert stats.skipped == 0

    count = db.execute("SELECT COUNT(*) FROM files").fetchone()
    assert count is not None and count[0] == 2


def test_scan_deduplicates_by_path(db: duckdb.DuckDBPyConnection, src_dir: Path) -> None:
    """Test that re-running scan doesn't insert duplicate files."""
    from vit_curator.preprocess.scan import scan_into_duckdb

    make_rgb_image(src_dir / "a.jpg")

    stats1 = scan_into_duckdb(db, src_dir, start_file_pk=1)
    assert stats1.inserted == 1

    stats2 = scan_into_duckdb(db, src_dir, start_file_pk=1)
    assert stats2.seen == 1


def test_scan_with_extension_filter(db: duckdb.DuckDBPyConnection, src_dir: Path) -> None:
    """Test that scan respects allow_exts filter."""
    from vit_curator.preprocess.scan import scan_into_duckdb

    make_rgb_image(src_dir / "a.jpg")
    make_rgb_image(src_dir / "b.png")
    (src_dir / "c.txt").write_text("not an image")

    stats = scan_into_duckdb(
        db, src_dir, start_file_pk=1, allow_exts={b".jpg", b".png"}, max_files=None
    )
    assert stats.seen >= 2


def test_scan_max_files(db: duckdb.DuckDBPyConnection, src_dir: Path) -> None:
    """Test that scan respects max_files limit."""
    from vit_curator.preprocess.scan import scan_into_duckdb

    for i in range(5):
        make_rgb_image(src_dir / f"img_{i}.jpg")

    stats = scan_into_duckdb(db, src_dir, start_file_pk=1, allow_exts=None, max_files=3)
    assert stats.inserted <= 3
