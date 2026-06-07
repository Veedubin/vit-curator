"""Tests for vit_curator.preprocess.perceptual_dedupe.

Tests for compute_phash and PerceptualDedupe.scan_and_mark are skipped when
imagehash is not installed (optional dependency).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import duckdb
import pytest

from vit_curator.preprocess.perceptual_dedupe import (
    DEFAULT_HASH_SIZE,
    DEFAULT_THRESHOLD,
    PerceptualDedupe,
    PerceptualDedupeConfig,
    PerceptualDedupeResult,
    _hamming_distance,
    run_perceptual_dedupe,
)
from vit_curator.shared.db import ensure_schema

SKIP_IMAGEHASH = not importlib.util.find_spec("imagehash")


def _insert_file(con: duckdb.DuckDBPyConnection, file_pk: int, rel_path: str) -> None:
    """Insert a minimal file row."""
    import os

    rel_blob = os.fsencode(rel_path)
    con.execute(
        "INSERT INTO files "
        "(file_pk, rel_path_blob, rel_path_hash, ext_blob, size_bytes, mtime_ns, "
        "status, is_exact_dupe, decode_status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [file_pk, rel_blob, rel_blob + b"_hash", os.fsencode("jpg"), 1234, 1000000, 1, False, 1],
    )


@pytest.fixture
def fresh_db(tmp_path) -> duckdb.DuckDBPyConnection:
    """Return an in-memory DuckDB with schema initialized."""
    con = duckdb.connect(str(tmp_path / "test.duckdb"))
    ensure_schema(con)
    return con


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


def test_perceptual_dedupe_config_defaults() -> None:
    """PerceptualDedupeConfig should use default threshold and hash_size."""
    cfg = PerceptualDedupeConfig(db_path=Path("index.duckdb"), src_root=Path("."))
    assert cfg.threshold == DEFAULT_THRESHOLD
    assert cfg.hash_size == DEFAULT_HASH_SIZE
    assert cfg.max_files is None
    assert cfg.dry_run is False


def test_perceptual_dedupe_config_custom() -> None:
    """PerceptualDedupeConfig should accept custom values."""
    cfg = PerceptualDedupeConfig(
        db_path=Path("db.duckdb"),
        src_root=Path("/src"),
        threshold=4,
        hash_size=16,
        max_files=100,
        dry_run=True,
    )
    assert cfg.threshold == 4
    assert cfg.hash_size == 16
    assert cfg.max_files == 100
    assert cfg.dry_run is True


def test_perceptual_dedupe_result_defaults() -> None:
    """PerceptualDedupeResult should default to zero."""
    result = PerceptualDedupeResult()
    assert result.total_scanned == 0
    assert result.near_dupes_found == 0
    assert result.canonicals == 0
    assert result.errors == 0


def test_perceptual_dedupe_result_fields() -> None:
    """PerceptualDedupeResult should store provided values."""
    result = PerceptualDedupeResult(total_scanned=10, near_dupes_found=2, canonicals=8, errors=0)
    assert result.total_scanned == 10
    assert result.near_dupes_found == 2
    assert result.canonicals == 8


# ---------------------------------------------------------------------------
# Hamming distance
# ---------------------------------------------------------------------------


def test_hamming_distance_same() -> None:
    """Distance between identical hashes should be 0."""
    h = "a" * 16
    assert _hamming_distance(h, h) == 0


def test_hamming_distance_max() -> None:
    """Distance between completely different hashes should be max (16 hex chars * 4 bits = 64)."""
    h1 = "0" * 16
    h2 = "f" * 16
    assert _hamming_distance(h1, h2) == 64


def test_hamming_distance_known() -> None:
    """Known hamming distance: 0 vs f should be 4 bits per char."""
    assert _hamming_distance("0", "f") == 4
    assert _hamming_distance("00", "ff") == 8


def test_hamming_distance_one_bit() -> None:
    """Two hashes differing by one bit should have distance 1."""
    # 0 = 0000, 1 = 0001 → distance 1
    assert _hamming_distance("0" * 16, "1" + "0" * 15) == 1


def test_hamming_distance_invalid() -> None:
    """Invalid hex strings should return max distance based on length."""
    assert _hamming_distance("zzzz", "0000") == 16  # 4 chars * 4 bits


# ---------------------------------------------------------------------------
# PerceptualDedupe class (no DB updates needed for init)
# ---------------------------------------------------------------------------


def test_deduper_init_default() -> None:
    """PerceptualDedupe should create a default config when none is given."""
    deduper = PerceptualDedupe()
    assert isinstance(deduper.config, PerceptualDedupeConfig)


def test_deduper_init_custom() -> None:
    """PerceptualDedupe should accept a custom config."""
    cfg = PerceptualDedupeConfig(db_path=Path("x.duckdb"), src_root=Path("/src"))
    deduper = PerceptualDedupe(cfg)
    assert deduper.config.src_root == Path("/src")


# ---------------------------------------------------------------------------
# scan_and_mark (skipped if imagehash not installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(SKIP_IMAGEHASH, reason="imagehash not installed")
def test_scan_and_mark_empty_db(fresh_db: duckdb.DuckDBPyConnection) -> None:
    """scan_and_mark with no files should return zeros."""
    deduper = PerceptualDedupe(PerceptualDedupeConfig(db_path=Path("x.duckdb"), src_root=Path(".")))
    result = deduper.scan_and_mark(fresh_db)
    assert result.total_scanned == 0
    assert result.near_dupes_found == 0
    assert result.canonicals == 0
    assert result.errors == 0


@pytest.mark.skipif(SKIP_IMAGEHASH, reason="imagehash not installed")
def test_scan_and_mark_single_image(fresh_db: duckdb.DuckDBPyConnection, tmp_path: Path) -> None:
    """A single canonical image should be scanned with no near-dupes."""
    from PIL import Image

    src = tmp_path / "src"
    src.mkdir()
    img = Image.new("RGB", (64, 64), (255, 0, 0))
    img.save(src / "img_00.jpg")

    _insert_file(fresh_db, 1, str(src / "img_00.jpg"))

    deduper = PerceptualDedupe(PerceptualDedupeConfig(db_path=Path("x.duckdb"), src_root=src))
    result = deduper.scan_and_mark(fresh_db)
    assert result.total_scanned == 1
    assert result.near_dupes_found == 0
    assert result.canonicals == 1


@pytest.mark.skipif(SKIP_IMAGEHASH, reason="imagehash not installed")
def test_scan_and_mark_near_duplicates(fresh_db: duckdb.DuckDBPyConnection, tmp_path: Path) -> None:
    """Two identical images should produce one near-duplicate."""
    from PIL import Image

    src = tmp_path / "src"
    src.mkdir()
    img = Image.new("RGB", (64, 64), (0, 255, 0))
    img.save(src / "a.jpg")
    img.save(src / "b.jpg")

    _insert_file(fresh_db, 1, "a.jpg")
    _insert_file(fresh_db, 2, "b.jpg")

    deduper = PerceptualDedupe(PerceptualDedupeConfig(db_path=Path("x.duckdb"), src_root=src))
    result = deduper.scan_and_mark(fresh_db)
    assert result.total_scanned == 2
    assert result.near_dupes_found == 1

    # Verify dupe_of_file_pk was set
    row = fresh_db.execute("SELECT dupe_of_file_pk FROM files WHERE file_pk = 2").fetchone()
    assert row is not None
    assert row[0] == 1


@pytest.mark.skipif(SKIP_IMAGEHASH, reason="imagehash not installed")
def test_scan_and_mark_dry_run(fresh_db: duckdb.DuckDBPyConnection, tmp_path: Path) -> None:
    """dry_run=True should compute hashes but not update the database."""
    from PIL import Image

    src = tmp_path / "src"
    src.mkdir()
    img = Image.new("RGB", (64, 64), (0, 0, 255))
    img.save(src / "a.jpg")
    img.save(src / "b.jpg")

    _insert_file(fresh_db, 1, "a.jpg")
    _insert_file(fresh_db, 2, "b.jpg")

    deduper = PerceptualDedupe(
        PerceptualDedupeConfig(db_path=Path("x.duckdb"), src_root=src, dry_run=True)
    )
    result = deduper.scan_and_mark(fresh_db)
    assert result.total_scanned == 2
    assert result.near_dupes_found == 1

    # Database should NOT be updated
    row = fresh_db.execute("SELECT dupe_of_file_pk FROM files WHERE file_pk = 2").fetchone()
    assert row is not None
    assert row[0] is None


# ---------------------------------------------------------------------------
# High-level entry point (skipped if imagehash not installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(SKIP_IMAGEHASH, reason="imagehash not installed")
def test_run_perceptual_dedupe(fresh_db: duckdb.DuckDBPyConnection, tmp_path: Path) -> None:
    """run_perceptual_dedupe should be a thin wrapper around PerceptualDedupe."""
    from PIL import Image

    src = tmp_path / "src"
    src.mkdir()
    img = Image.new("RGB", (64, 64), (255, 255, 0))
    img.save(src / "a.jpg")

    _insert_file(fresh_db, 1, "a.jpg")

    result = run_perceptual_dedupe(
        fresh_db,
        src_root=src,
        threshold=DEFAULT_THRESHOLD,
        hash_size=DEFAULT_HASH_SIZE,
    )
    assert result.total_scanned == 1
    assert result.near_dupes_found == 0


# ---------------------------------------------------------------------------
# compute_phash (skipped if imagehash not installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(SKIP_IMAGEHASH, reason="imagehash not installed")
def test_compute_phash_returns_hex(tmp_path: Path) -> None:
    """compute_phash should return a non-empty hex string."""
    from PIL import Image

    from vit_curator.preprocess.perceptual_dedupe import compute_phash

    img = Image.new("RGB", (64, 64), (255, 0, 0))
    path = tmp_path / "test.jpg"
    img.save(path)

    h = compute_phash(path)
    assert isinstance(h, str)
    assert len(h) > 0
    # Should be valid hex
    int(h, 16)


@pytest.mark.skipif(SKIP_IMAGEHASH, reason="imagehash not installed")
def test_compute_phash_same_image(tmp_path: Path) -> None:
    """compute_phash should return the same hash for the same image content."""
    from PIL import Image

    from vit_curator.preprocess.perceptual_dedupe import compute_phash

    img = Image.new("RGB", (64, 64), (0, 255, 0))
    p1 = tmp_path / "a.jpg"
    p2 = tmp_path / "b.jpg"
    img.save(p1)
    img.save(p2)

    assert compute_phash(p1) == compute_phash(p2)


@pytest.mark.skipif(SKIP_IMAGEHASH, reason="imagehash not installed")
def test_compute_phash_different_images(tmp_path: Path) -> None:
    """compute_phash should return different hashes for different images."""
    from PIL import Image

    from vit_curator.preprocess.perceptual_dedupe import compute_phash

    img1 = Image.new("RGB", (64, 64), (255, 0, 0))
    img2 = Image.new("RGB", (64, 64), (0, 255, 0))
    p1 = tmp_path / "red.jpg"
    p2 = tmp_path / "green.jpg"
    img1.save(p1)
    img2.save(p2)

    assert compute_phash(p1) != compute_phash(p2)
