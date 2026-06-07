"""End-to-end integration test for unified pipeline with synthetic dataset.

This test exercises the full flow:
  1. Create synthetic images (RGB)
  2. Ingest → scan into DuckDB
  3. Hash + dedupe
  4. Decode/derivative generation (if torch available)
  5. Verify database state at each stage

Requires torch for derivative generation; falls back to scan+hash+dedupe only.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pytest
from PIL import Image

from vit_curator.preprocess.dedupe import hash_and_mark_dupes
from vit_curator.preprocess.scan import scan_into_duckdb
from vit_curator.shared.db import ensure_schema

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """Create a small synthetic dataset with 5 unique images + 1 duplicate."""
    src = tmp_path / "synthetic_src"
    src.mkdir()

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    for i, color in enumerate(colors):
        img = Image.new("RGB", (64, 64), color)
        img.save(src / f"img_{i:02d}.jpg")

    # Create a duplicate (same content, different filename)
    img = Image.new("RGB", (64, 64), colors[0])
    img.save(src / "img_dup.jpg")

    return src


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_end_to_end_scan_dedupe(tmp_path: Path, synthetic_dataset: Path) -> None:
    """Test scan + hash + dedupe on synthetic dataset (no torch required)."""
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    db_path = out / "index.duckdb"

    con = duckdb.connect(str(db_path))
    ensure_schema(con)

    # Stage 1: Scan
    stats = scan_into_duckdb(
        con,
        src_root=synthetic_dataset,
        start_file_pk=1,
        allow_exts={b".jpg", b".jpeg"},
        insert_batch=100,
    )
    assert stats.seen == 6  # 5 unique + 1 duplicate

    # Stage 2: Hash + dedupe
    dedupe_stats = hash_and_mark_dupes(con, src_root=synthetic_dataset, num_workers=2)
    assert dedupe_stats.total_candidates == 6
    assert dedupe_stats.uniques == 5
    assert dedupe_stats.dupes == 1

    # Verify database state
    n_files = con.execute("SELECT COUNT(*) FROM files;").fetchone()[0]
    assert n_files == 6

    n_canonical = con.execute("SELECT COUNT(*) FROM files WHERE is_exact_dupe = FALSE;").fetchone()[
        0
    ]
    assert n_canonical == 5

    n_dupes = con.execute("SELECT COUNT(*) FROM files WHERE is_exact_dupe = TRUE;").fetchone()[0]
    assert n_dupes == 1

    n_claims = con.execute("SELECT COUNT(*) FROM content_claims;").fetchone()[0]
    assert n_claims == 5

    con.close()


@pytest.mark.skipif(
    not os.environ.get("VIT_CURATOR_TEST_TORCH"),
    reason="torch not installed; set VIT_CURATOR_TEST_TORCH=1 to enable",
)
def test_end_to_end_with_derivatives(tmp_path: Path, synthetic_dataset: Path) -> None:
    """Test full pipeline including derivative generation (requires torch)."""
    from vit_curator.config import LinkMode, RunConfig
    from vit_curator.preprocess.derivatives import run_pipeline

    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)

    cfg = RunConfig(
        src_root=synthetic_dataset,
        out_root=out,
        max_files=None,
        bucket_size=100,
        link_mode=LinkMode("copy"),
        hash_workers=2,
        scan_insert_batch=100,
        decode_backend="cpu",  # type: ignore[arg-type]
        device="cpu",  # type: ignore[arg-type]
        presets_arg="thumb-32=32",
        fmt="jpeg",  # type: ignore[arg-type]
        jpeg_quality=85,
        preserve_source=False,
        preserve_color=True,
        preserve_quality=False,
        decode_batch=16,
        inflight_batches=2,
        writer_workers=2,
        metrics_every_s=0,
        dali_batch_multiplier=4,
    )

    run_pipeline(cfg)

    db_path = out / "index.duckdb"
    assert db_path.exists()

    con = duckdb.connect(str(db_path))

    # Verify derivatives were created for all canonicals
    row = con.execute("SELECT COUNT(*) FROM image_derivatives WHERE status = 1;").fetchone()
    deriv_count = int(row[0]) if row and row[0] is not None else 0
    assert deriv_count == 5  # 5 canonicals, each gets 1 derivative

    # Verify file table state
    n_files = con.execute("SELECT COUNT(*) FROM files;").fetchone()[0]
    assert n_files == 6

    n_canonical = con.execute("SELECT COUNT(*) FROM files WHERE is_exact_dupe = FALSE;").fetchone()[
        0
    ]
    assert n_canonical == 5

    con.close()


def test_end_to_end_empty_source(tmp_path: Path) -> None:
    """Test pipeline with empty source directory."""
    src = tmp_path / "empty_src"
    src.mkdir()
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    db_path = out / "index.duckdb"

    con = duckdb.connect(str(db_path))
    ensure_schema(con)

    stats = scan_into_duckdb(
        con,
        src_root=src,
        start_file_pk=1,
        allow_exts={b".jpg"},
        insert_batch=100,
    )
    assert stats.seen == 0

    dedupe_stats = hash_and_mark_dupes(con, src_root=src, num_workers=1)
    assert dedupe_stats.total_candidates == 0

    con.close()


def test_end_to_end_non_image_files(tmp_path: Path) -> None:
    """Test that non-image files are ignored during scan."""
    src = tmp_path / "mixed_src"
    src.mkdir()
    (src / "image.jpg").write_bytes(b"fake_jpg")
    (src / "readme.txt").write_text("This is a text file")
    (src / "data.json").write_text('{"key": "value"}')

    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    db_path = out / "index.duckdb"

    con = duckdb.connect(str(db_path))
    ensure_schema(con)

    stats = scan_into_duckdb(
        con,
        src_root=src,
        start_file_pk=1,
        allow_exts={b".jpg"},
        insert_batch=100,
    )
    assert stats.seen == 3  # All files seen
    assert stats.inserted == 1  # Only .jpg inserted
    assert stats.skipped == 2  # .txt and .json skipped
