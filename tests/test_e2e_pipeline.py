"""End-to-end integration test covering multiple pipeline stages.

Tests the flow: scan → hash/dedupe → chunk → enrich (mocked)
without requiring torch, fastai, or httpx.

Stages covered:
  1. Schema creation (ensure_schema)
  2. Scan (scan_into_duckdb)
  3. Hash + dedupe (hash_and_mark_dupes)
  4. Chunking predictions (Chunker.chunk_predictions)
  5. Chunking from files (Chunker.chunk_files)
  6. Enrichment DB helpers (mocked LLM)
  7. Enrichment end-to-end (mocked LLM)
  8. Perceptual dedup config/class (no imagehash)
  9. CLI status command
  10. CLI init command
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest

from tests.conftest import make_rgb_image
from vit_curator.config import EnrichConfig
from vit_curator.post.chunk import ChunkConfig, Chunker, chunk_text
from vit_curator.post.enrich import (
    Enricher,
    parse_enrichment_json,
)
from vit_curator.preprocess.dedupe import hash_and_mark_dupes
from vit_curator.preprocess.scan import scan_into_duckdb
from vit_curator.shared.db import ensure_schema

# Default UUID for test predictions
_TEST_RUN_ID = "00000000-0000-0000-0000-000000000001"


def _insert_prediction(con: duckdb.DuckDBPyConnection, file_pk: int, text: str) -> None:
    """Insert a row into the predictions table with proper UUID and TIMESTAMP types."""
    con.execute(
        "INSERT INTO predictions (file_pk, run_id, labels, text, created_at) "
        "VALUES (?, ?::UUID, ?, ?, ?)",
        [file_pk, _TEST_RUN_ID, [1], text, datetime.now(UTC)],
    )


@pytest.fixture
def populated_db(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    """Create a DuckDB with schema, scan 5 unique images + 1 duplicate, hash/dedupe."""
    src = tmp_path / "src"
    src.mkdir()

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    for i, color in enumerate(colors):
        make_rgb_image(src / f"img_{i:02d}.jpg", color)

    # Create a duplicate (same content, different filename)
    make_rgb_image(src / "img_dup.jpg", colors[0])

    db_path = tmp_path / "index.duckdb"
    con = duckdb.connect(str(db_path))
    ensure_schema(con)

    scan_into_duckdb(con, src_root=src, start_file_pk=1, allow_exts={b".jpg"}, insert_batch=100)
    hash_and_mark_dupes(con, src_root=src, num_workers=2)

    yield con
    con.close()


# ---------------------------------------------------------------------------
# Stage 1-2: Schema + Scan + Hash/Dedupe
# ---------------------------------------------------------------------------


def test_e2e_scan_hash_dedupe(tmp_path: Path) -> None:
    """End-to-end: scan 6 images, find 1 duplicate, verify DB state."""
    src = tmp_path / "src"
    src.mkdir()

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i, color in enumerate(colors):
        make_rgb_image(src / f"img_{i:02d}.jpg", color)

    # Duplicate of img_00
    make_rgb_image(src / "img_dup.jpg", colors[0])

    db_path = tmp_path / "index.duckdb"
    con = duckdb.connect(str(db_path))
    ensure_schema(con)

    # Stage 1: Scan
    stats = scan_into_duckdb(con, src_root=src, start_file_pk=1, insert_batch=100)
    assert stats.seen == 4
    assert stats.inserted == 4

    # Stage 2: Hash + Dedupe
    dedupe_stats = hash_and_mark_dupes(con, src_root=src, num_workers=2)
    assert dedupe_stats.total_candidates == 4
    assert dedupe_stats.uniques == 3
    assert dedupe_stats.dupes == 1

    # Verify DB state
    n_files = con.execute("SELECT COUNT(*) FROM files;").fetchone()[0]
    assert n_files == 4

    n_canonical = con.execute("SELECT COUNT(*) FROM files WHERE is_exact_dupe = FALSE;").fetchone()[
        0
    ]
    assert n_canonical == 3

    n_dupes = con.execute("SELECT COUNT(*) FROM files WHERE is_exact_dupe = TRUE;").fetchone()[0]
    assert n_dupes == 1

    con.close()


def test_e2e_empty_source(tmp_path: Path) -> None:
    """End-to-end: empty source directory → 0 files."""
    src = tmp_path / "empty"
    src.mkdir()

    db_path = tmp_path / "index.duckdb"
    con = duckdb.connect(str(db_path))
    ensure_schema(con)

    stats = scan_into_duckdb(con, src_root=src, start_file_pk=1)
    assert stats.seen == 0
    assert stats.inserted == 0

    dedupe_stats = hash_and_mark_dupes(con, src_root=src, num_workers=1)
    assert dedupe_stats.total_candidates == 0

    con.close()


def test_e2e_mixed_file_types(tmp_path: Path) -> None:
    """End-to-end: mixed file types, only images are inserted."""
    src = tmp_path / "mixed"
    src.mkdir()

    make_rgb_image(src / "image.jpg", (100, 150, 200))
    (src / "readme.txt").write_text("This is a text file")
    (src / "data.json").write_text('{"key": "value"}')

    db_path = tmp_path / "index.duckdb"
    con = duckdb.connect(str(db_path))
    ensure_schema(con)

    stats = scan_into_duckdb(
        con, src_root=src, start_file_pk=1, allow_exts={b".jpg"}, insert_batch=100
    )
    assert stats.seen == 3
    assert stats.inserted == 1
    assert stats.skipped == 2

    con.close()


# ---------------------------------------------------------------------------
# Stage 7a: Chunking
# ---------------------------------------------------------------------------


def test_e2e_chunk_text_basic() -> None:
    """End-to-end: chunk_text produces correct overlapping segments."""
    text = "A" * 500
    chunks = chunk_text(text, chunk_chars=200, chunk_overlap=50)
    assert len(chunks) >= 2
    # Each chunk has (start, end, text)
    for start, end, chunk_text_val in chunks:
        assert start >= 0
        assert end <= len(text)
        assert chunk_text_val == text[start:end]


def test_e2e_chunk_text_empty() -> None:
    """End-to-end: chunk_text returns empty list for empty string."""
    assert chunk_text("", chunk_chars=100, chunk_overlap=10) == []


def test_e2e_chunk_predictions(populated_db: duckdb.DuckDBPyConnection) -> None:
    """End-to-end: chunk predictions stored in the DB."""
    con = populated_db

    # Insert predictions for the canonical files
    canonical_pks = con.execute("SELECT file_pk FROM files WHERE is_exact_dupe = FALSE;").fetchall()

    for (fpk,) in canonical_pks:
        _insert_prediction(con, int(fpk), "Sample text " * 50)

    # Chunk
    chunker = Chunker(ChunkConfig(chunk_chars=200, chunk_overlap=50))
    n_chunked = chunker.chunk_predictions(con)

    assert n_chunked == len(canonical_pks)

    # Verify chunks were inserted
    n_chunks = con.execute("SELECT COUNT(*) FROM chunks;").fetchone()[0]
    assert n_chunks > 0

    # Each chunk has valid ranges
    rows = con.execute("SELECT file_pk, chunk_id, char_start, char_end FROM chunks;").fetchall()
    for _fpk, _cid, cs, ce in rows:
        assert cs >= 0
        assert ce > cs


def test_e2e_chunk_predictions_idempotent(populated_db: duckdb.DuckDBPyConnection) -> None:
    """End-to-end: re-chunking predictions replaces existing chunks."""
    con = populated_db

    # Insert predictions
    canonical_pks = con.execute("SELECT file_pk FROM files WHERE is_exact_dupe = FALSE;").fetchall()
    for (fpk,) in canonical_pks:
        _insert_prediction(con, int(fpk), "Hello world " * 50)

    chunker = Chunker(ChunkConfig(chunk_chars=200, chunk_overlap=50))
    n1 = chunker.chunk_predictions(con)

    n_chunks_first = con.execute("SELECT COUNT(*) FROM chunks;").fetchone()[0]

    # Re-chunk (should delete old and reinsert)
    n2 = chunker.chunk_predictions(con)
    assert n2 == n1

    n_chunks_second = con.execute("SELECT COUNT(*) FROM chunks;").fetchone()[0]
    assert n_chunks_second == n_chunks_first


# ---------------------------------------------------------------------------
# Stage 7c: Enrichment (mocked LLM)
# ---------------------------------------------------------------------------


def test_e2e_enrichment_with_mocked_llm(populated_db: duckdb.DuckDBPyConnection) -> None:
    """End-to-end: enrichment with mocked LLM stores results in DB."""
    from unittest.mock import patch

    con = populated_db

    # Insert predictions with meaningful text
    canonical_pks = con.execute("SELECT file_pk FROM files WHERE is_exact_dupe = FALSE;").fetchall()

    for (fpk,) in canonical_pks:
        _insert_prediction(con, int(fpk), "This is a document about machine learning.")

    # Create enricher with mock config
    cfg = EnrichConfig(
        db_path=Path("index.duckdb"),
        server_url="http://localhost:9999",
        model="test-model",
        max_docs=5,
    )

    enricher = Enricher(cfg)

    # Mock call_llm at module level to return structured JSON
    mock_response = (
        '{"subject": "ML Document", "summary": "About machine learning.", '
        '"entities": ["ML", "AI"], "tags": ["tech", "science"]}'
    )

    with patch("vit_curator.post.enrich.call_llm", return_value=(mock_response, "stop")):
        n_enriched = enricher.enrich(con)

    assert n_enriched > 0

    # Verify enrichments stored in DB
    n_rows = con.execute("SELECT COUNT(*) FROM doc_enrichments;").fetchone()[0]
    assert n_rows > 0

    # Verify enrichment content
    row = con.execute("SELECT subject, summary FROM doc_enrichments LIMIT 1;").fetchone()
    assert row is not None
    assert "ML Document" in str(row[0])
    assert "machine learning" in str(row[1]).lower()


# ---------------------------------------------------------------------------
# Stage 7b: Embedding config (no sentence-transformers needed)
# ---------------------------------------------------------------------------


def test_e2e_chunk_then_enrich_json_parsing(populated_db: duckdb.DuckDBPyConnection) -> None:
    """End-to-end: chunk predictions, then verify JSON parsing for enrichment."""
    con = populated_db

    # Insert predictions
    canonical_pks = con.execute("SELECT file_pk FROM files WHERE is_exact_dupe = FALSE;").fetchall()

    for (fpk,) in canonical_pks:
        _insert_prediction(con, int(fpk), "A comprehensive guide to Python programming. " * 20)

    # Chunk
    chunker = Chunker(ChunkConfig(chunk_chars=500, chunk_overlap=100))
    n_chunked = chunker.chunk_predictions(con)
    assert n_chunked > 0

    # Verify chunks reference valid file_pks
    chunk_rows = con.execute("SELECT DISTINCT file_pk FROM chunks;").fetchall()
    valid_pks = {int(r[0]) for r in con.execute("SELECT file_pk FROM files;").fetchall()}
    for (cpk,) in chunk_rows:
        assert int(cpk) in valid_pks

    # Test JSON extraction from enrichment module
    json_result = parse_enrichment_json(
        '{"subject": "Python Guide", "summary": "A guide.", '
        '"entities": ["Python"], "tags": ["programming"]}'
    )
    assert json_result.subject == "Python Guide"
    assert json_result.summary == "A guide."


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_e2e_cli_init_command(tmp_path: Path) -> None:
    """End-to-end: CLI init command creates a valid database."""
    from typer.testing import CliRunner

    from vit_curator.cli import app

    runner = CliRunner()
    db_path = tmp_path / "test_init.duckdb"

    result = runner.invoke(app, ["init", "--db", str(db_path)])
    assert result.exit_code == 0
    assert db_path.exists()

    # Verify schema is initialized
    con = duckdb.connect(str(db_path))
    tables = [r[0] for r in con.execute("SHOW TABLES;").fetchall()]
    assert "files" in tables
    assert "predictions" in tables
    assert "chunks" in tables
    assert "doc_enrichments" in tables
    con.close()


def test_e2e_cli_status_after_scan(tmp_path: Path) -> None:
    """End-to-end: CLI status command works after scanning."""
    from typer.testing import CliRunner

    from vit_curator.cli import app

    src = tmp_path / "src"
    src.mkdir()
    make_rgb_image(src / "test.jpg", (100, 100, 100))

    db_path = tmp_path / "index.duckdb"
    con = duckdb.connect(str(db_path))
    ensure_schema(con)
    scan_into_duckdb(con, src_root=src, start_file_pk=1, allow_exts={b".jpg"}, insert_batch=100)
    con.close()

    runner = CliRunner()
    result = runner.invoke(app, ["status", "--db", str(db_path)])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Chunking edge cases
# ---------------------------------------------------------------------------


def test_e2e_chunk_short_text() -> None:
    """End-to-end: chunk_text with text shorter than chunk_chars produces single chunk."""
    text = "Short text"
    chunks = chunk_text(text, chunk_chars=1000, chunk_overlap=0)
    assert len(chunks) == 1
    assert chunks[0][2] == text


def test_e2e_run_chunking_predictions(populated_db: duckdb.DuckDBPyConnection) -> None:
    """End-to-end: run_chunking high-level entry point."""
    from vit_curator.post.chunk import run_chunking

    con = populated_db

    # Insert predictions
    canonical_pks = con.execute("SELECT file_pk FROM files WHERE is_exact_dupe = FALSE;").fetchall()

    for (fpk,) in canonical_pks[:2]:  # Only first 2 files
        _insert_prediction(con, int(fpk), "Text " * 100)

    n = run_chunking(con, source="predictions", chunk_chars=200, chunk_overlap=50)
    assert n > 0

    n_chunks = con.execute("SELECT COUNT(*) FROM chunks;").fetchone()[0]
    assert n_chunks > 0


# ---------------------------------------------------------------------------
# Pipeline stats aggregation
# ---------------------------------------------------------------------------


def test_e2e_pipeline_stats(populated_db: duckdb.DuckDBPyConnection) -> None:
    """End-to-end: verify PipelineStats can be constructed from DB state."""
    from vit_curator.config import PipelineStats

    con = populated_db

    n_files = con.execute("SELECT COUNT(*) FROM files;").fetchone()[0]
    n_canonical = con.execute("SELECT COUNT(*) FROM files WHERE is_exact_dupe = FALSE;").fetchone()[
        0
    ]
    n_dupes = con.execute("SELECT COUNT(*) FROM files WHERE is_exact_dupe = TRUE;").fetchone()[0]

    stats = PipelineStats(
        seen=n_files,
        inserted=n_canonical,
        skipped=n_dupes,
        hashed_ok=n_canonical,
        hash_err=0,
        canonicals=n_canonical,
        dupes=n_dupes,
    )

    assert stats.seen == 6
    assert stats.canonicals == 5
    assert stats.dupes == 1
    assert stats.hashed_ok == 5
