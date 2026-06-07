"""Tests for vit_curator.label.store — DuckDB store operations for labeling."""

from __future__ import annotations

import duckdb


def _insert_file(con: duckdb.DuckDBPyConnection, file_pk: int, rel_path: str) -> None:
    """Insert a test file row into the files table with all required fields."""
    import os

    from vit_curator.shared.hashing import xxh3_128

    rel_blob = os.fsencode(rel_path)
    rel_path_hash = xxh3_128(rel_blob)
    ext_blob = os.fsencode(os.path.splitext(rel_path)[1].lstrip("."))
    content_hash = xxh3_128(rel_blob + b" content")
    con.execute(
        "INSERT INTO files "
        "(file_pk, rel_path_blob, rel_path_hash, ext_blob, size_bytes, mtime_ns, "
        "status, content_hash, is_exact_dupe) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [file_pk, rel_blob, rel_path_hash, ext_blob, 1234, 1000000, 1, content_hash, False],
    )


def test_connect_label_db_creates_schema(tmp_path) -> None:
    """Test that connect_label_db creates all expected tables."""
    from vit_curator.label.store import connect_label_db

    db_path = tmp_path / "test_label.duckdb"
    con = connect_label_db(db_path)
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert "files" in tables
    assert "labels" in tables
    assert "runs" in tables
    assert "tasks" in tables
    assert "predictions" in tables
    con.close()


def test_ensure_run(db: duckdb.DuckDBPyConnection) -> None:
    """Test creating a run record."""
    from vit_curator.label.store import ensure_run

    run_id = "00000000-0000-0000-0000-000000000001"
    ensure_run(
        db,
        run_id=run_id,
        model="test-model",
        server_url="http://localhost:8000",
        prompt_version="v1",
        max_tokens=64,
    )

    row = db.execute("SELECT model, server_url FROM runs WHERE run_id = ?", [run_id]).fetchone()
    assert row is not None
    assert row[0] == "test-model"
    assert row[1] == "http://localhost:8000"


def test_ensure_tasks_for_run(db: duckdb.DuckDBPyConnection) -> None:
    """Test creating tasks for a run."""
    from vit_curator.label.store import ensure_run, ensure_tasks_for_run

    _insert_file(db, 1, "/test/image1.jpg")
    _insert_file(db, 2, "/test/image2.jpg")

    run_id = "00000000-0000-0000-0000-000000000002"
    ensure_run(
        db,
        run_id=run_id,
        model="m",
        server_url="http://localhost:8000",
        prompt_version="v1",
        max_tokens=64,
    )
    ensure_tasks_for_run(db, run_id=run_id, sample_pool=100)

    count = db.execute("SELECT COUNT(*) FROM tasks WHERE run_id = ?", [run_id]).fetchone()
    assert count is not None and count[0] == 2


def test_claim_pending_batch(db: duckdb.DuckDBPyConnection) -> None:
    """Test claiming pending tasks."""
    from vit_curator.label.store import (
        claim_pending_batch,
        ensure_run,
        ensure_tasks_for_run,
    )

    _insert_file(db, 1, "/test/a.jpg")

    run_id = "00000000-0000-0000-0000-000000000003"
    ensure_run(
        db,
        run_id=run_id,
        model="m",
        server_url="http://localhost:8000",
        prompt_version="v1",
        max_tokens=64,
    )
    ensure_tasks_for_run(db, run_id=run_id)

    claimed = claim_pending_batch(db, run_id=run_id, limit=10)
    assert len(claimed) == 1
    assert claimed[0][0] == 1  # file_pk


def test_crash_recover(db: duckdb.DuckDBPyConnection) -> None:
    """Test crash recovery resets processing tasks to pending."""
    from vit_curator.label.store import (
        claim_pending_batch,
        crash_recover,
        ensure_run,
        ensure_tasks_for_run,
    )

    _insert_file(db, 1, "/test/a.jpg")

    run_id = "00000000-0000-0000-0000-000000000004"
    ensure_run(
        db,
        run_id=run_id,
        model="m",
        server_url="http://localhost:8000",
        prompt_version="v1",
        max_tokens=64,
    )
    ensure_tasks_for_run(db, run_id=run_id)

    _ = claim_pending_batch(db, run_id=run_id, limit=10)

    recovered = crash_recover(db, run_id=run_id)
    assert recovered == 1

    row = db.execute("SELECT status FROM tasks WHERE run_id = ?", [run_id]).fetchone()
    assert row is not None and row[0] == "pending"


def test_summarize(db: duckdb.DuckDBPyConnection) -> None:
    """Test summarize function returns expected structure."""
    from vit_curator.label.store import (
        ensure_run,
        ensure_tasks_for_run,
        summarize,
    )

    _insert_file(db, 1, "/test/a.jpg")

    run_id = "00000000-0000-0000-0000-000000000005"
    ensure_run(
        db,
        run_id=run_id,
        model="m",
        server_url="http://localhost:8000",
        prompt_version="v1",
        max_tokens=64,
    )
    ensure_tasks_for_run(db, run_id=run_id)

    summary = summarize(db, run_id=run_id)
    assert isinstance(summary, dict)
    assert "total" in summary or "pending" in summary or len(summary) >= 1
