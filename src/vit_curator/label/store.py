"""DuckDB store for VLM labeling tasks and predictions.

This module adapts the ocrmj_labeler duckdb_store to use the unified schema
(file_pk instead of asset_id, unified tables from shared/db.py).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC).replace(tzinfo=None)


def connect_label_db(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection for labeling operations."""
    from vit_curator.shared.db import connect  # noqa: PLC0415

    db = connect(db_path)
    return db.con


def ensure_run(
    con: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    model: str,
    server_url: str,
    prompt_version: str,
    max_tokens: int,
    settings_json: str | None = None,
    settings_hash: str | None = None,
    stage: str = "label",
) -> None:
    """Create a run record if it doesn't exist."""
    row = con.execute("SELECT run_id FROM runs WHERE run_id = ?", [run_id]).fetchone()
    if row:
        return
    con.execute(
        "INSERT INTO runs (run_id, started_at, model, server_url, prompt_version, "
        "max_tokens, settings_json, settings_hash, stage) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);",
        [
            run_id,
            _utcnow(),
            model,
            server_url,
            prompt_version,
            max_tokens,
            settings_json,
            settings_hash,
            stage,
        ],
    )


def ensure_tasks_for_run(
    con: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    sample_pool: int = 100,
) -> int:
    """Create pending tasks for this run from deduplicated canonical files.

    Returns the number of pending tasks after creation.
    """
    if sample_pool < 1 or sample_pool > 100:
        raise ValueError("sample_pool must be 1..100")

    p = sample_pool / 100.0
    acc = 0.0

    batch: list[int] = []

    def flush() -> None:
        if not batch:
            return
        con.executemany(
            "INSERT INTO tasks (file_pk, run_id, status, attempt) "
            "VALUES (?, ?, 'pending', 0) "
            "ON CONFLICT (file_pk, run_id) DO NOTHING",
            [(fpk, run_id) for fpk in batch],
        )
        batch.clear()

    con.execute("BEGIN;")
    try:
        # Use canonical files (non-dupes) with content_hash
        cur = con.execute(
            "SELECT file_pk FROM files "
            "WHERE content_hash IS NOT NULL AND is_exact_dupe = FALSE AND status = 1 "
            "ORDER BY file_pk"
        )

        while True:
            rows = cur.fetchmany(50_000)
            if not rows:
                break
            for (file_pk,) in rows:
                acc += p
                if acc >= 1.0:
                    acc -= 1.0
                    batch.append(int(file_pk))
                    if len(batch) >= 10_000:
                        flush()
        flush()
        con.execute("COMMIT;")
    except Exception:
        con.execute("ROLLBACK;")
        raise

    row = con.execute(
        "SELECT COUNT(*) FROM tasks WHERE run_id = ? AND status = 'pending'",
        [run_id],
    ).fetchone()
    return int(row[0]) if row is not None else 0


def crash_recover(con: duckdb.DuckDBPyConnection, *, run_id: str) -> int:
    """Crash recovery: processing -> pending for the run."""
    con.execute(
        "UPDATE tasks SET status='pending', started_at=NULL, finished_at=NULL, "
        "latency_ms=NULL, finish_reason=NULL, next_run_at=NULL "
        "WHERE run_id = ? AND status='processing'",
        [run_id],
    )
    row = con.execute(
        "SELECT COUNT(*) FROM tasks WHERE run_id = ? AND status = 'pending'",
        [run_id],
    ).fetchone()
    return int(row[0]) if row is not None else 0


def claim_pending_batch(
    con: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    limit: int,
) -> list[tuple[int, str, int]]:
    """Atomically claim a batch of pending tasks.

    Returns list of (file_pk, path, attempt) tuples.
    """
    if limit <= 0:
        return []

    now = _utcnow()
    con.execute("BEGIN;")
    try:
        raw_rows = con.execute(
            "SELECT t.file_pk, f.rel_path_blob, t.attempt "
            "FROM tasks t "
            "JOIN files f ON f.file_pk = t.file_pk "
            "WHERE t.run_id = ? AND t.status = 'pending' "
            "  AND (t.next_run_at IS NULL OR t.next_run_at <= ?) "
            "ORDER BY f.rel_path_blob LIMIT ?",
            [run_id, now, limit],
        ).fetchall()

        file_pks = [int(r[0]) for r in raw_rows]
        if file_pks:
            con.executemany(
                "UPDATE tasks SET status='processing', started_at=?, attempt=attempt+1, "
                "last_error=NULL, next_run_at=NULL "
                "WHERE run_id = ? AND file_pk = ? AND status='pending'",
                [(now, run_id, fpk) for fpk in file_pks],
            )

        con.execute("COMMIT;")
    except Exception:
        con.execute("ROLLBACK;")
        raise

    import os  # noqa: PLC0415

    return [(int(r[0]), os.fsdecode(r[1]), int(r[2]) + 1) for r in raw_rows]


def mark_done(
    con: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    results: Sequence[
        tuple[
            int,
            list[int],
            str | None,
            str | None,
            list[str] | None,
            str | None,
            str,
            float,
            str | None,
        ]
    ],
) -> None:
    """Write predictions and mark tasks done.

    results:
        (file_pk, labels, text, subject, entities,
         summary, raw_json, latency_ms, finish_reason)
    """
    if not results:
        return

    now = _utcnow()
    con.execute("BEGIN;")
    try:
        con.executemany(
            "INSERT INTO predictions (file_pk, run_id, labels, text, subject, entities, "
            "summary, raw_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (file_pk, run_id) DO UPDATE SET "
            "  labels=excluded.labels, text=excluded.text, subject=excluded.subject, "
            "  entities=excluded.entities, summary=excluded.summary, "
            "  raw_json=excluded.raw_json, created_at=excluded.created_at",
            [
                (fpk, run_id, lbls, text, subject, entities, summary, raw, now)
                for (fpk, lbls, text, subject, entities, summary, raw, _lat, _fr) in results
            ],
        )
        con.executemany(
            "UPDATE tasks SET status='done', finished_at=?, latency_ms=?, "
            "finish_reason=?, next_run_at=NULL "
            "WHERE run_id = ? AND file_pk = ? AND status='processing'",
            [
                (now, lat, fr, run_id, fpk)
                for (fpk, _lbls, _text, _subject, _entities, _summary, _raw, lat, fr) in results
            ],
        )
        con.execute("COMMIT;")
    except Exception:
        con.execute("ROLLBACK;")
        raise


def mark_done_text(
    con: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    results: Sequence[tuple[int, float, str | None]],
) -> None:
    """Mark tasks done without writing predictions (text-only mode)."""
    if not results:
        return

    now = _utcnow()
    con.execute("BEGIN;")
    try:
        con.executemany(
            "UPDATE tasks SET status='done', finished_at=?, latency_ms=?, "
            "finish_reason=?, next_run_at=NULL "
            "WHERE run_id = ? AND file_pk = ? AND status='processing'",
            [(now, lat, fr, run_id, fpk) for (fpk, lat, fr) in results],
        )
        con.execute("COMMIT;")
    except Exception:
        con.execute("ROLLBACK;")
        raise


def mark_error(
    con: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    errors: Sequence[tuple[int, str]],
) -> None:
    """Mark tasks as permanently errored."""
    if not errors:
        return

    now = _utcnow()
    con.execute("BEGIN;")
    try:
        con.executemany(
            "UPDATE tasks SET status='error', finished_at=?, last_error=?, "
            "latency_ms=NULL, finish_reason=NULL, next_run_at=NULL "
            "WHERE run_id = ? AND file_pk = ? AND status='processing'",
            [(now, err, run_id, fpk) for (fpk, err) in errors],
        )
        con.execute("COMMIT;")
    except Exception:
        con.execute("ROLLBACK;")
        raise


def mark_retry(
    con: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    retries: Sequence[tuple[int, str, dt.datetime]],
) -> None:
    """Requeue tasks with a scheduled next_run_at."""
    if not retries:
        return

    now = _utcnow()
    con.execute("BEGIN;")
    try:
        con.executemany(
            "UPDATE tasks SET status='pending', finished_at=?, last_error=?, "
            "latency_ms=NULL, finish_reason=NULL, next_run_at=? "
            "WHERE run_id = ? AND file_pk = ? AND status='processing'",
            [(now, err, retry_at, run_id, fpk) for (fpk, err, retry_at) in retries],
        )
        con.execute("COMMIT;")
    except Exception:
        con.execute("ROLLBACK;")
        raise


def summarize(con: duckdb.DuckDBPyConnection, *, run_id: str) -> dict[str, int]:
    """Return status counts for a run."""
    rows = con.execute(
        "SELECT status, COUNT(*) AS c FROM tasks WHERE run_id = ? GROUP BY status",
        [run_id],
    ).fetchall()

    out: dict[str, int] = {"pending": 0, "processing": 0, "done": 0, "error": 0}
    for status, c in rows:
        out[str(status)] = int(c)
    return out


def get_last_run(con: duckdb.DuckDBPyConnection) -> dict[str, Any] | None:
    """Return the most recent run."""
    row = con.execute(
        "SELECT run_id, started_at, model, server_url, prompt_version, max_tokens, "
        "settings_json, settings_hash, notes, stage "
        "FROM runs ORDER BY started_at DESC LIMIT 1"
    ).fetchone()

    if not row:
        return None

    return {
        "run_id": str(row[0]),
        "started_at": str(row[1]),
        "model": str(row[2]),
        "server_url": str(row[3]),
        "prompt_version": str(row[4]),
        "max_tokens": int(row[5]),
        "settings_json": None if row[6] is None else str(row[6]),
        "settings_hash": None if row[7] is None else str(row[7]),
        "notes": None if row[8] is None else str(row[8]),
        "stage": str(row[9]) if row[9] else "label",
    }


def load_labels(
    con: duckdb.DuckDBPyConnection,
    *,
    labels: Sequence[tuple[int, str, str, bool, int]],
) -> None:
    """Load label definitions into the database."""
    con.execute("BEGIN;")
    try:
        con.executemany(
            "INSERT INTO labels (label_id, name, description, enabled, sort_order) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (label_id) DO UPDATE SET "
            "  name=excluded.name, description=excluded.description, "
            "  enabled=excluded.enabled, sort_order=excluded.sort_order",
            labels,
        )
        con.execute("COMMIT;")
    except Exception:
        con.execute("ROLLBACK;")
        raise
