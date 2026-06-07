"""Unified DuckDB schema, connection helpers, and migration framework.

This module merges the schemas from data_janitor (files, content_claims,
presets, image_derivatives, transform_configs, file_transform_runs) and
ocr-my-junk (labels, runs, tasks, predictions) into a single DuckDB
database with consistent terminology (file_pk, not asset_id).

Schema versioning follows an additive-migration pattern: CREATE TABLE IF NOT
EXISTS for base tables, ALTER TABLE ADD COLUMN for new columns, and a
schema_migrations table tracks applied versions.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

SCHEMA_SQL = r"""
-- Core file tracking (from data_janitor, unified terminology)
CREATE TABLE IF NOT EXISTS files (
  file_pk UBIGINT PRIMARY KEY,
  rel_path_blob BLOB NOT NULL UNIQUE,
  rel_path_hash BLOB NOT NULL UNIQUE,
  ext_blob BLOB NOT NULL,
  size_bytes UBIGINT NOT NULL,
  mtime_ns UBIGINT NOT NULL,

  -- Exact-hash gate
  content_hash BLOB,
  is_exact_dupe BOOLEAN NOT NULL DEFAULT FALSE,
  dupe_of_file_pk UBIGINT,

  -- Hash status
  status USMALLINT NOT NULL DEFAULT 0,
  err_code INTEGER,

  -- Decode status (canonicals only)
  decode_status USMALLINT NOT NULL DEFAULT 0,
  decode_err_code INTEGER,
  decode_err_msg VARCHAR,

  -- Optional image metadata
  orig_w INTEGER,
  orig_h INTEGER,

  -- Dense bucket packing (canonicals only)
  ok_index UBIGINT,
  bucket_id INTEGER,
  bucket_pos INTEGER
);

-- One canonical row per content_hash (first writer wins).
CREATE TABLE IF NOT EXISTS content_claims (
  content_hash BLOB PRIMARY KEY,
  canonical_file_pk UBIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS presets (
  preset_id USMALLINT PRIMARY KEY,
  name VARCHAR NOT NULL UNIQUE,
  width INTEGER NOT NULL,
  height INTEGER NOT NULL,
  fmt VARCHAR NOT NULL,
  jpeg_quality INTEGER NOT NULL
);

-- Canonical transform configs (dedupe identical settings by settings_hash).
CREATE TABLE IF NOT EXISTS transform_configs (
  transform_cfg_id UBIGINT PRIMARY KEY,
  settings_json VARCHAR NOT NULL,
  settings_hash BLOB NOT NULL UNIQUE,
  algo_version VARCHAR NOT NULL,
  created_at_ns UBIGINT
);

-- Per-file transform results for a given transform config.
CREATE TABLE IF NOT EXISTS file_transform_runs (
  run_id UBIGINT PRIMARY KEY,
  file_pk UBIGINT NOT NULL,
  transform_cfg_id UBIGINT NOT NULL,
  status USMALLINT NOT NULL DEFAULT 0,
  err_code INTEGER,
  err_msg VARCHAR,
  bg VARCHAR,
  crop_x0 INTEGER,
  crop_y0 INTEGER,
  crop_x1 INTEGER,
  crop_y1 INTEGER,
  crop_clamped BOOLEAN,
  deskew_angle_deg DOUBLE,
  deskew_confidence DOUBLE,
  preview_w INTEGER,
  preview_h INTEGER,
  analysis_ms DOUBLE,
  created_at_ns UBIGINT,
  updated_at_ns UBIGINT,
  UNIQUE (file_pk, transform_cfg_id)
);

-- Derived outputs (preset fanout). transform_cfg_id=0 is identity/no-op.
CREATE TABLE IF NOT EXISTS image_derivatives (
  deriv_pk UBIGINT PRIMARY KEY,
  file_pk UBIGINT NOT NULL,
  preset_id USMALLINT NOT NULL,
  transform_cfg_id UBIGINT NOT NULL DEFAULT 0,
  run_id UBIGINT,
  out_rel_path BLOB NOT NULL,
  width INTEGER NOT NULL,
  height INTEGER NOT NULL,
  fmt VARCHAR NOT NULL,
  status USMALLINT NOT NULL DEFAULT 0,
  err_code INTEGER,
  err_msg VARCHAR,
  created_at_ns UBIGINT,
  UNIQUE (file_pk, preset_id, transform_cfg_id)
);

-- Labeling (from ocr-my-junk, asset_id → file_pk)
CREATE TABLE IF NOT EXISTS labels (
  label_id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT NOT NULL,
  enabled BOOLEAN NOT NULL DEFAULT TRUE,
  sort_order INTEGER
);

CREATE TABLE IF NOT EXISTS runs (
  run_id UUID PRIMARY KEY,
  started_at TIMESTAMP NOT NULL,
  model TEXT NOT NULL,
  server_url TEXT,
  prompt_version TEXT,
  max_tokens INTEGER,
  settings_json TEXT,
  settings_hash TEXT,
  notes TEXT,
  stage TEXT NOT NULL DEFAULT 'label'
);

CREATE TABLE IF NOT EXISTS tasks (
  file_pk UBIGINT NOT NULL,
  run_id UUID NOT NULL,
  status TEXT NOT NULL,
  attempt INTEGER NOT NULL DEFAULT 0,
  last_error TEXT,
  started_at TIMESTAMP,
  finished_at TIMESTAMP,
  latency_ms DOUBLE,
  finish_reason TEXT,
  next_run_at TIMESTAMP,
  completion_tokens INTEGER,
  PRIMARY KEY(file_pk, run_id)
);

CREATE TABLE IF NOT EXISTS predictions (
  file_pk UBIGINT NOT NULL,
  run_id UUID NOT NULL,
  labels INTEGER[] NOT NULL,
  text TEXT,
  subject TEXT,
  entities TEXT[],
  summary TEXT,
  raw_json TEXT,
  created_at TIMESTAMP NOT NULL,
  PRIMARY KEY(file_pk, run_id)
);

-- Training artifacts (new)
CREATE TABLE IF NOT EXISTS models (
  model_id UUID PRIMARY KEY,
  run_id UUID NOT NULL,
  arch TEXT NOT NULL,
  path TEXT NOT NULL,
  metrics_json TEXT,
  exported_formats TEXT[],
  created_at TIMESTAMP
);

-- Schema bookkeeping
CREATE TABLE IF NOT EXISTS schema_migrations (
  version INTEGER PRIMARY KEY,
  applied_at TIMESTAMP NOT NULL
);

-- Post-processing: text chunks from predictions/OCR
CREATE TABLE IF NOT EXISTS chunks (
  file_pk UBIGINT NOT NULL,
  chunk_id INTEGER NOT NULL,
  char_start INTEGER NOT NULL,
  char_end INTEGER NOT NULL,
  text VARCHAR NOT NULL,
  created_at TIMESTAMP NOT NULL,
  PRIMARY KEY (file_pk, chunk_id)
);

-- Post-processing: semantic embeddings per chunk
CREATE TABLE IF NOT EXISTS embeddings (
  file_pk UBIGINT NOT NULL,
  chunk_id INTEGER NOT NULL,
  model_name VARCHAR NOT NULL,
  dim INTEGER NOT NULL,
  vector BLOB NOT NULL,
  created_at TIMESTAMP NOT NULL,
  PRIMARY KEY (file_pk, chunk_id, model_name)
);

-- Post-processing: document enrichment (LLM-based subject, summary, entities, tags)
CREATE TABLE IF NOT EXISTS doc_enrichments (
  file_pk UBIGINT NOT NULL,
  model_name VARCHAR NOT NULL,
  subject TEXT,
  summary TEXT,
  doc_type TEXT,
  entities_json TEXT,
  tags_json TEXT,
  finish_reason TEXT,
  truncated BOOLEAN NOT NULL DEFAULT FALSE,
  text_len INTEGER,
  word_count INTEGER,
  raw_payload TEXT,
  created_at TIMESTAMP NOT NULL,
  PRIMARY KEY (file_pk, model_name)
);

-- Metadata key-value store
CREATE TABLE IF NOT EXISTS meta (
  k VARCHAR PRIMARY KEY,
  v BLOB NOT NULL
);
"""

# Indexes
INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS files_status_idx ON files(status);",
    "CREATE INDEX IF NOT EXISTS files_decode_status_idx ON files(decode_status);",
    "CREATE INDEX IF NOT EXISTS files_bucket_idx ON files(bucket_id);",
    "CREATE INDEX IF NOT EXISTS files_dupe_idx ON files(dupe_of_file_pk);",
    "CREATE INDEX IF NOT EXISTS deriv_status_idx ON image_derivatives(status);",
    "CREATE INDEX IF NOT EXISTS deriv_file_idx ON image_derivatives(file_pk);",
    "CREATE INDEX IF NOT EXISTS tr_file_cfg_idx ON file_transform_runs(file_pk, transform_cfg_id);",
    "CREATE INDEX IF NOT EXISTS tr_status_idx ON file_transform_runs(status);",
    "CREATE INDEX IF NOT EXISTS chunks_file_idx ON chunks(file_pk);",
    "CREATE INDEX IF NOT EXISTS embed_model_idx ON embeddings(model_name);",
    "CREATE INDEX IF NOT EXISTS enrich_model_idx ON doc_enrichments(model_name);",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DB:
    """Holds an open DuckDB connection and its path."""

    con: duckdb.DuckDBPyConnection
    path: Path


@dataclass(frozen=True)
class Preset:
    """A derivative preset definition."""

    preset_id: int
    name: str
    width: int
    height: int
    fmt: str
    jpeg_quality: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _table_names(con: duckdb.DuckDBPyConnection) -> set[str]:
    rows = con.execute("SHOW TABLES;").fetchall()
    return {str(r[0]) for r in rows}


def _column_names(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    rows = con.execute(f"PRAGMA table_info('{table}');").fetchall()
    return {str(r[1]) for r in rows}


def _utcnow_ns() -> int:
    """Return current UTC time as nanoseconds since epoch."""
    return int(time.time() * 1_000_000_000)


# ---------------------------------------------------------------------------
# Schema init + migrations
# ---------------------------------------------------------------------------


def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Create base schema and apply additive migrations."""
    con.execute(SCHEMA_SQL)

    # Create indexes (best-effort)
    for stmt in INDEX_SQL:
        with contextlib.suppress(Exception):
            con.execute(stmt)

    # Ensure identity transform cfg exists
    _ensure_identity_transform_cfg(con)

    # Record schema version
    con.execute(
        "INSERT OR IGNORE INTO schema_migrations "
        "(version, applied_at) VALUES (?, CURRENT_TIMESTAMP);",
        [SCHEMA_VERSION],
    )


def _ensure_identity_transform_cfg(con: duckdb.DuckDBPyConnection) -> None:
    """Ensure transform_cfg_id=0 exists as an identity/no-op transform."""
    row = con.execute(
        "SELECT transform_cfg_id FROM transform_configs WHERE transform_cfg_id=0;"
    ).fetchone()
    if row:
        return
    settings_hash = b"\x00" * 16
    con.execute(
        "INSERT INTO transform_configs "
        "(transform_cfg_id, settings_json, settings_hash, algo_version, created_at_ns) "
        "VALUES (0, '{}', ?, 'identity_v1', 0);",
        [settings_hash],
    )


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def connect(db_path: Path) -> DB:
    """Open a DuckDB connection, create parent dir, init schema, return DB."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute("PRAGMA enable_progress_bar=false;")
    con.execute("PRAGMA threads=8;")
    ensure_schema(con)
    return DB(con=con, path=db_path)


# ---------------------------------------------------------------------------
# Meta key-value helpers
# ---------------------------------------------------------------------------


def set_meta(con: duckdb.DuckDBPyConnection, k: str, v: bytes) -> None:
    con.execute("INSERT OR REPLACE INTO meta (k, v) VALUES (?, ?);", [k, v])


def get_meta(con: duckdb.DuckDBPyConnection, k: str) -> bytes | None:
    row = con.execute("SELECT v FROM meta WHERE k = ?;", [k]).fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# PK generators
# ---------------------------------------------------------------------------


def next_file_pk(con: duckdb.DuckDBPyConnection) -> int:
    row = con.execute("SELECT COALESCE(MAX(file_pk) + 1, 1) FROM files;").fetchone()
    return int(row[0]) if row else 1


def next_deriv_pk(con: duckdb.DuckDBPyConnection) -> int:
    row = con.execute("SELECT COALESCE(MAX(deriv_pk) + 1, 1) FROM image_derivatives;").fetchone()
    return int(row[0]) if row else 1


def next_transform_cfg_pk(con: duckdb.DuckDBPyConnection) -> int:
    row = con.execute(
        "SELECT COALESCE(MAX(transform_cfg_id) + 1, 1) FROM transform_configs;"
    ).fetchone()
    return int(row[0]) if row else 1


def next_transform_run_pk(con: duckdb.DuckDBPyConnection) -> int:
    row = con.execute("SELECT COALESCE(MAX(run_id) + 1, 1) FROM file_transform_runs;").fetchone()
    return int(row[0]) if row else 1


# ---------------------------------------------------------------------------
# Preset helpers
# ---------------------------------------------------------------------------


def ensure_preset_rows(
    con: duckdb.DuckDBPyConnection,
    presets: list[tuple[str, int, int]],
    fmt: str,
    jpeg_quality: int,
) -> None:
    existing = {r[0] for r in con.execute("SELECT name FROM presets;").fetchall()}
    cur = con.execute("SELECT COALESCE(MAX(preset_id) + 1, 1) FROM presets;").fetchone()
    next_id = int(cur[0]) if cur else 1
    rows: list[tuple[int, str, int, int, str, int]] = []
    for name, w, h in presets:
        if name in existing:
            continue
        rows.append((next_id, name, w, h, fmt, jpeg_quality))
        next_id += 1
    if rows:
        con.executemany(
            "INSERT INTO presets (preset_id, name, width, height, fmt, jpeg_quality) "
            "VALUES (?, ?, ?, ?, ?, ?);",
            rows,
        )


def load_presets(con: duckdb.DuckDBPyConnection) -> list[Preset]:
    rows = con.execute(
        "SELECT preset_id, name, width, height, fmt, jpeg_quality FROM presets ORDER BY preset_id;"
    ).fetchall()
    return [
        Preset(int(pid), str(name), int(w), int(h), str(fmt), int(jq))
        for (pid, name, w, h, fmt, jq) in rows
    ]


# ---------------------------------------------------------------------------
# Transform config helpers
# ---------------------------------------------------------------------------


def get_or_create_transform_cfg(
    con: duckdb.DuckDBPyConnection,
    *,
    settings_json: str,
    settings_hash: bytes,
    algo_version: str,
) -> int:
    row = con.execute(
        "SELECT transform_cfg_id FROM transform_configs WHERE settings_hash=?;",
        [settings_hash],
    ).fetchone()
    if row and row[0] is not None:
        return int(row[0])
    cfg_id = next_transform_cfg_pk(con)
    con.execute(
        "INSERT INTO transform_configs "
        "(transform_cfg_id, settings_json, settings_hash, algo_version, created_at_ns) "
        "VALUES (?, ?, ?, ?, ?);",
        [int(cfg_id), str(settings_json), settings_hash, str(algo_version), 0],
    )
    return int(cfg_id)


# ---------------------------------------------------------------------------
# Preset parsing
# ---------------------------------------------------------------------------


def parse_presets_arg(presets: str) -> list[tuple[str, int, int]]:
    """Parse preset argument string into list of (name, width, height) tuples.

    Format: "name=WxH" or "name=Size" (square) or just "Name=Size"
    Examples:
        - "thumb-32=32" -> ("thumb-32", 32, 32)
        - "medium=128" -> ("medium", 128, 128)
        - "square=256x256" -> ("square", 256, 256)
        - "a=256,b=128" -> [("a", 256, 256), ("b", 128, 128)]
    """
    out: list[tuple[str, int, int]] = []
    for token in [t.strip() for t in presets.split(",") if t.strip()]:
        if "=" in token:
            name, spec = token.split("=", 1)
            name = name.strip()
        else:
            name, spec = token, token

        if "x" in spec:
            w_s, h_s = spec.lower().split("x", 1)
            w, h = int(w_s), int(h_s)
        else:
            w = h = int(spec)

        out.append((name, w, h))
    return out
