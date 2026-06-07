"""Tests for vit_curator.shared.db — schema creation and connection helpers."""

from __future__ import annotations

from pathlib import Path

import duckdb


def test_ensure_schema_creates_all_tables(db: duckdb.DuckDBPyConnection) -> None:
    """Test that ensure_schema creates all 13 expected tables."""
    tables = {r[0] for r in db.execute("SHOW TABLES").fetchall()}
    expected = {
        "files",
        "content_claims",
        "presets",
        "transform_configs",
        "file_transform_runs",
        "image_derivatives",
        "labels",
        "runs",
        "tasks",
        "predictions",
        "models",
        "schema_migrations",
        "meta",
    }
    assert expected.issubset(tables), f"Missing tables: {expected - tables}"


def test_ensure_schema_idempotent(db: duckdb.DuckDBPyConnection) -> None:
    """Test that calling ensure_schema twice doesn't error."""
    from vit_curator.shared.db import ensure_schema

    # Should not raise on second call
    ensure_schema(db)


def test_connect_creates_db(tmp_path: Path) -> None:
    """Test that connect() creates a new database file."""
    from vit_curator.shared.db import connect

    db_path = tmp_path / "test_connect.duckdb"
    database = connect(db_path)
    con = database.con
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert "files" in tables
    con.close()


def test_parse_presets_arg_single() -> None:
    """Test parsing a single preset."""
    from vit_curator.shared.db import parse_presets_arg

    result = parse_presets_arg("thumb-32=32")
    assert len(result) == 1
    name, w, h = result[0]
    assert name == "thumb-32"
    assert w == 32
    assert h == 32


def test_parse_presets_arg_multiple() -> None:
    """Test parsing multiple presets."""
    from vit_curator.shared.db import parse_presets_arg

    result = parse_presets_arg("thumb-32=32,medium-128=128")
    assert len(result) == 2
    assert result[0] == ("thumb-32", 32, 32)
    assert result[1] == ("medium-128", 128, 128)


def test_parse_presets_arg_rectangular() -> None:
    """Test parsing a rectangular preset (WxH)."""
    from vit_curator.shared.db import parse_presets_arg

    result = parse_presets_arg("square=256x256")
    assert result == [("square", 256, 256)]


def test_parse_presets_arg_empty() -> None:
    """Test parsing an empty presets string."""
    from vit_curator.shared.db import parse_presets_arg

    result = parse_presets_arg("")
    assert result == []


def test_files_table_schema_columns(db: duckdb.DuckDBPyConnection) -> None:
    """Test that files table has the expected column names."""
    columns = {r[0] for r in db.execute("DESCRIBE files").fetchall()}
    assert "file_pk" in columns
    assert "rel_path_blob" in columns
    assert "ext_blob" in columns
    assert "content_hash" in columns
    assert "is_exact_dupe" in columns
    assert "dupe_of_file_pk" in columns
    assert "status" in columns
