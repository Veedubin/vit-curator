"""Tests for vit_curator.preprocess — rescan and hash/dedupe integration."""

from __future__ import annotations

import os
from pathlib import Path

import duckdb

from tests.conftest import make_rgb_image


def test_rescan_resets_modified_file(tmp_path: Path) -> None:
    """Test that rescanning a modified file resets its status and clears derivatives."""
    from vit_curator.preprocess.dedupe import hash_and_mark_dupes
    from vit_curator.preprocess.scan import scan_into_duckdb
    from vit_curator.shared.db import ensure_schema

    src = tmp_path / "src"
    src.mkdir()

    img_path = src / "a.jpg"
    make_rgb_image(img_path, (10, 20, 30))

    con = duckdb.connect(str(tmp_path / "idx.duckdb"))
    ensure_schema(con)

    scan_into_duckdb(con, src, start_file_pk=1)
    stats = hash_and_mark_dupes(con, src, num_workers=1, metrics_every_s=0)
    assert stats.hashed_ok == 1

    row = con.execute("SELECT content_hash FROM files WHERE file_pk=1;").fetchone()
    assert row is not None and row[0] is not None
    old_hash = bytes(row[0])

    con.execute(
        "INSERT INTO image_derivatives "
        "(deriv_pk, file_pk, preset_id, transform_cfg_id, "
        "out_rel_path, width, height, fmt, status) "
        "VALUES (1, 1, 1, 0, 'a', 10, 10, 'jpeg', 1);"
    )

    make_rgb_image(img_path, (100, 110, 120))
    os.utime(img_path, None)

    scan_into_duckdb(con, src, start_file_pk=1)

    row2 = con.execute(
        "SELECT status, content_hash, is_exact_dupe, dupe_of_file_pk, decode_status, ok_index "
        "FROM files WHERE file_pk=1;"
    ).fetchone()
    assert row2 is not None
    status, content_hash, is_dupe, dupe_of, decode_status, ok_index = row2
    assert status == 0
    assert content_hash is None
    assert is_dupe is False or is_dupe is False
    assert dupe_of is None
    assert decode_status == 0
    assert ok_index is None

    deriv_count = con.execute("SELECT COUNT(*) FROM image_derivatives WHERE file_pk=1;").fetchone()
    assert deriv_count is not None and int(deriv_count[0]) == 0

    stats2 = hash_and_mark_dupes(con, src, num_workers=1, metrics_every_s=0)
    assert stats2.hashed_ok == 1
    row3 = con.execute("SELECT content_hash FROM files WHERE file_pk=1;").fetchone()
    assert row3 is not None and row3[0] is not None
    assert bytes(row3[0]) != old_hash

    con.close()
