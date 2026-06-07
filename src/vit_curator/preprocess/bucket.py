from __future__ import annotations

from dataclasses import dataclass

import duckdb
from rich.console import Console


@dataclass(frozen=True)
class BucketAssignment:
    file_pk: int
    rel_path_blob: bytes
    ext_blob: bytes
    bucket_id: int
    bucket_pos: int


def iter_bucket_assignments(
    con: duckdb.DuckDBPyConnection,
    *,
    bucket_size: int = 10_000,
    metrics_every_s: float = 2.0,
    console: Console | None = None,
) -> list[BucketAssignment]:
    con.execute(
        "SELECT COUNT(*) FROM files "
        "WHERE content_hash IS NOT NULL "
        "AND (is_exact_dupe = FALSE OR is_exact_dupe IS NULL) AND status = 1;"
    ).fetchone()

    rows = con.execute(
        "SELECT file_pk, rel_path_blob, ext_blob, ok_index "
        "FROM files "
        "WHERE content_hash IS NOT NULL "
        "AND (is_exact_dupe = FALSE OR is_exact_dupe IS NULL) AND status = 1 "
        "ORDER BY file_pk;"
    ).fetchall()

    unassigned: list[tuple[int, bytes, bytes]] = []
    for file_pk, rel_blob, ext_blob, ok_idx in rows:
        if ok_idx is not None:
            bid = int(ok_idx) // bucket_size
            bpos = int(ok_idx) % bucket_size
            con.execute(
                "UPDATE files SET bucket_id=?, bucket_pos=? WHERE file_pk=?;",
                [bid, bpos, int(file_pk)],
            )
        else:
            unassigned.append((int(file_pk), bytes(rel_blob), bytes(ext_blob)))

    results: list[BucketAssignment] = []
    cur_bucket = 0
    cur_pos = 0

    for file_pk, rel_blob, ext_blob in unassigned:
        bid = cur_bucket
        bpos = cur_pos
        ok_index = cur_bucket * bucket_size + cur_pos
        con.execute(
            "UPDATE files SET ok_index=?, bucket_id=?, bucket_pos=? WHERE file_pk=?;",
            [ok_index, bid, bpos, file_pk],
        )
        results.append(
            BucketAssignment(
                file_pk=file_pk,
                rel_path_blob=rel_blob,
                ext_blob=ext_blob,
                bucket_id=bid,
                bucket_pos=bpos,
            )
        )
        cur_pos += 1
        if cur_pos >= bucket_size:
            cur_bucket += 1
            cur_pos = 0

    if console:
        n_buckets = cur_bucket + 1 if results else 0
        console.print(
            f"[magenta]bucket[/magenta] assigned={len(results):,} buckets={n_buckets}",
            highlight=False,
        )

    return results
