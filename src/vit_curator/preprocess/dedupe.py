from __future__ import annotations

import os
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import duckdb
from rich.console import Console

from ..shared.errors import ErrorCode
from ..shared.hashing import xxh3_128_file


@dataclass(frozen=True)
class DedupeStats:
    total_candidates: int
    hashed_ok: int
    hash_err: int
    uniques: int
    dupes: int


def _rebuild_claims_if_needed(con: duckdb.DuckDBPyConnection) -> None:
    claim_count_row = con.execute("SELECT COUNT(*) FROM content_claims;").fetchone()
    claim_count = int(claim_count_row[0]) if claim_count_row else 0

    hashed_count_row = con.execute(
        "SELECT COUNT(*) FROM files WHERE content_hash IS NOT NULL;"
    ).fetchone()
    hashed_count = int(hashed_count_row[0]) if hashed_count_row else 0

    needs_rebuild = False
    if claim_count == 0 and hashed_count > 0:
        needs_rebuild = True
    else:
        bad_row = con.execute(
            "SELECT COUNT(*) "
            "FROM content_claims cc "
            "LEFT JOIN files f ON f.file_pk = cc.canonical_file_pk "
            "WHERE f.content_hash IS NULL OR f.content_hash != cc.content_hash;"
        ).fetchone()
        bad_count = int(bad_row[0]) if bad_row else 0
        missing_row = con.execute(
            "SELECT COUNT(*) "
            "FROM files f "
            "LEFT JOIN content_claims cc ON cc.content_hash = f.content_hash "
            "WHERE f.content_hash IS NOT NULL AND cc.content_hash IS NULL;"
        ).fetchone()
        missing_count = int(missing_row[0]) if missing_row else 0
        if bad_count > 0 or missing_count > 0:
            needs_rebuild = True

    if needs_rebuild:
        con.execute("DELETE FROM content_claims;")
        con.execute(
            "INSERT INTO content_claims "
            "SELECT content_hash, MIN(file_pk) AS canonical_file_pk "
            "FROM files WHERE content_hash IS NOT NULL GROUP BY content_hash;"
        )
        con.execute(
            "UPDATE files SET "
            "is_exact_dupe = (files.file_pk != cc.canonical_file_pk), "
            "dupe_of_file_pk = CASE "
            "WHEN files.file_pk != cc.canonical_file_pk "
            "THEN cc.canonical_file_pk ELSE NULL END "
            "FROM content_claims cc "
            "WHERE files.content_hash = cc.content_hash;"
        )


def _claim_or_mark_dupe(
    con: duckdb.DuckDBPyConnection, file_pk: int, digest: bytes
) -> tuple[bool, int | None]:
    con.execute(
        "INSERT OR IGNORE INTO content_claims(content_hash, canonical_file_pk) VALUES (?, ?);",
        [digest, int(file_pk)],
    )
    canon = con.execute(
        "SELECT canonical_file_pk FROM content_claims WHERE content_hash=?;",
        [digest],
    ).fetchone()
    canonical_pk = int(canon[0]) if canon and canon[0] is not None else int(file_pk)
    if canonical_pk == int(file_pk):
        con.execute(
            "UPDATE files SET is_exact_dupe=FALSE, dupe_of_file_pk=NULL WHERE file_pk=?;",
            [int(file_pk)],
        )
        return True, None
    con.execute(
        "UPDATE files SET is_exact_dupe=TRUE, dupe_of_file_pk=? WHERE file_pk=?;",
        [canonical_pk, int(file_pk)],
    )
    return False, canonical_pk


def hash_and_mark_dupes(
    con: duckdb.DuckDBPyConnection,
    src_root: Path,
    *,
    num_workers: int = 8,
    metrics_every_s: float = 2.0,
    console: Console | None = None,
    batch_rows: int = 25_000,
) -> DedupeStats:
    console = console or Console()

    _rebuild_claims_if_needed(con)

    total_candidates_row = con.execute(
        "SELECT COUNT(*) FROM files WHERE content_hash IS NULL;"
    ).fetchone()
    total_candidates = int(total_candidates_row[0]) if total_candidates_row else 0
    if total_candidates == 0:
        uniques_total_row = con.execute(
            "SELECT COUNT(*) FROM files "
            "WHERE dupe_of_file_pk IS NULL "
            "AND content_hash IS NOT NULL AND status=1;"
        ).fetchone()
        uniques_total = int(uniques_total_row[0]) if uniques_total_row else 0
        dupes_total_row = con.execute(
            "SELECT COUNT(*) FROM files WHERE dupe_of_file_pk IS NOT NULL;"
        ).fetchone()
        dupes_total = int(dupes_total_row[0]) if dupes_total_row else 0
        return DedupeStats(0, 0, 0, uniques_total, dupes_total)

    def _worker(file_pk: int, rel_blob: bytes) -> tuple[int, bytes | None, str | None]:
        rel = os.fsdecode(rel_blob)
        p = src_root / rel
        try:
            return file_pk, xxh3_128_file(p), None
        except Exception as e:
            return file_pk, None, f"{type(e).__name__}: {e}"

    hashed_ok = hash_err = uniques_added = dupes_added = 0
    done = 0

    t0 = time.time()
    last_print = t0

    with ThreadPoolExecutor(max_workers=max(1, int(num_workers))) as ex:
        last_pk: int | None = None
        while True:
            if last_pk is None:
                rows = con.execute(
                    "SELECT file_pk, rel_path_blob FROM files "
                    "WHERE content_hash IS NULL ORDER BY file_pk LIMIT ?;",
                    [int(batch_rows)],
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT file_pk, rel_path_blob FROM files "
                    "WHERE content_hash IS NULL AND file_pk > ? "
                    "ORDER BY file_pk LIMIT ?;",
                    [int(last_pk), int(batch_rows)],
                ).fetchall()

            if not rows:
                break

            last_pk = int(rows[-1][0])

            futs: list[Future] = [
                ex.submit(_worker, int(file_pk), rel_blob) for (file_pk, rel_blob) in rows
            ]
            for fut in as_completed(futs):
                file_pk, digest, err = fut.result()

                if err is not None or digest is None:
                    hash_err += 1
                    con.execute(
                        "UPDATE files SET status=2, err_code=? WHERE file_pk=?;",
                        [int(ErrorCode.ERR_HASH), int(file_pk)],
                    )
                else:
                    hashed_ok += 1
                    con.execute(
                        "UPDATE files SET content_hash=?, status=1, err_code=NULL WHERE file_pk=?;",
                        [digest, int(file_pk)],
                    )

                    is_unique, _canon = _claim_or_mark_dupe(con, int(file_pk), digest)
                    if is_unique:
                        uniques_added += 1
                    else:
                        dupes_added += 1

                done = hashed_ok + hash_err
                now = time.time()
                if metrics_every_s > 0 and (now - last_print) >= metrics_every_s:
                    rate = done / max(1e-9, (now - t0))
                    console.print(
                        f"[cyan]hash+dedupe[/cyan] {done:,}/{total_candidates:,} ({rate:,.1f}/s)  "
                        f"uniques+{uniques_added:,} dupes+{dupes_added:,} err+{hash_err:,}",
                        highlight=False,
                    )
                    last_print = now

    uniques_total_row = con.execute(
        "SELECT COUNT(*) FROM files "
        "WHERE dupe_of_file_pk IS NULL "
        "AND content_hash IS NOT NULL AND status=1;"
    ).fetchone()
    uniques_total = int(uniques_total_row[0]) if uniques_total_row else 0
    dupes_total_row = con.execute(
        "SELECT COUNT(*) FROM files WHERE dupe_of_file_pk IS NOT NULL;"
    ).fetchone()
    dupes_total = int(dupes_total_row[0]) if dupes_total_row else 0

    return DedupeStats(
        total_candidates=total_candidates,
        hashed_ok=hashed_ok,
        hash_err=hash_err,
        uniques=uniques_total,
        dupes=dupes_total,
    )
