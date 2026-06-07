from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import duckdb

from ..shared.hashing import fsencode_relpath, xxh3_128

DEFAULT_IMAGE_EXTS: set[bytes] = {
    b".jpg",
    b".jpeg",
    b".png",
    b".bmp",
    b".tif",
    b".tiff",
    b".webp",
    b".pdf",
}


@dataclass(frozen=True)
class ScanStats:
    seen: int
    inserted: int
    skipped: int


def iter_files(root: Path) -> Iterable[os.DirEntry]:
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for ent in it:
                    try:
                        if ent.is_dir(follow_symlinks=False):
                            stack.append(Path(ent.path))
                        elif ent.is_file(follow_symlinks=False):
                            yield ent
                    except OSError:
                        continue
        except OSError:
            continue


def scan_into_duckdb(
    con: duckdb.DuckDBPyConnection,
    src_root: Path,
    start_file_pk: int,
    allow_exts: set[bytes] | None = None,
    max_files: int | None = None,
    insert_batch: int = 50_000,
) -> ScanStats:
    rows: list[tuple[int, bytes, bytes, bytes, int, int]] = []

    seen = inserted = skipped = 0
    file_pk = start_file_pk

    sql = (
        "INSERT OR IGNORE INTO files "
        "(file_pk, rel_path_blob, rel_path_hash, ext_blob, size_bytes, mtime_ns) "
        "VALUES (?, ?, ?, ?, ?, ?);"
    )

    def _flush_rows(rows: list[tuple[int, bytes, bytes, bytes, int, int]]) -> int:
        if not rows:
            return 0

        con.executemany(sql, rows)

        start_pk = int(rows[0][0])
        end_pk = int(rows[-1][0])
        inserted_row = con.execute(
            "SELECT COUNT(*) FROM files WHERE file_pk >= ? AND file_pk <= ?;",
            [start_pk, end_pk],
        ).fetchone()
        inserted_count = int(inserted_row[0]) if inserted_row else 0

        rel_hashes = [r[2] for r in rows]
        existing = con.execute(
            "SELECT rel_path_hash, file_pk, size_bytes, mtime_ns "
            "FROM files WHERE rel_path_hash = ANY(?);",
            [rel_hashes],
        ).fetchall()
        existing_map = {
            bytes(rel_hash): (int(file_pk), int(size_bytes), int(mtime_ns))
            for rel_hash, file_pk, size_bytes, mtime_ns in existing
        }

        updates: list[tuple[int, int, bytes, int]] = []
        changed_pks: list[int] = []
        for _file_pk, _rel_blob, rel_hash, ext, size_bytes, mtime_ns in rows:
            existing_row = existing_map.get(rel_hash)
            if existing_row is None:
                continue
            existing_pk, existing_size, existing_mtime = existing_row
            if int(size_bytes) == existing_size and int(mtime_ns) == existing_mtime:
                continue
            updates.append((int(size_bytes), int(mtime_ns), ext, int(existing_pk)))
            changed_pks.append(int(existing_pk))

        if updates:
            con.executemany(
                "UPDATE files SET "
                "size_bytes=?, mtime_ns=?, ext_blob=?, "
                "content_hash=NULL, is_exact_dupe=FALSE, dupe_of_file_pk=NULL, "
                "status=0, err_code=NULL, "
                "decode_status=0, decode_err_code=NULL, decode_err_msg=NULL, "
                "orig_w=NULL, orig_h=NULL, "
                "ok_index=NULL, bucket_id=NULL, bucket_pos=NULL "
                "WHERE file_pk=?;",
                updates,
            )
            con.execute(
                "DELETE FROM image_derivatives WHERE file_pk = ANY(?);",
                [changed_pks],
            )
            con.execute(
                "DELETE FROM file_transform_runs WHERE file_pk = ANY(?);",
                [changed_pks],
            )

        rows.clear()
        return inserted_count

    for ent in iter_files(src_root):
        if max_files is not None and seen >= max_files:
            break

        seen += 1

        p = Path(ent.path)
        ext = os.fsencode(p.suffix.lower())
        if allow_exts is not None and ext not in allow_exts:
            skipped += 1
            continue

        try:
            st = ent.stat(follow_symlinks=False)
        except OSError:
            skipped += 1
            continue

        rel = os.path.relpath(p, src_root)
        rel_blob = fsencode_relpath(rel)
        rel_hash = xxh3_128(rel_blob)

        rows.append((file_pk, rel_blob, rel_hash, ext, int(st.st_size), int(st.st_mtime_ns)))
        file_pk += 1

        if len(rows) >= insert_batch:
            inserted += _flush_rows(rows)

    if rows:
        inserted += _flush_rows(rows)

    return ScanStats(seen=seen, inserted=inserted, skipped=skipped)
