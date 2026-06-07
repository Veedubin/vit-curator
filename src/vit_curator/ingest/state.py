"""SQLite-based state tracking for ingest pipeline.

Tracks the status of each URL/file through download, extraction, and sorting.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .fsops import ensure_dir


@dataclass(frozen=True)
class IngestState:
    db_path: Path

    def open(self) -> sqlite3.Connection:
        ensure_dir(self.db_path.parent)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        self._init(conn)
        return conn

    def _init(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ingest_items (
              item_id INTEGER PRIMARY KEY AUTOINCREMENT,
              kind TEXT NOT NULL,
              src TEXT NOT NULL,
              local_path TEXT,
              status TEXT NOT NULL,
              err TEXT,
              updated_ts REAL NOT NULL
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ingest_items_status ON ingest_items(status);")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_ingest_items_kind_src ON ingest_items(kind, src);"
        )
        conn.commit()

    def upsert(
        self,
        conn: sqlite3.Connection,
        kind: str,
        src: str,
        status: str,
        local_path: str | None = None,
        err: str | None = None,
    ) -> None:
        now = time.time()
        conn.execute(
            """
            INSERT INTO ingest_items(kind, src, local_path, status, err, updated_ts)
            VALUES(?,?,?,?,?,?)
            ON CONFLICT(kind, src) DO UPDATE SET
              local_path=COALESCE(excluded.local_path, ingest_items.local_path),
              status=excluded.status,
              err=excluded.err,
              updated_ts=excluded.updated_ts;
            """,
            (kind, src, local_path, status, err, now),
        )
        conn.commit()
