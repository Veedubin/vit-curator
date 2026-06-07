"""Post-processing: document chunking for text extracted by OCR or VLM.

Reads text from predictions table (or external .txt files) and produces
overlapping character-based chunks stored in the chunks DuckDB table.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from pathlib import Path

import duckdb

# ---------------------------------------------------------------------------
# Chunking logic
# ---------------------------------------------------------------------------


def chunk_text(text: str, chunk_chars: int, chunk_overlap: int) -> list[tuple[int, int, str]]:
    """Simple character-based chunking with overlap.

    Returns list of (char_start, char_end, chunk_text).
    """
    n = len(text)
    if n == 0:
        return []
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_chars:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_chars")

    chunks: list[tuple[int, int, str]] = []
    step = chunk_chars - chunk_overlap
    start = 0
    while start < n:
        end = min(start + chunk_chars, n)
        chunk = text[start:end]
        chunks.append((start, end, chunk))
        if end == n:
            break
        start += step
    return chunks


# ---------------------------------------------------------------------------
# Chunker class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkConfig:
    """Configuration for document chunking.

    Attributes:
        chunk_chars: Maximum characters per chunk.
        chunk_overlap: Character overlap between consecutive chunks.
        source_column: DuckDB column to read text from ('text' for predictions.text).
    """

    chunk_chars: int = 1200
    chunk_overlap: int = 200
    source_column: str = "text"


class Chunker:
    """Chunks documents from predictions or files and stores in DuckDB."""

    def __init__(self, config: ChunkConfig | None = None) -> None:
        self.config = config or ChunkConfig()

    def chunk_predictions(
        self,
        con: duckdb.DuckDBPyConnection,
        run_id: str | None = None,
        max_docs: int | None = None,
    ) -> int:
        """Read text from predictions table, chunk, and insert into chunks table.

        Args:
            con: DuckDB connection.
            run_id: Optional run_id filter. If None, chunks all predictions.
            max_docs: Optional limit on number of docs to process.

        Returns:
            Number of documents chunked.
        """
        # Build query to fetch text
        sql = "SELECT file_pk, text FROM predictions WHERE text IS NOT NULL AND text != ''"
        params: list = []
        if run_id is not None:
            sql += " AND run_id = ?"
            params.append(run_id)
        if max_docs is not None:
            sql += f" LIMIT {int(max_docs)}"

        rows = con.execute(sql, params).fetchall()
        if not rows:
            return 0

        # Clear existing chunks for these files (re-chunk idempotently)
        file_pks = [int(r[0]) for r in rows]
        placeholders = ",".join("?" * len(file_pks))
        con.execute(f"DELETE FROM chunks WHERE file_pk IN ({placeholders});", file_pks)

        processed = 0
        now_iso = _utc_iso()
        for file_pk, text in rows:
            if not text:
                continue
            chunks = chunk_text(str(text), self.config.chunk_chars, self.config.chunk_overlap)
            if not chunks:
                continue

            # Insert chunks
            chunk_rows = [
                (int(file_pk), idx, start, end, chunk_text_val, now_iso)
                for idx, (start, end, chunk_text_val) in enumerate(chunks)
            ]
            con.executemany(
                "INSERT INTO chunks (file_pk, chunk_id, char_start, char_end, text, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?);",
                chunk_rows,
            )
            processed += 1

        return processed

    def chunk_files(
        self,
        con: duckdb.DuckDBPyConnection,
        text_dir: Path,
        pattern: str = "*.txt",
        max_docs: int | None = None,
    ) -> int:
        """Read .txt files from disk, chunk, and insert into chunks table.

        Args:
            con: DuckDB connection.
            text_dir: Directory containing .txt files (one per source file).
            pattern: Glob pattern for text files.
            max_docs: Optional limit.

        Returns:
            Number of documents chunked.
        """
        files = sorted(text_dir.glob(pattern))
        if max_docs is not None:
            files = files[:max_docs]

        if not files:
            return 0

        processed = 0
        now_iso = _utc_iso()
        for txt_path in files:
            text = txt_path.read_text(encoding="utf-8", errors="replace")
            if not text:
                continue
            chunks = chunk_text(text, self.config.chunk_chars, self.config.chunk_overlap)
            if not chunks:
                continue

            # Derive file_pk from filename or insert a new file record
            rel_name = txt_path.stem
            row = con.execute(
                "SELECT file_pk FROM files WHERE rel_path_blob = ?;", [rel_name.encode()]
            ).fetchone()
            if row is None:
                # Skip files not in the files table
                continue
            file_pk = int(row[0])

            # Delete old chunks for this file
            con.execute("DELETE FROM chunks WHERE file_pk = ?;", [file_pk])

            chunk_rows = [
                (file_pk, idx, start, end, chunk_text_val, now_iso)
                for idx, (start, end, chunk_text_val) in enumerate(chunks)
            ]
            con.executemany(
                "INSERT INTO chunks (file_pk, chunk_id, char_start, char_end, text, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?);",
                chunk_rows,
            )
            processed += 1

        return processed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    from datetime import datetime  # noqa: PLC0415

    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------


def run_chunking(
    con: duckdb.DuckDBPyConnection,
    source: str = "predictions",
    chunk_chars: int = 1200,
    chunk_overlap: int = 200,
    run_id: str | None = None,
    text_dir: Path | None = None,
    max_docs: int | None = None,
) -> int:
    """High-level entry point for chunking.

    Args:
        con: DuckDB connection.
        source: 'predictions' to read from predictions table, 'files' to read from disk.
        chunk_chars: Max characters per chunk.
        chunk_overlap: Character overlap between chunks.
        run_id: Filter predictions by run_id (predictions source only).
        text_dir: Directory with .txt files (files source only).
        max_docs: Optional limit on docs to process.

    Returns:
        Number of documents chunked.
    """
    cfg = ChunkConfig(chunk_chars=chunk_chars, chunk_overlap=chunk_overlap)
    chunker = Chunker(cfg)

    if source == "predictions":
        return chunker.chunk_predictions(con, run_id=run_id, max_docs=max_docs)
    if source == "files":
        if text_dir is None:
            raise ValueError("text_dir required when source='files'")
        return chunker.chunk_files(con, text_dir, max_docs=max_docs)
    raise ValueError(f"Unknown chunking source: {source}")
