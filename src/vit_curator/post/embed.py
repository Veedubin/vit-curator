"""Post-processing: semantic embeddings for chunked text.

Reads chunks from the chunks table, encodes them with an embedding model,
and stores vectors in the embeddings table.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import duckdb


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbedConfig:
    """Configuration for embedding generation.

    Attributes:
        model_name: HuggingFace model identifier (e.g., 'sentence-transformers/all-MiniLM-L6-v2').
        device: Compute device ('cpu' or 'cuda').
        batch_size: Number of chunks to encode per batch.
        max_length: Maximum token length per chunk.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 64
    max_length: int = 512


class Embedder:
    """Encodes text chunks into dense vectors using a sentence-transformers model."""

    def __init__(self, config: EmbedConfig | None = None) -> None:
        self.config = config or EmbedConfig()
        self._model: object | None = None

    def _load_model(self) -> object:
        """Lazy-load the embedding model."""
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install with: uv add sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(self.config.model_name, device=self.config.device)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into dense vectors.

        Args:
            texts: List of text strings to encode.

        Returns:
            Array of shape (len(texts), dim) with dtype float32.
        """
        model = self._load_model()
        vectors = model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype="float32")

    def embed_chunks(
        self,
        con: duckdb.DuckDBPyConnection,
        run_id: str | None = None,
        max_chunks: int | None = None,
    ) -> int:
        """Embed all unembedded chunks and store results.

        Args:
            con: DuckDB connection.
            run_id: Optional filter on predictions.run_id (via chunks -> predictions join).
            max_chunks: Optional limit on total chunks to embed.

        Returns:
            Total number of chunks embedded.
        """
        model_name = self.config.model_name
        total_unembedded = _count_unembedded(con, model_name)
        if total_unembedded == 0:
            return 0

        target = min(total_unembedded, max_chunks) if max_chunks else total_unembedded
        done = 0
        batch_num = 0

        while done < target:
            remain = target - done
            batch_size = min(self.config.batch_size, remain)
            rows = _fetch_unembedded(con, model_name, batch_size)
            if not rows:
                break

            batch_num += 1
            file_pks = [int(r[0]) for r in rows]
            chunk_ids = [int(r[1]) for r in rows]
            texts = [str(r[2]) for r in rows]

            vectors = self.encode(texts)
            _insert_embeddings(con, model_name, file_pks, chunk_ids, vectors)

            done += len(texts)

        return done


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _count_unembedded(con: duckdb.DuckDBPyConnection, model_name: str) -> int:
    row = con.execute(
        """
        SELECT COUNT(*)
        FROM chunks c
        LEFT JOIN embeddings e
          ON e.file_pk = c.file_pk AND e.chunk_id = c.chunk_id AND e.model_name = ?
        WHERE e.file_pk IS NULL
        """,
        [model_name],
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _fetch_unembedded(
    con: duckdb.DuckDBPyConnection,
    model_name: str,
    batch_size: int,
) -> list[tuple]:
    return con.execute(
        """
        SELECT c.file_pk, c.chunk_id, c.text
        FROM chunks c
        LEFT JOIN embeddings e
          ON e.file_pk = c.file_pk AND e.chunk_id = c.chunk_id AND e.model_name = ?
        WHERE e.file_pk IS NULL
        ORDER BY c.file_pk, c.chunk_id
        LIMIT ?
        """,
        [model_name, batch_size],
    ).fetchall()


def _insert_embeddings(
    con: duckdb.DuckDBPyConnection,
    model_name: str,
    file_pks: list[int],
    chunk_ids: list[int],
    vectors: np.ndarray,
) -> None:
    from datetime import datetime  # noqa: PLC0415

    dim = vectors.shape[1]
    now = datetime.now(UTC).isoformat()
    rows = [
        (int(fp), int(cid), model_name, int(dim), vec.astype("float32").tobytes(), now)
        for fp, cid, vec in zip(file_pks, chunk_ids, vectors, strict=False)
    ]
    con.executemany(
        """
        INSERT OR REPLACE INTO embeddings
        (file_pk, chunk_id, model_name, dim, vector, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def run_embedding(
    con: duckdb.DuckDBPyConnection,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
    batch_size: int = 64,
    max_chunks: int | None = None,
) -> int:
    """Embed all unembedded chunks with a sentence-transformers model.

    Args:
        con: DuckDB connection.
        model_name: HuggingFace model identifier.
        device: 'cpu' or 'cuda'.
        batch_size: Encoding batch size.
        max_chunks: Optional limit on total chunks.

    Returns:
        Number of chunks embedded.
    """
    cfg = EmbedConfig(model_name=model_name, device=device, batch_size=batch_size)
    embedder = Embedder(cfg)
    return embedder.embed_chunks(con, max_chunks=max_chunks)
