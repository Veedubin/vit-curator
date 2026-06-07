"""Data loading from DuckDB for training.

Adapted from ocrmj_labeler to use the unified schema (file_pk, files table).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def load_training_data(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    min_confidence: float = 0.0,
    exclude_labels: list[int] | None = None,
) -> pd.DataFrame:
    """Load training data from predictions table.

    Returns DataFrame with columns:
    - path: image path (decoded from rel_path_blob)
    - labels: list of label IDs
    - label_names: list of label names
    - text: OCR text (optional)
    """
    exclude_labels = exclude_labels or []
    exclude_filter = ""
    params: list = [run_id]

    if exclude_labels:
        exclude_ids = [int(lid) for lid in exclude_labels]
        exclude_filter = "AND NOT list_has_any(p.labels, ?)"
        params.append(exclude_ids)

    query = f"""
        SELECT
            os.fsencode(f.rel_path_blob) AS path,
            p.labels,
            p.text,
            p.subject,
            p.entities,
            p.summary,
            t.latency_ms,
            t.finish_reason
        FROM predictions p
        JOIN files f ON f.file_pk = p.file_pk
        JOIN tasks t ON t.file_pk = p.file_pk AND t.run_id = p.run_id
        WHERE p.run_id = ?
          AND t.status = 'done'
          {exclude_filter}
        ORDER BY f.rel_path_blob
    """

    df = conn.execute(query, params).fetchdf()

    if len(df) == 0:
        return pd.DataFrame(columns=["path", "labels", "label_names", "text"])

    label_vocab = get_label_vocab(conn)

    df["labels"] = df["labels"].apply(lambda x: list(x) if x is not None else [])

    def get_label_names(label_ids):
        return [label_vocab.get(int(lid), f"label_{lid}") for lid in label_ids]

    df["label_names"] = df["labels"].apply(get_label_names)
    df["path"] = df["path"].astype(str)

    return df


def get_label_vocab(conn: duckdb.DuckDBPyConnection) -> dict[int, str]:
    """Get label ID to name mapping from labels table."""
    rows = conn.execute(
        "SELECT label_id, name FROM labels WHERE enabled = TRUE ORDER BY label_id"
    ).fetchall()
    return {int(row[0]): str(row[1]) for row in rows}


def create_datablock(
    db_path: Path,
    run_id: str,
    img_size: int = 224,
    batch_size: int = 64,
    valid_pct: float = 0.2,
    seed: int = 42,
):
    """Create FastAI DataBlock for multi-label classification."""
    from fastai.vision.all import (  # noqa: PLC0415
        DataBlock,
        ImageBlock,
        MultiCategoryBlock,
        Normalize,
        RandomSplitter,
        Resize,
        aug_transforms,
        imagenet_stats,
    )

    def get_x(row):
        return row["path"]

    def get_y(row):
        return row["label_names"]

    item_tfms = [Resize(img_size)]
    batch_tfms = [
        *aug_transforms(
            mult=1.0,
            do_flip=True,
            flip_vert=False,
            max_rotate=15.0,
            max_zoom=1.1,
            max_lighting=0.2,
            max_warp=0.2,
        ),
        Normalize.from_stats(*imagenet_stats),
    ]

    datablock = DataBlock(
        blocks=[ImageBlock, MultiCategoryBlock],
        get_x=get_x,
        get_y=get_y,
        splitter=RandomSplitter(valid_pct=valid_pct, seed=seed),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
    )

    return datablock


def create_dataloaders(
    db_path: Path,
    run_id: str,
    img_size: int = 224,
    batch_size: int = 64,
    valid_pct: float = 0.2,
    seed: int = 42,
    num_workers: int = 4,
):
    """Create DataLoaders from DuckDB data."""

    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("PRAGMA enable_progress_bar=false;")
        from vit_curator.shared.db import ensure_schema  # noqa: PLC0415

        ensure_schema(conn)
        df = load_training_data(conn, run_id)

        if len(df) == 0:
            raise ValueError(f"No training data found for run_id={run_id}")

        datablock = create_datablock(
            db_path=db_path,
            run_id=run_id,
            img_size=img_size,
            batch_size=batch_size,
            valid_pct=valid_pct,
            seed=seed,
        )

        dls = datablock.dataloaders(df, bs=batch_size, num_workers=num_workers)
        return dls
    finally:
        conn.close()
