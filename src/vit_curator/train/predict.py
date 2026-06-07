"""Batch prediction using trained models."""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np

from vit_curator.train.data import get_label_vocab


def load_trained_model(model_path: Path):
    """Load a trained model."""
    from fastai.vision.all import load_learner  # noqa: PLC0415

    return load_learner(model_path)


def predict_batch(
    learner,
    image_paths: list[Path],
    *,
    threshold: float = 0.5,
) -> list[dict]:
    """Predict labels for a batch of images."""
    from fastai.vision.all import PILImage  # noqa: PLC0415

    results = []

    for img_path in image_paths:
        try:
            img = PILImage.create(img_path)
        except Exception as e:
            results.append(
                {
                    "path": str(img_path),
                    "predictions": [],
                    "top_label": None,
                    "label_ids": [],
                    "error": str(e),
                }
            )
            continue

        _pred, _, probs = learner.predict(img)
        probs_np = probs.numpy() if hasattr(probs, "numpy") else np.array(probs)

        predictions = []
        label_ids = []
        vocab = learner.dls.vocab

        for i, prob in enumerate(probs_np):
            if prob >= threshold:
                label_name = vocab[i] if i < len(vocab) else f"class_{i}"
                predictions.append((label_name, float(prob)))
                label_ids.append(i)

        predictions.sort(key=lambda x: x[1], reverse=True)
        top_label = predictions[0] if predictions else None

        results.append(
            {
                "path": str(img_path),
                "predictions": predictions,
                "top_label": top_label,
                "label_ids": label_ids,
            }
        )

    return results


def predict_run(
    model_path: Path,
    db_path: Path,
    target_run_id: str,
    *,
    source_run_id: str | None = None,
    threshold: float = 0.5,
    batch_size: int = 64,
) -> int:
    """Run prediction on all files in a run.

    Returns the number of predictions made.
    """

    learner = load_trained_model(model_path)
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("PRAGMA enable_progress_bar=false;")
        from vit_curator.shared.db import ensure_schema  # noqa: PLC0415

        ensure_schema(conn)
        label_vocab = get_label_vocab(conn)
        label_name_to_id = {v: k for k, v in label_vocab.items()}

        if source_run_id:
            rows = conn.execute(
                "SELECT DISTINCT f.file_pk, f.rel_path_blob "
                "FROM files f "
                "JOIN tasks t ON t.file_pk = f.file_pk "
                "WHERE t.run_id = ? "
                "ORDER BY f.rel_path_blob",
                [source_run_id],
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT DISTINCT f.file_pk, f.rel_path_blob "
                "FROM files f "
                "JOIN tasks t ON t.file_pk = f.file_pk "
                "WHERE t.run_id = ? "
                "ORDER BY f.rel_path_blob",
                [target_run_id],
            ).fetchall()

        if not rows:
            print("No files found for prediction")
            return 0

        print(f"Found {len(rows)} files to predict")

        total = 0
        for i in range(0, len(rows), batch_size):
            batch_rows = rows[i : i + batch_size]
            image_paths = [Path(os.fsdecode(row[1])) for row in batch_rows]
            predictions = predict_batch(learner, image_paths, threshold=threshold)
            write_predictions_to_db(conn, target_run_id, predictions, label_name_to_id)
            total += len(predictions)
            print(f"Processed {total}/{len(rows)} images")

        return total
    finally:
        conn.close()


def write_predictions_to_db(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    predictions: list[dict],
    label_name_to_id: dict[str, int],
) -> None:
    """Write predictions to database."""
    if not predictions:
        return

    import datetime as dt  # noqa: PLC0415

    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)

    conn.execute("BEGIN;")
    try:
        for pred in predictions:
            path = pred["path"]

            row = conn.execute(
                "SELECT file_pk FROM files WHERE rel_path_blob = ?",
                [os.fsencode(path)],
            ).fetchone()

            if not row:
                continue

            file_pk = int(row[0])

            label_ids = []
            for label_name, _prob in pred["predictions"]:
                label_id = label_name_to_id.get(label_name)
                if label_id is not None:
                    label_ids.append(label_id)

            label_ids = sorted(label_ids)

            conn.execute(
                "INSERT INTO predictions (file_pk, run_id, labels, raw_json, created_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT (file_pk, run_id) DO UPDATE SET "
                "labels=excluded.labels, "
                "raw_json=excluded.raw_json, "
                "created_at=excluded.created_at",
                [file_pk, run_id, label_ids, str(pred["predictions"]), now],
            )

            conn.execute(
                "UPDATE tasks "
                "SET status='done', finished_at=?, latency_ms=NULL, finish_reason=NULL "
                "WHERE run_id=? AND file_pk=?",
                [now, run_id, file_pk],
            )

        conn.execute("COMMIT;")
    except Exception:
        conn.execute("ROLLBACK;")
        raise


def predict_images(
    model_path: Path,
    image_paths: list[Path],
    *,
    threshold: float = 0.5,
    output_json: Path | None = None,
) -> list[dict]:
    """Predict labels for a list of images and optionally save to JSON."""
    import json  # noqa: PLC0415

    learner = load_trained_model(model_path)
    results = predict_batch(learner, image_paths, threshold=threshold)

    if output_json:
        output_json = Path(output_json)
        output_json.write_text(json.dumps(results, indent=2))
        print(f"Predictions saved to {output_json}")

    return results
