"""Evaluation metrics and threshold tuning."""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)

from vit_curator.train.data import get_label_vocab
from vit_curator.train.predict import load_trained_model


def evaluate_run(
    db_path: Path,
    run_id: str,
    model_path: Path | None = None,
) -> dict:
    """Evaluate predictions for a run.

    Returns dict with:
    - accuracy, f1_macro, f1_micro, precision, recall, hamming_loss
    - per_label_metrics: dict of per-label metrics (if model provided)
    """
    conn = duckdb.connect(str(db_path))
    try:
        if model_path:
            return _evaluate_with_model(conn, run_id, model_path)
        return _evaluate_predictions(conn, run_id)
    finally:
        conn.close()


def _evaluate_predictions(conn: duckdb.DuckDBPyConnection, run_id: str) -> dict:
    """Evaluate existing predictions in the database."""
    rows = conn.execute(
        "SELECT p.labels, p.file_pk FROM predictions p WHERE p.run_id = ? ORDER BY p.file_pk",
        [run_id],
    ).fetchall()

    if not rows:
        return {"error": "No predictions found for this run"}

    label_vocab = get_label_vocab(conn)
    label_counts: dict[str, int] = {}
    total_images = len(rows)

    for labels, _ in rows:
        for label_id in labels or []:
            label_name = label_vocab.get(int(label_id), f"label_{label_id}")
            label_counts[label_name] = label_counts.get(label_name, 0) + 1

    label_distribution = dict(sorted(label_counts.items(), key=lambda x: x[1], reverse=True))
    total_labels = sum(len(r[0] or []) for r in rows)
    avg_labels = total_labels / total_images if total_images > 0 else 0

    return {
        "total_images": total_images,
        "num_classes": len(label_vocab),
        "avg_labels_per_image": avg_labels,
        "label_distribution": label_distribution,
    }


def _evaluate_with_model(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    model_path: Path,
) -> dict:
    """Evaluate model predictions against ground truth."""
    learner = load_trained_model(model_path)
    vocab = learner.dls.vocab

    rows = conn.execute(
        "SELECT os.fsencode(f.rel_path_blob), p.labels "
        "FROM predictions p "
        "JOIN files f ON f.file_pk = p.file_pk "
        "WHERE p.run_id = ? "
        "ORDER BY f.rel_path_blob",
        [run_id],
    ).fetchall()

    if not rows:
        return {"error": "No ground truth data found"}

    num_samples = len(rows)
    num_classes = len(vocab)

    y_true = np.zeros((num_samples, num_classes), dtype=np.int32)
    y_pred = np.zeros((num_samples, num_classes), dtype=np.int32)
    y_scores = np.zeros((num_samples, num_classes), dtype=np.float32)

    vocab_to_idx = {name: i for i, name in enumerate(vocab)}
    label_vocab = get_label_vocab(conn)

    from fastai.vision.all import PILImage  # noqa: PLC0415

    for i, (path_blob, gt_labels) in enumerate(rows):
        path = os.fsdecode(path_blob) if isinstance(path_blob, bytes) else str(path_blob)
        for label_id in gt_labels or []:
            label_name = label_vocab.get(int(label_id), f"label_{label_id}")
            if label_name in vocab_to_idx:
                y_true[i, vocab_to_idx[label_name]] = 1

        try:
            img = PILImage.create(path)
            _pred, _, probs = learner.predict(img)
            for pred_label in _pred if isinstance(_pred, list) else [_pred]:
                if pred_label in vocab_to_idx:
                    y_pred[i, vocab_to_idx[pred_label]] = 1
            probs_np = probs.numpy() if hasattr(probs, "numpy") else np.array(probs)
            y_scores[i] = probs_np
        except Exception as e:
            print(f"Error predicting {path}: {e}")
            continue

    results = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }

    per_label = {}
    for i, label_name in enumerate(vocab):
        label_true = y_true[:, i]
        label_pred = y_pred[:, i]
        if label_true.sum() > 0:
            per_label[label_name] = {
                "precision": float(precision_score(label_true, label_pred, zero_division=0)),
                "recall": float(recall_score(label_true, label_pred, zero_division=0)),
                "f1": float(f1_score(label_true, label_pred, zero_division=0)),
                "support": int(label_true.sum()),
            }

    results["per_label_metrics"] = per_label
    return results


def tune_thresholds(
    db_path: Path,
    run_id: str,
    model_path: Path,
    *,
    min_threshold: float = 0.1,
    max_threshold: float = 0.9,
    step: float = 0.05,
    metric: str = "f1",
) -> dict[float, float]:
    """Tune probability thresholds for optimal F1."""
    metric_fn = {
        "f1": f1_score,
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
    }.get(metric)

    if metric_fn is None:
        raise ValueError(f"Unknown metric: {metric}")

    learner = load_trained_model(model_path)
    vocab = learner.dls.vocab
    conn = duckdb.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT os.fsencode(f.rel_path_blob), p.labels "
            "FROM predictions p "
            "JOIN files f ON f.file_pk = p.file_pk "
            "WHERE p.run_id = ? "
            "ORDER BY f.rel_path_blob",
            [run_id],
        ).fetchall()

        if not rows:
            return {}

        num_samples = len(rows)
        num_classes = len(vocab)
        y_true = np.zeros((num_samples, num_classes), dtype=np.int32)
        y_scores = np.zeros((num_samples, num_classes), dtype=np.float32)

        vocab_to_idx = {name: i for i, name in enumerate(vocab)}
        label_vocab = get_label_vocab(conn)

        from fastai.vision.all import PILImage  # noqa: PLC0415

        for i, (path_blob, gt_labels) in enumerate(rows):
            path = os.fsdecode(path_blob) if isinstance(path_blob, bytes) else str(path_blob)
            for label_id in gt_labels or []:
                label_name = label_vocab.get(int(label_id), f"label_{label_id}")
                if label_name in vocab_to_idx:
                    y_true[i, vocab_to_idx[label_name]] = 1

            try:
                img = PILImage.create(path)
                _, _, probs = learner.predict(img)
                probs_np = probs.numpy() if hasattr(probs, "numpy") else np.array(probs)
                y_scores[i] = probs_np
            except Exception as e:
                print(f"Error predicting {path}: {e}")
                continue

        results: dict[float, float] = {}
        threshold = min_threshold
        while threshold <= max_threshold:
            y_pred = (y_scores >= threshold).astype(np.int32)
            if metric == "accuracy":
                score = accuracy_score(y_true, y_pred)
            elif metric == "precision":
                score = precision_score(y_true, y_pred, average="macro", zero_division=0)
            elif metric == "recall":
                score = recall_score(y_true, y_pred, average="macro", zero_division=0)
            else:
                score = f1_score(y_true, y_pred, average="macro", zero_division=0)
            results[round(threshold, 2)] = float(score)
            threshold += step

        return results
    finally:
        conn.close()


def generate_report(
    db_path: Path,
    run_id: str,
    output_path: Path,
    model_path: Path | None = None,
) -> None:
    """Generate detailed evaluation report (Markdown)."""
    output_path = Path(output_path)
    results = evaluate_run(db_path, run_id, model_path)

    lines = ["# Evaluation Report\n", "## Summary\n"]

    if "error" in results:
        lines.append(f"Error: {results['error']}\n")
    else:
        if "total_images" in results:
            lines.append(f"- Total Images: {results['total_images']}")
            lines.append(f"- Number of Classes: {results['num_classes']}")
            lines.append(f"- Avg Labels per Image: {results['avg_labels_per_image']:.2f}")
            lines.append("")

        if "accuracy" in results:
            lines.append(f"- Accuracy: {results['accuracy']:.4f}")
            lines.append(f"- F1 Macro: {results['f1_macro']:.4f}")
            lines.append(f"- F1 Micro: {results['f1_micro']:.4f}")
            lines.append(f"- Precision Macro: {results['precision_macro']:.4f}")
            lines.append(f"- Recall Macro: {results['recall_macro']:.4f}")
            lines.append(f"- Hamming Loss: {results['hamming_loss']:.4f}")
            lines.append("")

        if "label_distribution" in results:
            lines.append("## Label Distribution\n")
            lines.append("| Label | Count |")
            lines.append("|-------|-------|")
            for label, count in results["label_distribution"].items():
                lines.append(f"| {label} | {count} |")
            lines.append("")

        if "per_label_metrics" in results:
            lines.append("## Per-Label Metrics\n")
            lines.append("| Label | Precision | Recall | F1 | Support |")
            lines.append("|-------|-----------|--------|----|----------|")
            for label, metrics in results["per_label_metrics"].items():
                lines.append(
                    f"| {label} | {metrics['precision']:.4f} | "
                    f"{metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['support']} |"
                )

    output_path.write_text("\n".join(lines))
    print(f"Report saved to {output_path}")
