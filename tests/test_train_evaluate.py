"""Tests for vit_curator.train.evaluate — evaluation, threshold tuning, reports.

Uses pytest.importorskip for optional fastai dependency.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("fastai")


# ---------------------------------------------------------------------------
# evaluate_run
# ---------------------------------------------------------------------------


def test_evaluate_run_no_predictions() -> None:
    """evaluate_run should return error dict when no predictions."""
    from vit_curator.train.evaluate import evaluate_run

    db_path = Path("/tmp/test.duckdb")

    with patch("vit_curator.train.evaluate.duckdb.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.execute.return_value.fetchall.return_value = []

        result = evaluate_run(db_path=db_path, run_id="test-run")

    assert "error" in result
    assert result["error"] == "No predictions found for this run"


def test_evaluate_run_with_predictions() -> None:
    """evaluate_run should return label distribution from predictions."""
    from vit_curator.train.evaluate import evaluate_run

    db_path = Path("/tmp/test.duckdb")

    with patch("vit_curator.train.evaluate.duckdb.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        # Mock predictions query
        mock_conn.execute.return_value.fetchall.return_value = [
            ([1, 2], 1),
            ([1, 3], 2),
        ]

        # Mock get_label_vocab
        with patch(
            "vit_curator.train.evaluate.get_label_vocab",
            return_value={1: "cat", 2: "dog", 3: "bird"},
        ):
            result = evaluate_run(db_path=db_path, run_id="test-run")

    assert "total_images" in result
    assert result["total_images"] == 2
    assert result["num_classes"] == 3
    assert result["avg_labels_per_image"] == 2.0
    assert "label_distribution" in result
    assert result["label_distribution"]["cat"] == 2
    assert result["label_distribution"]["dog"] == 1
    assert result["label_distribution"]["bird"] == 1


def test_evaluate_run_with_model_path() -> None:
    """evaluate_run should call _evaluate_with_model when model_path provided."""
    from vit_curator.train.evaluate import evaluate_run

    db_path = Path("/tmp/test.duckdb")
    model_path = Path("/tmp/model.pkl")

    with (
        patch("vit_curator.train.evaluate.duckdb.connect") as mock_connect,
        patch("vit_curator.train.evaluate._evaluate_with_model") as mock_eval_model,
    ):
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_eval_model.return_value = {"accuracy": 0.95, "f1_macro": 0.94}

        result = evaluate_run(db_path=db_path, run_id="test-run", model_path=model_path)

    assert result["accuracy"] == 0.95
    mock_eval_model.assert_called_once()


# ---------------------------------------------------------------------------
# _evaluate_predictions
# ---------------------------------------------------------------------------


def test_evaluate_predictions_empty() -> None:
    """_evaluate_predictions should return error for empty results."""
    from vit_curator.train.evaluate import _evaluate_predictions

    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = []

    result = _evaluate_predictions(mock_conn, "test-run")
    assert result == {"error": "No predictions found for this run"}


def test_evaluate_predictions_with_data() -> None:
    """_evaluate_predictions should compute label distribution."""
    from vit_curator.train.evaluate import _evaluate_predictions

    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = [
        ([1, 2], 1),
        ([1], 2),
    ]

    with patch("vit_curator.train.evaluate.get_label_vocab", return_value={1: "cat", 2: "dog"}):
        result = _evaluate_predictions(mock_conn, "test-run")

    assert result["total_images"] == 2
    assert result["num_classes"] == 2
    assert result["avg_labels_per_image"] == 1.5
    assert result["label_distribution"]["cat"] == 2
    assert result["label_distribution"]["dog"] == 1


# ---------------------------------------------------------------------------
# _evaluate_with_model
# ---------------------------------------------------------------------------


def test_evaluate_with_model_no_data() -> None:
    """_evaluate_with_model should return error when no ground truth."""
    from vit_curator.train.evaluate import _evaluate_with_model

    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = []

    with patch("vit_curator.train.evaluate.load_trained_model") as mock_load:
        mock_learner = MagicMock()
        mock_learner.dls.vocab = ["cat"]
        mock_load.return_value = mock_learner

        result = _evaluate_with_model(mock_conn, "test-run", Path("/tmp/model.pkl"))
    assert result == {"error": "No ground truth data found"}


def test_evaluate_with_model_success() -> None:
    """_evaluate_with_model should compute metrics with mock model."""
    from vit_curator.train.evaluate import _evaluate_with_model

    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = [
        (b"/images/cat.jpg", [1]),
    ]

    # Mock the learner
    mock_learner = MagicMock()
    mock_learner.dls.vocab = ["cat", "dog"]

    # Mock PILImage.create and learner.predict
    with (
        patch("vit_curator.train.evaluate.load_trained_model", return_value=mock_learner),
        patch("vit_curator.train.evaluate.get_label_vocab", return_value={1: "cat", 2: "dog"}),
        patch("fastai.vision.all.PILImage.create") as mock_pil_create,
    ):
        mock_img = MagicMock()
        mock_pil_create.return_value = mock_img

        # Mock predict to return (pred, _, probs)
        mock_learner.predict.return_value = (["cat"], None, np.array([0.9, 0.1]))

        result = _evaluate_with_model(mock_conn, "test-run", Path("/tmp/model.pkl"))

    assert "accuracy" in result
    assert "f1_macro" in result
    assert "f1_micro" in result
    assert "per_label_metrics" in result


# ---------------------------------------------------------------------------
# tune_thresholds
# ---------------------------------------------------------------------------


def test_tune_thresholds_no_data() -> None:
    """tune_thresholds should return empty dict when no data."""
    from vit_curator.train.evaluate import tune_thresholds

    db_path = Path("/tmp/test.duckdb")
    model_path = Path("/tmp/model.pkl")

    with (
        patch("vit_curator.train.evaluate.duckdb.connect") as mock_connect,
        patch("vit_curator.train.evaluate.load_trained_model") as mock_load,
    ):
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.execute.return_value.fetchall.return_value = []

        mock_learner = MagicMock()
        mock_learner.dls.vocab = ["cat", "dog"]
        mock_load.return_value = mock_learner

        result = tune_thresholds(db_path=db_path, run_id="test-run", model_path=model_path)

    assert result == {}


def test_tune_thresholds_invalid_metric() -> None:
    """tune_thresholds should raise ValueError for unknown metric."""
    from vit_curator.train.evaluate import tune_thresholds

    db_path = Path("/tmp/test.duckdb")
    model_path = Path("/tmp/model.pkl")

    with pytest.raises(ValueError, match="Unknown metric"):
        tune_thresholds(
            db_path=db_path,
            run_id="test-run",
            model_path=model_path,
            metric="unknown",
        )


def test_tune_thresholds_with_data() -> None:
    """tune_thresholds should return threshold->score mapping."""
    from vit_curator.train.evaluate import tune_thresholds

    db_path = Path("/tmp/test.duckdb")
    model_path = Path("/tmp/model.pkl")

    with (
        patch("vit_curator.train.evaluate.duckdb.connect") as mock_connect,
        patch("vit_curator.train.evaluate.load_trained_model") as mock_load,
        patch("vit_curator.train.evaluate.get_label_vocab", return_value={1: "cat"}),
        patch("fastai.vision.all.PILImage.create") as mock_pil_create,
    ):
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.execute.return_value.fetchall.return_value = [
            (b"/images/cat.jpg", [1]),
        ]

        mock_learner = MagicMock()
        mock_learner.dls.vocab = ["cat"]
        mock_load.return_value = mock_learner

        mock_img = MagicMock()
        mock_pil_create.return_value = mock_img
        mock_learner.predict.return_value = (["cat"], None, np.array([0.8]))

        result = tune_thresholds(
            db_path=db_path,
            run_id="test-run",
            model_path=model_path,
            min_threshold=0.1,
            max_threshold=0.5,
            step=0.2,
        )

    assert len(result) > 0
    for threshold, score in result.items():
        assert isinstance(threshold, float)
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------


def test_generate_report_no_predictions(tmp_path) -> None:
    """generate_report should write error report when no predictions."""
    from vit_curator.train.evaluate import generate_report

    db_path = Path("/tmp/test.duckdb")
    output_path = tmp_path / "report.md"

    with patch("vit_curator.train.evaluate.duckdb.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.execute.return_value.fetchall.return_value = []

        generate_report(db_path=db_path, run_id="test-run", output_path=output_path)

    assert output_path.exists()
    content = output_path.read_text()
    assert "Evaluation Report" in content
    assert "No predictions found" in content


def test_generate_report_with_data(tmp_path) -> None:
    """generate_report should write detailed report with data."""
    from vit_curator.train.evaluate import generate_report

    db_path = Path("/tmp/test.duckdb")
    output_path = tmp_path / "report.md"

    with patch("vit_curator.train.evaluate.duckdb.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        # Mock predictions
        mock_conn.execute.return_value.fetchall.return_value = [
            ([1, 2], 1),
        ]

        with patch("vit_curator.train.evaluate.get_label_vocab", return_value={1: "cat", 2: "dog"}):
            generate_report(db_path=db_path, run_id="test-run", output_path=output_path)

    assert output_path.exists()
    content = output_path.read_text()
    assert "Evaluation Report" in content
    assert "Total Images" in content
    assert "cat" in content


def test_generate_report_with_model(tmp_path) -> None:
    """generate_report should include per-label metrics when model provided."""
    from vit_curator.train.evaluate import generate_report

    db_path = Path("/tmp/test.duckdb")
    output_path = tmp_path / "report.md"
    model_path = Path("/tmp/model.pkl")

    with (
        patch("vit_curator.train.evaluate.duckdb.connect") as mock_connect,
        patch("vit_curator.train.evaluate._evaluate_with_model") as mock_eval,
    ):
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        mock_eval.return_value = {
            "total_images": 1,
            "num_classes": 2,
            "avg_labels_per_image": 1.0,
            "accuracy": 0.95,
            "f1_macro": 0.94,
            "f1_micro": 0.95,
            "precision_macro": 0.94,
            "recall_macro": 0.94,
            "hamming_loss": 0.05,
            "per_label_metrics": {
                "cat": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1},
            },
        }

        generate_report(
            db_path=db_path,
            run_id="test-run",
            output_path=output_path,
            model_path=model_path,
        )

    assert output_path.exists()
    content = output_path.read_text()
    assert "Accuracy" in content
    assert "F1 Macro" in content
    assert "Per-Label Metrics" in content
    assert "cat" in content
