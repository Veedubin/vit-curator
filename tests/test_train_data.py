"""Tests for vit_curator.train.data — data loading and DataBlock creation.

Uses pytest.importorskip for optional fastai dependency.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

pytest.importorskip("fastai")


# ---------------------------------------------------------------------------
# load_training_data
# ---------------------------------------------------------------------------


def test_load_training_data_empty() -> None:
    """load_training_data should return empty DataFrame when no data."""
    from vit_curator.train.data import load_training_data

    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchdf.return_value = pd.DataFrame()

    df = load_training_data(mock_conn, run_id="test-run")
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["path", "labels", "label_names", "text"]
    assert len(df) == 0


def test_load_training_data_with_data() -> None:
    """load_training_data should process data correctly."""
    from vit_curator.train.data import load_training_data

    mock_conn = MagicMock()

    # Mock the SQL query result
    raw_df = pd.DataFrame(
        {
            "path": [b"/images/test.jpg"],
            "labels": [[1, 2]],
            "text": ["some text"],
            "subject": ["a subject"],
            "entities": [["entity1"]],
            "summary": ["a summary"],
            "latency_ms": [100.0],
            "finish_reason": ["stop"],
        }
    )
    mock_conn.execute.return_value.fetchdf.return_value = raw_df

    # Mock get_label_vocab
    with patch("vit_curator.train.data.get_label_vocab", return_value={1: "cat", 2: "dog"}):
        df = load_training_data(mock_conn, run_id="test-run")

    assert len(df) == 1
    assert "path" in df.columns
    assert "labels" in df.columns
    assert "label_names" in df.columns
    assert df["path"].iloc[0] == "/images/test.jpg"
    assert df["labels"].iloc[0] == [1, 2]
    assert df["label_names"].iloc[0] == ["cat", "dog"]


def test_load_training_data_with_exclude_labels() -> None:
    """load_training_data should pass exclude_labels to query."""
    from vit_curator.train.data import load_training_data

    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchdf.return_value = pd.DataFrame()

    df = load_training_data(mock_conn, run_id="test-run", exclude_labels=[99])
    assert isinstance(df, pd.DataFrame)

    # Verify the query included the exclude filter
    call_args = mock_conn.execute.call_args
    assert call_args is not None
    sql = call_args[0][0]
    assert "list_has_any" in sql


def test_load_training_data_none_labels() -> None:
    """load_training_data should handle None labels."""
    from vit_curator.train.data import load_training_data

    mock_conn = MagicMock()

    raw_df = pd.DataFrame(
        {
            "path": [b"/images/test.jpg"],
            "labels": [None],
            "text": [None],
            "subject": [None],
            "entities": [None],
            "summary": [None],
            "latency_ms": [None],
            "finish_reason": [None],
        }
    )
    mock_conn.execute.return_value.fetchdf.return_value = raw_df

    with patch("vit_curator.train.data.get_label_vocab", return_value={}):
        df = load_training_data(mock_conn, run_id="test-run")

    assert len(df) == 1
    assert df["labels"].iloc[0] == []


# ---------------------------------------------------------------------------
# get_label_vocab
# ---------------------------------------------------------------------------


def test_get_label_vocab() -> None:
    """get_label_vocab should return label ID to name mapping."""
    from vit_curator.train.data import get_label_vocab

    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = [(1, "cat"), (2, "dog"), (3, "bird")]

    vocab = get_label_vocab(mock_conn)
    assert vocab == {1: "cat", 2: "dog", 3: "bird"}


def test_get_label_vocab_empty() -> None:
    """get_label_vocab should return empty dict when no labels."""
    from vit_curator.train.data import get_label_vocab

    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = []

    vocab = get_label_vocab(mock_conn)
    assert vocab == {}


# ---------------------------------------------------------------------------
# create_datablock
# ---------------------------------------------------------------------------


def test_create_datablock() -> None:
    """create_datablock should return a DataBlock."""
    from vit_curator.train.data import create_datablock

    db_path = Path("/tmp/test.duckdb")
    datablock = create_datablock(
        db_path=db_path,
        run_id="test-run",
        img_size=224,
        batch_size=64,
        valid_pct=0.2,
        seed=42,
    )

    assert datablock is not None
    # DataBlock should have blocks configured
    assert hasattr(datablock, "blocks")


def test_create_datablock_custom_params() -> None:
    """create_datablock should accept custom parameters."""
    from vit_curator.train.data import create_datablock

    db_path = Path("/tmp/test.duckdb")
    datablock = create_datablock(
        db_path=db_path,
        run_id="test-run",
        img_size=512,
        batch_size=32,
        valid_pct=0.1,
        seed=123,
    )

    assert datablock is not None


# ---------------------------------------------------------------------------
# create_dataloaders
# ---------------------------------------------------------------------------


def test_create_dataloaders_no_data() -> None:
    """create_dataloaders should raise ValueError when no training data."""
    from vit_curator.train.data import create_dataloaders

    db_path = Path("/tmp/test.duckdb")

    with (
        patch("vit_curator.train.data.duckdb.connect") as mock_connect,
        patch("vit_curator.train.data.load_training_data", return_value=pd.DataFrame()),
    ):
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        with pytest.raises(ValueError, match="No training data found"):
            create_dataloaders(
                db_path=db_path,
                run_id="test-run",
                img_size=224,
                batch_size=64,
            )


def test_create_dataloaders_with_data() -> None:
    """create_dataloaders should create DataLoaders with data."""
    from vit_curator.train.data import create_dataloaders

    db_path = Path("/tmp/test.duckdb")

    df = pd.DataFrame(
        {
            "path": ["/images/test.jpg"],
            "labels": [[1, 2]],
            "label_names": [["cat", "dog"]],
            "text": ["some text"],
        }
    )

    with (
        patch("vit_curator.train.data.duckdb.connect") as mock_connect,
        patch("vit_curator.train.data.load_training_data", return_value=df),
        patch("vit_curator.train.data.create_datablock") as mock_create_db,
    ):
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        mock_datablock = MagicMock()
        mock_dls = MagicMock()
        mock_datablock.dataloaders.return_value = mock_dls
        mock_create_db.return_value = mock_datablock

        dls = create_dataloaders(
            db_path=db_path,
            run_id="test-run",
            img_size=224,
            batch_size=64,
        )

        assert dls is mock_dls
        mock_datablock.dataloaders.assert_called_once()
