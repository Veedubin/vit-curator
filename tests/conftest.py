"""Shared test fixtures and markers for vit_curator tests."""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pytest


@pytest.fixture
def db(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    """Create a fresh DuckDB connection with the unified schema."""
    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(db_path))
    from vit_curator.shared.db import ensure_schema

    ensure_schema(con)
    yield con
    con.close()


@pytest.fixture
def src_dir(tmp_path: Path) -> Path:
    """Create a temporary source directory for test images."""
    src = tmp_path / "src"
    src.mkdir()
    return src


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory."""
    out = tmp_path / "out"
    out.mkdir()
    return out


def make_rgb_image(
    path: Path, color: tuple[int, int, int] = (128, 128, 128), size: int = 64
) -> None:
    """Create a small test RGB image at the given path."""
    from PIL import Image

    img = Image.new("RGB", (size, size), color)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


# --- Pytest markers ---

# Mark tests that require torch (not installed by default)
torch = pytest.mark.skipif(
    not os.environ.get("VIT_CURATOR_TEST_TORCH"),
    reason="torch not installed; set VIT_CURATOR_TEST_TORCH=1 to enable",
)

# Mark tests that require fastai (not installed by default)
fastai = pytest.mark.skipif(
    not os.environ.get("VIT_CURATOR_TEST_FASTAI"),
    reason="fastai not installed; set VIT_CURATOR_TEST_FASTAI=1 to enable",
)

# Mark tests that require DALI (not installed by default)
dali = pytest.mark.skipif(
    not os.environ.get("VIT_CURATOR_TEST_DALI"),
    reason="DALI not installed; set VIT_CURATOR_TEST_DALI=1 to enable",
)

# Mark tests that require nvidia-ml-py (not installed by default)
nvidia = pytest.mark.skipif(
    not os.environ.get("VIT_CURATOR_TEST_NVIDIA"),
    reason="nvidia-ml-py not installed; set VIT_CURATOR_TEST_NVIDIA=1 to enable",
)

# Mark slow integration tests
slow = pytest.mark.slow
