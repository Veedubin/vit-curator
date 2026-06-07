"""Integration tests for the full preprocess pipeline (scan → hash → dedupe → derivatives)."""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pytest

from tests.conftest import make_rgb_image

# Skip entire module if torch is not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("VIT_CURATOR_TEST_TORCH"),
    reason="torch not installed; set VIT_CURATOR_TEST_TORCH=1 to enable",
)


def _import_run_config():
    from vit_curator.config import LinkMode, RunConfig

    return LinkMode, RunConfig


def test_pipeline_small_run(src_dir: Path, out_dir: Path) -> None:
    """Test end-to-end pipeline with small dataset (requires torch)."""
    LinkMode, RunConfig = _import_run_config()
    from vit_curator.preprocess.derivatives import run_pipeline

    make_rgb_image(src_dir / "a.jpg", (10, 20, 30))
    (src_dir / "b.jpg").write_bytes((src_dir / "a.jpg").read_bytes())  # duplicate

    cfg = RunConfig(
        src_root=src_dir,
        out_root=out_dir,
        max_files=None,
        bucket_size=100,
        link_mode=LinkMode("copy"),
        hash_workers=2,
        scan_insert_batch=100,
        decode_backend="cpu",
        device="cpu",
        presets_arg="thumb-32=32",
        fmt="jpeg",
        jpeg_quality=85,
        preserve_source=False,
        preserve_color=True,
        preserve_quality=False,
        decode_batch=16,
        inflight_batches=2,
        writer_workers=2,
        metrics_every_s=0,
        dali_batch_multiplier=4,
    )

    run_pipeline(cfg)

    db_path = out_dir / "index.duckdb"
    assert db_path.exists()

    con = duckdb.connect(str(db_path))
    try:
        deriv_count = con.execute(
            "SELECT COUNT(*) FROM image_derivatives WHERE status=1"
        ).fetchone()
        assert deriv_count is not None and deriv_count[0] >= 1
    finally:
        con.close()


def test_error_hierarchy_accessible() -> None:
    """Test that error hierarchy is accessible even without torch."""
    from vit_curator.shared.errors import (
        CLIError,
        DecodeError,
        EncodeError,
        HashError,
        WriteError,
    )

    assert issubclass(HashError, Exception)
    assert issubclass(DecodeError, Exception)
    assert issubclass(EncodeError, Exception)
    assert issubclass(WriteError, Exception)
    assert issubclass(CLIError, Exception)

    err = HashError("test error", file_pk=42)
    assert err.message == "test error"
    assert err.code == 1001
    assert err.file_pk == 42


def test_config_link_modes() -> None:
    """Test LinkMode dataclass."""
    from vit_curator.config import LinkMode

    assert LinkMode("hardlink").mode == "hardlink"
    assert LinkMode("symlink").mode == "symlink"
    assert LinkMode("copy").mode == "copy"


def test_parse_presets() -> None:
    """Test preset argument parsing."""
    from vit_curator.shared.db import parse_presets_arg

    result = parse_presets_arg("thumb-32=32,medium-128=128")
    assert len(result) == 2
    assert result[0] == ("thumb-32", 32, 32)
    assert result[1] == ("medium-128", 128, 128)

    result = parse_presets_arg("square=256x256")
    assert result == [("square", 256, 256)]

    result = parse_presets_arg("")
    assert result == []
