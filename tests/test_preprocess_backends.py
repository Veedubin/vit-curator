"""Tests for vit_curator preprocess backends — CPU/DALI derivative generation.

Ported from data_janitor/tests/test_backends.py, adapted for vit_curator API.
Tests requiring torch are guarded by VIT_CURATOR_TEST_TORCH env var.
Tests requiring DALI are guarded by VIT_CURATOR_TEST_DALI env var.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pytest
from typer.testing import CliRunner

from tests.conftest import make_rgb_image

runner = CliRunner()


# ---------------------------------------------------------------------------
# GPU/DALI guard
# ---------------------------------------------------------------------------

SKIP_TORCH = not os.environ.get("VIT_CURATOR_TEST_TORCH")
SKIP_DALI = not os.environ.get("VIT_CURATOR_TEST_DALI")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_pipeline_cfg(src: Path, out: Path, presets: str = "thumb-32=32", **overrides):  # type: ignore[no-untyped-def]
    """Create a RunConfig for testing with sensible defaults."""
    from vit_curator.config import LinkMode, RunConfig

    defaults = dict(
        src_root=src,
        out_root=out,
        max_files=None,
        bucket_size=100,
        link_mode=LinkMode("copy"),
        hash_workers=2,
        scan_insert_batch=100,
        decode_backend="cpu",
        device="cpu",
        presets_arg=presets,
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
    defaults.update(overrides)
    return RunConfig(**defaults)


# ---------------------------------------------------------------------------
# Tests: Derivative generation (requires torch)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(SKIP_TORCH, reason="torch not installed; set VIT_CURATOR_TEST_TORCH=1")
def test_derivatives_cpu_backend(tmp_path: Path) -> None:
    """Test CPU backend derivative generation."""
    from vit_curator.preprocess.derivatives import run_pipeline

    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir()

    make_rgb_image(src / "a.jpg", (255, 0, 0))
    make_rgb_image(src / "b.jpg", (0, 255, 0))

    cfg = _make_pipeline_cfg(src, out)
    run_pipeline(cfg)

    db_path = out / "index.duckdb"
    assert db_path.exists()

    con = duckdb.connect(str(db_path))
    row = con.execute("SELECT COUNT(*) FROM image_derivatives WHERE status=1").fetchone()
    count = int(row[0]) if row and row[0] is not None else 0
    assert count == 2
    con.close()


@pytest.mark.skipif(SKIP_TORCH, reason="torch not installed; set VIT_CURATOR_TEST_TORCH=1")
def test_derivatives_cpu_grayscale(tmp_path: Path) -> None:
    """Test CPU backend with grayscale conversion."""
    from vit_curator.preprocess.derivatives import run_pipeline

    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir()

    make_rgb_image(src / "color.jpg", (100, 150, 200))

    cfg = _make_pipeline_cfg(src, out, presets="thumb-16=16", preserve_color=False)
    run_pipeline(cfg)

    con = duckdb.connect(str(out / "index.duckdb"))
    row = con.execute("SELECT COUNT(*) FROM image_derivatives WHERE status=1").fetchone()
    count = int(row[0]) if row and row[0] is not None else 0
    assert count == 1
    con.close()


@pytest.mark.skipif(SKIP_TORCH, reason="torch not installed; set VIT_CURATOR_TEST_TORCH=1")
def test_derivatives_multiple_presets(tmp_path: Path) -> None:
    """Test CPU backend with multiple presets."""
    from vit_curator.preprocess.derivatives import run_pipeline

    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir()

    make_rgb_image(src / "test.jpg", (50, 100, 150))

    cfg = _make_pipeline_cfg(
        src, out, presets="small=64,medium=128,large=256", hash_workers=2, writer_workers=2
    )
    run_pipeline(cfg)

    con = duckdb.connect(str(out / "index.duckdb"))
    row = con.execute("SELECT COUNT(*) FROM image_derivatives WHERE status=1").fetchone()
    count = int(row[0]) if row and row[0] is not None else 0
    assert count == 3
    con.close()


@pytest.mark.skipif(SKIP_TORCH, reason="torch not installed; set VIT_CURATOR_TEST_TORCH=1")
def test_cli_dali_batch_multiplier(tmp_path: Path) -> None:
    """Test that dali_batch_multiplier is passed through correctly."""
    from vit_curator.config import LinkMode, RunConfig

    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir()

    make_rgb_image(src / "test.jpg", (100, 100, 100))

    cfg = RunConfig(
        src_root=src,
        out_root=out,
        max_files=None,
        bucket_size=100,
        link_mode=LinkMode("copy"),
        hash_workers=1,
        scan_insert_batch=100,
        decode_backend="cpu",
        device="cpu",
        presets_arg="test-16=16",
        fmt="jpeg",
        jpeg_quality=80,
        preserve_source=False,
        preserve_color=True,
        preserve_quality=False,
        decode_batch=8,
        inflight_batches=2,
        writer_workers=1,
        metrics_every_s=0,
        dali_batch_multiplier=8,
    )

    assert cfg.dali_batch_multiplier == 8


@pytest.mark.skipif(SKIP_DALI, reason="DALI not installed; set VIT_CURATOR_TEST_DALI=1")
def test_derivatives_dali_backend(tmp_path: Path) -> None:
    """Test DALI backend derivative generation (skipped if DALI not available)."""
    from vit_curator.preprocess.derivatives import run_pipeline

    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir()

    make_rgb_image(src / "a.jpg", (255, 0, 0))
    make_rgb_image(src / "b.jpg", (0, 255, 0))

    cfg = _make_pipeline_cfg(src, out, decode_backend="dali")
    run_pipeline(cfg)

    db_path = out / "index.duckdb"
    assert db_path.exists()

    con = duckdb.connect(str(db_path))
    row = con.execute("SELECT COUNT(*) FROM image_derivatives WHERE status=1").fetchone()
    count = int(row[0]) if row and row[0] is not None else 0
    assert count == 2
    con.close()


# ---------------------------------------------------------------------------
# Tests: CLI (mixed torch/non-torch)
# ---------------------------------------------------------------------------


def test_errors_module_exists() -> None:
    """Test that centralized errors module exists and has expected values."""
    from vit_curator.shared.errors import (
        ERR_DECODE,
        ERR_ENCODE,
        ERR_GPU,
        ERR_HASH,
        ERR_WRITE,
    )

    assert ERR_HASH == 1001
    assert ERR_DECODE == 2001
    assert ERR_GPU == 2002
    assert ERR_WRITE == 2003
    assert ERR_ENCODE == 2004


def test_cli_help_output() -> None:
    """Test CLI help output is informative."""
    from vit_curator.cli import app

    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "pipeline" in result.output.lower()
    assert "preprocess" in result.output.lower()
    assert "ingest" in result.output.lower()

    result = runner.invoke(app, ["preprocess", "--help"])
    assert result.exit_code == 0
    assert "--src" in result.output
    assert "--out" in result.output
    assert "--presets" in result.output


def test_error_hierarchy() -> None:
    """Test that error hierarchy and codes are consistent."""
    from vit_curator.shared.errors import (
        ERR_DECODE,
        ERR_ENCODE,
        ERR_GPU,
        ERR_HASH,
        ERR_WRITE,
        DecodeError,
        EncodeError,
        GPUError,
        HashError,
        UnifiedPipelineError,
        WriteError,
    )

    # All specific errors inherit from UnifiedPipelineError
    assert issubclass(HashError, UnifiedPipelineError)
    assert issubclass(DecodeError, UnifiedPipelineError)
    assert issubclass(GPUError, UnifiedPipelineError)
    assert issubclass(WriteError, UnifiedPipelineError)
    assert issubclass(EncodeError, UnifiedPipelineError)

    # Error codes are integers
    for code in [ERR_HASH, ERR_DECODE, ERR_GPU, ERR_WRITE, ERR_ENCODE]:
        assert isinstance(code, int)

    # Instantiation with file_pk
    err = HashError("hash failed", file_pk=42)
    assert err.code == ERR_HASH
    assert err.file_pk == 42
    assert "hash failed" in str(err)


@pytest.mark.skipif(SKIP_TORCH, reason="torch not installed; set VIT_CURATOR_TEST_TORCH=1")
def test_cli_preprocess_with_minimal_args(tmp_path: Path) -> None:
    """Test CLI preprocess command with minimal required arguments."""
    from vit_curator.cli import app

    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir()

    make_rgb_image(src / "test.jpg", (100, 150, 200))

    result = runner.invoke(
        app,
        [
            "preprocess",
            "--src",
            str(src),
            "--out",
            str(out),
            "--presets",
            "test-16=16",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"


@pytest.mark.skipif(SKIP_TORCH, reason="torch not installed; set VIT_CURATOR_TEST_TORCH=1")
def test_cli_preprocess_with_all_args(tmp_path: Path) -> None:
    """Test CLI preprocess command with multiple presets and options."""
    from vit_curator.cli import app

    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir()

    make_rgb_image(src / "a.jpg", (255, 0, 0))
    make_rgb_image(src / "b.jpg", (0, 255, 0))

    result = runner.invoke(
        app,
        [
            "preprocess",
            "--src",
            str(src),
            "--out",
            str(out),
            "--presets",
            "small=64,medium=128,large=256",
            "--fmt",
            "jpeg",
            "--preserve-color",
            "--bucket-size",
            "100",
            "--hash-workers",
            "2",
            "--writer-workers",
            "2",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"


def test_cli_preprocess_validation_errors() -> None:
    """Test CLI preprocess command validation for missing required args."""
    from vit_curator.cli import app

    result = runner.invoke(app, ["preprocess"])
    # Should fail because --src and --out are required (but --out has no default)
    # Actually --out is required but --src defaults to None which triggers CLIError
    assert result.exit_code != 0


@pytest.mark.skipif(SKIP_TORCH, reason="torch not installed; set VIT_CURATOR_TEST_TORCH=1")
def test_cli_status_command(tmp_path: Path) -> None:
    """Test CLI status command after a preprocess run."""
    from vit_curator.cli import app

    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir()

    make_rgb_image(src / "test.jpg", (100, 100, 100))

    result = runner.invoke(
        app,
        ["preprocess", "--src", str(src), "--out", str(out), "--presets", "test-16=16"],
    )
    assert result.exit_code == 0

    result = runner.invoke(app, ["status", "--db", str(out / "index.duckdb")])
    assert result.exit_code == 0
    # Status command should show file count and metrics
    output_lower = result.output.lower()
    assert "files" in output_lower or "metric" in output_lower or "status" in output_lower
