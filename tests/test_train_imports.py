"""Tests for vit_curator.train — module imports and smoke tests."""

from __future__ import annotations

import os

import pytest

# All tests in this module require fastai/torch
pytestmark = pytest.mark.skipif(
    not os.environ.get("VIT_CURATOR_TEST_FASTAI"),
    reason="fastai/torch not installed; set VIT_CURATOR_TEST_FASTAI=1 to enable",
)


def test_config_classes_importable() -> None:
    """Test that config classes are importable without optional deps."""
    from vit_curator.config import (
        IngestConfig,
        LinkMode,
        RunConfig,
    )

    assert callable(IngestConfig)
    assert callable(RunConfig)
    assert callable(LinkMode)


def test_error_classes_importable() -> None:
    """Test that error classes are importable."""
    from vit_curator.shared.errors import (
        CLIError,
        DecodeError,
        EncodeError,
        GPUError,
        HashError,
        IngestError,
        LabelError,
        TrainError,
        UnifiedPipelineError,
        WriteError,
    )

    assert issubclass(HashError, UnifiedPipelineError)
    assert issubclass(DecodeError, UnifiedPipelineError)
    assert issubclass(EncodeError, UnifiedPipelineError)
    assert issubclass(WriteError, UnifiedPipelineError)
    assert issubclass(GPUError, UnifiedPipelineError)
    assert issubclass(LabelError, UnifiedPipelineError)
    assert issubclass(TrainError, UnifiedPipelineError)
    assert issubclass(IngestError, UnifiedPipelineError)
    assert issubclass(CLIError, UnifiedPipelineError)


def test_db_module_importable() -> None:
    """Test that shared.db module is importable and functional."""
    from vit_curator.shared.db import connect, ensure_schema, parse_presets_arg

    assert callable(connect)
    assert callable(ensure_schema)
    assert callable(parse_presets_arg)


def test_scan_module_importable() -> None:
    """Test that scan module is importable."""
    from vit_curator.preprocess.scan import scan_into_duckdb

    assert callable(scan_into_duckdb)


def test_dedupe_module_importable() -> None:
    """Test that dedupe module is importable."""
    from vit_curator.preprocess.dedupe import hash_and_mark_dupes

    assert callable(hash_and_mark_dupes)


def test_label_store_importable() -> None:
    """Test that label store module is importable."""
    from vit_curator.label.store import connect_label_db

    assert callable(connect_label_db)


def test_label_prompt_importable() -> None:
    """Test that label prompt module is importable."""
    from vit_curator.label.prompt import build_prompt, load_labelset

    assert callable(build_prompt)
    assert callable(load_labelset)


def test_label_scheduler_importable() -> None:
    """Test that label scheduler module is importable."""
    from vit_curator.label.scheduler import AutoTune, DynamicConcurrency, Ema

    assert callable(Ema)
    assert callable(DynamicConcurrency)
    assert callable(AutoTune)


def test_archive_module_importable() -> None:
    """Test that ingest archive module is importable."""
    from vit_curator.ingest.archive import extract_archive, is_archive

    assert callable(extract_archive)
    assert callable(is_archive)


def test_hashing_module_importable() -> None:
    """Test that hashing module is importable."""
    from vit_curator.shared.hashing import xxh3_128, xxh3_128_file

    assert callable(xxh3_128)
    assert callable(xxh3_128_file)
