"""Tests for vit_curator.shared.errors — error hierarchy and error codes."""

from __future__ import annotations


def test_error_hierarchy() -> None:
    """Test that all error classes exist and inherit from UnifiedPipelineError."""
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
    assert issubclass(GPUError, UnifiedPipelineError)
    assert issubclass(WriteError, UnifiedPipelineError)
    assert issubclass(EncodeError, UnifiedPipelineError)
    assert issubclass(LabelError, UnifiedPipelineError)
    assert issubclass(TrainError, UnifiedPipelineError)
    assert issubclass(IngestError, UnifiedPipelineError)
    assert issubclass(CLIError, UnifiedPipelineError)


def test_error_codes() -> None:
    """Test that error code constants have expected values."""
    from vit_curator.shared.errors import (
        ERR_DECODE,
        ERR_HASH,
        ERR_WRITE,
        ErrorCode,
    )

    assert ErrorCode.ERR_HASH == 1001
    assert ErrorCode.ERR_DECODE == 2001
    assert ErrorCode.ERR_GPU == 2002
    assert ErrorCode.ERR_WRITE == 2003
    assert ErrorCode.ERR_ENCODE == 2004
    assert ErrorCode.ERR_INGEST == 3001
    assert ErrorCode.ERR_LABEL == 4001
    assert ErrorCode.ERR_TRAIN == 5001

    # Also check module-level constants
    assert ERR_HASH == 1001
    assert ERR_DECODE == 2001
    assert ERR_WRITE == 2003


def test_hash_error_with_file_pk() -> None:
    """Test that HashError stores file_pk and error code."""
    from vit_curator.shared.errors import ERR_HASH, HashError

    err = HashError("test error", file_pk=42)
    assert err.message == "test error"
    assert err.code == ERR_HASH
    assert err.file_pk == 42


def test_decode_error_with_file_pk() -> None:
    """Test that DecodeError stores file_pk and error code."""
    from vit_curator.shared.errors import ERR_DECODE, DecodeError

    err = DecodeError("decode failed", file_pk=7)
    assert err.message == "decode failed"
    assert err.code == ERR_DECODE
    assert err.file_pk == 7


def test_error_string_representation() -> None:
    """Test that errors display useful string representation."""
    from vit_curator.shared.errors import HashError

    err = HashError("file not found", file_pk=10)
    s = str(err)
    assert "file not found" in s
