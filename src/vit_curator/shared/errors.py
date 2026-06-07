"""Error codes and exception hierarchy for vit_curator.

Merges error codes from data_janitor and adds pipeline-stage-specific
error codes for labeling, training, etc.
"""

from __future__ import annotations


class UnifiedPipelineError(Exception):
    """Base exception for vit-curator errors.

    .. deprecated::
        Use :class:`ViTCuratorError` instead. This alias is kept for
        backwards compatibility with the legacy ``unified-pipeline``
        name; it will be removed in a future major release.
    """

    def __init__(self, message: str, *, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ViTCuratorError(UnifiedPipelineError):
    """Base exception for vit-curator errors (preferred name)."""


class HashError(UnifiedPipelineError):
    """Raised when file hashing fails."""

    def __init__(self, message: str, *, file_pk: int | None = None) -> None:
        super().__init__(message, code=ERR_HASH)
        self.file_pk = file_pk


class DecodeError(UnifiedPipelineError):
    """Raised when image decoding fails."""

    def __init__(self, message: str, *, file_pk: int | None = None) -> None:
        super().__init__(message, code=ERR_DECODE)
        self.file_pk = file_pk


class GPUError(UnifiedPipelineError):
    """Raised when GPU/DALI operations fail."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code=ERR_GPU)


class WriteError(UnifiedPipelineError):
    """Raised when file write/link operations fail."""

    def __init__(self, message: str, *, file_pk: int | None = None) -> None:
        super().__init__(message, code=ERR_WRITE)
        self.file_pk = file_pk


class EncodeError(UnifiedPipelineError):
    """Raised when image encoding fails."""

    def __init__(self, message: str, *, file_pk: int | None = None) -> None:
        super().__init__(message, code=ERR_ENCODE)
        self.file_pk = file_pk


class LabelError(UnifiedPipelineError):
    """Raised when VLM labeling fails."""

    def __init__(self, message: str, *, file_pk: int | None = None) -> None:
        super().__init__(message, code=ERR_LABEL)
        self.file_pk = file_pk


class TrainError(UnifiedPipelineError):
    """Raised when training fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code=ERR_TRAIN)


class IngestError(UnifiedPipelineError):
    """Raised when ingestion (download/extract/sort) fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code=ERR_INGEST)


class CLIError(UnifiedPipelineError):
    """Raised for CLI argument validation errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code=None)


# ---------------------------------------------------------------------------
# Error code constants (range-separated by pipeline stage)
# ---------------------------------------------------------------------------


class ErrorCode:
    """Namespace for error code constants, accessible as ErrorCode.ERR_HASH etc."""

    # Preprocessing (1000-1999)
    ERR_HASH = 1001
    ERR_DECODE = 2001
    ERR_GPU = 2002
    ERR_WRITE = 2003
    ERR_ENCODE = 2004

    # Ingest (3000-3999)
    ERR_INGEST = 3001

    # Labeling (4000-4999)
    ERR_LABEL = 4001

    # Training (5000-5999)
    ERR_TRAIN = 5001


# Module-level constants for convenience
ERR_HASH = ErrorCode.ERR_HASH
ERR_DECODE = ErrorCode.ERR_DECODE
ERR_GPU = ErrorCode.ERR_GPU
ERR_WRITE = ErrorCode.ERR_WRITE
ERR_ENCODE = ErrorCode.ERR_ENCODE
ERR_INGEST = ErrorCode.ERR_INGEST
ERR_LABEL = ErrorCode.ERR_LABEL
ERR_TRAIN = ErrorCode.ERR_TRAIN
