"""xxh3_128 hashing helpers for content deduplication and path hashing."""

from __future__ import annotations

import os
from pathlib import Path

import xxhash


def xxh3_128(blob: bytes) -> bytes:
    """Compute xxh3_128 hash of a byte string.

    Args:
        blob: Input bytes to hash.

    Returns:
        16-byte hash digest.
    """
    return xxhash.xxh3_128_digest(blob)


def xxh3_128_file(path: Path, chunk_size: int = 1024 * 1024) -> bytes:
    """Compute xxh3_128 over file bytes using a streaming reader.

    Args:
        path: Path to the file to hash.
        chunk_size: Size of chunks to read at a time (default 1MB).

    Returns:
        16-byte hash digest.
    """
    h = xxhash.xxh3_128()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.digest()


def fsencode_relpath(relpath: str) -> bytes:
    """Encode a relative path string to bytes using filesystem encoding.

    Args:
        relpath: Relative path as string.

    Returns:
        Encoded bytes representation.
    """
    return os.fsencode(relpath)


def fsdecode_relpath(relpath_blob: bytes) -> str:
    """Decode bytes to a relative path string using filesystem encoding.

    Args:
        relpath_blob: Relative path as bytes.

    Returns:
        Decoded string representation.
    """
    return os.fsdecode(relpath_blob)
