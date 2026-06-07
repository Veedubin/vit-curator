"""Tests for vit_curator.shared.hashing — xxh3_128 hash functions."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_xxh3_128_deterministic() -> None:
    """Test that xxh3_128 produces deterministic hashes for the same input."""
    from vit_curator.shared.hashing import xxh3_128

    data = b"hello world"
    h1 = xxh3_128(data)
    h2 = xxh3_128(data)
    assert h1 == h2
    assert isinstance(h1, bytes)
    assert len(h1) == 16  # 128 bits = 16 bytes


def test_xxh3_128_different_inputs() -> None:
    """Test that xxh3_128 produces different hashes for different inputs."""
    from vit_curator.shared.hashing import xxh3_128

    h1 = xxh3_128(b"input A")
    h2 = xxh3_128(b"input B")
    assert h1 != h2


def test_xxh3_128_file(tmp_path: Path) -> None:
    """Test xxh3_128_file produces the same hash as xxh3_128 for file contents."""
    from vit_curator.shared.hashing import xxh3_128, xxh3_128_file

    content = b"test file content for hashing"
    f = tmp_path / "test.bin"
    f.write_bytes(content)

    file_hash = xxh3_128_file(f)
    mem_hash = xxh3_128(content)
    assert file_hash == mem_hash


def test_xxh3_128_empty() -> None:
    """Test xxh3_128 with empty bytes."""
    from vit_curator.shared.hashing import xxh3_128

    h = xxh3_128(b"")
    assert isinstance(h, bytes)
    assert len(h) == 16


def test_xxh3_128_file_nonexistent(tmp_path: Path) -> None:
    """Test xxh3_128_file raises for nonexistent file."""
    from vit_curator.shared.hashing import xxh3_128_file

    with pytest.raises((FileNotFoundError, OSError)):
        xxh3_128_file(tmp_path / "nonexistent.bin")
