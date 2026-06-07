"""Tests for vit_curator.ingest.archive — archive extraction and path traversal security."""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest


def test_extract_zip_path_traversal(tmp_path: Path) -> None:
    """Test that extract_archive blocks zip path traversal attacks."""
    from vit_curator.ingest.archive import extract_archive

    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("../evil.txt", "bad")

    with pytest.raises(RuntimeError, match="Unsafe"):
        extract_archive(zip_path, tmp_path / "out")


def test_extract_tar_path_traversal(tmp_path: Path) -> None:
    """Test that extract_archive blocks tar path traversal attacks."""
    from vit_curator.ingest.archive import extract_archive

    tar_path = tmp_path / "evil.tar"
    with tarfile.open(tar_path, "w") as tf:
        info = tarfile.TarInfo(name="../../evil.txt")
        data = b"bad"
        info.size = len(data)
        tf.addfile(info, fileobj=io.BytesIO(data))

    with pytest.raises(RuntimeError, match="Unsafe"):
        extract_archive(tar_path, tmp_path / "out2")


def test_extract_zip_normal(tmp_path: Path) -> None:
    """Test that extract_archive works for normal zip files."""
    from vit_curator.ingest.archive import extract_archive

    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("hello.txt", "hello world")
        z.writestr("subdir/nested.txt", "nested content")

    out_dir = tmp_path / "extracted"
    extract_archive(zip_path, out_dir)

    assert (out_dir / "hello.txt").read_text() == "hello world"
    assert (out_dir / "subdir" / "nested.txt").read_text() == "nested content"


def test_extract_tar_normal(tmp_path: Path) -> None:
    """Test that extract_archive works for normal tar files."""
    from vit_curator.ingest.archive import extract_archive

    tar_path = tmp_path / "test.tar"
    with tarfile.open(tar_path, "w") as tf:
        info = tarfile.TarInfo(name="data.txt")
        data = b"tar content"
        info.size = len(data)
        tf.addfile(info, fileobj=io.BytesIO(data))

    out_dir = tmp_path / "extracted"
    extract_archive(tar_path, out_dir)

    assert (out_dir / "data.txt").read_text() == "tar content"


def test_is_archive(tmp_path: Path) -> None:
    """Test is_archive recognizes common archive formats."""
    from vit_curator.ingest.archive import is_archive

    assert is_archive(tmp_path / "test.zip")
    assert is_archive(tmp_path / "test.tar")
    assert is_archive(tmp_path / "test.tar.gz")
    assert is_archive(tmp_path / "test.tgz")
    assert is_archive(tmp_path / "test.7z")
    assert not is_archive(tmp_path / "test.jpg")
    assert not is_archive(tmp_path / "test.png")
    assert not is_archive(tmp_path / "test.txt")
