"""Filesystem helpers for the ingest pipeline."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _nearest_existing_dir(p: Path) -> Path:
    cur = p
    while not cur.exists() and cur != cur.parent:
        cur = cur.parent
    return cur


def same_filesystem(a: Path, b: Path) -> bool:
    try:
        return os.stat(a).st_dev == os.stat(b.parent).st_dev
    except FileNotFoundError:
        parent = _nearest_existing_dir(b.parent)
        if parent.exists():
            return os.stat(a).st_dev == os.stat(parent).st_dev
        return False


def link_or_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        return
    if same_filesystem(src, dst):
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def move_into(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        raise FileExistsError(str(dst))
    shutil.move(str(src), str(dst))


def safe_relpath(root: Path, p: Path) -> Path:
    rp = p.resolve()
    rr = root.resolve()
    try:
        return rp.relative_to(rr)
    except Exception as e:
        raise ValueError(f"Path {p} is not under root {root}") from e
