from __future__ import annotations

import shutil
import tarfile
import zipfile
from pathlib import Path

try:
    import py7zr
except Exception:
    py7zr = None


def is_archive(p: Path) -> bool:
    name = p.name.lower()
    if name.endswith((".tar.gz", ".tar.bz2", ".tar.xz")):
        return True
    return p.suffix.lower() in {".zip", ".7z", ".rar", ".tar", ".tgz", ".tbz", ".tbz2", ".txz"}


def extract_archive(src: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = src.name.lower()

    if name.endswith(".zip") or src.suffix.lower() == ".zip":
        _extract_zip(src, dest_dir)
        return

    if name.endswith((".tar", ".tgz", ".tar.gz", ".tbz", ".tbz2", ".tar.bz2", ".txz", ".tar.xz")):
        _extract_tar(src, dest_dir)
        return

    if src.suffix.lower() == ".7z":
        _extract_7z(src, dest_dir)
        return

    _extract_via_7z_cli(src, dest_dir)


def _extract_zip(src: Path, dest_dir: Path) -> None:
    with zipfile.ZipFile(src, "r") as z:
        dest_root = dest_dir.resolve()
        for member in z.namelist():
            mpath = (dest_dir / member).resolve()
            try:
                mpath.relative_to(dest_root)
            except ValueError:
                raise RuntimeError(f"Unsafe zip member path: {member}") from None
        z.extractall(dest_dir)


def _extract_tar(src: Path, dest_dir: Path) -> None:
    with tarfile.open(src, "r:*") as tf:
        dest_root = dest_dir.resolve()
        for m in tf.getmembers():
            mpath = (dest_dir / m.name).resolve()
            try:
                mpath.relative_to(dest_root)
            except ValueError:
                raise RuntimeError(f"Unsafe tar member path: {m.name}") from None
        tf.extractall(dest_dir, filter="data")


def _extract_7z(src: Path, dest_dir: Path) -> None:
    if py7zr is None:
        raise RuntimeError("py7zr is not installed; cannot extract .7z archives.")
    with py7zr.SevenZipFile(src, mode="r") as z:
        z.extractall(path=dest_dir)


def _extract_via_7z_cli(src: Path, dest_dir: Path) -> None:
    exe = shutil.which("7z") or shutil.which("7zz") or shutil.which("7za")
    if not exe:
        raise RuntimeError("7z CLI not found. Install 7-Zip/p7zip.")

    import subprocess  # noqa: PLC0415

    cmd = [exe, "x", "-y", f"-o{dest_dir!s}", str(src)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"7z extraction failed for {src}")
