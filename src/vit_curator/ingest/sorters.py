from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".gif", ".heic"}
PDF_EXTS = {".pdf"}
DOC_EXTS = {
    ".txt",
    ".csv",
    ".tsv",
    ".json",
    ".xml",
    ".html",
    ".md",
    ".rtf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
}


@dataclass(frozen=True)
class BucketLayout:
    sorted_root: Path
    bucket_images: str = "images"
    bucket_pdfs: str = "pdfs"
    bucket_docs: str = "docs"
    bucket_other: str = "other"

    def bucket_dir(self, bucket_name: str) -> Path:
        return self.sorted_root / bucket_name


def choose_bucket(path: Path, layout: BucketLayout) -> str:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return layout.bucket_images
    if ext in PDF_EXTS:
        return layout.bucket_pdfs
    if ext in DOC_EXTS:
        return layout.bucket_docs
    return layout.bucket_other


def collision_safe_target(dst_dir: Path, rel: Path, src_hint: str = "") -> Path:
    target = dst_dir / rel
    if not target.exists():
        return target

    stem = target.stem
    suff = target.suffix
    h = hashlib.sha1((str(rel) + "|" + src_hint).encode("utf-8", errors="ignore")).hexdigest()[:10]
    return target.with_name(f"{stem}__{h}{suff}")
