from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UrlItem:
    url: str


def iter_urls(path: str | Path) -> Iterator[UrlItem]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    if p.suffix.lower() == ".csv":
        yield from _iter_urls_csv(p)
        return

    yield from _iter_urls_txt(p)


def _iter_urls_txt(p: Path) -> Iterator[UrlItem]:
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        yield UrlItem(url=s)


def _iter_urls_csv(p: Path) -> Iterator[UrlItem]:
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return

    header = [c.strip() for c in rows[0]]
    header_l = [c.lower() for c in header]

    url_idx: int | None = None
    for cand in ("url", "link", "href"):
        if cand in header_l:
            url_idx = header_l.index(cand)
            break

    if (
        url_idx is None
        and len(rows[0]) >= 1
        and (rows[0][0] or "").strip().lower().startswith("http")
    ):
        start = 0
        idx = 0
    else:
        start = 1
        idx = url_idx if url_idx is not None else 0

    for r in rows[start:]:
        if not r:
            continue
        if idx >= len(r):
            continue
        u = (r[idx] or "").strip()
        if not u or u.startswith("#"):
            continue
        yield UrlItem(url=u)
