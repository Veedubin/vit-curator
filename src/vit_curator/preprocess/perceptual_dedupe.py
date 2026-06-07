"""Perceptual deduplication: detect near-duplicate images using phash.

Provides perceptual hashing (phash) to find visually similar images that
exact content hashing misses. Integrates with the DuckDB files table to
mark near-duplicates.

Requires optional dependency: imagehash (pip install imagehash)
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.theme import Theme

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD = 8  # Hamming distance threshold for near-duplicates
DEFAULT_HASH_SIZE = 8  # phash size (8x8 = 64-bit hash)

_THEME = Theme(
    {
        "info": "cyan",
        "ok": "green",
        "warn": "yellow",
        "error": "bold red",
        "stat": "magenta",
    }
)


# ---------------------------------------------------------------------------
# Config and result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerceptualDedupeConfig:
    """Configuration for perceptual deduplication.

    Attributes:
        db_path: Path to the DuckDB database.
        src_root: Root directory containing source image files.
        threshold: Hamming distance threshold (lower = stricter).
        hash_size: phash size (8 = 64-bit hash).
        max_files: Optional cap on number of files to process.
        dry_run: If True, compute hashes but don't update the database.
    """

    db_path: Path
    src_root: Path
    threshold: int = DEFAULT_THRESHOLD
    hash_size: int = DEFAULT_HASH_SIZE
    max_files: int | None = None
    dry_run: bool = False


@dataclass
class PerceptualDedupeResult:
    """Statistics from a perceptual deduplication run."""

    total_scanned: int = 0
    near_dupes_found: int = 0
    canonicals: int = 0
    errors: int = 0


# ---------------------------------------------------------------------------
# phash computation
# ---------------------------------------------------------------------------


def compute_phash(image_path: Path, hash_size: int = DEFAULT_HASH_SIZE) -> str:
    """Compute perceptual hash for an image file.

    Args:
        image_path: Path to the image file.
        hash_size: Size of the phash (8 = 64-bit hash).

    Returns:
        Hex string of the perceptual hash.

    Raises:
        ImportError: If imagehash is not installed.
    """
    try:
        from imagehash import phash  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "imagehash and Pillow are required for perceptual deduplication. "
            "Install with: uv add imagehash Pillow"
        ) from exc

    img = Image.open(image_path)
    h = phash(img, hash_size=hash_size)
    return str(h)


def _hamming_distance(h1: str, h2: str) -> int:
    """Compute the Hamming distance between two hex hash strings.

    Compares the binary representations bit by bit.
    """
    try:
        v1 = int(h1, 16)
        v2 = int(h2, 16)
        return bin(v1 ^ v2).count("1")
    except ValueError:
        return len(h1) * 4  # max distance on parse failure


# ---------------------------------------------------------------------------
# PerceptualDedupe class
# ---------------------------------------------------------------------------


class PerceptualDedupe:
    """Detects near-duplicate images using perceptual hashing."""

    def __init__(self, config: PerceptualDedupeConfig | None = None) -> None:
        self.config = config or PerceptualDedupeConfig(
            db_path=Path("index.duckdb"),
            src_root=Path("."),
        )

    def scan_and_mark(
        self,
        con: duckdb.DuckDBPyConnection,
        console: Console | None = None,
    ) -> PerceptualDedupeResult:
        """Scan canonical images and mark near-duplicates.

        Reads canonical (non-exact-duplicate) file paths from DuckDB,
        computes perceptual hashes, clusters by Hamming distance,
        and marks near-duplicates.

        Args:
            con: DuckDB connection with schema initialized.
            console: Rich console for progress output.

        Returns:
            PerceptualDedupeResult with statistics.
        """
        if console is None:
            console = Console(theme=_THEME)

        cfg = self.config
        result = PerceptualDedupeResult()

        # Fetch canonical (non-exact-duplicate) image paths
        sql = """
            SELECT f.file_pk, f.rel_path_blob
            FROM files f
            WHERE f.is_exact_dupe = FALSE
              AND f.decode_status = 1
            ORDER BY f.file_pk
        """
        if cfg.max_files is not None:
            sql += f" LIMIT {int(cfg.max_files)}"

        rows = con.execute(sql).fetchall()
        if not rows:
            console.print("[warn]No canonical decoded files found for phash scanning.[/warn]")
            return result

        console.print(f"[info]Scanning {len(rows)} canonical files for phash…[/info]")

        # Compute perceptual hashes
        phash_map: dict[int, str] = {}  # file_pk -> phash_hex
        cluster_map: dict[str, list[int]] = defaultdict(list)  # phash -> [file_pks]

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as prog:
            task = prog.add_task("[info]Computing phash…[/info]", total=len(rows))

            for file_pk, rel_blob in rows:
                rel_path = os.fsdecode(rel_blob) if isinstance(rel_blob, bytes) else str(rel_blob)
                full_path = cfg.src_root / rel_path

                try:
                    _ph = compute_phash(full_path, hash_size=cfg.hash_size)
                    phash_map[file_pk] = _ph
                    cluster_map[_ph].append(file_pk)
                    result.total_scanned += 1
                except Exception as exc:
                    console.log(f"[error]phash error for file_pk={file_pk}: {exc}[/error]")
                    result.errors += 1
                prog.update(task, advance=1)

        # Exact phash duplicates (same hash)
        for _ph, file_pks in cluster_map.items():
            if len(file_pks) > 1:
                # First file_pk is canonical, rest are near-dupes
                canonical_pk = file_pks[0]
                for dupe_pk in file_pks[1:]:
                    if not cfg.dry_run:
                        con.execute(
                            "UPDATE files SET dupe_of_file_pk = ?, is_exact_dupe = TRUE "
                            "WHERE file_pk = ?;",
                            [canonical_pk, dupe_pk],
                        )
                    result.near_dupes_found += 1

        # Near-duplicates (similar hash within threshold)
        # For each unique phash, check against all others
        unique_hashes = list(cluster_map.keys())
        if cfg.threshold > 0 and len(unique_hashes) > 1:
            # Group similar hashes using union-find approach
            groups: dict[int, int] = {}  # file_pk -> group representative

            for i, h1 in enumerate(unique_hashes):
                for j in range(i + 1, len(unique_hashes)):
                    h2 = unique_hashes[j]
                    dist = _hamming_distance(h1, h2)
                    if dist <= cfg.threshold and dist > 0:
                        # These clusters are similar — merge
                        canon1 = cluster_map[h1][0]
                        canon2 = cluster_map[h2][0]
                        # canon1 stays canonical, canon2 becomes near-duplicate
                        if canon2 not in groups:
                            groups[canon2] = canon1

            # Mark near-duplicates from groupings
            for dupe_pk, canon_pk in groups.items():
                # Only mark if not already marked as exact dupe
                existing = con.execute(
                    "SELECT dupe_of_file_pk FROM files WHERE file_pk = ?;",
                    [dupe_pk],
                ).fetchone()
                if existing and existing[0] is None:
                    if not cfg.dry_run:
                        con.execute(
                            "UPDATE files SET dupe_of_file_pk = ?, is_exact_dupe = TRUE "
                            "WHERE file_pk = ?;",
                            [canon_pk, dupe_pk],
                        )
                    result.near_dupes_found += 1

        # Count canonicals (non-dupes after this run)
        canon_row = con.execute(
            "SELECT COUNT(*) FROM files "
            "WHERE (dupe_of_file_pk IS NULL OR dupe_of_file_pk = 0) "
            "AND decode_status = 1;"
        ).fetchone()
        result.canonicals = int(canon_row[0]) if canon_row and canon_row[0] else 0

        _print_summary(result, cfg.dry_run, console)
        return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _print_summary(
    result: PerceptualDedupeResult,
    dry_run: bool,
    console: Console,
) -> None:
    """Print perceptual deduplication run summary."""
    from rich.table import Table  # noqa: PLC0415

    mode = "DRY RUN" if dry_run else "LIVE"
    table = Table(title=f"Perceptual Deduplication Summary ({mode})")
    table.add_column("Metric", style="stat")
    table.add_column("Count", justify="right")
    table.add_row("Files scanned", str(result.total_scanned))
    table.add_row("Near-duplicates found", str(result.near_dupes_found))
    table.add_row("Canonicals (kept)", str(result.canonicals))
    table.add_row("Errors", str(result.errors))
    console.print(table)


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def run_perceptual_dedupe(
    con: duckdb.DuckDBPyConnection,
    src_root: Path,
    threshold: int = DEFAULT_THRESHOLD,
    hash_size: int = DEFAULT_HASH_SIZE,
    max_files: int | None = None,
    dry_run: bool = False,
    console: Console | None = None,
) -> PerceptualDedupeResult:
    """High-level entry point for perceptual deduplication.

    Args:
        con: DuckDB connection with schema initialized.
        src_root: Root directory containing source image files.
        threshold: Hamming distance threshold for near-duplicate detection.
        hash_size: phash size (8 = 64-bit hash).
        max_files: Optional cap on number of files to process.
        dry_run: If True, compute hashes but don't update the database.
        console: Rich console for progress output.

    Returns:
        PerceptualDedupeResult with statistics.
    """
    config = PerceptualDedupeConfig(
        db_path=Path("index.duckdb"),  # only used for display, con is passed
        src_root=src_root,
        threshold=threshold,
        hash_size=hash_size,
        max_files=max_files,
        dry_run=dry_run,
    )
    deduper = PerceptualDedupe(config)
    return deduper.scan_and_mark(con, console=console)
