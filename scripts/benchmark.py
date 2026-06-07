"""Performance benchmark for vit-curator core stages.

Usage:
    python scripts/benchmark.py

Benchmarks the following pipeline stages on synthetic data:
  - Scan:     vit_curator.preprocess.scan.scan_into_duckdb
  - Hash/Dedupe: vit_curator.preprocess.dedupe.hash_and_mark_dupes
  - Chunking: vit_curator.post.chunk.run_chunking
  - Enrichment: vit_curator.post.enrich.run_enrichment (mocked LLM call)

Creates synthetic test data (small images, text files) using PIL and simple
text generation inside a TemporaryDirectory.  All working directories are
cleaned up automatically.

Optional dependencies (torch, fastai, imagehash) are handled gracefully:
if missing, the corresponding benchmark stage reports a skip message instead
of crashing.

Results are printed as a Rich table (falls back to plain text if Rich is
not installed).
"""

from __future__ import annotations

import random
import string
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional imports — handled gracefully
# ---------------------------------------------------------------------------
try:
    from PIL import Image as PILImage

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import duckdb

    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

try:
    from rich.console import Console
    from rich.table import Table

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# ---------------------------------------------------------------------------
# Pipeline imports — all from vit_curator.*
# ---------------------------------------------------------------------------
try:
    from vit_curator.post.chunk import run_chunking as _chunk
    from vit_curator.post.enrich import EnrichmentResult, _insert_enrichment
    from vit_curator.preprocess.dedupe import hash_and_mark_dupes as _hd
    from vit_curator.preprocess.scan import scan_into_duckdb as _scan
    from vit_curator.shared.db import connect, next_file_pk

    HAS_PIPELINE = True
except ImportError:
    _chunk = None  # type: ignore[assignment,misc]
    EnrichmentResult = None  # type: ignore[assignment,misc]
    _insert_enrichment = None  # type: ignore[assignment,misc]
    _hd = None  # type: ignore[assignment,misc]
    _scan = None  # type: ignore[assignment,misc]
    connect = None  # type: ignore[assignment,misc]
    next_file_pk = None  # type: ignore[assignment,misc]
    HAS_PIPELINE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_IMAGES = 500
NUM_TEXT_FILES = 200
IMAGE_SIZE = (64, 64)
TEXT_WORD_COUNT = 500
BENCHMARK_ITERATIONS = 1  # single pass; increase for multi-run averaging


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Single benchmark stage result."""

    stage: str
    items_processed: int = 0
    elapsed_s: float = 0.0
    rate: float = 0.0  # items/sec
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""

    results: list[BenchmarkResult] = field(default_factory=list)

    def add(self, result: BenchmarkResult) -> None:
        self.results.append(result)


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def _random_color() -> tuple[int, int, int]:
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def create_synthetic_images(root: Path, count: int) -> int:
    """Create ``count`` small random-coloured PNG images under *root*.

    Returns the number of files actually created.
    """
    if not HAS_PIL:
        print("  [SKIP] PIL not available — cannot create synthetic images")
        return 0

    created = 0
    for i in range(count):
        img = PILImage.new("RGB", IMAGE_SIZE, color=_random_color())
        # Add a tiny amount of noise so hashes differ
        pixels = img.load()
        if pixels is not None:
            px = random.randint(0, IMAGE_SIZE[0] - 1)
            py = random.randint(0, IMAGE_SIZE[1] - 1)
            pixels[px, py] = _random_color()

        out_path = root / f"img_{i:04d}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        created += 1

    return created


def create_synthetic_text_files(
    root: Path, count: int, words_per_file: int = TEXT_WORD_COUNT
) -> int:
    """Create ``count`` text files with random word content under *root*.

    Returns the number of files actually created.
    """
    created = 0
    for i in range(count):
        words = [
            "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))
            for _ in range(words_per_file)
        ]
        text = " ".join(words)
        out_path = root / f"doc_{i:04d}.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        created += 1

    return created


# ---------------------------------------------------------------------------
# Individual benchmark stages
# ---------------------------------------------------------------------------


def bench_scan(src_root: Path, con: duckdb.DuckDBPyConnection, num_files: int) -> BenchmarkResult:
    """Benchmark scan_into_duckdb — measures files/sec."""
    start_pk = next_file_pk(con)

    t0 = time.perf_counter()
    stats = _scan(
        con,
        src_root,
        start_file_pk=start_pk,
        allow_exts=None,
        max_files=None,
    )
    elapsed = time.perf_counter() - t0

    total = stats.seen
    rate = total / max(elapsed, 1e-9)

    return BenchmarkResult(
        stage="Scan",
        items_processed=total,
        elapsed_s=elapsed,
        rate=rate,
    )


def bench_hash_dedupe(src_root: Path, con: duckdb.DuckDBPyConnection) -> BenchmarkResult:
    """Benchmark hash_and_mark_dupes — measures files/sec."""
    t0 = time.perf_counter()
    stats = _hd(
        con,
        src_root,
        num_workers=4,
        metrics_every_s=0,  # suppress progress printing
        console=None,
    )
    elapsed = time.perf_counter() - t0

    total = stats.hashed_ok
    rate = total / max(elapsed, 1e-9)

    return BenchmarkResult(
        stage="Hash/Dedupe",
        items_processed=total,
        elapsed_s=elapsed,
        rate=rate,
    )


def bench_chunking(con: duckdb.DuckDBPyConnection) -> BenchmarkResult:
    """Benchmark run_chunking — measures chunks/sec.

    Requires a predictions table with text content.  Creates synthetic
    prediction rows if the table is empty.
    """
    # Ensure there is text to chunk — insert synthetic predictions if needed
    row_count = con.execute(
        "SELECT COUNT(*) FROM predictions WHERE text IS NOT NULL AND text != '';"
    ).fetchone()
    existing = int(row_count[0]) if row_count else 0

    if existing == 0:
        # Insert synthetic predictions with random text
        synth_texts = []
        for i in range(NUM_TEXT_FILES):
            words = [
                "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))
                for _ in range(TEXT_WORD_COUNT)
            ]
            text = " ".join(words)
            synth_texts.append((i + 1, text))

        con.executemany(
            "INSERT INTO predictions (file_pk, run_id, labels, text, created_at) "
            "VALUES (?, '00000000-0000-0000-0000-000000000001', [], ?, CURRENT_TIMESTAMP);",
            [(fp, t) for fp, t in synth_texts],
        )

    # Count total characters for reference (informational)
    char_row = con.execute(
        "SELECT SUM(LENGTH(text)) FROM predictions WHERE text IS NOT NULL AND text != '';"
    ).fetchone()
    if char_row and char_row[0] is not None:
        print(f"  Total characters in corpus: {int(char_row[0]):,}")

    t0 = time.perf_counter()
    _chunk(
        con,
        source="predictions",
        chunk_chars=1200,
        chunk_overlap=200,
        run_id=None,
        text_dir=None,
        max_docs=None,
    )
    elapsed = time.perf_counter() - t0

    # Count total chunks produced
    chunk_row = con.execute("SELECT COUNT(*) FROM chunks;").fetchone()
    total_chunks = int(chunk_row[0]) if chunk_row else 0

    rate = total_chunks / max(elapsed, 1e-9)

    return BenchmarkResult(
        stage="Chunking",
        items_processed=total_chunks,
        elapsed_s=elapsed,
        rate=rate,
    )


def bench_enrichment(con: duckdb.DuckDBPyConnection) -> BenchmarkResult:
    """Benchmark enrichment — mocks the LLM call, measures enrichments/sec.

    Instead of calling an external LLM, we insert EnrichmentResult rows
    directly using _insert_enrichment() to measure the DB write rate
    for the enrichment pipeline.
    """
    # Ensure we have predictions to enrich
    row_count = con.execute(
        "SELECT COUNT(*) FROM predictions WHERE text IS NOT NULL AND text != '';"
    ).fetchone()
    existing = int(row_count[0]) if row_count else 0

    if existing == 0:
        # Insert minimal predictions
        synth_texts = []
        for i in range(NUM_TEXT_FILES):
            words = [
                "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))
                for _ in range(TEXT_WORD_COUNT)
            ]
            text = " ".join(words)
            synth_texts.append((i + 1, text))

        con.executemany(
            "INSERT INTO predictions (file_pk, run_id, labels, text, created_at) "
            "VALUES (?, '00000000-0000-0000-0000-000000000002', [], ?, CURRENT_TIMESTAMP);",
            [(fp, t) for fp, t in synth_texts],
        )

    # Fetch file_pks and text lengths for enrichment
    rows = con.execute(
        "SELECT file_pk, LENGTH(text) FROM predictions "
        "WHERE text IS NOT NULL AND text != '' LIMIT 200;"
    ).fetchall()

    if not rows:
        return BenchmarkResult(
            stage="Enrichment", skipped=True, skip_reason="No predictions to enrich"
        )

    target_count = len(rows)
    model_name = "bench-mock"

    # Mock enrichment — measure DB write throughput for enrichment
    t0 = time.perf_counter()
    for file_pk, raw_text_len in rows:
        text_len = int(raw_text_len) if raw_text_len else 0
        word_count = max(1, text_len // 5)  # rough estimate
        result = EnrichmentResult(
            subject=f"Benchmark subject for file {file_pk}",
            summary="Mock summary for benchmarking enrichment write throughput.",
            doc_type="benchmark",
            entities_json=(
                '{"persons": [], "organizations": [], "locations": [], '
                '"dates": [], "case_numbers": [], "other": []}'
            ),
            tags_json='["benchmark", "mock"]',
        )
        _insert_enrichment(
            con,
            file_pk=int(file_pk),
            model_name=model_name,
            result=result,
            finish_reason="stop",
            truncated=False,
            text_len=text_len,
            word_count=word_count,
            raw_payload='{"subject": "benchmark", "summary": "mock"}',
        )
    elapsed = time.perf_counter() - t0

    rate = target_count / max(elapsed, 1e-9)

    return BenchmarkResult(
        stage="Enrichment (mocked LLM)",
        items_processed=target_count,
        elapsed_s=elapsed,
        rate=rate,
    )


# ---------------------------------------------------------------------------
# Result printing
# ---------------------------------------------------------------------------


def _format_rate(rate: float) -> str:
    """Format a rate with appropriate units."""
    if rate >= 1_000_000:
        return f"{rate / 1_000_000:.1f}M/s"
    if rate >= 1_000:
        return f"{rate / 1_000:.1f}K/s"
    return f"{rate:.1f}/s"


def print_results(suite: BenchmarkSuite) -> None:
    """Print benchmark results as a table.

    Uses Rich if available, otherwise plain text.
    """
    if HAS_RICH:
        table = Table(title="ViT-Curator Performance Benchmark")
        table.add_column("Stage", style="bold cyan", min_width=22)
        table.add_column("Items", justify="right", style="green")
        table.add_column("Elapsed (s)", justify="right", style="yellow")
        table.add_column("Rate", justify="right", style="magenta")

        for r in suite.results:
            if r.skipped:
                table.add_row(r.stage, "SKIPPED", "-", r.skip_reason or "N/A")
            else:
                table.add_row(
                    r.stage,
                    f"{r.items_processed:,}",
                    f"{r.elapsed_s:.3f}",
                    _format_rate(r.rate),
                )

        console = Console()
        console.print()
        console.print(table)
        console.print()
    else:
        # Fallback plain-text output
        print("\nViT-Curator Performance Benchmark")
        print("=" * 70)
        fmt = "{:<22} {:>10} {:>12} {:>14}"
        print(fmt.format("Stage", "Items", "Elapsed (s)", "Rate"))
        print("-" * 70)
        for r in suite.results:
            if r.skipped:
                print(fmt.format(r.stage, "SKIPPED", "-", r.skip_reason or "N/A"))
            else:
                print(
                    fmt.format(
                        r.stage,
                        f"{r.items_processed:,}",
                        f"{r.elapsed_s:.3f}",
                        _format_rate(r.rate),
                    )
                )
        print("=" * 70)
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_benchmarks() -> None:
    """Set up synthetic data and run all benchmark stages."""
    # Dependency checks
    missing: list[str] = []
    if not HAS_DUCKDB:
        missing.append("duckdb")
    if not HAS_PIL:
        missing.append("pillow")
    if not HAS_PIPELINE:
        missing.append("vit_curator")

    if missing:
        print(f"ERROR: Missing required dependencies: {', '.join(missing)}")
        print("Install with: uv add " + " ".join(missing))
        sys.exit(1)

    suite = BenchmarkSuite()

    with tempfile.TemporaryDirectory(prefix="up_bench_") as tmpdir:
        tmp_path = Path(tmpdir)
        src_root = tmp_path / "source"
        out_root = tmp_path / "output"
        db_path = out_root / "index.duckdb"

        src_root.mkdir(parents=True, exist_ok=True)
        out_root.mkdir(parents=True, exist_ok=True)

        # --- Create synthetic data ---
        print(f"\nCreating {NUM_IMAGES} synthetic images ({IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}) ...")
        n_images = create_synthetic_images(src_root / "images", NUM_IMAGES)
        print(f"  Created {n_images} images")

        print(f"Creating {NUM_TEXT_FILES} synthetic text files ({TEXT_WORD_COUNT} words each) ...")
        n_texts = create_synthetic_text_files(src_root / "docs", NUM_TEXT_FILES)
        print(f"  Created {n_texts} text files")

        # --- Open database ---
        db = connect(db_path)
        con = db.con

        # --- Stage 1: Scan ---
        print("\n[Benchmark] Scan stage ...")
        r = bench_scan(src_root, con, n_images + n_texts)
        suite.add(r)
        print(
            f"  Scanned {r.items_processed:,} files in {r.elapsed_s:.3f}s ({_format_rate(r.rate)})"
        )

        # --- Stage 2: Hash/Dedupe ---
        print("\n[Benchmark] Hash/Dedupe stage ...")
        r = bench_hash_dedupe(src_root, con)
        suite.add(r)
        print(
            f"  Hashed {r.items_processed:,} files in {r.elapsed_s:.3f}s ({_format_rate(r.rate)})"
        )

        # --- Stage 3: Chunking ---
        print("\n[Benchmark] Chunking stage ...")
        try:
            r = bench_chunking(con)
            suite.add(r)
            rate = _format_rate(r.rate)
            print(f"  Chunked into {r.items_processed:,} chunks in {r.elapsed_s:.3f}s ({rate})")
        except Exception as exc:
            print(f"  [ERROR] Chunking benchmark failed: {exc}")
            suite.add(BenchmarkResult(stage="Chunking", skipped=True, skip_reason=str(exc)))

        # --- Stage 4: Enrichment (mocked) ---
        print("\n[Benchmark] Enrichment stage (mocked LLM) ...")
        try:
            r = bench_enrichment(con)
            suite.add(r)
            print(
                f"  Enriched {r.items_processed:,} documents in "
                f"{r.elapsed_s:.3f}s ({_format_rate(r.rate)})"
            )
        except Exception as exc:
            print(f"  [ERROR] Enrichment benchmark failed: {exc}")
            suite.add(
                BenchmarkResult(stage="Enrichment (mocked LLM)", skipped=True, skip_reason=str(exc))
            )

        # --- Cleanup ---
        con.close()

    # --- Print results ---
    print_results(suite)


if __name__ == "__main__":
    run_benchmarks()
