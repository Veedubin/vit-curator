"""Shared configuration dataclasses for all pipeline stages.

Merges RunConfig + IngestConfig (data_janitor) with RunParams + AppPaths
(ocr-my-junk) into a single unified configuration hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Ingest (Stage 0)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IngestConfig:
    """Configuration for the download/unarchive pipeline.

    Attributes:
        dest_dir: Destination directory for downloads and extracts.
        download_urls_file: Path to file containing URLs to download.
        unarchive_source_dir: Source directory of archives to extract.
        download_workers: Number of parallel download workers.
        unarchive_workers: Number of parallel unarchive workers.
        sort_workers: Number of workers for sorting extracted files.
        retries: Number of retries for failed downloads.
        timeout_s: Timeout in seconds for downloads.
        include_non_archives_in_unarchive_mode: Include non-archive files when unarchiving.
    """

    dest_dir: Path
    download_urls_file: Path | None = None
    unarchive_source_dir: Path | None = None
    download_workers: int = 8
    unarchive_workers: int = 4
    sort_workers: int = 4
    retries: int = 3
    timeout_s: int = 60
    include_non_archives_in_unarchive_mode: bool = True


# ---------------------------------------------------------------------------
# Preprocess (Stage 1-2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinkMode:
    """Configuration for how files should be materialized."""

    mode: str  # "hardlink" | "symlink" | "copy"


@dataclass(frozen=True)
class RunConfig:
    """Configuration for the preprocess pipeline (scan → hash → dedupe → derivatives).

    Attributes:
        src_root: Source directory to scan for input files.
        out_root: Output root directory for database and results.
        max_files: Optional maximum number of files to process.
        bucket_size: Number of files per output bucket directory.
        link_mode: How to materialize canonical original files.
        hash_workers: Number of threads for xxh3 hashing.
        scan_insert_batch: Batch size for database inserts during scan.
        decode_backend: Backend for image decoding ('cpu' or 'dali').
        device: Compute device for resizing ('cpu' or 'cuda').
        presets_arg: Comma-separated preset specifications.
        fmt: Output format for derivatives ('jpeg', 'png', 'webp', 'tiff').
        jpeg_quality: JPEG quality percentage (1-100).
        preserve_source: Whether to preserve source container/extension.
        preserve_color: Whether to keep color (False = grayscale).
        preserve_quality: Whether to use high-quality defaults for JPEG.
        decode_batch: Batch size for derivative scheduling.
        inflight_batches: Maximum in-flight batches to writer.
        writer_workers: Number of threads for encode/link/copy operations.
        metrics_every_s: Interval in seconds for progress logging.
        dali_batch_multiplier: DALI batch multiplier.
        crop: Enable canonical crop detection.
        deskew: Enable canonical deskew.
        preview_long_edge: Preview edge size for crop/deskew analysis.
        crop_bg: Crop background mode.
        crop_padding_px: Padding around detected crop box.
        crop_white_bg_thresh: Threshold for white background detection.
        crop_black_bg_thresh: Threshold for black background detection.
        max_crop_margin_ratio: Maximum fraction cropped from any edge.
        min_retained_area_ratio: Minimum area retained after cropping.
        deskew_max_angle_deg: Maximum deskew search range in degrees.
        deskew_step_deg: Deskew search step in degrees.
        deskew_min_conf: Minimum confidence required to apply deskew.
    """

    src_root: Path
    out_root: Path
    max_files: int | None
    bucket_size: int
    link_mode: LinkMode
    hash_workers: int
    scan_insert_batch: int
    decode_backend: Literal["cpu", "dali"]
    device: Literal["cpu", "cuda"]
    presets_arg: str
    fmt: Literal["jpeg", "png", "webp", "tiff"]
    jpeg_quality: int
    preserve_source: bool
    preserve_color: bool
    preserve_quality: bool
    decode_batch: int
    inflight_batches: int
    writer_workers: int
    metrics_every_s: float
    dali_batch_multiplier: int
    crop: bool = False
    deskew: bool = False
    preview_long_edge: int = 1024
    crop_bg: Literal["auto", "white", "black"] = "auto"
    crop_padding_px: int = 8
    crop_white_bg_thresh: int = 245
    crop_black_bg_thresh: int = 10
    max_crop_margin_ratio: float = 0.25
    min_retained_area_ratio: float = 0.60
    deskew_max_angle_deg: float = 2.0
    deskew_step_deg: float = 0.5
    deskew_min_conf: float = 0.15


# ---------------------------------------------------------------------------
# Post-processing (Stage 7: Chunk, Embed, Enrich)
# ---------------------------------------------------------------------------


# ChunkConfig and EmbedConfig are defined in their respective modules
# (post/chunk.py and post/embed.py) to keep optional dependencies local.


@dataclass(frozen=True)
class EnrichConfig:
    """Configuration for document enrichment (subject, summary, entities, tags).

    Attributes:
        db_path: Path to the DuckDB database.
        server_url: OpenAI-compatible server URL for the text LLM.
        api_key: API key for the LLM server.
        model: LLM model identifier (e.g., 'Qwen2.5-7B-Instruct').
        max_tokens: Maximum input tokens for the model (used to estimate char budget).
        max_output_tokens: Maximum tokens the model generates (JSON output).
        tokens_per_word: Heuristic tokens per word for char budget estimation.
        chars_per_word: Heuristic chars per word for char budget estimation.
        skip_too_long: If True, skip docs whose text exceeds max_chars instead of truncating.
        reprocess_existing: If True, re-enrich docs that already have an enrichment row.
        max_docs: Optional cap on number of docs to enrich (for testing).
    """

    db_path: Path = Path("index.duckdb")
    server_url: str = "http://localhost:9001"
    api_key: str = ""
    model: str = "Qwen2.5-7B-Instruct"
    max_tokens: int = 8192
    max_output_tokens: int = 512
    tokens_per_word: float = 1.4
    chars_per_word: float = 5.0
    skip_too_long: bool = False
    reprocess_existing: bool = False
    max_docs: int | None = None
