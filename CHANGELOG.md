# Changelog

All notable changes to ViT-Curator are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-07

### Changed
- **Renamed project from `unified-pipeline` to `vit-curator`** across package, CLI,
  module imports, env-var prefixes, docs, and lockfile.
- CLI entry point: `unified-pipeline` / `up` are now `vit-curator` / `up`.
- Module path: `unified_pipeline.*` is now `vit_curator.*`.
- Env-var prefix: `UNIFIED_PIPELINE_TEST_*` is now `VIT_CURATOR_TEST_*`.

### Fixed
- 18 ruff lint errors in `scripts/benchmark.py` (unused imports, top-level import
  placement, long lines, unused locals, loop-variable overwrite).
- Stale `# Future: Run-all` comment in `cli.py` — the command was already
  implemented below it.

## [0.1.0] - 2026-05-30

Initial release as `unified-pipeline`.

### Features
- **Stage 0 — Ingest**: parallel download, unarchive, and sort into structured
  directory layout.
- **Stages 1-2 — Preprocess**: scan, xxh3 hash, exact-hash dedupe, decode
  (CPU + optional NVIDIA DALI), per-preset derivative generation with
  crop and deskew analysis. Hardlink / symlink / copy materialization.
- **Perceptual dedupe**: phash-based near-duplicate detection as a separate
  CLI command (`vit-curator perceptual-dedupe`).
- **Stage 3 — Label**: async VLM dispatcher with dynamic concurrency,
  auto-tune (EMA halflife, p95 / TTFT / tok/s targets), error-rate backoff,
  retry-with-jitter, live `MetricsDashboard` and progress.
- **Stages 4-6 — Train / Evaluate / Predict / Export**: FastAI ViT and
  ResNet trainers, threshold tuning, batch prediction, ONNX and TorchScript
  export.
- **Stage 7 — Post-processing**: `chunk` (overlapping text chunks),
  `embed` (sentence-transformers vectors), `enrich` (LLM subject / summary /
  entities / tags).
- **Textual TUI dashboard**: 5 screens (Dashboard, Runs, Assets, Stats,
  Settings), GPU meter, latency histogram, activity log.
- **`run-all`**: YAML config-driven chain executor with `--dry-run` and
  per-stage filtering across 9 stage adapters.
- **CLI utilities**: `status`, `init` for ad-hoc DB inspection.
- **Single DuckDB schema** carries every artifact (files, derivatives,
  labels, runs, tasks, predictions, models, chunks, embeddings,
  enrichments) with an additive-migration framework.
- 104 tests passing (31 gated for torch / fastai / DALI / nvidia-ml-py).
