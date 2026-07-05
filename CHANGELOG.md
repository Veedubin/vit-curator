# Changelog

All notable changes to ViT-Curator are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-07-04

### Added
- **P0-2: libvips decode backend** — `decode_rgb_u8_chw_vips()` with PIL fallback. `backend="auto"|"vips"|"pil"` parameter. 3-10x faster batch decode.
- **P1-4: NetworkX pipeline DAG** — `--parallel` flag on `run-all`. `_build_pipeline_dag()`, `_run_stages_parallel()` with `ThreadPoolExecutor`. `networkx>=3.2` added to core deps.
- **P1-5: NetworkX document layout graphs** — New `post/layout_graph.py` with `DocumentLayoutGraph` class. New `layout-graph` CLI command. Reading order inference, table detection, region grouping, GraphML export.
- **P2: NetworkX knowledge graph** — New `post/knowledge_graph.py` with `ImageKnowledgeGraph` class. Cross-document entity linking, Jaccard similarity search, co-occurrence analysis, concept hierarchy. New `knowledge-graph` CLI command.
- **P3: LangGraph batch pipelines** — New `langgraph_pipeline.py` with `PipelineState` TypedDict, `_build_pipeline_graph()` (StateGraph with 9 stages + quality gate + conditional retry), `LangGraphExecutor` class. `--langgraph` flag on `run-all`. Mutual exclusion with `--parallel`.

### Changed
- **New CLI commands**: `layout-graph`, `knowledge-graph`, `run-all --parallel`, `run-all --langgraph`.
- **New optional dependencies**: `[vips]` — pyvips>=2.2.0; `[langgraph]` — langgraph>=0.2.0, langgraph-checkpoint-sqlite>=2.0.0.

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
