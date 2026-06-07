# ViT-Curator

> End-to-end ML curation pipeline: image ingest → preprocess → VLM label →
> train/predict → chunk/embed/enrich, plus a live TUI dashboard.
> Single CLI, single DuckDB.

[![CI](https://github.com/Veedubin/vit-curator/actions/workflows/ci.yml/badge.svg)](https://github.com/Veedubin/vit-curator/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://docs.astral.sh/ruff/)

## What it does

ViT-Curator is a single command-line tool that turns a giant pile of messy
images into a deduplicated, labeled, searchable knowledge base — and lets
you train a custom model to auto-label new images.

```
download/unzip
    → scan → hash → dedupe → crop/deskew → resize to N presets
        → VLM (Qwen3-VL) labels every image
            → train a fastai ViT/ResNet on the labels
                → predict on new images at low cost
                    → chunk / embed / enrich the captions + text
                        → Textual TUI watches it all run
```

Everything lives in **one DuckDB file** — `files`, derivatives, labels,
runs, predictions, trained models, chunks, embeddings, and enrichments
all share a single schema with an additive-migration framework.

## Installation

```bash
uv sync                    # Core dependencies
uv sync --extra train      # + FastAI training
uv sync --extra label      # + nvidia-ml-py for GPU monitoring
uv sync --extra tui        # + Textual dashboard
uv sync --extra dali       # + NVIDIA DALI for GPU decode
uv sync --extra embed      # + sentence-transformers for embeddings
uv sync --extra dev        # + pytest, ruff, pyright
```

The CLI entry point is `vit-curator` (alias `up`).

## Usage

```bash
# Run directly via uv (recommended — no venv activation needed)

# 0. Ingest: download URLs, extract archives, sort into a directory layout
vit-curator ingest --dest /data --download urls.txt

# 1-2. Preprocess: scan, hash, dedupe, decode, generate per-preset derivatives
vit-curator preprocess --src /data/sorted --out /data \
    --presets "vit-train-256=256,thumb-64=64"

# 1.5. Near-duplicate detection (phash)
vit-curator perceptual-dedupe --src /data/sorted --threshold 8

# 3. VLM labeling against a running vLLM server
vit-curator label --db /data/index.duckdb --server-url http://localhost:8000

# 4-6. Train, evaluate, predict, export
vit-curator train      --db /data/index.duckdb --run-id <uuid>
vit-curator evaluate   --db /data/index.duckdb --run-id <uuid> --model model.pkl
vit-curator predict    --db /data/index.duckdb --model model.pkl --target-run-id <uuid>
vit-curator export-model --model model.pkl --formats onnx,torchscript

# 7. Post-processing (RAG)
vit-curator chunk   --db /data/index.duckdb --source predictions --run-id <uuid>
vit-curator embed   --db /data/index.duckdb --model sentence-transformers/all-MiniLM-L6-v2
vit-curator enrich  --db /data/index.duckdb --server-url http://localhost:9001

# Live Textual TUI dashboard
vit-curator dashboard --db /data/index.duckdb

# YAML config-driven full chain
vit-curator run-all --config pipeline.yaml --dry-run
vit-curator run-all --config pipeline.yaml --stages preprocess,train
```

## Real-world jobs it's good for

- **E-commerce cataloging** at scale (dedupe, crop, deskew, auto-tag, train a custom classifier).
- **Document / archive digitization** with skew correction + semantic search over the OCR'd text.
- **Dataset curation** before a vision-model fine-tune.
- **Reverse image search** / near-duplicate detection for a content platform.
- **RAG from a visual corpus** — turn a museum or archive into a queryable knowledge base.

See [CHANGELOG.md](CHANGELOG.md) for the full feature list.

## Architecture

```
src/vit_curator/
├── cli.py              # Typer entrypoint — every stage as a subcommand
├── config.py           # Frozen dataclasses for all stage configs
├── shared/             # DB schema, hashing, errors, progress, signals
├── ingest/             # Stage 0: download / unarchive / sort
├── preprocess/         # Stages 1-2: scan, hash, dedupe, decode, derivatives, crop/deskew, perceptual dedupe
├── label/              # Stage 3: VLM dispatcher + metrics + scheduler
├── train/              # Stages 4-6: train, evaluate, predict, export
├── post/               # Stage 7: chunk, embed, enrich
└── tui/                # Textual dashboard (5 screens, GPU meter, latency histogram)
```

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest -m "not torch and not fastai and not dali and not nvidia and not slow"
uv run python scripts/benchmark.py    # smoke benchmark of all 4 core stages
```

The default test deselection skips tests that need optional heavy stacks
(torch, fastai, DALI, nvidia-ml-py). To run those locally:

```bash
VIT_CURATOR_TEST_TORCH=1   uv run pytest
VIT_CURATOR_TEST_FASTAI=1  uv run pytest
VIT_CURATOR_TEST_DALI=1    uv run pytest
VIT_CURATOR_TEST_NVIDIA=1  uv run pytest
```

## License

MIT — see [LICENSE](LICENSE).
