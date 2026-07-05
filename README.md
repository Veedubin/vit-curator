# ViT-Curator

> End-to-end ML curation pipeline: image ingest → preprocess → VLM label → train/predict → chunk/embed/enrich, plus a live TUI dashboard. Single CLI, single DuckDB.

[![CI](https://github.com/Veedubin/vit-curator/actions/workflows/ci.yml/badge.svg)](https://github.com/Veedubin/vit-curator/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://docs.astral.sh/ruff/)

## 1. Overview

`vit-curator` is a professional-grade command-line tool designed to transform massive, unstructured collections of images into high-quality, deduplicated, labeled, and searchable knowledge bases. By integrating a single DuckDB backend with a modular pipeline, it allows users to bridge the gap between raw data ingestion and production-ready visual RAG (Retrieval-Augmented Generation) or custom model training.

### Pipeline Flow
```text
download/unzip
    → scan → hash → dedupe → crop/deskew → resize to N presets
        → VLM (e.g., Qwen3-VL) labels every image
            → train a fastai ViT/ResNet on the labels
                → predict on new images at low cost
                    → chunk / embed / enrich the captions + text
                        → Textual TUI watches it all run
```

---

## 2. Installation

### via PyPI
```bash
pip install vit-curator
```

### via `uv` (Recommended)
For full feature parity, install with optional extras based on your hardware and goals:

```bash
uv sync --all-extras
# OR specific extras:
uv sync --extra vips --extra train --extra tui
```

### Extras Reference
- `[vips]`: Enables `pyvips` for 3-10x faster image decoding and processing.
- `[dali]`: Enables `nvidia-dali-cuda120` for GPU-accelerated decoding.
- `[train]`: Adds `fastai`, `torch`, and `torchvision` for model training.
- `[label]`: Adds `nvidia-ml-py` for real-time GPU monitoring during VLM labeling.
- `[tui]`: Adds `textual` for the live monitoring dashboard.
- `[embed]`: Adds `sentence-transformers` for generating text embeddings.
- `[langgraph]`: Adds `langgraph` for checkpointed, stateful pipeline execution.
- `[dev]`: Adds `pytest`, `ruff`, and `pyright` for development.

---

## 3. Quick Start

### Minimal Working Example
Go from raw URLs to enriched data in a few steps:

```bash
# 1. Ingest images from a list of URLs
vit-curator ingest --dest ./data --download urls.txt

# 2. Preprocess (Scan, Hash, Resize)
vit-curator preprocess --src ./data/sorted --out ./data --presets "vit-train-256=256,thumb-64=64"

# 3. Label with VLM (assuming vLLM server is running)
vit-curator label --db ./data/index.duckdb --server-url http://localhost:8000

# 4. Enrich labels into searchable text
vit-curator enrich --db ./data/index.duckdb --server-url http://localhost:9001
```

### YAML Config Approach
For production pipelines, use a `pipeline.yaml` to define the entire workflow:

```bash
vit-curator run-all --config pipeline.yaml
```

---

## 4. Full Command Reference

### Data Ingestion & Preparation
- **`ingest`**: Collects and organizes raw images.
    - `--dest`: Destination directory.
    - `--download`: Path to `urls.txt`.
    - `--max-files`: Cap total images.
    - `--sort-by`: Sorting logic for organization.
    - `--link-mode`: symlink, hardlink, or copy.
- **`preprocess`**: The core image engine.
    - `--src`: Source directory.
    - `--out`: Output directory.
    - `--presets`: Comma-separated list (e.g., `name=size`).
    - `--backend`: `auto` (detects vips), `vips`, or `pil`.
    - `--crop`: Enable auto-cropping.
    - `--deskew`: Enable deskewing for scanned documents.
    - `--max-files`: Cap total images processed.
- **`perceptual-dedupe`**: Removes visually similar images.
    - `--src`: Source directory.
    - `--threshold`: pHash distance threshold.
    - `--dry-run`: List duplicates without deleting.

### Labeling & Training
- **`label`**: VLM-based automated tagging.
    - `--db`: Path to DuckDB file.
    - `--server-url`: VLM server endpoint (e.g., vLLM).
    - `--model`: Specific VLM model name.
    - `--concurrency`: Number of parallel requests.
    - `--max-tokens`: Max output tokens per label.
    - `--temperature`: Sampling temperature.
- **`train`**: Trains a custom classifier on VLM labels.
    - `--db`: Path to DuckDB file.
    - `--run-id`: Unique identifier for this training run.
    - `--arch`: Architecture (e.g., `vit_small_patch16_224`).
    - `--epochs`: Number of training epochs.
    - `--lr`: Learning rate.
    - `--batch-size`: Training batch size.
- **`evaluate`**: Validates trained model performance.
    - `--db`: Path to DuckDB file.
    - `--run-id`: Run ID to evaluate.
    - `--model`: Path to the `.pkl` model file.
- **`predict`**: Runs the trained model on new/unlabeled data.
    - `--db`: Path to DuckDB file.
    - `--model`: Path to the `.pkl` model file.
    - `--target-run-id`: Run ID to associate predictions with.
- **`export-model`**: Converts models for production.
    - `--model`: Path to `.pkl` model.
    - `--formats`: `onnx`, `torchscript`.

### RAG & Knowledge Base
- **`chunk`**: Splits text labels into manageable pieces.
    - `--db`: Path to DuckDB file.
    - `--source`: Source table (e.g., `predictions`).
    - `--run-id`: Specific run to chunk.
    - `--chunk-size`: Tokens per chunk.
    - `--overlap`: Overlap between chunks.
- **`embed`**: Generates semantic vectors for text.
    - `--db`: Path to DuckDB file.
    - `--model`: Embedding model (e.g., `sentence-transformers/...`).
- **`enrich`**: Enhances labels using another VLM/LLM.
    - `--db`: Path to DuckDB file.
    - `--server-url`: Server endpoint.
    - `--model`: Enrichment model.
    - `--max-chars`: Max character limit per enrichment.

### System & Management
- **`dashboard`**: Launches the Textual TUI.
    - `--db`: Path to DuckDB file.
- **`run-all`**: Orchestrates the full pipeline.
    - `--config`: Path to `pipeline.yaml`.
    - `--stages`: Comma-separated stages to run.
    - `--dry-run`: Validate config without executing.
    - `--parallel`: Execute independent stages in parallel.
    - `--langgraph`: Use LangGraph for stateful execution.
- **`layout-graph`**: Visualizes image spatial/layout relationships.
    - `--db`: Path to DuckDB file.
    - `--run-id`: Run ID.
    - `--output`: Output file path.
    - `--format`: Output format (e.g., `png`, `json`).
- **`knowledge-graph`**: Generates a semantic KG from labels.
    - `--db`: Path to DuckDB file.
    - `--run-id`: Run ID.
    - `--output`: Output file path.
    - `--format`: Output format.
- **`status`**: Summary of DB state and pipeline progress.
    - `--db`: Path to DuckDB file.
- **`init`**: Initializes the DuckDB schema.
    - `--db`: Path to the desired DuckDB file.

---

## 5. YAML Configuration Reference

A comprehensive `pipeline.yaml` allows for reproducible, automated workflows.

```yaml
db: "./data/index.duckdb"

ingest:
  dest: "./data"
  download: "urls.txt"

preprocess:
  src: "./data/sorted"
  out: "./data"
  presets:
    - "vit-train-256=256"
    - "thumb-64=64"
  backend: "auto"

label:
  server_url: "http://localhost:8000"
  model: "Qwen3-VL-8B"
  concurrency: 4

train:
  arch: "vit_small_patch16_224"
  epochs: 10

post:
  chunk_size: 512
  embed_model: "sentence-transformers/all-MiniLM-L6-v2"
```

---

## 6. Pipeline Architecture

### Stage-by-Stage Breakdown
1. **Ingest**: Handles the "wild" phase—downloading from URLs, extracting archives, and sorting files into a structured layout.
2. **Preprocess**: The technical foundation. It performs hashing for exact deduplication, utilizes **libvips** for high-performance resizing, and applies deskewing/cropping to clean the signal.
3. **Label**: The semantic phase. It dispatches images to a VLM (like Qwen3-VL) to generate detailed descriptive captions.
4. **Train**: The ability to distill VLM knowledge. By training a smaller ViT or ResNet on the VLM's labels, you create a fast, local classifier that replicates the VLM's logic at a fraction of the cost.
5. **Post (RAG)**: Converts labels into a searchable index via chunking, embedding (vectorization), and further enrichment.

### Core Technologies
- **DuckDB Backend**: All metadata is stored in a single-file DuckDB database. It uses an **additive-migration framework**, ensuring that as you add new pipeline stages, your existing data remains intact and compatible.
- **Parallel Execution**: When `--parallel` is used, `vit-curator` constructs a **NetworkX DAG** (Directed Acyclic Graph) of tasks and executes them via a `ThreadPoolExecutor`, maximizing CPU/GPU utilization.
- **LangGraph Integration**: The `--langgraph` mode transforms the pipeline into a **StateGraph**. This adds industrial-grade reliability: checkpointing (save/resume), quality gates (verify labels before training), and automatic retries.
- **Libvips Backend**: By using `pyvips`, the pipeline achieves 3-10x faster image decoding compared to PIL, significantly reducing bottlenecks in large-scale datasets.
- **Graph Analytics**: The `layout-graph` and `knowledge-graph` tools allow you to move beyond flat lists, mapping how images relate to each other spatially or semantically.

---

## 7. Optional Dependencies

| Extra | Package | Enables |
|-------|---------|---------|
| `vips` | `pyvips` | 3-10x faster image decode/resize |
| `dali` | `nvidia-dali-cuda120` | GPU-accelerated decode and augment |
| `train` | `fastai`, `torch`, `torchvision` | Model training, evaluation, and prediction |
| `label` | `nvidia-ml-py` | GPU memory/utilization monitoring during labeling |
| `tui` | `textual` | Live interactive monitoring dashboard |
| `embed` | `sentence-transformers` | Vector embeddings for semantic search |
| `langgraph` | `langgraph` | Checkpointed, stateful pipeline execution |

---

## 8. Real-World Use Cases

- **E-commerce Cataloging**: Automatically deduplicate product images, crop out backgrounds, auto-tag attributes (color, material), and train a custom classifier for new arrivals.
- **Document/Archive Digitization**: Batch-process scanned pages, apply deskewing to fix tilts, and build a searchable RAG system over the extracted visual/textual content.
- **Dataset Curation**: Curate massive web-scraped sets for vision-model fine-tuning by removing near-duplicates and filtering via VLM-based quality scoring.
- **Reverse Image Search**: Use perceptual hashing and embeddings to build a high-speed similarity search engine for content platforms.
- **RAG from Visual Corpus**: Transform a museum archive or corporate asset library into a queryable knowledge base where text queries find visually relevant images.

---

## 9. Development

### Setup
```bash
uv sync --extra dev
```

### Quality Assurance
```bash
# Linting
uv run ruff check .

# Testing (Standard)
uv run pytest -m "not torch and not fastai and not dali and not nvidia and not slow"

# Testing (Full Stack)
VIT_CURATOR_TEST_TORCH=1   uv run pytest
VIT_CURATOR_TEST_FASTAI=1  uv run pytest
VIT_CURATOR_TEST_DALI=1    uv run pytest
VIT_CURATOR_TEST_NVIDIA=1  uv run pytest
```

### Benchmarking
```bash
uv run python scripts/benchmark.py
```

---

## 10. Common Patterns / Recipes

### "I have a folder of images and want to train a classifier"
1. `vit-curator preprocess --src ./images --out ./processed --presets "train-256=256"`
2. `vit-curator label --db ./index.duckdb --server-url http://localhost:8000`
3. `vit-curator train --db ./index.duckdb --run-id my-first-model`

### "I want to deduplicate and label with a VLM"
1. `vit-curator preprocess --src ./images --out ./processed`
2. `vit-curator perceptual-dedupe --src ./processed --threshold 8`
3. `vit-curator label --db ./index.duckdb --server-url http://localhost:8000`

### "I want to build a searchable knowledge base"
1. Run `ingest` $\rightarrow$ `preprocess` $\rightarrow$ `label`.
2. `vit-curator chunk --db ./index.duckdb --source labels`
3. `vit-curator embed --db ./index.duckdb --model sentence-transformers/all-MiniLM-L6-v2`
4. Use the `dashboard` to explore results.

### "I want to run the full pipeline with checkpointing"
Create a `pipeline.yaml` and run:
```bash
vit-curator run-all --config pipeline.yaml --langgraph
```

---

## License

MIT — see [LICENSE](LICENSE).
