"""Unified CLI for the image-processing/ML pipeline.

Provides commands for all pipeline stages:
  - ingest:     Download, extract, sort archives
  - preprocess:   Scan, hash, dedupe, decode, transform, derivatives
  - label:      VLM inference dispatch
  - train:      FastAI model training
  - evaluate:   Model evaluation + threshold tuning
  - predict:    Batch prediction with trained model
  - export-model: Export to ONNX/TorchScript
  - dashboard:  TUI control center
  - chunk:      Text chunking from predictions or files
  - embed:      Semantic embeddings for chunks
  - enrich:     LLM-based document enrichment
  - perceptual-dedupe: Near-duplicate image detection via phash
  - run-all:    YAML config-driven pipeline chaining
"""

from __future__ import annotations

import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from vit_curator.shared.db import DB, connect
from vit_curator.shared.errors import CLIError

app = typer.Typer(
    add_completion=False,
    help=(
        "ViT-Curator: ingest → preprocess → label → train → "
        "predict → dashboard → chunk/embed/enrich."
    ),
)
console = Console()


def _open_db(db_path: Path) -> DB:
    """Open a DuckDB connection with schema initialization."""
    if not db_path.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
    return connect(db_path)


# ---------------------------------------------------------------------------
# Stage 0: Ingest
# ---------------------------------------------------------------------------


@app.command()
def ingest(
    dest_dir: Path = typer.Option(
        ..., "--dest", help="Destination directory for downloads/extracts"
    ),
    download: Path | None = typer.Option(
        None, "--download", help="Path to file containing URLs to download"
    ),
    unarchive: Path | None = typer.Option(
        None, "--unarchive", help="Source directory of archives to extract"
    ),
    download_workers: int = typer.Option(
        8, "--download-workers", help="Number of download workers"
    ),
    unarchive_workers: int = typer.Option(
        4, "--unarchive-workers", help="Number of unarchive workers"
    ),
    sort_workers: int = typer.Option(4, "--sort-workers", help="Number of sort workers"),
    retries: int = typer.Option(3, "--retries", help="Number of download retries"),
    timeout_s: int = typer.Option(60, "--timeout-s", help="Download timeout in seconds"),
) -> None:
    """Download, extract, and sort archives into a structured directory."""
    from vit_curator.config import IngestConfig  # noqa: PLC0415
    from vit_curator.ingest.pipeline import run_ingest  # noqa: PLC0415

    if download and unarchive:
        raise CLIError("Use either --download or --unarchive, not both.")

    cfg = IngestConfig(
        dest_dir=dest_dir,
        download_urls_file=download,
        unarchive_source_dir=unarchive,
        download_workers=download_workers,
        unarchive_workers=unarchive_workers,
        sort_workers=sort_workers,
        retries=retries,
        timeout_s=timeout_s,
    )
    result_dir = run_ingest(cfg)
    console.print(f"[green]Ingest complete. Sorted files at: {result_dir}[/]")


# ---------------------------------------------------------------------------
# Stage 1-2: Preprocess
# ---------------------------------------------------------------------------


@app.command()
def preprocess(
    src: Path = typer.Option(None, "--src", help="Source directory to scan"),
    out: Path = typer.Option(..., "--out", help="Output root directory"),
    db: Path | None = typer.Option(None, "--db", help="DuckDB path (default: out/index.duckdb)"),
    max_files: int | None = typer.Option(None, "--max-files", help="Max files to process"),
    bucket_size: int = typer.Option(10_000, "--bucket-size", help="Files per bucket directory"),
    link_mode: str = typer.Option(
        "hardlink", "--link-mode", help="Materialization mode: hardlink|symlink|copy"
    ),
    hash_workers: int = typer.Option(8, "--hash-workers", help="Number of hash threads"),
    presets: str = typer.Option(
        "vit-train-256=256", "--presets", help="Preset specs: name=WxH,..."
    ),
    fmt: str = typer.Option("jpeg", "--fmt", help="Output format: jpeg|png|webp|tiff"),
    jpeg_quality: int = typer.Option(80, "--jpeg-quality", help="JPEG quality (1-100)"),
    decode_backend: str = typer.Option("cpu", "--decode-backend", help="Decode backend: cpu|dali"),
    device: str = typer.Option("cpu", "--device", help="Compute device: cpu|cuda"),
    crop: bool = typer.Option(False, "--crop", help="Enable crop detection"),
    deskew: bool = typer.Option(False, "--deskew", help="Enable deskew"),
    preserve_color: bool = typer.Option(
        False, "--preserve-color", help="Preserve color in output (False = grayscale)"
    ),
    writer_workers: int = typer.Option(4, "--writer-workers", help="Number of writer threads"),
) -> None:
    """Scan, hash, dedupe, decode, and generate derivative images."""
    from vit_curator.config import LinkMode, RunConfig  # noqa: PLC0415
    from vit_curator.preprocess.derivatives import run_pipeline  # noqa: PLC0415

    if src is None:
        raise CLIError("Provide --src or use --download/--unarchive to generate source.")

    cfg = RunConfig(
        src_root=src,
        out_root=out,
        max_files=max_files,
        bucket_size=bucket_size,
        link_mode=LinkMode(link_mode),
        hash_workers=hash_workers,
        scan_insert_batch=20_000,
        decode_backend=decode_backend,  # type: ignore[arg-type]
        device=device,  # type: ignore[arg-type]
        presets_arg=presets,
        fmt=fmt,  # type: ignore[arg-type]
        jpeg_quality=jpeg_quality,
        preserve_source=False,
        preserve_color=preserve_color,
        preserve_quality=False,
        decode_batch=64,
        inflight_batches=4,
        writer_workers=writer_workers,
        metrics_every_s=2.0,
        dali_batch_multiplier=4,
        crop=crop,
        deskew=deskew,
    )
    run_pipeline(cfg)
    db_path = db or (out / "index.duckdb")
    console.print(f"[green]Preprocess complete. Database: {db_path}[/]")


# ---------------------------------------------------------------------------
# Stage 3: Label
# ---------------------------------------------------------------------------


@app.command()
def label(
    db: Path = typer.Option(Path("var/duckdb/labels.duckdb"), "--db", help="DuckDB path"),
    labels: Path = typer.Option(
        Path("configs/labels.default.json"), "--labels", help="Labels JSON path"
    ),
    server: str = typer.Option("http://localhost:8000", "--server", help="vLLM base URL"),
    model: str = typer.Option("Qwen/Qwen3-VL-7B-Instruct", "--model", help="vLLM model id"),
    max_inflight: int = typer.Option(32, "--max-inflight", help="Max concurrent requests"),
    sample_pool: int = typer.Option(
        100, "--sample-pool", help="Percent of unique files to process (1-100)"
    ),
    new_run: bool = typer.Option(False, "--new-run", help="Force a new run"),
) -> None:
    """Run VLM labeling on ingested images."""
    import asyncio  # noqa: PLC0415

    from vit_curator.label.dispatcher import DispatchConfig, run_dispatch_loop  # noqa: PLC0415
    from vit_curator.label.prompt import build_prompt, load_labelset  # noqa: PLC0415
    from vit_curator.label.store import connect_label_db  # noqa: PLC0415

    conn = connect_label_db(db)
    labelset = load_labelset(str(labels)) if labels.exists() else None
    if labelset is not None:
        bundle = build_prompt(labelset)
        prompt_text = bundle.prompt
        prompt_schema = bundle.schema
    else:
        prompt_text = ""
        prompt_schema = None

    cfg = DispatchConfig(
        run_id="",
        server_url=server,
        model=model,
        prompt=prompt_text,
        schema=prompt_schema,
        include_text=True,
        include_subject=False,
        include_entities=False,
        include_summary=False,
        text_output_dir=None,
        output_root=None,
        output_ext="json",
        max_inflight=max_inflight,
        batch_size=64,
        max_tokens=64,
        temperature=0.0,
        timeout_s=120.0,
        stream=False,
        stream_include_usage=False,
        dynamic_concurrency=False,
        min_inflight=8,
        max_inflight_cap=256,
        ema_halflife_s=30.0,
        auto_tune=False,
        min_batch_size=32,
        max_batch_size=128,
        batch_step=16,
        target_p95_ms=5000.0,
        target_ttft_ms=None,
        min_tok_s=None,
        max_err_rate=0.05,
        warmup_batches=3,
        tune_interval_s=30.0,
        max_attempts=3,
        retry_backoff_s=1.0,
        retry_backoff_mult=2.0,
        retry_backoff_cap_s=60.0,
        uncertain_label_ids=(),
        use_dashboard=True,
        metrics_interval_s=2.0,
    )
    asyncio.run(run_dispatch_loop(conn=conn, cfg=cfg, console=console))
    console.print("[green]Labeling complete.[/]")


# ---------------------------------------------------------------------------
# Stage 4: Train
# ---------------------------------------------------------------------------


@app.command()
def train(
    db: Path = typer.Option(Path("var/duckdb/labels.duckdb"), "--db", help="DuckDB path"),
    run_id: str = typer.Option(..., "--run-id", help="Training run UUID"),
    model_arch: str = typer.Option("vit", "--model-arch", help="Architecture: vit|resnet"),
    epochs: int = typer.Option(10, "--epochs", help="Number of training epochs"),
    lr: float = typer.Option(1e-3, "--lr", help="Learning rate"),
    batch_size: int = typer.Option(64, "--batch-size", help="Training batch size"),
) -> None:
    """Train a FastAI model on labeled data."""
    from vit_curator.train.train import train_model  # noqa: PLC0415

    train_model(
        db_path=db,
        run_id=run_id,
        output_path=db.parent / "models" / f"{run_id}.pkl",
        model_arch=model_arch,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )
    console.print("[green]Training complete. Model saved.[/]")


# ---------------------------------------------------------------------------
# Stage 5: Evaluate
# ---------------------------------------------------------------------------


@app.command()
def evaluate(
    db: Path = typer.Option(Path("var/duckdb/labels.duckdb"), "--db", help="DuckDB path"),
    run_id: str = typer.Option(..., "--run-id", help="Run UUID to evaluate"),
    model: Path = typer.Option(..., "--model", help="Path to trained model (.pkl)"),
    tune: bool = typer.Option(False, "--tune", help="Tune classification thresholds"),
) -> None:
    """Evaluate a trained model on labeled data."""
    from vit_curator.train.evaluate import evaluate_run  # noqa: PLC0415

    metrics = evaluate_run(db_path=db, run_id=run_id, model_path=model)
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in metrics.items():
        table.add_row(str(k), f"{v:.4f}" if isinstance(v, float) else str(v))
    console.print(table)


# ---------------------------------------------------------------------------
# Stage 6: Predict
# ---------------------------------------------------------------------------


@app.command()
def predict(
    db: Path = typer.Option(Path("var/duckdb/labels.duckdb"), "--db", help="DuckDB path"),
    model: Path = typer.Option(..., "--model", help="Path to trained model (.pkl)"),
    run_id: str | None = typer.Option(
        None, "--target-run-id", help="Target run UUID for predictions"
    ),
) -> None:
    """Run batch prediction with a trained model."""
    from vit_curator.train.predict import predict_run  # noqa: PLC0415

    if run_id is None:
        raise CLIError("Provide --target-run-id for predictions")

    n = predict_run(
        model_path=model,
        db_path=db,
        target_run_id=run_id,
    )
    console.print(f"[green]Predictions complete: {n} files processed[/]")


# ---------------------------------------------------------------------------
# Stage 6b: Export Model
# ---------------------------------------------------------------------------


@app.command("export-model")
def export_model(
    model: Path = typer.Option(..., "--model", help="Path to trained model (.pkl)"),
    formats: str = typer.Option(
        "onnx,torchscript", "--formats", help="Export formats: onnx,torchscript,pkl,checkpoint"
    ),
    out_dir: Path = typer.Option(Path("exported_models"), "--out-dir", help="Output directory"),
) -> None:
    """Export a trained model to ONNX and/or TorchScript formats."""
    from vit_curator.train.export import export_all_formats  # noqa: PLC0415
    from vit_curator.train.predict import load_trained_model  # noqa: PLC0415

    learner = load_trained_model(model)
    results = export_all_formats(
        learner=learner,
        output_dir=out_dir,
        base_name=model.stem,
    )
    table = Table(title="Export Results")
    table.add_column("Format", style="cyan")
    table.add_column("Path", style="green")
    for fmt_name, path in results.items():
        table.add_row(fmt_name, path)
    console.print(table)


# ---------------------------------------------------------------------------
# Dashboard (TUI)
# ---------------------------------------------------------------------------


@app.command()
def dashboard(
    db: Path = typer.Option(Path("var/duckdb/labels.duckdb"), "--db", help="DuckDB path"),
    refresh_rate: float = typer.Option(
        1.0, "--refresh-rate", help="Dashboard refresh rate in seconds"
    ),
) -> None:
    """Launch the Textual TUI dashboard for monitoring pipeline runs."""
    from vit_curator.tui.app import PipelineApp  # noqa: PLC0415

    app = PipelineApp(db=db, refresh_rate=refresh_rate)
    app.run()


# ---------------------------------------------------------------------------
# Run-all: YAML config-driven pipeline chaining
# ---------------------------------------------------------------------------


@app.command("run-all")
def run_all(
    config: Path = typer.Option(..., "--config", help="YAML config file for pipeline chaining"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config without executing"),
    stages: str | None = typer.Option(
        None, "--stages", help="Comma-separated stage list to run (default: all)"
    ),
) -> None:
    """Run the full pipeline from a YAML configuration file.

    Stages are run in order: ingest → preprocess → label → train → evaluate → predict.
    Each stage is optional and only runs if its config section is present.
    """
    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        console.print("[red]PyYAML is required for run-all. Install with: uv add pyyaml[/]")
        raise typer.Exit(1) from None

    if not config.exists():
        console.print(f"[red]Config file not found: {config}[/]")
        raise typer.Exit(1)

    with config.open() as fh:
        cfg_data = yaml.safe_load(fh)

    if not isinstance(cfg_data, dict):
        console.print("[red]Config must be a YAML mapping (dictionary).[/]")
        raise typer.Exit(1)

    # Determine which stages to run
    available_stages = [
        "ingest",
        "preprocess",
        "label",
        "train",
        "evaluate",
        "predict",
        "chunk",
        "embed",
        "enrich",
    ]
    selected_stages = [s.strip() for s in (stages or "").split(",") if s.strip()] if stages else []
    if selected_stages:
        # Validate stage names
        invalid = [s for s in selected_stages if s not in available_stages]
        if invalid:
            console.print(f"[red]Invalid stages: {invalid}. Valid: {available_stages}[/]")
            raise typer.Exit(1)
        stages_to_run = [s for s in available_stages if s in selected_stages]
    else:
        stages_to_run = [s for s in available_stages if s in cfg_data]

    if not stages_to_run:
        console.print(
            "[yellow]No stages configured. Add sections: "
            "ingest, preprocess, label, train, evaluate, predict, chunk, embed, enrich[/]"
        )
        raise typer.Exit(0)

    # Dry run: just validate and report
    if dry_run:
        console.print(f"[cyan]{'=' * 60}[/]")
        console.print("[cyan]DRY RUN — Configuration Valid[/]")
        console.print(f"[cyan]Config file: {config}[/]")
        console.print(f"[cyan]Stages to run: {', '.join(stages_to_run)}[/]")
        for stage in stages_to_run:
            stage_cfg = cfg_data.get(stage, {})
            console.print(f"\n[bold]{stage}:[/]")
            for k, v in stage_cfg.items():
                console.print(f"  {k}: {v}")
        console.print(f"\n[cyan]{'=' * 60}[/]")
        return

    # Execute stages sequentially
    pipeline_name = cfg_data.get("pipeline", {}).get("name", "vit-curator")
    console.print(f"[green]{'=' * 60}[/]")
    console.print(f"[bold green]Running: {pipeline_name}[/]")
    console.print(f"[green]Stages: {', '.join(stages_to_run)}[/]")
    console.print(f"[green]{'=' * 60}[/]\n")

    overall_ok = True
    for stage in stages_to_run:
        stage_cfg = cfg_data.get(stage, {})
        if not stage_cfg:
            console.print(f"[yellow]Skipping {stage} — no configuration.[/]")
            continue

        console.print(f"[bold cyan]→ Stage: {stage}[/]")
        start = time.time()
        try:
            _run_stage(stage, stage_cfg, console)
            elapsed = time.time() - start
            console.print(f"[green]  ✓ {stage} complete ({elapsed:.1f}s)[/]\n")
        except Exception as exc:
            elapsed = time.time() - start
            console.print(f"[red]  ✗ {stage} failed after {elapsed:.1f}s: {exc}[/]\n")
            overall_ok = False
            break

    if overall_ok:
        console.print(f"[bold green]{'=' * 60}[/]")
        console.print("[bold green]Pipeline complete![/]")
        console.print(f"[bold green]{'=' * 60}[/]")
    else:
        console.print(f"[bold red]{'=' * 60}[/]")
        console.print("[bold red]Pipeline stopped due to error.[/]")
        console.print(f"[bold red]{'=' * 60}[/]")
        raise typer.Exit(1)


def _run_stage(stage: str, cfg: dict, console: Console) -> None:
    """Execute a single pipeline stage from its configuration dict."""
    if stage == "ingest":
        from vit_curator.config import IngestConfig  # noqa: PLC0415
        from vit_curator.ingest.pipeline import run_ingest  # noqa: PLC0415

        ingest_cfg = IngestConfig(
            dest_dir=Path(cfg.get("dest_dir", "./ingested")),
            download_urls_file=Path(cfg["download_urls_file"])
            if cfg.get("download_urls_file")
            else None,
            unarchive_source_dir=Path(cfg["unarchive_source_dir"])
            if cfg.get("unarchive_source_dir")
            else None,
            download_workers=cfg.get("download_workers", 8),
            unarchive_workers=cfg.get("unarchive_workers", 4),
            sort_workers=cfg.get("sort_workers", 4),
            retries=cfg.get("retries", 3),
            timeout_s=cfg.get("timeout_s", 60),
        )
        result_dir = run_ingest(ingest_cfg)
        console.print(f"  Ingested files: {result_dir}")

    elif stage == "preprocess":
        from vit_curator.config import LinkMode, RunConfig  # noqa: PLC0415
        from vit_curator.preprocess import run_pipeline  # noqa: PLC0415

        run_cfg = RunConfig(
            src_root=Path(cfg.get("src", "./ingested")),
            out_root=Path(cfg.get("out", "./preprocessed")),
            max_files=cfg.get("max_files"),
            bucket_size=cfg.get("bucket_size", 10_000),
            link_mode=LinkMode(cfg.get("link_mode", "hardlink")),
            hash_workers=cfg.get("hash_workers", 8),
            scan_insert_batch=cfg.get("scan_insert_batch", 20_000),
            decode_backend=cfg.get("decode_backend", "cpu"),  # type: ignore[arg-type]
            device=cfg.get("device", "cpu"),  # type: ignore[arg-type]
            presets_arg=cfg.get("presets", "thumb-64=64"),
            fmt=cfg.get("fmt", "jpeg"),  # type: ignore[arg-type]
            jpeg_quality=cfg.get("jpeg_quality", 80),
            preserve_source=cfg.get("preserve_source", False),
            preserve_color=cfg.get("preserve_color", False),
            preserve_quality=cfg.get("preserve_quality", False),
            decode_batch=cfg.get("decode_batch", 64),
            inflight_batches=cfg.get("inflight_batches", 4),
            writer_workers=cfg.get("writer_workers", 4),
            metrics_every_s=cfg.get("metrics_every_s", 2.0),
            dali_batch_multiplier=cfg.get("dali_batch_multiplier", 4),
            crop=cfg.get("crop", False),
            deskew=cfg.get("deskew", False),
        )
        run_pipeline(run_cfg)
        console.print(f"  Preprocessed to: {run_cfg.out_root}")

    elif stage == "label":
        import asyncio  # noqa: PLC0415

        from vit_curator.label.dispatcher import (  # noqa: PLC0415
            DispatchConfig,
            run_dispatch_loop,
        )
        from vit_curator.label.prompt import build_prompt, load_labelset  # noqa: PLC0415
        from vit_curator.label.store import connect_label_db  # noqa: PLC0415

        db_path = Path(
            cfg.get("db", cfg.get("preprocess", {}).get("out", "./preprocessed")) / "index.duckdb"
        )
        conn = connect_label_db(db_path)
        labels_path = Path(cfg.get("labels_file", "configs/labels.default.json"))
        labelset = load_labelset(str(labels_path)) if labels_path.exists() else None
        if labelset is not None:
            bundle = build_prompt(labelset)
            prompt_text = bundle.prompt
            prompt_schema = bundle.schema
        else:
            prompt_text = ""
            prompt_schema = None

        dispatch_cfg = DispatchConfig(
            run_id="",
            server_url=cfg.get("server_url", "http://localhost:8000"),
            model=cfg.get("model", "Qwen/Qwen3-VL-7B-Instruct"),
            prompt=prompt_text,
            schema=prompt_schema,
            include_text=True,
            include_subject=False,
            include_entities=False,
            include_summary=False,
            text_output_dir=None,
            output_root=None,
            output_ext="json",
            max_inflight=cfg.get("max_inflight", 32),
            batch_size=cfg.get("batch_size", 64),
            max_tokens=cfg.get("max_tokens", 64),
            temperature=cfg.get("temperature", 0.0),
            timeout_s=cfg.get("timeout_s", 120.0),
            stream=False,
            stream_include_usage=False,
            dynamic_concurrency=cfg.get("dynamic_concurrency", False),
            min_inflight=cfg.get("min_inflight", 8),
            max_inflight_cap=cfg.get("max_inflight_cap", 256),
            ema_halflife_s=cfg.get("ema_halflife_s", 30.0),
            auto_tune=cfg.get("auto_tune", False),
            min_batch_size=cfg.get("min_batch_size", 32),
            max_batch_size=cfg.get("max_batch_size", 128),
            batch_step=cfg.get("batch_step", 16),
            target_p95_ms=cfg.get("target_p95_ms", 5000.0),
            target_ttft_ms=cfg.get("target_ttft_ms"),
            min_tok_s=cfg.get("min_tok_s"),
            max_err_rate=cfg.get("max_err_rate", 0.05),
            warmup_batches=cfg.get("warmup_batches", 3),
            tune_interval_s=cfg.get("tune_interval_s", 30.0),
            max_attempts=cfg.get("max_attempts", 3),
            retry_backoff_s=cfg.get("retry_backoff_s", 1.0),
            retry_backoff_mult=cfg.get("retry_backoff_mult", 2.0),
            retry_backoff_cap_s=cfg.get("retry_backoff_cap_s", 60.0),
            uncertain_label_ids=(),
            use_dashboard=True,
            metrics_interval_s=cfg.get("metrics_interval_s", 2.0),
        )
        asyncio.run(run_dispatch_loop(conn=conn, cfg=dispatch_cfg, console=console))

    elif stage == "train":
        from vit_curator.train.train import train_model  # noqa: PLC0415

        db_path = Path(cfg.get("db", "./preprocessed/index.duckdb"))
        train_model(
            db_path=db_path,
            run_id=cfg.get("run_id", ""),
            output_path=Path(cfg.get("output_path", "./models/model.pkl")),
            model_arch=cfg.get("model_arch", "vit"),
            epochs=cfg.get("epochs", 10),
            lr=cfg.get("lr", 1e-3),
            batch_size=cfg.get("batch_size", 64),
        )

    elif stage == "evaluate":
        from vit_curator.train.evaluate import evaluate_run  # noqa: PLC0415

        metrics = evaluate_run(
            db_path=Path(cfg.get("db", "./preprocessed/index.duckdb")),
            run_id=cfg.get("run_id", ""),
            model_path=Path(cfg.get("model", "./models/model.pkl")),
        )
        console.print(f"  Evaluation metrics: {metrics}")

    elif stage == "predict":
        from vit_curator.train.predict import predict_run  # noqa: PLC0415

        n = predict_run(
            model_path=Path(cfg.get("model", "./models/model.pkl")),
            db_path=Path(cfg.get("db", "./preprocessed/index.duckdb")),
            target_run_id=cfg.get("target_run_id"),
        )
        console.print(f"  Predictions: {n} files processed")

    elif stage == "chunk":
        from vit_curator.post.chunk import ChunkConfig, Chunker  # noqa: PLC0415

        database = _open_db(Path(cfg.get("db", "./preprocessed/index.duckdb")))
        chunk_cfg = ChunkConfig(
            chunk_chars=cfg.get("chunk_chars", 1200),
            chunk_overlap=cfg.get("chunk_overlap", 200),
            source_column=cfg.get("source_column", "text"),
        )
        chunker = Chunker(chunk_cfg)
        source = cfg.get("source", "predictions")
        if source == "predictions":
            n = chunker.chunk_predictions(
                database.con,
                run_id=cfg.get("run_id"),
                max_docs=cfg.get("max_docs"),
            )
        elif source == "files":
            n = chunker.chunk_files(
                database.con,
                text_dir=Path(cfg.get("text_dir", ".")),
                max_docs=cfg.get("max_docs"),
            )
        else:
            raise ValueError(f"Unknown chunk source: {source}")
        console.print(f"  Chunked {n} documents")

    elif stage == "embed":
        from vit_curator.post.embed import run_embedding  # noqa: PLC0415

        database = _open_db(Path(cfg.get("db", "./preprocessed/index.duckdb")))
        n = run_embedding(
            database.con,
            model_name=cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            device=cfg.get("device", "cpu"),
            batch_size=cfg.get("batch_size", 64),
            max_chunks=cfg.get("max_chunks"),
        )
        console.print(f"  Embedded {n} chunks")

    elif stage == "enrich":
        from vit_curator.config import EnrichConfig  # noqa: PLC0415
        from vit_curator.post.enrich import Enricher  # noqa: PLC0415

        database = _open_db(Path(cfg.get("db", "./preprocessed/index.duckdb")))
        enrich_cfg = EnrichConfig(
            db_path=Path(cfg.get("db", "./preprocessed/index.duckdb")),
            server_url=cfg.get("server_url", "http://localhost:9001"),
            api_key=cfg.get("api_key", ""),
            model=cfg.get("model", "Qwen2.5-7B-Instruct"),
            max_tokens=cfg.get("max_tokens", 8192),
            max_output_tokens=cfg.get("max_output_tokens", 512),
            tokens_per_word=cfg.get("tokens_per_word", 1.4),
            chars_per_word=cfg.get("chars_per_word", 5.0),
            skip_too_long=cfg.get("skip_too_long", False),
            reprocess_existing=cfg.get("reprocess_existing", False),
            max_docs=cfg.get("max_docs"),
        )
        enricher = Enricher(enrich_cfg)
        n = enricher.enrich(database.con, console=console)
        console.print(f"  Enriched {n} documents")

    else:
        raise ValueError(f"Unknown stage: {stage}")


# ---------------------------------------------------------------------------
# Stage 7: Post-processing (chunk, embed, enrich)
# ---------------------------------------------------------------------------


@app.command()
def chunk(
    db: Path = typer.Option(Path("var/duckdb/labels.duckdb"), "--db", help="DuckDB path"),
    source: str = typer.Option("predictions", "--source", help="Source: 'predictions' or 'files'"),
    chunk_chars: int = typer.Option(1200, "--chunk-chars", help="Max characters per chunk"),
    chunk_overlap: int = typer.Option(
        200, "--chunk-overlap", help="Character overlap between chunks"
    ),
    run_id: str | None = typer.Option(
        None, "--run-id", help="Filter by run_id (predictions source)"
    ),
    text_dir: Path | None = typer.Option(
        None, "--text-dir", help="Directory with .txt files (files source)"
    ),
    max_docs: int | None = typer.Option(None, "--max-docs", help="Cap on number of docs to chunk"),
) -> None:
    """Chunk text from predictions or files into overlapping segments."""
    from vit_curator.post.chunk import ChunkConfig, Chunker  # noqa: PLC0415

    database = _open_db(db)
    con = database.con

    cfg = ChunkConfig(chunk_chars=chunk_chars, chunk_overlap=chunk_overlap, source_column=source)
    chunker = Chunker(cfg)

    if source == "predictions":
        n = chunker.chunk_predictions(con, run_id=run_id, max_docs=max_docs)
    elif source == "files":
        if text_dir is None:
            raise CLIError("--text-dir is required when --source=files")
        n = chunker.chunk_files(con, text_dir, max_docs=max_docs)
    else:
        raise CLIError(f"Unknown chunking source: {source}. Use 'predictions' or 'files'.")

    console.print(f"[green]Chunked {n} documents.[/]")


@app.command()
def embed(
    db: Path = typer.Option(Path("var/duckdb/labels.duckdb"), "--db", help="DuckDB path"),
    model_name: str = typer.Option(
        "sentence-transformers/all-MiniLM-L6-v2", "--model", help="Embedding model name"
    ),
    device: str = typer.Option("cpu", "--device", help="Compute device: cpu|cuda"),
    batch_size: int = typer.Option(64, "--batch-size", help="Encoding batch size"),
    max_chunks: int | None = typer.Option(None, "--max-chunks", help="Cap on total chunks"),
) -> None:
    """Generate semantic embeddings for chunked text."""
    from vit_curator.post.embed import run_embedding  # noqa: PLC0415

    database = _open_db(db)
    con = database.con

    n = run_embedding(
        con,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        max_chunks=max_chunks,
    )
    console.print(f"[green]Embedded {n} chunks.[/]")


@app.command()
def enrich(
    db: Path = typer.Option(Path("var/duckdb/labels.duckdb"), "--db", help="DuckDB path"),
    server_url: str = typer.Option(
        "http://localhost:9001", "--server-url", help="OpenAI-compatible LLM server URL"
    ),
    api_key: str = typer.Option("", "--api-key", help="API key for the LLM server"),
    model: str = typer.Option("Qwen2.5-7B-Instruct", "--model", help="LLM model name"),
    max_tokens: int = typer.Option(8192, "--max-tokens", help="Max input tokens"),
    max_output_tokens: int = typer.Option(512, "--max-output-tokens", help="Max output tokens"),
    max_docs: int | None = typer.Option(None, "--max-docs", help="Cap on docs to enrich"),
    skip_too_long: bool = typer.Option(
        False, "--skip-too-long", help="Skip docs exceeding max chars instead of truncating"
    ),
    reprocess_existing: bool = typer.Option(
        False, "--reprocess-existing", help="Re-enrich docs that already have results"
    ),
) -> None:
    """Enrich documents with subject, summary, entities, and tags via LLM."""
    from vit_curator.config import EnrichConfig  # noqa: PLC0415
    from vit_curator.post.enrich import Enricher  # noqa: PLC0415

    database = _open_db(db)

    cfg = EnrichConfig(
        db_path=db,
        server_url=server_url,
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
        max_output_tokens=max_output_tokens,
        skip_too_long=skip_too_long,
        reprocess_existing=reprocess_existing,
        max_docs=max_docs,
    )
    enricher = Enricher(cfg)
    n = enricher.enrich(database.con, console=console)
    console.print(f"[green]Enriched {n} documents.[/]")


# ---------------------------------------------------------------------------
# Perceptual deduplication
# ---------------------------------------------------------------------------


@app.command("perceptual-dedupe")
def perceptual_dedupe(
    db: Path = typer.Option(Path("var/duckdb/labels.duckdb"), "--db", help="DuckDB path"),
    src: Path = typer.Option(..., "--src", help="Source directory containing image files"),
    threshold: int = typer.Option(
        8, "--threshold", help="Hamming distance threshold for near-duplicates"
    ),
    hash_size: int = typer.Option(8, "--hash-size", help="phash size (8 = 64-bit)"),
    max_files: int | None = typer.Option(None, "--max-files", help="Cap on files to scan"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Compute hashes but don't update the database"
    ),
) -> None:
    """Detect near-duplicate images using perceptual hashing."""
    from vit_curator.preprocess.perceptual_dedupe import run_perceptual_dedupe  # noqa: PLC0415

    database = _open_db(db)
    result = run_perceptual_dedupe(
        database.con,
        src_root=src,
        threshold=threshold,
        hash_size=hash_size,
        max_files=max_files,
        dry_run=dry_run,
        console=console,
    )
    console.print(
        f"[green]Perceptual dedup complete: "
        f"{result.total_scanned} scanned, "
        f"{result.near_dupes_found} near-dupes found, "
        f"{result.canonicals} canonicals.[/]"
    )


# ---------------------------------------------------------------------------
# Utility commands
# ---------------------------------------------------------------------------


@app.command()
def status(
    db: Path = typer.Option(Path("var/duckdb/labels.duckdb"), "--db", help="DuckDB path"),
) -> None:
    """Show quick status summary of the database."""
    database = _open_db(db)
    con = database.con

    n_files = con.execute("SELECT COUNT(*) FROM files;").fetchone()[0]
    n_labels = (
        con.execute("SELECT COUNT(*) FROM labels;").fetchall()[0][0]
        if "labels" in [r[0] for r in con.execute("SHOW TABLES;").fetchall()]
        else 0
    )

    table = Table(title="Pipeline Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Database", str(db))
    table.add_row("Files", f"{n_files:,}")
    table.add_row("Labels", f"{n_labels:,}")
    console.print(table)


@app.command()
def init(
    db: Path = typer.Option(Path("var/duckdb/labels.duckdb"), "--db", help="DuckDB path"),
    labels: Path | None = typer.Option(None, "--labels", help="Labels JSON path to load"),
) -> None:
    """Initialize the database schema (and optionally load labels)."""
    _open_db(db)
    console.print(f"[green]Database initialized:[/] {db}")

    if labels is not None:
        from vit_curator.label.prompt import load_labelset  # noqa: PLC0415

        loaded = load_labelset(str(labels))
        console.print(f"[green]Labels loaded from:[/] {labels} ({len(loaded.labels)} labels)")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
