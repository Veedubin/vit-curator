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
  - layout-graph: Build spatial relationship graph from OCR/label output
  - run-all:    YAML config-driven pipeline chaining
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import networkx as nx
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
    parallel: bool = typer.Option(
        False, "--parallel", help="Run independent stages in parallel using NetworkX DAG analysis"
    ),
    langgraph: bool = typer.Option(
        False,
        "--langgraph",
        help="Use LangGraph for stateful, resumable execution with checkpointing",
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

    if parallel and langgraph:
        console.print("[red]--parallel and --langgraph are mutually exclusive.[/]")
        raise typer.Exit(1)

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
        if parallel:
            G = _build_pipeline_dag(stages_to_run)
            if nx.is_directed_acyclic_graph(G):
                generations = list(nx.topological_generations(G))
                console.print("[cyan]DAG execution plan (parallel groups):[/]")
                for gen_idx, generation in enumerate(generations):
                    console.print(f"  Gen {gen_idx}: {', '.join(generation)}")
                try:
                    critical_path = nx.dag_longest_path(G)
                    console.print(f"[cyan]Critical path: {' → '.join(critical_path)}[/]")
                except Exception:
                    pass
            else:
                console.print("[red]DAG contains cycles — would fall back to sequential[/]")
        for stage in stages_to_run:
            stage_cfg = cfg_data.get(stage, {})
            console.print(f"\n[bold]{stage}:[/]")
            for k, v in stage_cfg.items():
                console.print(f"  {k}: {v}")
        console.print(f"\n[cyan]{'=' * 60}[/]")
        return

    pipeline_name = cfg_data.get("pipeline", {}).get("name", "vit-curator")
    console.print(f"[green]{'=' * 60}[/]")
    console.print(f"[bold green]Running: {pipeline_name}[/]")
    console.print(f"[green]Stages: {', '.join(stages_to_run)}[/]")
    console.print(f"[green]{'=' * 60}[/]\n")

    # Execute stages (langgraph, parallel, or sequential)
    if langgraph:
        try:
            import vit_curator.langgraph_pipeline as _lg  # noqa: PLC0415, F401
        except ImportError:
            console.print(
                "[red]langgraph not installed. Install with: pip install vit-curator[langgraph][/]"
            )
            raise typer.Exit(1) from None

        console.print("[cyan]LangGraph mode enabled — stateful execution with checkpointing[/]")
        overall_ok = _run_stages_langgraph(stages_to_run, cfg_data, config, console)
    elif parallel:
        console.print("[cyan]Parallel mode enabled — using NetworkX DAG analysis[/]")
        overall_ok = _run_stages_parallel(stages_to_run, cfg_data, console)
    else:
        overall_ok = _run_stages_sequential(stages_to_run, cfg_data, console)

    if overall_ok:
        console.print(f"[bold green]{'=' * 60}[/]")
        console.print("[bold green]Pipeline complete![/]")
        console.print(f"[bold green]{'=' * 60}[/]")
    else:
        console.print(f"[bold red]{'=' * 60}[/]")
        console.print("[bold red]Pipeline stopped due to error.[/]")
        console.print(f"[bold red]{'=' * 60}[/]")
        raise typer.Exit(1)


def _build_pipeline_dag(stages: list[str]) -> nx.DiGraph:
    """Build a directed acyclic graph of pipeline stage dependencies.

    Returns a NetworkX DiGraph where edges represent dependencies
    (A → B means B depends on A). Stages with no edges between them
    can run in parallel.
    """

    G = nx.DiGraph()

    # Add all stages as nodes
    for stage in stages:
        G.add_node(stage)

    # Define known dependencies between stages
    dependencies = [
        ("ingest", "preprocess"),  # preprocess needs ingested files
        ("preprocess", "label"),  # label needs preprocessed images
        ("label", "train"),  # train needs labels
        ("train", "evaluate"),  # evaluate needs trained model
        ("train", "predict"),  # predict needs trained model
        ("evaluate", "predict"),  # predict should run after evaluate
        ("predict", "chunk"),  # chunk needs predictions
        ("chunk", "embed"),  # embed needs chunks
        ("embed", "enrich"),  # enrich can use embeddings
    ]

    # Only add edges for stages that are actually in the pipeline
    for src, dst in dependencies:
        if src in stages and dst in stages:
            G.add_edge(src, dst)

    return G


def _run_stages_sequential(
    stages_to_run: list[str],
    cfg_data: dict,
    console: Console,
) -> bool:
    """Run pipeline stages sequentially (original behavior)."""
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
    return overall_ok


def _run_stages_parallel(
    stages_to_run: list[str],
    cfg_data: dict,
    console: Console,
) -> bool:
    """Run pipeline stages in parallel where possible using NetworkX DAG analysis.

    Uses topological generations to identify independent stages that can
    run concurrently via ThreadPoolExecutor.
    """
    G = _build_pipeline_dag(stages_to_run)

    if not nx.is_directed_acyclic_graph(G):
        console.print("[red]Pipeline DAG contains cycles! Falling back to sequential.[/]")
        return _run_stages_sequential(stages_to_run, cfg_data, console)

    # Get topological generations (groups of independent stages)
    generations = list(nx.topological_generations(G))

    # Find critical path for reporting
    try:
        critical_path = nx.dag_longest_path(G)
        console.print(f"[dim]Critical path: {' → '.join(critical_path)}[/]")
    except Exception:
        pass

    overall_ok = True
    stage_results: dict[str, bool] = {}

    for gen_idx, generation in enumerate(generations):
        if len(generation) == 1:
            # Single stage — run directly
            stage = generation[0]
            stage_cfg = cfg_data.get(stage, {})
            if not stage_cfg:
                console.print(f"[yellow]Skipping {stage} — no configuration.[/]")
                stage_results[stage] = True
                continue

            console.print(f"[bold cyan]→ Stage: {stage} (gen {gen_idx})[/]")
            start = time.time()
            try:
                _run_stage(stage, stage_cfg, console)
                elapsed = time.time() - start
                console.print(f"[green]  ✓ {stage} complete ({elapsed:.1f}s)[/]\n")
                stage_results[stage] = True
            except Exception as exc:
                elapsed = time.time() - start
                console.print(f"[red]  ✗ {stage} failed after {elapsed:.1f}s: {exc}[/]\n")
                stage_results[stage] = False
                overall_ok = False
                break
        else:
            # Multiple independent stages — run in parallel
            console.print(
                f"[bold cyan]→ Generation {gen_idx}: {', '.join(generation)} (parallel)[/]"
            )

            def _run_one(stage: str) -> tuple[str, bool, str]:
                stage_cfg = cfg_data.get(stage, {})
                if not stage_cfg:
                    return (stage, True, "skipped")
                t0 = time.time()
                try:
                    _run_stage(stage, stage_cfg, console)
                    elapsed = time.time() - t0
                    return (stage, True, f"{elapsed:.1f}s")
                except Exception as exc:
                    elapsed = time.time() - t0
                    return (stage, False, f"failed after {elapsed:.1f}s: {exc}")

            with ThreadPoolExecutor(max_workers=len(generation)) as executor:
                futures = {executor.submit(_run_one, stage): stage for stage in generation}
                for future in as_completed(futures):
                    stage, ok, msg = future.result()
                    if ok:
                        console.print(f"[green]  ✓ {stage} ({msg})[/]")
                    else:
                        console.print(f"[red]  ✗ {stage} ({msg})[/]")
                        overall_ok = False
                    stage_results[stage] = ok

            console.print()

            if not overall_ok:
                break

    return overall_ok


def _run_stages_langgraph(
    stages_to_run: list[str],
    cfg_data: dict,
    config_path: Path,
    console: Console,
) -> bool:
    """Run pipeline stages using LangGraph for stateful, resumable execution.

    Features:
    - Checkpoint/resume: survives crashes, resumes from last completed stage
    - Quality gates: pauses at label stage for confidence review
    - Conditional retry: retries label with different model on failure
    """
    from vit_curator.langgraph_pipeline import LangGraphExecutor, PipelineState  # noqa: PLC0415

    # Determine checkpoint directory
    out_dir = cfg_data.get("pipeline", {}).get(
        "out_dir",
        str(config_path.parent / "checkpoints"),
    )
    checkpoint_dir = Path(out_dir)

    executor = LangGraphExecutor(checkpoint_dir=checkpoint_dir)

    # Build initial state
    thread_id = config_path.stem
    initial_state: PipelineState = {
        "config_path": str(config_path),
        "stages_to_run": stages_to_run,
        "stage_results": {},
        "current_stage": "",
        "cfg_data": cfg_data,
        "out_dir": str(out_dir),
        "errors": [],
        "quality_gate_approvals": {},
        "thread_id": thread_id,
        "overall_ok": True,
    }

    # Check for existing checkpoint
    existing_state = executor.get_state(thread_id)
    if existing_state and existing_state.get("stage_results"):
        completed = [
            s for s, r in existing_state["stage_results"].items() if r.get("status") == "ok"
        ]
        console.print(
            f"[cyan]Resuming from checkpoint. Completed stages: {', '.join(completed)}[/]"
        )
        # Use existing state but update stages_to_run
        existing_state["stages_to_run"] = stages_to_run
        existing_state["cfg_data"] = cfg_data
        initial_state = existing_state

    # Run pipeline
    console.print(f"[cyan]Thread ID: {thread_id}[/]")
    console.print(f"[cyan]Checkpoint dir: {checkpoint_dir}[/]\n")

    try:
        for event in executor.run(initial_state):
            for node_name, node_state in event.items():
                stage = node_state.get("current_stage", node_name)
                result = node_state.get("stage_results", {}).get(stage, {})

                if result.get("status") == "ok":
                    console.print(
                        f"[green]  ✓ {stage} complete ({result.get('elapsed', 0):.1f}s)[/]"
                    )
                elif result.get("status") == "error":
                    console.print(f"[red]  ✗ {stage} failed: {result.get('error', 'unknown')}[/]")
                elif result.get("status") == "skipped":
                    console.print(f"[yellow]  - {stage} skipped[//]")

        return initial_state.get("overall_ok", True)

    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted. State saved to checkpoint.[/]")
        console.print(
            f"[yellow]Resume with: vit-curator run-all --config {config_path} --langgraph[/]"
        )
        return False
    except Exception as exc:
        console.print(f"[red]Pipeline failed: {exc}[/]")
        console.print("[yellow]State saved to checkpoint. Resume with --langgraph[/]")
        return False


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
# Layout graph
# ---------------------------------------------------------------------------


@app.command("layout-graph")
def layout_graph(
    db: Path = typer.Option(Path("var/duckdb/labels.duckdb"), "--db", help="DuckDB path"),
    run_id: str | None = typer.Option(None, "--run-id", help="Filter by run_id for label data"),
    output: Path | None = typer.Option(None, "--output", help="Output path for GraphML file"),
    row_tolerance: float = typer.Option(
        20.0, "--row-tolerance", help="Vertical pixel tolerance for same-row detection"
    ),
    col_tolerance: float = typer.Option(
        20.0, "--col-tolerance", help="Horizontal pixel tolerance for same-column detection"
    ),
    max_blocks: int | None = typer.Option(
        None, "--max-blocks", help="Maximum number of blocks to process"
    ),
) -> None:
    """Build a spatial relationship graph from OCR/label output.

    Analyzes document layout using NetworkX graph algorithms:
    - Reading order inference
    - Table detection via community detection
    - Region grouping via connected components

    Outputs GraphML for visualization in tools like Gephi or Cytoscape.
    """
    import json  # noqa: PLC0415

    from vit_curator.post.layout_graph import (  # noqa: PLC0415
        DocumentLayoutGraph,
        LayoutBlock,
    )

    database = _open_db(db)
    con = database.con

    # Query label data from the database
    query = """
        SELECT l.text, l.bbox, l.label, l.confidence
        FROM labels l
        JOIN files f ON l.file_pk = f.file_pk
        WHERE 1=1
    """
    params: list[Any] = []

    if run_id:
        query += " AND l.run_id = ?"
        params.append(run_id)

    if max_blocks:
        query += " LIMIT ?"
        params.append(max_blocks)

    try:
        rows = con.execute(query, params).fetchall()
    except Exception as e:
        console.print(f"[red]Failed to query labels: {e}[/]")
        console.print("[yellow]Tip: Run 'label' stage first to populate label data.[/]")
        raise typer.Exit(1) from e

    if not rows:
        console.print("[yellow]No label data found. Run the 'label' stage first.[/]")
        raise typer.Exit(0)

    # Build blocks from query results
    blocks: list[LayoutBlock] = []
    for row in rows:
        text = str(row[0]) if row[0] else ""
        bbox_raw = row[1]

        # Parse bbox (could be JSON string or list)
        if isinstance(bbox_raw, str):
            try:
                bbox_raw = json.loads(bbox_raw)
            except json.JSONDecodeError:
                bbox_raw = [0, 0, 0, 0]

        if isinstance(bbox_raw, list) and len(bbox_raw) >= 4:
            if len(bbox_raw) == 8:
                xs = [bbox_raw[0], bbox_raw[2], bbox_raw[4], bbox_raw[6]]
                ys = [bbox_raw[1], bbox_raw[3], bbox_raw[5], bbox_raw[7]]
                bbox = (min(xs), min(ys), max(xs), max(ys))
            else:
                bbox = (
                    float(bbox_raw[0]),
                    float(bbox_raw[1]),
                    float(bbox_raw[2]),
                    float(bbox_raw[3]),
                )
        else:
            bbox = (0.0, 0.0, 0.0, 0.0)

        blocks.append(
            LayoutBlock(
                text=text,
                bbox=bbox,
                label=str(row[2]) if len(row) > 2 and row[2] else "text",
                confidence=float(row[3]) if len(row) > 3 and row[3] else 1.0,
            )
        )

    # Build and analyze the graph
    graph = DocumentLayoutGraph(row_tolerance=row_tolerance, col_tolerance=col_tolerance)
    graph.add_blocks(blocks)
    result = graph.analyze()

    # Display results
    table = Table(title="Layout Graph Analysis")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Blocks", str(result.num_blocks))
    table.add_row("Edges", str(result.num_edges))
    table.add_row("Reading order length", str(len(result.reading_order)))
    table.add_row("Tables detected", str(len(result.tables)))
    table.add_row("Regions detected", str(len(result.regions)))
    console.print(table)

    # Show table details
    if result.tables:
        console.print("\n[bold]Tables detected:[/]")
        for i, table_nodes in enumerate(result.tables):
            texts = [graph.G.nodes[n]["text"][:30] for n in table_nodes[:5]]
            console.print(f"  Table {i + 1}: {len(table_nodes)} cells — {', '.join(texts)}...")

    # Show region details
    if result.regions:
        console.print(f"\n[bold]Regions: {len(result.regions)}[/]")
        for i, region_nodes in enumerate(result.regions[:5]):
            texts = [graph.G.nodes[n]["text"][:30] for n in region_nodes[:3]]
            console.print(f"  Region {i + 1}: {len(region_nodes)} blocks — {', '.join(texts)}")

    # Save GraphML if requested
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result.graph_ml)
        console.print(f"\n[green]GraphML saved to: {output}[/]")


# ---------------------------------------------------------------------------
# Knowledge graph command
# ---------------------------------------------------------------------------


@app.command("knowledge-graph")
def knowledge_graph(
    db: Path = typer.Option(Path("var/duckdb/labels.duckdb"), "--db", help="DuckDB path"),
    run_id: str | None = typer.Option(None, "--run-id", help="Filter by run_id for label data"),
    output: Path | None = typer.Option(None, "--output", help="Output path for GraphML file"),
    query_entity: str | None = typer.Option(
        None, "--query-entity", help="Find connections for a specific entity"
    ),
    query_image: str | None = typer.Option(
        None, "--query-image", help="Find images similar to this image ID"
    ),
    max_entities: int | None = typer.Option(
        None, "--max-entities", help="Maximum number of entities to process"
    ),
    top_k: int = typer.Option(10, "--top-k", help="Number of top results to show"),
) -> None:
    """Build a cross-document knowledge graph from extracted entities.

    Links entities across documents, enabling:
    - Cross-document entity search ("find all images with 'Acme Corp'")
    - Similar image discovery via shared entities
    - Entity co-occurrence analysis
    - Concept hierarchy building

    Outputs GraphML for visualization in tools like Gephi or Cytoscape.
    """
    from vit_curator.post.knowledge_graph import (  # noqa: PLC0415
        EntityInfo,
        ImageKnowledgeGraph,
    )

    database = _open_db(db)
    con = database.con

    # Query label data from the database
    query = """
        SELECT f.file_pk, l.text, l.label, l.confidence
        FROM labels l
        JOIN files f ON l.file_pk = f.file_pk
        WHERE l.text IS NOT NULL AND l.text != ''
    """
    params: list[Any] = []

    if run_id:
        query += " AND l.run_id = ?"
        params.append(run_id)

    if max_entities:
        query += " LIMIT ?"
        params.append(max_entities)

    try:
        rows = con.execute(query, params).fetchall()
    except Exception as e:
        console.print(f"[red]Failed to query labels: {e}[/]")
        console.print("[yellow]Tip: Run 'label' stage first to populate label data.[/]")
        raise typer.Exit(1) from e

    if not rows:
        console.print("[yellow]No label data found. Run the 'label' stage first.[/]")
        raise typer.Exit(0)

    # Build knowledge graph
    kg = ImageKnowledgeGraph()

    # Group entities by image
    from collections import defaultdict  # noqa: PLC0415

    image_entities: dict[str, list[EntityInfo]] = defaultdict(list)

    for row in rows:
        file_pk = str(row[0])
        text = str(row[1]) if row[1] else ""
        label = str(row[2]) if len(row) > 2 and row[2] else "entity"
        confidence = float(row[3]) if len(row) > 3 and row[3] else 1.0

        if text:
            image_entities[file_pk].append(
                EntityInfo(text=text, label=label, confidence=confidence, image_id=file_pk)
            )

    for image_id, entities in image_entities.items():
        kg.load_from_entities(image_id, entities)

    # Display stats
    stats = kg.get_stats()
    table = Table(title="Knowledge Graph Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for key, value in stats.items():
        table.add_row(key.replace("_", " ").title(), f"{value:,}")
    console.print(table)

    # Top entities
    top_entities = kg.get_top_entities(top_k=top_k)
    if top_entities:
        console.print(f"\n[bold]Top {min(top_k, len(top_entities))} Entities:[/]")
        for entity, count in top_entities:
            console.print(f"  {entity}: {count} images")

    # Query specific entity if requested
    if query_entity:
        console.print(f"\n[bold]Entity Connections for '{query_entity}':[/]")
        connections = kg.find_entity_connections(query_entity)
        if connections:
            for conn in connections[:top_k]:
                console.print(f"  {conn['source']} --[{conn['relation']}]--> {conn['target']}")
        else:
            console.print("  No connections found.")

        co_occurring = kg.find_co_occurring_entities(query_entity, top_k=top_k)
        if co_occurring:
            console.print(f"\n[bold]Co-occurring with '{query_entity}':[/]")
            for entity, count in co_occurring:
                console.print(f"  {entity}: {count}x")

        images = kg.get_images_for_entity(query_entity)
        if images:
            console.print(f"\n[bold]Images containing '{query_entity}': {len(images)}[/]")
            for img in images[:top_k]:
                console.print(f"  {img}")

    # Query similar images if requested
    if query_image:
        console.print(f"\n[bold]Images similar to '{query_image}':[/]")
        similar = kg.find_similar_images(query_image, top_k=top_k)
        if similar:
            for img_id, score in similar:
                console.print(f"  {img_id}: {score:.4f}")
        else:
            console.print("  No similar images found.")

    # Save GraphML if requested
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(kg.to_graphml())
        console.print(f"\n[green]GraphML saved to: {output}[/]")


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
