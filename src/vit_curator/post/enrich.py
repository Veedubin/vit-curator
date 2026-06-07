"""Post-processing: document-level enrichment (subject, summary, entities, tags).

Reads text from the predictions table, calls an OpenAI-compatible LLM,
and stores structured enrichment results in the doc_enrichments table.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from datetime import UTC
from typing import TYPE_CHECKING

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

from vit_curator.config import EnrichConfig

if TYPE_CHECKING:
    import duckdb

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

_THEME = Theme(
    {
        "info": "cyan",
        "ok": "green",
        "warn": "yellow",
        "error": "bold red",
        "stat": "magenta",
    }
)


def _default_console() -> Console:
    return Console(theme=_THEME)


# ---------------------------------------------------------------------------
# EnrichmentResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnrichmentResult:
    """Parsed enrichment from a single document."""

    subject: str | None = None
    summary: str | None = None
    doc_type: str | None = None
    entities_json: str | None = None
    tags_json: str | None = None


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_system_prompt() -> str:
    """System prompt for document-level enrichment.

    We want strict JSON back with subject, summary, entities, tags.
    """
    return (
        "You are an information extraction assistant working on OCR'd documents.\n"
        "You must read the document and output ONLY a single JSON object with the "
        "following shape:\n\n"
        "{\n"
        '  "subject": string,\n'
        '  "summary": string,\n'
        '  "entities": {\n'
        '    "persons": [string, ...],\n'
        '    "organizations": [string, ...],\n'
        '    "locations": [string, ...],\n'
        '    "dates": [string, ...],\n'
        '    "case_numbers": [string, ...],\n'
        '    "other": [string, ...]\n'
        "  },\n"
        '  "tags": [string, ...]\n'
        "}\n\n"
        "Rules:\n"
        "- Only output JSON, no extra commentary.\n"
        "- Keep subject short but informative (like an email subject or "
        "document title).\n"
        "- summary should be 1-3 sentences.\n"
        '- tags should be high-level categories like "court_filing", "email", '
        '"financial", "memorandum", etc.\n'
    )


def build_user_prompt(doc_text: str, file_path: str) -> str:
    """User message content for the LLM."""
    return (
        f"File path: {file_path}\n\n"
        "Document text begins below this line:\n"
        "----------------------------------------\n"
        f"{doc_text}\n"
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def call_llm(
    server_url: str,
    api_key: str,
    model: str,
    max_output_tokens: int,
    doc_text: str,
    file_path: str,
) -> tuple[str, str]:
    """Call an OpenAI-compatible /v1/chat/completions endpoint.

    Returns:
        (content, finish_reason) tuple.
    """
    try:
        import httpx  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "httpx is required for document enrichment. Install with: uv add httpx"
        ) from exc

    url = server_url.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "max_tokens": max_output_tokens,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(doc_text, file_path)},
        ],
    }

    with httpx.Client(timeout=900) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()

    data = resp.json()
    choice = data["choices"][0]
    content = choice["message"]["content"]
    if isinstance(content, list):
        content = "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content, choice.get("finish_reason", "stop")


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _extract_json_from_text(text: str) -> str:
    """Extract JSON object from LLM output, stripping code fences."""
    s = text.strip()
    if s.startswith("```"):
        _first_line, _, rest = s.partition("\n")
        if rest.rstrip().endswith("```"):
            rest = rest.rstrip()[:-3].rstrip()
        s = rest
    return s.strip()


def parse_enrichment_json(raw_text: str) -> EnrichmentResult:
    """Parse model output into structured EnrichmentResult."""
    s = _extract_json_from_text(raw_text)
    try:
        obj = json.loads(s)
    except Exception:
        return EnrichmentResult()

    subject = obj.get("subject")
    summary = obj.get("summary")
    entities = obj.get("entities", {})
    tags = obj.get("tags", [])

    doc_type = None
    if isinstance(tags, list) and tags:
        first = tags[0]
        if isinstance(first, str):
            doc_type = first

    entities_json = None
    tags_json = None
    if "entities" in obj:
        with contextlib.suppress(Exception):
            entities_json = json.dumps(entities, ensure_ascii=False)
    if "tags" in obj:
        with contextlib.suppress(Exception):
            tags_json = json.dumps(tags, ensure_ascii=False)

    return EnrichmentResult(
        subject=subject,
        summary=summary,
        doc_type=doc_type,
        entities_json=entities_json,
        tags_json=tags_json,
    )


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _count_target_files(
    con: duckdb.DuckDBPyConnection,
    model_name: str,
    reprocess_existing: bool,
) -> int:
    """Count files eligible for enrichment."""
    if reprocess_existing:
        row = con.execute(
            "SELECT COUNT(*) FROM predictions WHERE text IS NOT NULL AND text != '';"
        ).fetchone()
    else:
        row = con.execute(
            """
            SELECT COUNT(*)
            FROM predictions p
            LEFT JOIN doc_enrichments e
              ON e.file_pk = p.file_pk AND e.model_name = ?
            WHERE p.text IS NOT NULL AND p.text != ''
              AND e.file_pk IS NULL
            """,
            [model_name],
        ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _fetch_target_files(
    con: duckdb.DuckDBPyConnection,
    model_name: str,
    max_docs: int | None,
    reprocess_existing: bool,
) -> list[tuple[int, str, str]]:
    """Fetch (file_pk, text, rel_path_blob) tuples for enrichment."""
    if reprocess_existing:
        sql = """
            SELECT p.file_pk, p.text, f.rel_path_blob
            FROM predictions p
            JOIN files f ON f.file_pk = p.file_pk
            WHERE p.text IS NOT NULL AND p.text != ''
            ORDER BY p.file_pk
        """
        params: list = []
    else:
        sql = """
            SELECT p.file_pk, p.text, f.rel_path_blob
            FROM predictions p
            JOIN files f ON f.file_pk = p.file_pk
            LEFT JOIN doc_enrichments e
              ON e.file_pk = p.file_pk AND e.model_name = ?
            WHERE p.text IS NOT NULL AND p.text != ''
              AND e.file_pk IS NULL
            ORDER BY p.file_pk
        """
        params = [model_name]

    if max_docs is not None:
        sql += f" LIMIT {int(max_docs)}"

    import os  # noqa: PLC0415

    rows = con.execute(sql, params).fetchall()
    return [(int(r[0]), str(r[1]), os.fsdecode(r[2]) if r[2] else f"file_{r[0]}") for r in rows]


def _insert_enrichment(
    con: duckdb.DuckDBPyConnection,
    file_pk: int,
    model_name: str,
    result: EnrichmentResult,
    finish_reason: str,
    truncated: bool,
    text_len: int,
    word_count: int,
    raw_payload: str,
) -> None:
    """Insert enrichment result into doc_enrichments table."""
    from datetime import datetime  # noqa: PLC0415

    con.execute(
        """
        INSERT OR REPLACE INTO doc_enrichments (
            file_pk, model_name, subject, summary, doc_type,
            entities_json, tags_json, finish_reason, truncated,
            text_len, word_count, raw_payload, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            file_pk,
            model_name,
            result.subject,
            result.summary,
            result.doc_type,
            result.entities_json,
            result.tags_json,
            finish_reason,
            truncated,
            text_len,
            word_count,
            raw_payload,
            datetime.now(UTC).isoformat(),
        ),
    )


# ---------------------------------------------------------------------------
# Enricher class
# ---------------------------------------------------------------------------


class Enricher:
    """Enriches documents by calling an LLM for subject, summary, entities, tags."""

    def __init__(self, config: EnrichConfig | None = None) -> None:
        self.config = config or EnrichConfig()

    def enrich(
        self,
        con: duckdb.DuckDBPyConnection,
        console: Console | None = None,
    ) -> int:
        """Enrich documents from predictions table via LLM.

        Args:
            con: DuckDB connection with schema initialized.
            console: Rich console for progress output.

        Returns:
            Number of documents enriched.
        """
        if console is None:
            console = _default_console()

        cfg = self.config
        model_name = cfg.model

        # Compute character budget from token heuristics
        if cfg.tokens_per_word <= 0 or cfg.chars_per_word <= 0:
            raise ValueError("tokens_per_word and chars_per_word must be > 0")
        chars_per_token = cfg.chars_per_word / cfg.tokens_per_word
        max_chars = int(cfg.max_tokens * chars_per_token)

        # Count targets
        total_targets = _count_target_files(con, model_name, cfg.reprocess_existing)
        if total_targets == 0:
            console.print("[warn]No documents to enrich for this configuration.[/warn]")
            return 0

        # Fetch targets
        targets = _fetch_target_files(con, model_name, cfg.max_docs, cfg.reprocess_existing)
        if not targets:
            console.print("[warn]No documents to enrich.[/warn]")
            return 0

        num_targets = len(targets)

        # Print config
        _print_config(cfg, total_targets, num_targets, max_chars, console)

        enriched = 0
        too_long_skipped = 0
        errors = 0
        truncated_count = 0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as prog:
            task = prog.add_task("[info]Enriching documents…[/info]", total=num_targets)

            for file_pk, raw_text, file_path in targets:
                text_len = len(raw_text)
                word_count = len(raw_text.split())
                truncated = False
                doc_text = raw_text

                # Truncate or skip long texts
                if text_len > max_chars:
                    if cfg.skip_too_long:
                        console.print(
                            f"[warn]File {file_pk} ({file_path}) "
                            f"length {text_len} > max_chars {max_chars}, skipping.[/warn]"
                        )
                        too_long_skipped += 1
                        prog.update(task, advance=1)
                        continue
                    doc_text = raw_text[:max_chars]
                    truncated = True
                    truncated_count += 1

                # Call LLM
                try:
                    raw_content, finish_reason = call_llm(
                        cfg.server_url,
                        cfg.api_key,
                        cfg.model,
                        cfg.max_output_tokens,
                        doc_text,
                        file_path,
                    )
                except Exception as exc:
                    console.print(f"[error]File {file_pk} ({file_path}): LLM error: {exc}[/error]")
                    errors += 1
                    prog.update(task, advance=1)
                    continue

                # Parse result
                result = parse_enrichment_json(raw_content)
                if result.subject is None and result.summary is None:
                    console.print(
                        f"[warn]File {file_pk} ({file_path}): "
                        f"failed to parse JSON, storing raw payload only.[/warn]"
                    )
                    errors += 1

                # Store enrichment
                _insert_enrichment(
                    con,
                    file_pk=file_pk,
                    model_name=model_name,
                    result=result,
                    finish_reason=finish_reason,
                    truncated=truncated,
                    text_len=text_len,
                    word_count=word_count,
                    raw_payload=raw_content,
                )
                enriched += 1
                prog.update(task, advance=1)

        _print_summary(enriched, too_long_skipped, errors, truncated_count, console)
        return enriched


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _print_config(
    cfg: EnrichConfig,
    total_docs: int,
    target_docs: int,
    max_chars: int,
    console: Console,
) -> None:
    """Print enrichment configuration summary."""
    from rich.table import Table  # noqa: PLC0415

    if not isinstance(cfg, EnrichConfig):
        return

    approx_words = int(max_chars / cfg.chars_per_word) if cfg.chars_per_word > 0 else 0

    table = Table(title="Enrichment Configuration")
    table.add_column("Setting", style="stat")
    table.add_column("Value")
    table.add_row("Database", str(cfg.db_path))
    table.add_row("Server URL", cfg.server_url)
    table.add_row("Model", cfg.model)
    table.add_row("Max input tokens", str(cfg.max_tokens))
    table.add_row("Max output tokens", str(cfg.max_output_tokens))
    table.add_row("Max document size (chars)", f"{max_chars} (~{approx_words} words)")
    table.add_row("Total eligible docs", str(total_docs))
    table.add_row("Docs to enrich", str(target_docs))
    table.add_row("Reprocess existing", str(cfg.reprocess_existing))
    table.add_row("Skip too-long docs", str(cfg.skip_too_long))
    console.print(table)
    console.print()


def _print_summary(
    enriched: int,
    too_long_skipped: int,
    errors: int,
    truncated_count: int,
    console: Console,
) -> None:
    """Print enrichment run summary."""
    from rich.table import Table  # noqa: PLC0415

    table = Table(title="Enrichment Summary")
    table.add_column("Metric", style="stat")
    table.add_column("Count", justify="right")
    table.add_row("Docs enriched", str(enriched))
    table.add_row("Too long (skipped)", str(too_long_skipped))
    table.add_row("Truncated to max_chars", str(truncated_count))
    table.add_row("Errors", str(errors))
    console.print(table)


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def run_enrichment(
    con: duckdb.DuckDBPyConnection,
    config: EnrichConfig | None = None,
    console: Console | None = None,
) -> int:
    """High-level entry point for document enrichment.

    Args:
        con: DuckDB connection with schema initialized.
        config: Enrichment configuration. Uses defaults if None.
        console: Rich console for progress output.

    Returns:
        Number of documents enriched.
    """
    if config is None:
        config = EnrichConfig()
    enricher = Enricher(config)
    return enricher.enrich(con, console=console)
