"""VLM label dispatcher — async dispatch loop with dynamic concurrency.

Ported from ocrmj_labeler.pipeline.dispatcher, adapted to use unified schema
(file_pk instead of asset_id).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import orjson
from rich.console import Console

from vit_curator.label import store as label_store
from vit_curator.label.client import VllmClient
from vit_curator.label.metrics import MetricsDashboard, RunMetrics
from vit_curator.label.scheduler import AutoTune, DynamicConcurrency


@dataclass
class DispatchConfig:
    run_id: str
    server_url: str
    model: str
    prompt: str
    schema: dict | None
    include_text: bool
    include_subject: bool
    include_entities: bool
    include_summary: bool
    text_output_dir: Path | None
    output_root: Path | None
    output_ext: str
    max_inflight: int
    batch_size: int
    max_tokens: int
    temperature: float
    timeout_s: float
    stream: bool
    stream_include_usage: bool
    dynamic_concurrency: bool
    min_inflight: int
    max_inflight_cap: int
    ema_halflife_s: float
    auto_tune: bool
    min_batch_size: int
    max_batch_size: int
    batch_step: int
    target_p95_ms: float
    target_ttft_ms: float | None
    min_tok_s: float | None
    max_err_rate: float
    warmup_batches: int
    tune_interval_s: float
    max_attempts: int
    retry_backoff_s: float
    retry_backoff_mult: float
    retry_backoff_cap_s: float
    uncertain_label_ids: tuple[int, ...]
    use_dashboard: bool = True
    metrics_interval_s: float = 2.0
    metrics_history_window: int = 10000


async def run_dispatch_loop(
    *,
    conn: Any,
    cfg: DispatchConfig,
    console: Console,
) -> None:
    client = VllmClient(base_url=cfg.server_url, timeout_s=cfg.timeout_s)
    metrics = RunMetrics(max_history_window=cfg.metrics_history_window)
    metrics.lat.max_samples = cfg.metrics_history_window
    metrics.ttft.max_samples = cfg.metrics_history_window
    metrics.db_fetch_time_ms.max_samples = cfg.metrics_history_window
    metrics.db_write_time_ms.max_samples = cfg.metrics_history_window
    dyn = DynamicConcurrency(
        min_inflight=cfg.min_inflight,
        max_inflight=cfg.max_inflight_cap,
        target_p95_ms=cfg.target_p95_ms,
        ema_halflife_s=cfg.ema_halflife_s,
    )
    tuner = AutoTune(
        min_inflight=cfg.min_inflight,
        max_inflight=cfg.max_inflight_cap,
        min_batch_size=cfg.min_batch_size,
        max_batch_size=cfg.max_batch_size,
        batch_step=cfg.batch_step,
        target_p95_ms=cfg.target_p95_ms,
        target_ttft_ms=cfg.target_ttft_ms,
        min_tok_s=cfg.min_tok_s,
        max_err_rate=cfg.max_err_rate,
        warmup_batches=cfg.warmup_batches,
        cooldown_s=cfg.tune_interval_s,
    )
    uncertain_ids = set(cfg.uncertain_label_ids)

    inflight_limit = cfg.max_inflight
    sem = asyncio.Semaphore(inflight_limit)
    batch_limit = max(cfg.min_batch_size, min(cfg.max_batch_size, cfg.batch_size))
    batch_limit = max(batch_limit, inflight_limit)
    text_mode = cfg.text_output_dir is not None

    limits = httpx.Limits(
        max_keepalive_connections=cfg.max_inflight_cap,
        max_connections=cfg.max_inflight_cap,
    )
    async with httpx.AsyncClient(timeout=cfg.timeout_s, limits=limits) as http:
        last_report = time.perf_counter()

        async def one(file_pk: int, path: str, attempt: int):
            nonlocal sem
            async with sem:
                img_path = Path(path)
                try:
                    res = await client.classify_one(
                        http=http,
                        model=cfg.model,
                        prompt=cfg.prompt,
                        image_path=img_path,
                        max_tokens=cfg.max_tokens,
                        temperature=cfg.temperature,
                        schema=cfg.schema,
                        stream=cfg.stream,
                        stream_include_usage=cfg.stream_include_usage,
                    )
                    if text_mode:
                        content = res.content.strip()
                        if not content:
                            raise ValueError("empty content response")
                        metrics.processed += 1
                        metrics.ok += 1
                        metrics.lat.add(res.latency_ms)
                        if res.ttft_ms is not None:
                            metrics.ttft.add(res.ttft_ms)
                            gen_ms = res.latency_ms - res.ttft_ms
                            if gen_ms > 0:
                                metrics.generation_s += gen_ms / 1000.0
                        if res.usage and res.usage.completion_tokens:
                            metrics.completion_tokens += res.usage.completion_tokens
                        return ("text", file_pk, content, res.latency_ms, res.finish_reason)

                    data = orjson.loads(res.content.encode("utf-8"))
                    labels = data.get("labels")
                    if not isinstance(labels, list) or any(not isinstance(x, int) for x in labels):
                        raise ValueError(f"invalid labels payload: {data!r}")
                    labels = sorted(set(int(x) for x in labels))

                    def _get_str(d: dict[str, Any], key: str, *, required: bool) -> str | None:
                        if key not in d or d[key] is None:
                            if required:
                                raise ValueError(f"missing '{key}' in response: {d!r}")
                            return None
                        val = d[key]
                        if not isinstance(val, str):
                            raise ValueError(f"invalid '{key}' type: {type(val).__name__}")
                        return val

                    def _get_str_list(
                        d: dict[str, Any], key: str, *, required: bool
                    ) -> list[str] | None:
                        if key not in d or d[key] is None:
                            if required:
                                raise ValueError(f"missing '{key}' in response: {d!r}")
                            return None
                        val = d[key]
                        if not isinstance(val, list) or any(not isinstance(x, str) for x in val):
                            raise ValueError(f"invalid '{key}' list: {d!r}")
                        return [str(x) for x in val]

                    text = _get_str(data, "text", required=cfg.include_text)
                    subject = _get_str(data, "subject", required=cfg.include_subject)
                    entities = _get_str_list(data, "entities", required=cfg.include_entities)
                    summary = _get_str(data, "summary", required=cfg.include_summary)
                    if uncertain_ids:
                        matched = sorted(set(labels) & uncertain_ids)
                        if matched:
                            raise ValueError(f"uncertain_label_ids={matched}")
                    metrics.processed += 1
                    metrics.ok += 1
                    metrics.lat.add(res.latency_ms)
                    if res.ttft_ms is not None:
                        metrics.ttft.add(res.ttft_ms)
                        gen_ms = res.latency_ms - res.ttft_ms
                        if gen_ms > 0:
                            metrics.generation_s += gen_ms / 1000.0
                    if res.usage and res.usage.completion_tokens:
                        metrics.completion_tokens += res.usage.completion_tokens
                    return (
                        "ok",
                        file_pk,
                        labels,
                        text,
                        subject,
                        entities,
                        summary,
                        res.content,
                        res.latency_ms,
                        res.finish_reason,
                    )
                except Exception as e:
                    metrics.processed += 1
                    metrics.err += 1
                    return ("err", file_pk, str(e), attempt)

        dashboard = MetricsDashboard(console) if cfg.use_dashboard else None

        while True:
            t_fetch_start = time.perf_counter()
            pending = label_store.claim_pending_batch(conn, run_id=cfg.run_id, limit=batch_limit)
            t_fetch_ms = (time.perf_counter() - t_fetch_start) * 1000.0
            metrics.db_fetch_time_ms.add(t_fetch_ms)

            if not pending:
                break

            metrics.add_batch_fill(len(pending), batch_limit)

            tasks = [asyncio.create_task(one(fpk, p, attempt)) for (fpk, p, attempt) in pending]
            results = await asyncio.gather(*tasks)

            oks = []
            text_oks = []
            errs = []
            for r in results:
                if r[0] == "ok":
                    _, fpk, labels, text, subject, entities, summary, raw, lat, fr = r
                    oks.append((fpk, labels, text, subject, entities, summary, raw, lat, fr))
                elif r[0] == "text":
                    _, fpk, content, lat, fr = r
                    text_oks.append((fpk, content, lat, fr))
                else:
                    _, fpk, err, attempt = r
                    errs.append((fpk, err, attempt))

            t_write_start = time.perf_counter()
            if text_oks:
                label_store.mark_done_text(
                    conn,
                    run_id=cfg.run_id,
                    results=[(fpk, lat, fr) for (fpk, _content, lat, fr) in text_oks],
                )
            if oks:
                label_store.mark_done(conn, run_id=cfg.run_id, results=oks)
            if errs:
                retries = []
                final_errs = []
                now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
                for fpk, err, attempt in errs:
                    if attempt < cfg.max_attempts:
                        backoff = cfg.retry_backoff_s * (cfg.retry_backoff_mult ** (attempt - 1))
                        backoff = min(backoff, cfg.retry_backoff_cap_s)
                        retry_at = now + dt.timedelta(seconds=backoff)
                        retries.append((fpk, err, retry_at))
                    else:
                        final_errs.append((fpk, err))
                label_store.mark_retry(conn, run_id=cfg.run_id, retries=retries)
                label_store.mark_error(conn, run_id=cfg.run_id, errors=final_errs)
            t_write_ms = (time.perf_counter() - t_write_start) * 1000.0
            metrics.db_write_time_ms.add(t_write_ms)

            tuner.note_batch()

            now_perf = time.perf_counter()
            if now_perf - last_report > cfg.metrics_interval_s:
                summ = label_store.summarize(conn, run_id=cfg.run_id)
                metrics.add_queue_depth(summ.get("pending", 0))
                metrics.sample_throughput()

                if dashboard:
                    tuner_state = tuner.get_state() if cfg.auto_tune else None
                    console.clear()
                    console.print(
                        dashboard.render(
                            metrics=metrics,
                            summ=summ,
                            run_id=cfg.run_id,
                            model=cfg.model,
                            inflight=inflight_limit,
                            batch_size=batch_limit,
                            tuner_state=tuner_state,
                        )
                    )
                else:
                    console.clear()
                    console.print(
                        metrics.table(
                            inflight=inflight_limit,
                            batch_size=batch_limit,
                            pending=summ.get("pending", 0),
                            done=summ.get("done", 0),
                            error=summ.get("error", 0),
                        )
                    )
                last_report = now_perf

                if cfg.auto_tune and len(metrics.lat.samples_ms) >= 20:
                    p95 = metrics.lat.percentile(95)
                    ttft_p95 = metrics.ttft.percentile(95)
                    ttft_p95_ms = None if math.isnan(ttft_p95) else ttft_p95
                    tok_s = metrics.tokens_per_s()
                    err_rate = metrics.error_rate()
                    new_inflight, new_batch = tuner.suggest(
                        current_inflight=inflight_limit,
                        current_batch=batch_limit,
                        p95_ms=p95,
                        ttft_p95_ms=ttft_p95_ms,
                        tok_s=tok_s,
                        err_rate=err_rate,
                    )
                    if new_inflight != inflight_limit:
                        inflight_limit = new_inflight
                        sem = asyncio.Semaphore(inflight_limit)
                    batch_limit = new_batch
                elif cfg.dynamic_concurrency and len(metrics.lat.samples_ms) >= 20:
                    p95 = metrics.lat.percentile(95)
                    new_limit = dyn.suggest(inflight_limit, p95)
                    if new_limit != inflight_limit:
                        inflight_limit = new_limit
                        sem = asyncio.Semaphore(inflight_limit)
                        batch_limit = max(batch_limit, inflight_limit)

    # Final summary
    summ = label_store.summarize(conn, run_id=cfg.run_id)
    if dashboard:
        tuner_state = tuner.get_state() if cfg.auto_tune else None
        console.print(
            dashboard.render(
                metrics=metrics,
                summ=summ,
                run_id=cfg.run_id,
                model=cfg.model,
                inflight=inflight_limit,
                batch_size=batch_limit,
                tuner_state=tuner_state,
            )
        )
    else:
        console.print(
            metrics.table(
                inflight=inflight_limit,
                batch_size=batch_limit,
                pending=summ.get("pending", 0),
                done=summ.get("done", 0),
                error=summ.get("error", 0),
            )
        )
