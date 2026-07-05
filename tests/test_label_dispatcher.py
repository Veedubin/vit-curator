"""Tests for vit_curator.label.dispatcher — dispatch loop and config.

All network calls are mocked; no real HTTP requests are made.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

# ---------------------------------------------------------------------------
# DispatchConfig
# ---------------------------------------------------------------------------


def test_dispatch_config_defaults() -> None:
    """DispatchConfig should accept all fields and provide defaults."""
    from vit_curator.label.dispatcher import DispatchConfig

    cfg = DispatchConfig(
        run_id="test-run",
        server_url="http://localhost:8000",
        model="test-model",
        prompt="classify this",
        schema=None,
        include_text=False,
        include_subject=False,
        include_entities=False,
        include_summary=False,
        text_output_dir=None,
        output_root=None,
        output_ext=".json",
        max_inflight=4,
        batch_size=8,
        max_tokens=64,
        temperature=0.0,
        timeout_s=30.0,
        stream=False,
        stream_include_usage=False,
        dynamic_concurrency=True,
        min_inflight=1,
        max_inflight_cap=16,
        ema_halflife_s=30.0,
        auto_tune=False,
        min_batch_size=1,
        max_batch_size=64,
        batch_step=16,
        target_p95_ms=2500.0,
        target_ttft_ms=None,
        min_tok_s=None,
        max_err_rate=5.0,
        warmup_batches=2,
        tune_interval_s=2.0,
        max_attempts=3,
        retry_backoff_s=1.0,
        retry_backoff_mult=2.0,
        retry_backoff_cap_s=60.0,
        uncertain_label_ids=(),
    )

    assert cfg.run_id == "test-run"
    assert cfg.server_url == "http://localhost:8000"
    assert cfg.model == "test-model"
    assert cfg.max_inflight == 4
    assert cfg.batch_size == 8
    assert cfg.max_attempts == 3
    assert cfg.use_dashboard is True
    assert cfg.metrics_interval_s == 2.0


def test_dispatch_config_text_mode() -> None:
    """DispatchConfig with text_output_dir should indicate text mode."""
    from vit_curator.label.dispatcher import DispatchConfig

    cfg = DispatchConfig(
        run_id="r",
        server_url="http://localhost:8000",
        model="m",
        prompt="p",
        schema=None,
        include_text=False,
        include_subject=False,
        include_entities=False,
        include_summary=False,
        text_output_dir=Path("/tmp/out"),
        output_root=None,
        output_ext=".txt",
        max_inflight=4,
        batch_size=8,
        max_tokens=64,
        temperature=0.0,
        timeout_s=30.0,
        stream=False,
        stream_include_usage=False,
        dynamic_concurrency=True,
        min_inflight=1,
        max_inflight_cap=16,
        ema_halflife_s=30.0,
        auto_tune=False,
        min_batch_size=1,
        max_batch_size=64,
        batch_step=16,
        target_p95_ms=2500.0,
        target_ttft_ms=None,
        min_tok_s=None,
        max_err_rate=5.0,
        warmup_batches=2,
        tune_interval_s=2.0,
        max_attempts=3,
        retry_backoff_s=1.0,
        retry_backoff_mult=2.0,
        retry_backoff_cap_s=60.0,
        uncertain_label_ids=(),
    )

    assert cfg.text_output_dir == Path("/tmp/out")


def test_dispatch_config_uncertain_ids() -> None:
    """DispatchConfig should store uncertain_label_ids as a tuple."""
    from vit_curator.label.dispatcher import DispatchConfig

    cfg = DispatchConfig(
        run_id="r",
        server_url="http://localhost:8000",
        model="m",
        prompt="p",
        schema=None,
        include_text=False,
        include_subject=False,
        include_entities=False,
        include_summary=False,
        text_output_dir=None,
        output_root=None,
        output_ext=".json",
        max_inflight=4,
        batch_size=8,
        max_tokens=64,
        temperature=0.0,
        timeout_s=30.0,
        stream=False,
        stream_include_usage=False,
        dynamic_concurrency=True,
        min_inflight=1,
        max_inflight_cap=16,
        ema_halflife_s=30.0,
        auto_tune=False,
        min_batch_size=1,
        max_batch_size=64,
        batch_step=16,
        target_p95_ms=2500.0,
        target_ttft_ms=None,
        min_tok_s=None,
        max_err_rate=5.0,
        warmup_batches=2,
        tune_interval_s=2.0,
        max_attempts=3,
        retry_backoff_s=1.0,
        retry_backoff_mult=2.0,
        retry_backoff_cap_s=60.0,
        uncertain_label_ids=(1, 2, 3),
    )

    assert cfg.uncertain_label_ids == (1, 2, 3)


# ---------------------------------------------------------------------------
# Helper: create a minimal DispatchConfig for testing
# ---------------------------------------------------------------------------


def _make_cfg(
    run_id: str = "test-run",
    **overrides: object,
) -> object:
    from vit_curator.label.dispatcher import DispatchConfig

    defaults: dict[str, object] = dict(
        run_id=run_id,
        server_url="http://localhost:8000",
        model="m",
        prompt="p",
        schema=None,
        include_text=False,
        include_subject=False,
        include_entities=False,
        include_summary=False,
        text_output_dir=None,
        output_root=None,
        output_ext=".json",
        max_inflight=4,
        batch_size=8,
        max_tokens=64,
        temperature=0.0,
        timeout_s=30.0,
        stream=False,
        stream_include_usage=False,
        dynamic_concurrency=False,
        min_inflight=1,
        max_inflight_cap=16,
        ema_halflife_s=30.0,
        auto_tune=False,
        min_batch_size=1,
        max_batch_size=64,
        batch_step=16,
        target_p95_ms=2500.0,
        target_ttft_ms=None,
        min_tok_s=None,
        max_err_rate=5.0,
        warmup_batches=2,
        tune_interval_s=2.0,
        max_attempts=3,
        retry_backoff_s=1.0,
        retry_backoff_mult=2.0,
        retry_backoff_cap_s=60.0,
        uncertain_label_ids=(),
    )
    defaults.update(overrides)
    return DispatchConfig(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# run_dispatch_loop — empty task queue
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_dispatch_loop_empty_queue() -> None:
    """run_dispatch_loop should exit gracefully when there are no pending tasks."""
    from vit_curator.label.dispatcher import run_dispatch_loop

    cfg = _make_cfg(run_id="empty-run")

    mock_conn = MagicMock()
    mock_claim = MagicMock(return_value=[])
    mock_summarize = MagicMock(return_value={"pending": 0, "done": 0, "error": 0})
    with (
        patch("vit_curator.label.dispatcher.label_store.claim_pending_batch", mock_claim),
        patch("vit_curator.label.dispatcher.label_store.summarize", mock_summarize),
        patch("vit_curator.label.dispatcher.httpx.AsyncClient") as mock_client_cls,
    ):
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        console = Console()
        await run_dispatch_loop(conn=mock_conn, cfg=cfg, console=console)

    mock_claim.assert_called_once_with(mock_conn, run_id="empty-run", limit=8)


# ---------------------------------------------------------------------------
# run_dispatch_loop — with results (ok / err / retry)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_dispatch_loop_with_ok_results() -> None:
    """run_dispatch_loop should process ok results and call mark_done."""
    from vit_curator.label.dispatcher import run_dispatch_loop
    from vit_curator.label.schemas import ChatResult, ChatUsage

    cfg = _make_cfg(
        run_id="ok-run",
        schema={
            "type": "object",
            "properties": {"labels": {"type": "array", "items": {"type": "integer"}}},
        },
    )

    mock_conn = MagicMock()

    # First call returns one task, second call returns empty -> loop runs once then exits
    pending_calls = [
        [(1, "/tmp/test.jpg", 0)],  # (file_pk, path, attempt)
        [],
    ]

    mock_claim = MagicMock(side_effect=pending_calls)
    mock_mark_done = MagicMock()
    mock_mark_error = MagicMock()
    mock_mark_retry = MagicMock()
    mock_summarize = MagicMock(return_value={"pending": 0, "done": 1, "error": 0})
    with (
        patch("vit_curator.label.dispatcher.label_store.claim_pending_batch", mock_claim),
        patch("vit_curator.label.dispatcher.label_store.mark_done", mock_mark_done),
        patch("vit_curator.label.dispatcher.label_store.mark_error", mock_mark_error),
        patch("vit_curator.label.dispatcher.label_store.mark_retry", mock_mark_retry),
        patch("vit_curator.label.dispatcher.label_store.summarize", mock_summarize),
        patch("vit_curator.label.dispatcher.httpx.AsyncClient") as mock_client_cls,
        patch("vit_curator.label.dispatcher.VllmClient") as mock_vllm_cls,
    ):
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        mock_vllm = AsyncMock()
        mock_vllm_cls.return_value = mock_vllm
        mock_vllm.classify_one.return_value = ChatResult(
            content='{"labels": [1, 2, 3]}',
            finish_reason="stop",
            latency_ms=150.0,
            ttft_ms=50.0,
            usage=ChatUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        )

        console = Console()
        await run_dispatch_loop(conn=mock_conn, cfg=cfg, console=console)

    # Should have called mark_done with the result
    mock_mark_done.assert_called_once()
    kwargs = mock_mark_done.call_args.kwargs
    assert kwargs["run_id"] == "ok-run"
    assert len(kwargs["results"]) == 1
    fpk, labels, *_ = kwargs["results"][0]
    assert fpk == 1
    assert labels == [1, 2, 3]


@pytest.mark.asyncio
async def test_run_dispatch_loop_with_errors() -> None:
    """run_dispatch_loop should handle errors and retry logic."""
    from vit_curator.label.dispatcher import run_dispatch_loop

    cfg = _make_cfg(run_id="err-run")

    mock_conn = MagicMock()

    pending_calls = [
        [(1, "/tmp/test.jpg", 0)],
        [],
    ]

    mock_claim = MagicMock(side_effect=pending_calls)
    mock_mark_done = MagicMock()
    mock_mark_error = MagicMock()
    mock_mark_retry = MagicMock()
    mock_summarize = MagicMock(return_value={"pending": 0, "done": 0, "error": 1})
    with (
        patch("vit_curator.label.dispatcher.label_store.claim_pending_batch", mock_claim),
        patch("vit_curator.label.dispatcher.label_store.mark_done", mock_mark_done),
        patch("vit_curator.label.dispatcher.label_store.mark_error", mock_mark_error),
        patch("vit_curator.label.dispatcher.label_store.mark_retry", mock_mark_retry),
        patch("vit_curator.label.dispatcher.label_store.summarize", mock_summarize),
        patch("vit_curator.label.dispatcher.httpx.AsyncClient") as mock_client_cls,
        patch("vit_curator.label.dispatcher.VllmClient") as mock_vllm_cls,
    ):
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        mock_vllm = AsyncMock()
        mock_vllm_cls.return_value = mock_vllm
        mock_vllm.classify_one.side_effect = RuntimeError("server unreachable")

        console = Console()
        await run_dispatch_loop(conn=mock_conn, cfg=cfg, console=console)

    # With attempt=0 and max_attempts=3, should retry (not final error)
    mock_mark_retry.assert_called_once()
    # mark_error is also called but with empty list
    mock_mark_error.assert_called_once()
    mock_mark_done.assert_not_called()

    # Verify retry was called with actual retries (not empty)
    _args, kwargs = mock_mark_retry.call_args
    assert len(kwargs["retries"]) == 1


@pytest.mark.asyncio
async def test_run_dispatch_loop_max_attempts_exceeded() -> None:
    """run_dispatch_loop should mark as final error when max_attempts exceeded."""
    from vit_curator.label.dispatcher import run_dispatch_loop

    cfg = _make_cfg(run_id="max-attempts-run", max_attempts=1)

    mock_conn = MagicMock()

    # attempt=1 with max_attempts=1 means this is the last attempt -> final error
    pending_calls = [
        [(1, "/tmp/test.jpg", 1)],
        [],
    ]

    mock_claim = MagicMock(side_effect=pending_calls)
    mock_mark_done = MagicMock()
    mock_mark_error = MagicMock()
    mock_mark_retry = MagicMock()
    mock_summarize = MagicMock(return_value={"pending": 0, "done": 0, "error": 1})
    with (
        patch("vit_curator.label.dispatcher.label_store.claim_pending_batch", mock_claim),
        patch("vit_curator.label.dispatcher.label_store.mark_done", mock_mark_done),
        patch("vit_curator.label.dispatcher.label_store.mark_error", mock_mark_error),
        patch("vit_curator.label.dispatcher.label_store.mark_retry", mock_mark_retry),
        patch("vit_curator.label.dispatcher.label_store.summarize", mock_summarize),
        patch("vit_curator.label.dispatcher.httpx.AsyncClient") as mock_client_cls,
        patch("vit_curator.label.dispatcher.VllmClient") as mock_vllm_cls,
    ):
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        mock_vllm = AsyncMock()
        mock_vllm_cls.return_value = mock_vllm
        mock_vllm.classify_one.side_effect = RuntimeError("server unreachable")

        console = Console()
        await run_dispatch_loop(conn=mock_conn, cfg=cfg, console=console)

    # With attempt=1 and max_attempts=1, should be final error
    mock_mark_error.assert_called_once()
    mock_mark_retry.assert_called_once()  # called but with empty list
    mock_mark_done.assert_not_called()

    # Verify error was called with actual errors (not empty)
    _args, kwargs = mock_mark_error.call_args
    assert len(kwargs["errors"]) == 1

    # Verify retry was called with empty list
    _args, kwargs = mock_mark_retry.call_args
    assert len(kwargs["retries"]) == 0


@pytest.mark.asyncio
async def test_run_dispatch_loop_text_mode() -> None:
    """run_dispatch_loop should handle text-mode results."""
    from vit_curator.label.dispatcher import run_dispatch_loop
    from vit_curator.label.schemas import ChatResult, ChatUsage

    cfg = _make_cfg(
        run_id="text-run",
        text_output_dir=Path("/tmp/text_out"),
        output_ext=".txt",
    )

    mock_conn = MagicMock()

    pending_calls = [
        [(1, "/tmp/test.jpg", 0)],
        [],
    ]

    with (
        patch("vit_curator.label.store.claim_pending_batch", side_effect=pending_calls),
        patch("vit_curator.label.store.mark_done_text") as mock_mark_done_text,
        patch("vit_curator.label.store.mark_done") as mock_mark_done,
        patch("vit_curator.label.store.mark_error"),
        patch("vit_curator.label.store.mark_retry"),
        patch(
            "vit_curator.label.store.summarize", return_value={"pending": 0, "done": 1, "error": 0}
        ),
        patch("vit_curator.label.dispatcher.httpx.AsyncClient") as mock_client_cls,
        patch("vit_curator.label.dispatcher.VllmClient") as mock_vllm_cls,
    ):
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        mock_vllm = AsyncMock()
        mock_vllm_cls.return_value = mock_vllm
        mock_vllm.classify_one.return_value = ChatResult(
            content="  This is a description of the image.  ",
            finish_reason="stop",
            latency_ms=100.0,
            ttft_ms=30.0,
            usage=ChatUsage(prompt_tokens=50, completion_tokens=10, total_tokens=60),
        )

        console = Console()
        await run_dispatch_loop(conn=mock_conn, cfg=cfg, console=console)

    # Should have called mark_done_text (not mark_done)
    mock_mark_done_text.assert_called_once()
    mock_mark_done.assert_not_called()


@pytest.mark.asyncio
async def test_run_dispatch_loop_empty_content_raises() -> None:
    """run_dispatch_loop should handle empty content in text mode as error."""
    from vit_curator.label.dispatcher import run_dispatch_loop
    from vit_curator.label.schemas import ChatResult, ChatUsage

    cfg = _make_cfg(
        run_id="empty-text-run",
        text_output_dir=Path("/tmp/text_out"),
        output_ext=".txt",
    )

    mock_conn = MagicMock()

    pending_calls = [
        [(1, "/tmp/test.jpg", 0)],
        [],
    ]

    with (
        patch("vit_curator.label.store.claim_pending_batch", side_effect=pending_calls),
        patch("vit_curator.label.store.mark_retry") as mock_mark_retry,
        patch("vit_curator.label.store.mark_error"),
        patch(
            "vit_curator.label.store.summarize", return_value={"pending": 0, "done": 0, "error": 1}
        ),
        patch("vit_curator.label.dispatcher.httpx.AsyncClient") as mock_client_cls,
        patch("vit_curator.label.dispatcher.VllmClient") as mock_vllm_cls,
    ):
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        mock_vllm = AsyncMock()
        mock_vllm_cls.return_value = mock_vllm
        mock_vllm.classify_one.return_value = ChatResult(
            content="   ",
            finish_reason="stop",
            latency_ms=50.0,
            ttft_ms=10.0,
            usage=ChatUsage(prompt_tokens=50, completion_tokens=0, total_tokens=50),
        )

        console = Console()
        await run_dispatch_loop(conn=mock_conn, cfg=cfg, console=console)

    # Should have retried (not final error)
    mock_mark_retry.assert_called_once()
    _args, kwargs = mock_mark_retry.call_args
    assert len(kwargs["retries"]) == 1


@pytest.mark.asyncio
async def test_run_dispatch_loop_invalid_labels() -> None:
    """run_dispatch_loop should handle invalid labels payload as error."""
    from vit_curator.label.dispatcher import run_dispatch_loop
    from vit_curator.label.schemas import ChatResult, ChatUsage

    cfg = _make_cfg(
        run_id="bad-labels-run",
        schema={
            "type": "object",
            "properties": {"labels": {"type": "array", "items": {"type": "integer"}}},
        },
    )

    mock_conn = MagicMock()

    pending_calls = [
        [(1, "/tmp/test.jpg", 0)],
        [],
    ]

    with (
        patch("vit_curator.label.store.claim_pending_batch", side_effect=pending_calls),
        patch("vit_curator.label.store.mark_retry") as mock_mark_retry,
        patch("vit_curator.label.store.mark_error"),
        patch(
            "vit_curator.label.store.summarize", return_value={"pending": 0, "done": 0, "error": 1}
        ),
        patch("vit_curator.label.dispatcher.httpx.AsyncClient") as mock_client_cls,
        patch("vit_curator.label.dispatcher.VllmClient") as mock_vllm_cls,
    ):
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        mock_vllm = AsyncMock()
        mock_vllm_cls.return_value = mock_vllm
        mock_vllm.classify_one.return_value = ChatResult(
            content='{"labels": ["a", "b", "c"]}',
            finish_reason="stop",
            latency_ms=100.0,
            ttft_ms=30.0,
            usage=ChatUsage(prompt_tokens=50, completion_tokens=10, total_tokens=60),
        )

        console = Console()
        await run_dispatch_loop(conn=mock_conn, cfg=cfg, console=console)

    # Should have retried
    mock_mark_retry.assert_called_once()
    _args, kwargs = mock_mark_retry.call_args
    assert len(kwargs["retries"]) == 1


@pytest.mark.asyncio
async def test_run_dispatch_loop_uncertain_labels() -> None:
    """run_dispatch_loop should error when uncertain labels are matched."""
    from vit_curator.label.dispatcher import run_dispatch_loop
    from vit_curator.label.schemas import ChatResult, ChatUsage

    cfg = _make_cfg(
        run_id="uncertain-run",
        schema={
            "type": "object",
            "properties": {"labels": {"type": "array", "items": {"type": "integer"}}},
        },
        uncertain_label_ids=(99,),
    )

    mock_conn = MagicMock()

    pending_calls = [
        [(1, "/tmp/test.jpg", 0)],
        [],
    ]

    with (
        patch("vit_curator.label.store.claim_pending_batch", side_effect=pending_calls),
        patch("vit_curator.label.store.mark_retry") as mock_mark_retry,
        patch("vit_curator.label.store.mark_error"),
        patch(
            "vit_curator.label.store.summarize", return_value={"pending": 0, "done": 0, "error": 1}
        ),
        patch("vit_curator.label.dispatcher.httpx.AsyncClient") as mock_client_cls,
        patch("vit_curator.label.dispatcher.VllmClient") as mock_vllm_cls,
    ):
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        mock_vllm = AsyncMock()
        mock_vllm_cls.return_value = mock_vllm
        mock_vllm.classify_one.return_value = ChatResult(
            content='{"labels": [1, 99]}',
            finish_reason="stop",
            latency_ms=100.0,
            ttft_ms=30.0,
            usage=ChatUsage(prompt_tokens=50, completion_tokens=10, total_tokens=60),
        )

        console = Console()
        await run_dispatch_loop(conn=mock_conn, cfg=cfg, console=console)

    # Should have retried due to uncertain label match
    mock_mark_retry.assert_called_once()
    _args, kwargs = mock_mark_retry.call_args
    assert len(kwargs["retries"]) == 1
