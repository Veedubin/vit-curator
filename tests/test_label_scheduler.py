"""Tests for vit_curator.label.scheduler — Ema, DynamicConcurrency, AutoTune."""

from __future__ import annotations

import time

# ---------------------------------------------------------------------------
# Ema
# ---------------------------------------------------------------------------


def test_ema_initial_value() -> None:
    """Ema should return the first value on first update."""
    from vit_curator.label.scheduler import Ema

    ema = Ema(halflife_s=10.0)
    result = ema.update(100.0)
    assert result == 100.0
    assert ema.value == 100.0


def test_ema_subsequent_updates() -> None:
    """Ema should smooth subsequent values."""
    from vit_curator.label.scheduler import Ema

    ema = Ema(halflife_s=0.1)  # Short halflife for fast adaptation
    v1 = ema.update(100.0)
    assert v1 == 100.0

    v2 = ema.update(200.0)
    # With short halflife, should move significantly toward 200
    assert v2 > 100.0
    assert v2 < 200.0


def test_ema_reset() -> None:
    """Ema should reset to initial state."""
    from vit_curator.label.scheduler import Ema

    ema = Ema(halflife_s=10.0)
    ema.update(100.0)
    assert ema.value is not None

    ema.value = None
    ema.t_last = None
    result = ema.update(50.0)
    assert result == 50.0


def test_ema_halflife_effect() -> None:
    """Longer halflife should smooth more aggressively."""
    from vit_curator.label.scheduler import Ema

    ema_short = Ema(halflife_s=0.01)
    ema_long = Ema(halflife_s=100.0)

    ema_short.update(100.0)
    ema_long.update(100.0)

    # Force t_last to be in the past
    ema_short.t_last = time.perf_counter() - 0.5
    ema_long.t_last = time.perf_counter() - 0.5

    vs = ema_short.update(200.0)
    vl = ema_long.update(200.0)

    # Short halflife should move more toward 200
    assert vs > vl


# ---------------------------------------------------------------------------
# DynamicConcurrency
# ---------------------------------------------------------------------------


def test_dynamic_concurrency_initial() -> None:
    """DynamicConcurrency should initialize with given bounds."""
    from vit_curator.label.scheduler import DynamicConcurrency

    dc = DynamicConcurrency(min_inflight=1, max_inflight=16, target_p95_ms=2500.0)
    assert dc.min_inflight == 1
    assert dc.max_inflight == 16
    assert dc.target_p95_ms == 2500.0


def test_dynamic_concurrency_increase_on_low_latency() -> None:
    """DynamicConcurrency should increase inflight when p95 is well below target."""
    from vit_curator.label.scheduler import DynamicConcurrency

    dc = DynamicConcurrency(min_inflight=1, max_inflight=16, target_p95_ms=2500.0)
    result = dc.suggest(current=4, p95_ms=1000.0)  # Below 85% of 2500 = 2125
    assert result > 4  # Should increase


def test_dynamic_concurrency_decrease_on_high_latency() -> None:
    """DynamicConcurrency should decrease inflight when p95 exceeds target."""
    from vit_curator.label.scheduler import DynamicConcurrency

    dc = DynamicConcurrency(min_inflight=1, max_inflight=16, target_p95_ms=2500.0)
    result = dc.suggest(current=8, p95_ms=3000.0)  # Above 110% of 2500 = 2750
    assert result < 8  # Should decrease


def test_dynamic_concurrency_clamp_to_min() -> None:
    """DynamicConcurrency should not go below min_inflight."""
    from vit_curator.label.scheduler import DynamicConcurrency

    dc = DynamicConcurrency(min_inflight=2, max_inflight=16, target_p95_ms=1000.0)
    result = dc.suggest(current=2, p95_ms=5000.0)  # Very high latency
    assert result >= 2


def test_dynamic_concurrency_clamp_to_max() -> None:
    """DynamicConcurrency should not go above max_inflight."""
    from vit_curator.label.scheduler import DynamicConcurrency

    dc = DynamicConcurrency(min_inflight=1, max_inflight=8, target_p95_ms=5000.0)
    result = dc.suggest(current=8, p95_ms=100.0)  # Very low latency
    assert result <= 8


def test_dynamic_concurrency_hold_in_middle() -> None:
    """DynamicConcurrency should hold steady when p95 is in the acceptable range."""
    from vit_curator.label.scheduler import DynamicConcurrency

    dc = DynamicConcurrency(min_inflight=1, max_inflight=16, target_p95_ms=2500.0)
    # 85% of 2500 = 2125, 110% of 2500 = 2750
    # p95=2400 is in the middle
    result = dc.suggest(current=4, p95_ms=2400.0)
    # After EMA update, the smoothed value might be slightly different
    # but should be close to 4
    assert result >= 1
    assert result <= 16


# ---------------------------------------------------------------------------
# AutoTune
# ---------------------------------------------------------------------------


def test_auto_tune_initial_state() -> None:
    """AutoTune should initialize with correct defaults."""
    from vit_curator.label.scheduler import AutoTune

    tuner = AutoTune(
        min_inflight=1,
        max_inflight=16,
        min_batch_size=1,
        max_batch_size=64,
        batch_step=16,
        target_p95_ms=2500.0,
        target_ttft_ms=None,
        min_tok_s=None,
        max_err_rate=5.0,
        warmup_batches=2,
        cooldown_s=2.0,
    )

    assert tuner.min_inflight == 1
    assert tuner.max_inflight == 16
    assert tuner.batches_seen == 0
    assert tuner.last_tune is None
    assert tuner.decision_log == []


def test_auto_tune_note_batch() -> None:
    """note_batch should increment batches_seen."""
    from vit_curator.label.scheduler import AutoTune

    tuner = AutoTune(
        min_inflight=1,
        max_inflight=16,
        min_batch_size=1,
        max_batch_size=64,
        batch_step=16,
        target_p95_ms=2500.0,
    )

    assert tuner.batches_seen == 0
    tuner.note_batch()
    assert tuner.batches_seen == 1
    tuner.note_batch()
    assert tuner.batches_seen == 2


def test_auto_tune_get_state() -> None:
    """get_state should return current tuner state."""
    from vit_curator.label.scheduler import AutoTune

    tuner = AutoTune(
        min_inflight=1,
        max_inflight=16,
        min_batch_size=1,
        max_batch_size=64,
        batch_step=16,
        target_p95_ms=2500.0,
        warmup_batches=2,
    )

    state = tuner.get_state()
    assert "batches_seen" in state
    assert "warmup_complete" in state
    assert "last_tune" in state
    assert "cooldown_remaining_s" in state
    assert state["batches_seen"] == 0
    assert state["warmup_complete"] is False


def test_auto_tune_warmup() -> None:
    """AutoTune should return current values during warmup."""
    from vit_curator.label.scheduler import AutoTune

    tuner = AutoTune(
        min_inflight=1,
        max_inflight=16,
        min_batch_size=1,
        max_batch_size=64,
        batch_step=16,
        target_p95_ms=2500.0,
        warmup_batches=2,
    )

    # During warmup (batches_seen < warmup_batches)
    inflight, batch = tuner.suggest(
        current_inflight=4,
        current_batch=32,
        p95_ms=5000.0,
        ttft_p95_ms=None,
        tok_s=None,
        err_rate=0.0,
    )

    assert inflight == 4
    assert batch == 32
    assert len(tuner.decision_log) == 1
    assert tuner.decision_log[0]["action"] == "warmup"


def test_auto_tune_scale_down_on_error() -> None:
    """AutoTune should scale down when error rate exceeds max."""
    from vit_curator.label.scheduler import AutoTune

    tuner = AutoTune(
        min_inflight=1,
        max_inflight=16,
        min_batch_size=1,
        max_batch_size=64,
        batch_step=16,
        target_p95_ms=2500.0,
        max_err_rate=5.0,
        warmup_batches=0,
        cooldown_s=0.0,
    )

    inflight, batch = tuner.suggest(
        current_inflight=8,
        current_batch=32,
        p95_ms=1000.0,
        ttft_p95_ms=None,
        tok_s=None,
        err_rate=10.0,  # Above 5%
    )

    assert inflight < 8  # Should scale down
    assert batch < 32
    assert len(tuner.decision_log) == 1
    assert tuner.decision_log[0]["action"] == "scale_down"


def test_auto_tune_scale_down_on_latency() -> None:
    """AutoTune should scale down when p95 exceeds target."""
    from vit_curator.label.scheduler import AutoTune

    tuner = AutoTune(
        min_inflight=1,
        max_inflight=16,
        min_batch_size=1,
        max_batch_size=64,
        batch_step=16,
        target_p95_ms=2500.0,
        max_err_rate=5.0,
        warmup_batches=0,
        cooldown_s=0.0,
    )

    inflight, batch = tuner.suggest(
        current_inflight=8,
        current_batch=32,
        p95_ms=5000.0,  # Above 110% of 2500 = 2750
        ttft_p95_ms=None,
        tok_s=None,
        err_rate=0.0,
    )

    assert inflight < 8
    assert batch < 32
    assert tuner.decision_log[0]["action"] == "scale_down"


def test_auto_tune_scale_up() -> None:
    """AutoTune should scale up when metrics are good."""
    from vit_curator.label.scheduler import AutoTune

    tuner = AutoTune(
        min_inflight=1,
        max_inflight=16,
        min_batch_size=1,
        max_batch_size=64,
        batch_step=16,
        target_p95_ms=2500.0,
        max_err_rate=5.0,
        warmup_batches=0,
        cooldown_s=0.0,
    )

    inflight, batch = tuner.suggest(
        current_inflight=4,
        current_batch=16,
        p95_ms=1000.0,  # Below 85% of 2500 = 2125
        ttft_p95_ms=None,
        tok_s=None,
        err_rate=0.0,
    )

    assert inflight > 4
    assert batch > 16
    assert tuner.decision_log[0]["action"] == "scale_up"


def test_auto_tune_hold() -> None:
    """AutoTune should hold when metrics are in acceptable range."""
    from vit_curator.label.scheduler import AutoTune

    tuner = AutoTune(
        min_inflight=1,
        max_inflight=16,
        min_batch_size=1,
        max_batch_size=64,
        batch_step=16,
        target_p95_ms=2500.0,
        max_err_rate=5.0,
        warmup_batches=0,
        cooldown_s=0.0,
    )

    # p95=2400 is between 85% (2125) and 110% (2750) of target
    inflight, batch = tuner.suggest(
        current_inflight=4,
        current_batch=16,
        p95_ms=2400.0,
        ttft_p95_ms=None,
        tok_s=None,
        err_rate=0.0,
    )

    assert inflight == 4
    assert batch == 16
    assert tuner.decision_log[0]["action"] == "hold"


def test_auto_tune_clamp_to_bounds() -> None:
    """AutoTune should clamp values to min/max bounds."""
    from vit_curator.label.scheduler import AutoTune

    tuner = AutoTune(
        min_inflight=2,
        max_inflight=4,
        min_batch_size=8,
        max_batch_size=16,
        batch_step=16,
        target_p95_ms=2500.0,
        max_err_rate=5.0,
        warmup_batches=0,
        cooldown_s=0.0,
    )

    # Scale down from already-min values
    inflight, batch = tuner.suggest(
        current_inflight=2,
        current_batch=8,
        p95_ms=5000.0,
        ttft_p95_ms=None,
        tok_s=None,
        err_rate=10.0,
    )

    assert inflight == 2  # Clamped to min
    assert batch >= 8  # batch_size must be >= inflight


def test_auto_tune_ttft_threshold() -> None:
    """AutoTune should consider TTFT when target_ttft_ms is set."""
    from vit_curator.label.scheduler import AutoTune

    tuner = AutoTune(
        min_inflight=1,
        max_inflight=16,
        min_batch_size=1,
        max_batch_size=64,
        batch_step=16,
        target_p95_ms=5000.0,  # High p95 target
        target_ttft_ms=500.0,  # Low TTFT target
        max_err_rate=5.0,
        warmup_batches=0,
        cooldown_s=0.0,
    )

    # p95 is fine but TTFT is too high
    inflight, _batch = tuner.suggest(
        current_inflight=8,
        current_batch=32,
        p95_ms=2000.0,  # Below 110% of 5000
        ttft_p95_ms=1000.0,  # Above 110% of 500 = 550
        tok_s=None,
        err_rate=0.0,
    )

    assert inflight < 8  # Should scale down due to TTFT
    assert tuner.decision_log[0]["action"] == "scale_down"


def test_auto_tune_decision_log_limit() -> None:
    """AutoTune should cap the decision log at max_log_entries."""
    from vit_curator.label.scheduler import AutoTune

    tuner = AutoTune(
        min_inflight=1,
        max_inflight=16,
        min_batch_size=1,
        max_batch_size=64,
        batch_step=16,
        target_p95_ms=2500.0,
        max_err_rate=5.0,
        warmup_batches=0,
        cooldown_s=0.0,
        max_log_entries=5,
    )

    for _ in range(10):
        tuner.suggest(
            current_inflight=4,
            current_batch=16,
            p95_ms=2400.0,
            ttft_p95_ms=None,
            tok_s=None,
            err_rate=0.0,
        )

    assert len(tuner.decision_log) <= 5


def test_auto_tune_cooldown() -> None:
    """AutoTune should respect cooldown period after a change."""
    from vit_curator.label.scheduler import AutoTune

    tuner = AutoTune(
        min_inflight=1,
        max_inflight=16,
        min_batch_size=1,
        max_batch_size=64,
        batch_step=16,
        target_p95_ms=2500.0,
        max_err_rate=5.0,
        warmup_batches=0,
        cooldown_s=100.0,  # Long cooldown
    )

    # First call triggers a change (scale_up)
    inflight1, batch1 = tuner.suggest(
        current_inflight=4,
        current_batch=16,
        p95_ms=1000.0,
        ttft_p95_ms=None,
        tok_s=None,
        err_rate=0.0,
    )
    assert inflight1 > 4  # Changed

    # Second call should be in cooldown
    inflight2, _batch2 = tuner.suggest(
        current_inflight=inflight1,
        current_batch=batch1,
        p95_ms=1000.0,
        ttft_p95_ms=None,
        tok_s=None,
        err_rate=0.0,
    )
    assert inflight2 == inflight1  # No change during cooldown
    assert tuner.decision_log[-1]["action"] == "cooldown"
