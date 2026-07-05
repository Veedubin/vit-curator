"""Tests for vit_curator.label.metrics — LatencyStats, RunMetrics, utilities."""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# LatencyStats
# ---------------------------------------------------------------------------


def test_latency_stats_empty() -> None:
    """LatencyStats should return NaN for percentiles when empty."""
    from vit_curator.label.metrics import LatencyStats

    ls = LatencyStats()
    assert math.isnan(ls.percentile(50))
    assert math.isnan(ls.percentile(95))
    assert math.isnan(ls.percentile(99))
    assert math.isnan(ls.mean())


def test_latency_stats_add() -> None:
    """LatencyStats.add should append samples."""
    from vit_curator.label.metrics import LatencyStats

    ls = LatencyStats()
    ls.add(100.0)
    ls.add(200.0)
    assert len(ls.samples_ms) == 2
    assert ls.samples_ms == [100.0, 200.0]


def test_latency_stats_percentile() -> None:
    """LatencyStats.percentile should compute correct percentiles."""
    from vit_curator.label.metrics import LatencyStats

    ls = LatencyStats()
    for v in range(1, 101):
        ls.add(float(v))

    assert ls.percentile(50) == 50.0
    assert ls.percentile(95) == 95.0
    assert ls.percentile(99) == 99.0
    assert ls.percentile(0) == 1.0
    assert ls.percentile(100) == 100.0


def test_latency_stats_mean() -> None:
    """LatencyStats.mean should compute the average."""
    from vit_curator.label.metrics import LatencyStats

    ls = LatencyStats()
    ls.add(10.0)
    ls.add(20.0)
    ls.add(30.0)
    assert ls.mean() == 20.0


def test_latency_stats_max_samples() -> None:
    """LatencyStats should cap samples at max_samples."""
    from vit_curator.label.metrics import LatencyStats

    ls = LatencyStats(max_samples=5)
    for v in range(10):
        ls.add(float(v))

    assert len(ls.samples_ms) == 5
    assert ls.samples_ms == [5.0, 6.0, 7.0, 8.0, 9.0]


# ---------------------------------------------------------------------------
# RunMetrics
# ---------------------------------------------------------------------------


def test_run_metrics_initial() -> None:
    """RunMetrics should initialize with zeros."""
    from vit_curator.label.metrics import RunMetrics

    m = RunMetrics()
    assert m.processed == 0
    assert m.ok == 0
    assert m.err == 0
    assert m.completion_tokens == 0
    assert m.generation_s == 0.0


def test_run_metrics_throughput() -> None:
    """RunMetrics.throughput should return images per second."""
    from vit_curator.label.metrics import RunMetrics

    m = RunMetrics()
    # No processed items -> 0 / dt
    tp = m.throughput()
    assert tp >= 0.0

    m.processed = 10
    tp = m.throughput()
    assert tp > 0.0


def test_run_metrics_error_rate() -> None:
    """RunMetrics.error_rate should return percentage."""
    from vit_curator.label.metrics import RunMetrics

    m = RunMetrics()
    assert m.error_rate() == 0.0  # No processed items

    m.processed = 100
    m.err = 5
    assert m.error_rate() == 5.0

    m.processed = 0
    m.err = 0
    assert m.error_rate() == 0.0


def test_run_metrics_tokens_per_s() -> None:
    """RunMetrics.tokens_per_s should return None when no tokens."""
    from vit_curator.label.metrics import RunMetrics

    m = RunMetrics()
    assert m.tokens_per_s() is None

    m.completion_tokens = 100
    m.generation_s = 10.0
    assert m.tokens_per_s() == 10.0


def test_run_metrics_add_batch_fill() -> None:
    """RunMetrics.add_batch_fill should record fill percentage."""
    from vit_curator.label.metrics import RunMetrics

    m = RunMetrics()
    m.add_batch_fill(actual=5, limit=10)
    assert len(m.batch_fills) == 1
    assert m.batch_fills[0] == 50.0

    m.add_batch_fill(actual=10, limit=10)
    assert m.batch_fills[1] == 100.0


def test_run_metrics_batch_efficiency() -> None:
    """RunMetrics.batch_efficiency should return average fill."""
    from vit_curator.label.metrics import RunMetrics

    m = RunMetrics()
    assert m.batch_efficiency() == 0.0  # No fills

    m.add_batch_fill(actual=5, limit=10)
    m.add_batch_fill(actual=8, limit=10)
    assert m.batch_efficiency() == 65.0


def test_run_metrics_add_queue_depth() -> None:
    """RunMetrics.add_queue_depth should record queue depth."""
    from vit_curator.label.metrics import RunMetrics

    m = RunMetrics()
    m.add_queue_depth(10)
    m.add_queue_depth(20)
    assert m.queue_depth_samples == [10, 20]


def test_run_metrics_queue_depth_p50() -> None:
    """RunMetrics.queue_depth_p50 should return median."""
    from vit_curator.label.metrics import RunMetrics

    m = RunMetrics()
    assert m.queue_depth_p50() == 0  # No samples

    m.add_queue_depth(10)
    m.add_queue_depth(20)
    m.add_queue_depth(30)
    assert m.queue_depth_p50() == 20


def test_run_metrics_sample_throughput() -> None:
    """RunMetrics.sample_throughput should record throughput history."""
    from vit_curator.label.metrics import RunMetrics

    m = RunMetrics()
    m.sample_throughput()
    assert len(m.throughput_history) == 1


def test_run_metrics_record_batch() -> None:
    """RunMetrics should track processed/ok/err counts."""
    from vit_curator.label.metrics import RunMetrics

    m = RunMetrics()
    m.processed = 50
    m.ok = 45
    m.err = 5
    assert m.processed == 50
    assert m.ok == 45
    assert m.err == 5


# ---------------------------------------------------------------------------
# sparkline
# ---------------------------------------------------------------------------


def test_sparkline_empty() -> None:
    """sparkline should return empty Text for empty input."""
    from vit_curator.label.metrics import sparkline

    result = sparkline([])
    assert str(result) == ""


def test_sparkline_single_value() -> None:
    """sparkline should handle single value."""
    from vit_curator.label.metrics import sparkline

    result = sparkline([100.0])
    assert str(result) != ""


def test_sparkline_renders_chars() -> None:
    """sparkline should render sparkline characters."""
    from vit_curator.label.metrics import sparkline

    result = sparkline([1.0, 2.0, 3.0, 4.0, 5.0], width=5)
    rendered = str(result)
    assert len(rendered) == 5


def test_sparkline_width() -> None:
    """sparkline should respect width parameter."""
    from vit_curator.label.metrics import sparkline

    values = list(range(20))
    result = sparkline(values, width=10)
    assert len(str(result)) == 10

    result = sparkline(values, width=5)
    assert len(str(result)) == 5


# ---------------------------------------------------------------------------
# histogram_buckets
# ---------------------------------------------------------------------------


def test_histogram_buckets_empty() -> None:
    """histogram_buckets should return empty list for empty input."""
    from vit_curator.label.metrics import histogram_buckets

    assert histogram_buckets([]) == []


def test_histogram_buckets_single_value() -> None:
    """histogram_buckets should handle single unique value."""
    from vit_curator.label.metrics import histogram_buckets

    buckets = histogram_buckets([100.0, 100.0, 100.0])
    assert len(buckets) == 1
    assert buckets[0][0] == 100.0
    assert buckets[0][1] == 3


def test_histogram_buckets_distribution() -> None:
    """histogram_buckets should distribute values into bins."""
    from vit_curator.label.metrics import histogram_buckets

    samples = [10.0, 20.0, 30.0, 40.0, 50.0]
    buckets = histogram_buckets(samples, bins=5)
    assert len(buckets) == 5
    total = sum(c for _, c in buckets)
    assert total == 5


def test_histogram_buckets_bin_count() -> None:
    """histogram_buckets should respect bins parameter."""
    from vit_curator.label.metrics import histogram_buckets

    samples = list(range(100))
    buckets = histogram_buckets(samples, bins=10)
    assert len(buckets) == 10

    buckets = histogram_buckets(samples, bins=30)
    assert len(buckets) == 30


# ---------------------------------------------------------------------------
# color_threshold
# ---------------------------------------------------------------------------


def test_color_threshold_green() -> None:
    """color_threshold should return green for low values."""
    from vit_curator.label.metrics import color_threshold

    assert color_threshold(1.0, green=2.0, yellow=5.0) == "green"


def test_color_threshold_yellow() -> None:
    """color_threshold should return yellow for medium values."""
    from vit_curator.label.metrics import color_threshold

    assert color_threshold(3.0, green=2.0, yellow=5.0) == "yellow"


def test_color_threshold_red() -> None:
    """color_threshold should return red for high values."""
    from vit_curator.label.metrics import color_threshold

    assert color_threshold(6.0, green=2.0, yellow=5.0) == "red"


# ---------------------------------------------------------------------------
# network_gauge
# ---------------------------------------------------------------------------


def test_network_gauge_zero_max() -> None:
    """network_gauge should handle zero max connections."""
    from vit_curator.label.metrics import network_gauge

    result = network_gauge(0, 0)
    assert "░" in result


def test_network_gauge_format() -> None:
    """network_gauge should format correctly."""
    from vit_curator.label.metrics import network_gauge

    result = network_gauge(5, 10, width=10)
    assert "█" in result
    assert "░" in result
    assert "5/10" in result


# ---------------------------------------------------------------------------
# print_summary
# ---------------------------------------------------------------------------


def test_print_summary_no_error(capsys) -> None:
    """print_summary should not raise with empty metrics."""
    from rich.console import Console

    from vit_curator.label.metrics import RunMetrics, print_summary

    console = Console()
    metrics = RunMetrics()
    # Should not raise
    print_summary(console, "Test Summary", metrics)


# ---------------------------------------------------------------------------
# MetricsDashboard
# ---------------------------------------------------------------------------


def test_metrics_dashboard_init() -> None:
    """MetricsDashboard should initialize with layout."""
    from rich.console import Console

    from vit_curator.label.metrics import MetricsDashboard

    console = Console()
    dashboard = MetricsDashboard(console)
    assert dashboard.console is console
    assert dashboard.layout is not None


def test_metrics_dashboard_render() -> None:
    """MetricsDashboard.render should return a Layout."""
    from rich.console import Console

    from vit_curator.label.metrics import MetricsDashboard, RunMetrics

    console = Console()
    dashboard = MetricsDashboard(console)
    metrics = RunMetrics()
    metrics.processed = 10
    metrics.ok = 8
    metrics.err = 2

    layout = dashboard.render(
        metrics=metrics,
        summ={"pending": 5, "done": 8, "error": 2, "processing": 0},
        run_id="test-run-123",
        model="test-model",
        inflight=4,
        batch_size=8,
    )

    assert layout is not None


def test_metrics_dashboard_render_with_tuner() -> None:
    """MetricsDashboard.render should accept tuner_state."""
    from rich.console import Console

    from vit_curator.label.metrics import MetricsDashboard, RunMetrics

    console = Console()
    dashboard = MetricsDashboard(console)
    metrics = RunMetrics()

    layout = dashboard.render(
        metrics=metrics,
        summ={"pending": 0, "done": 0, "error": 0, "processing": 0},
        run_id="r",
        model="m",
        inflight=4,
        batch_size=8,
        tuner_state={
            "warmup_complete": True,
            "cooldown_remaining_s": 0.0,
            "batches_seen": 5,
            "last_tune": 100.0,
        },
    )

    assert layout is not None


# ---------------------------------------------------------------------------
# RunMetrics.table
# ---------------------------------------------------------------------------


def test_run_metrics_table() -> None:
    """RunMetrics.table should return a Rich Table."""
    from vit_curator.label.metrics import RunMetrics

    m = RunMetrics()
    m.processed = 100
    m.ok = 95
    m.err = 5
    m.lat.add(100.0)
    m.lat.add(200.0)

    tbl = m.table(inflight=4, pending=10, done=90, error=5, batch_size=8)
    assert tbl is not None
    assert tbl.title == "Labeler Metrics"
