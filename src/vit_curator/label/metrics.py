"""GPU monitoring, latency percentiles, and metric dashboards for VLM dispatch."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]

try:
    import pynvml
except ImportError:
    pynvml = None  # type: ignore[assignment]


def sample_gpu_info() -> dict[str, int | float] | None:
    """Sample GPU stats using nvidia-ml-py."""
    if pynvml is None:
        return None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        return {
            "util": float(util.gpu),
            "mem_used_mib": int(mem.used / 1024 / 1024),
            "mem_total_mib": int(mem.total / 1024 / 1024),
            "temp_c": int(temp),
        }
    except Exception:
        return None


@dataclass
class LatencyStats:
    samples_ms: list[float] = field(default_factory=list)
    max_samples: int = 10000

    def add(self, v_ms: float) -> None:
        self.samples_ms.append(v_ms)
        if len(self.samples_ms) > self.max_samples:
            self.samples_ms = self.samples_ms[-self.max_samples :]

    def percentile(self, p: float) -> float:
        if not self.samples_ms:
            return float("nan")
        xs = sorted(self.samples_ms)
        k = math.ceil((p / 100.0) * len(xs)) - 1
        k = max(0, min(k, len(xs) - 1))
        return xs[k]

    def mean(self) -> float:
        if not self.samples_ms:
            return float("nan")
        return sum(self.samples_ms) / len(self.samples_ms)


@dataclass
class RunMetrics:
    t0: float = field(default_factory=time.perf_counter)
    processed: int = 0
    ok: int = 0
    err: int = 0
    completion_tokens: int = 0
    lat: LatencyStats = field(default_factory=LatencyStats)
    ttft: LatencyStats = field(default_factory=LatencyStats)
    generation_s: float = 0.0
    batch_fills: list[float] = field(default_factory=list)
    queue_depth_samples: list[int] = field(default_factory=list)
    db_fetch_time_ms: LatencyStats = field(default_factory=LatencyStats)
    db_write_time_ms: LatencyStats = field(default_factory=LatencyStats)
    throughput_history: list[float] = field(default_factory=list)
    max_history_window: int = 10000
    gpu_util_history: list[float] = field(default_factory=list)
    gpu_vram_history: list[int] = field(default_factory=list)
    gpu_temp_history: list[int] = field(default_factory=list)

    def throughput(self) -> float:
        dt_s = max(1e-9, time.perf_counter() - self.t0)
        return self.processed / dt_s

    def error_rate(self) -> float:
        if self.processed == 0:
            return 0.0
        return (self.err / self.processed) * 100.0

    def tokens_per_s(self) -> float | None:
        if self.completion_tokens <= 0:
            return None
        if self.generation_s > 0:
            return self.completion_tokens / self.generation_s
        dt_s = max(1e-9, time.perf_counter() - self.t0)
        return self.completion_tokens / dt_s

    def add_batch_fill(self, actual: int, limit: int) -> None:
        if limit > 0:
            self.batch_fills.append(100.0 * actual / limit)
            if len(self.batch_fills) > self.max_history_window:
                self.batch_fills = self.batch_fills[-self.max_history_window :]

    def batch_efficiency(self) -> float:
        if not self.batch_fills:
            return 0.0
        return sum(self.batch_fills) / len(self.batch_fills)

    def add_queue_depth(self, depth: int) -> None:
        self.queue_depth_samples.append(depth)
        if len(self.queue_depth_samples) > self.max_history_window:
            self.queue_depth_samples = self.queue_depth_samples[-self.max_history_window :]

    def queue_depth_p50(self) -> int:
        if not self.queue_depth_samples:
            return 0
        sorted_samples = sorted(self.queue_depth_samples)
        return sorted_samples[len(sorted_samples) // 2]

    def sample_throughput(self) -> None:
        self.throughput_history.append(self.throughput())
        if len(self.throughput_history) > self.max_history_window:
            self.throughput_history = self.throughput_history[-self.max_history_window]

    def get_system_stats(self) -> dict[str, float] | None:
        if psutil is None:
            return None
        try:
            proc = psutil.Process()
            return {
                "rss_mb": proc.memory_info().rss / 1024 / 1024,
                "cpu_percent": proc.cpu_percent(interval=0.0),
            }
        except Exception:
            return None

    def sample_gpu(self) -> dict[str, int | float] | None:
        info = sample_gpu_info()
        if info is None:
            return None
        self.gpu_util_history.append(float(info["util"]))
        self.gpu_vram_history.append(int(info["mem_used_mib"]))
        self.gpu_temp_history.append(int(info["temp_c"]))
        if len(self.gpu_util_history) > self.max_history_window:
            self.gpu_util_history = self.gpu_util_history[-self.max_history_window :]
        if len(self.gpu_vram_history) > self.max_history_window:
            self.gpu_vram_history = self.gpu_vram_history[-self.max_history_window :]
        if len(self.gpu_temp_history) > self.max_history_window:
            self.gpu_temp_history = self.gpu_temp_history[-self.max_history_window :]
        return info

    def table(
        self,
        *,
        inflight: int,
        pending: int,
        done: int,
        error: int,
        batch_size: int,
    ) -> Table:
        tbl = Table(title="Labeler Metrics")
        tbl.add_column("processed", justify="right")
        tbl.add_column("ok", justify="right")
        tbl.add_column("err", justify="right")
        tbl.add_column("err%", justify="right")
        tbl.add_column("img/s", justify="right")
        tbl.add_column("tok/s", justify="right")
        tbl.add_column("ttft p50", justify="right")
        tbl.add_column("ttft p95", justify="right")
        tbl.add_column("p50 ms", justify="right")
        tbl.add_column("p95 ms", justify="right")
        tbl.add_column("inflight", justify="right")
        tbl.add_column("batch", justify="right")
        tbl.add_column("pending", justify="right")
        tbl.add_column("done", justify="right")
        tbl.add_column("error", justify="right")

        tok_s = self.tokens_per_s()
        tok_s_str = "-" if tok_s is None else f"{tok_s:.1f}"

        ttft_p50 = self.ttft.percentile(50)
        ttft_p95 = self.ttft.percentile(95)
        ttft_p50_str = "-" if math.isnan(ttft_p50) else f"{ttft_p50:.1f}"
        ttft_p95_str = "-" if math.isnan(ttft_p95) else f"{ttft_p95:.1f}"

        tbl.add_row(
            str(self.processed),
            str(self.ok),
            str(self.err),
            f"{self.error_rate():.1f}",
            f"{self.throughput():.2f}",
            tok_s_str,
            ttft_p50_str,
            ttft_p95_str,
            f"{self.lat.percentile(50):.1f}",
            f"{self.lat.percentile(95):.1f}",
            str(inflight),
            str(batch_size),
            str(pending),
            str(done),
            str(error),
        )
        return tbl


def sparkline(values: list[float], width: int = 20) -> Text:
    if not values:
        return Text("")
    ticks = " ▁▂▃▄▅▆▇█"
    recent = values[-width:]
    if not recent:
        return Text("")
    mn, mx = min(recent), max(recent)
    rng = max(1e-9, mx - mn)
    chars = [ticks[int(((v - mn) / rng) * (len(ticks) - 1))] for v in recent]
    return Text("".join(chars), style="cyan")


def histogram_buckets(samples: list[float], bins: int = 30) -> list[tuple[float, int]]:
    if not samples:
        return []
    mn, mx = min(samples), max(samples)
    if mn == mx:
        return [(mn, len(samples))]
    width = (mx - mn) / bins
    counts = [0] * bins
    for s in samples:
        idx = min(bins - 1, int((s - mn) / max(1e-9, width)))
        counts[idx] += 1
    return [(mn + i * width, counts[i]) for i in range(bins)]


def render_histogram(buckets: list[tuple[float, int]], max_width: int = 40) -> str:
    if not buckets:
        return ""
    max_count = max(c for _, c in buckets)
    if max_count == 0:
        return ""
    lines = []
    for ms, count in buckets:
        bar_len = int((count / max_count) * max_width)
        bar = "█" * bar_len
        lines.append(f"{ms:6.0f} | {bar} {count}")
    return "\n".join(lines[-15:])


def color_threshold(value: float, green: float, yellow: float) -> str:
    if value < green:
        return "green"
    if value < yellow:
        return "yellow"
    return "red"


def network_gauge(active: int, max_conns: int, width: int = 20) -> str:
    if max_conns <= 0:
        return "[" + "░" * width + "]"
    filled = int((active / max_conns) * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {active}/{max_conns}"


def print_summary(console: Console, title: str, metrics: RunMetrics) -> None:
    console.rule(title)
    tok_s = metrics.tokens_per_s()
    tok_s_str = "-" if tok_s is None else f"{tok_s:.1f}"
    ttft_p50 = metrics.ttft.percentile(50)
    ttft_p95 = metrics.ttft.percentile(95)
    ttft_p50_str = "-" if math.isnan(ttft_p50) else f"{ttft_p50:.1f}"
    ttft_p95_str = "-" if math.isnan(ttft_p95) else f"{ttft_p95:.1f}"
    console.print(
        f"processed={metrics.processed} ok={metrics.ok} err={metrics.err} "
        f"err%={metrics.error_rate():.1f} "
        f"throughput={metrics.throughput():.2f} img/s tok/s={tok_s_str} "
        f"ttft50={ttft_p50_str}ms ttft95={ttft_p95_str}ms "
        f"p50={metrics.lat.percentile(50):.1f}ms "
        f"p95={metrics.lat.percentile(95):.1f}ms "
        f"p99={metrics.lat.percentile(99):.1f}ms"
    )


class MetricsDashboard:
    """Rich Layout-based dashboard for live metrics display."""

    def __init__(self, console: Console):
        self.console = console
        self.layout = Layout()
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=10),
        )
        self.layout["main"].split_row(Layout(name="left"), Layout(name="right"))

    def render(
        self,
        metrics: RunMetrics,
        summ: dict[str, Any],
        run_id: str,
        model: str,
        inflight: int,
        batch_size: int,
        tuner_state: dict[str, Any] | None = None,
    ) -> Layout:
        self.layout["header"].update(
            Panel(
                f"[bold cyan]Run:[/] {run_id[:8]}... | [bold yellow]Model:[/] {model}",
                style="bold",
            )
        )
        self.layout["left"].update(self._throughput_panel(metrics))
        self.layout["right"].update(
            self._queue_panel(summ, inflight, batch_size, metrics, tuner_state)
        )
        self.layout["footer"].update(self._latency_panel(metrics))
        return self.layout

    def _throughput_panel(self, m: RunMetrics) -> Panel:
        tok_s = m.tokens_per_s()
        tok_s_str = f"[cyan]{tok_s:.1f}[/]" if tok_s is not None else "[dim]N/A[/]"

        err_rate = m.error_rate()
        err_color = color_threshold(err_rate, 2.0, 5.0)

        trend = sparkline(m.throughput_history, width=15)
        trend_str = f"  Trend: {trend}" if len(m.throughput_history) > 1 else ""

        sys_stats = m.get_system_stats()
        sys_str = ""
        if sys_stats:
            sys_str = (
                f"\n  Memory:       [cyan]{sys_stats['rss_mb']:.0f}[/] MB"
                f"\n  CPU:          [cyan]{sys_stats['cpu_percent']:.1f}[/]%"
            )

        gpu_str = ""
        if m.gpu_util_history or m.gpu_vram_history or m.gpu_temp_history:
            util = m.gpu_util_history[-1] if m.gpu_util_history else 0.0
            vram_used = m.gpu_vram_history[-1] if m.gpu_vram_history else 0
            temp = m.gpu_temp_history[-1] if m.gpu_temp_history else 0
            temp_color = color_threshold(temp, 80, 85)
            vram_gb = vram_used / 1024
            gpu_info = sample_gpu_info()
            total_gb = (gpu_info["mem_total_mib"] / 1024) if gpu_info else 24.0
            gpu_str = (
                f"\n  GPU:          [cyan]{util:.0f}%[/] | "
                f"VRAM [cyan]{vram_gb:.1f}[/]/[dim]{total_gb:.0f}[/]GB | "
                f"[{temp_color}]{temp}°C[/]"
            )

        content = f"""[bold]Throughput[/]
  Images/s:     [green]{m.throughput():.2f}[/]{trend_str}
  Tokens/s:     {tok_s_str}
  Total:        {m.processed}
  OK:           [green]{m.ok}[/]
  Errors:       [{err_color}]{m.err}[/] ([{err_color}]{err_rate:.1f}%[/]){sys_str}{gpu_str}
        """
        return Panel(content.strip(), border_style="green", title="📊 Performance")

    def _queue_panel(
        self,
        summ: dict[str, Any],
        inflight: int,
        batch_size: int,
        m: RunMetrics,
        tuner_state: dict[str, Any] | None,
    ) -> Panel:
        pending = summ.get("pending", 0)
        done = summ.get("done", 0)
        processing = summ.get("processing", 0)
        error = summ.get("error", 0)
        total = pending + done + processing + error

        progress_pct = (done / max(1, total)) * 100
        eta_s = pending / max(0.1, m.throughput())
        eta_str = (
            f"{eta_s:.0f}s"
            if eta_s < 120
            else f"{eta_s / 60:.1f}m"
            if eta_s < 3600
            else f"{eta_s / 3600:.1f}h"
        )

        batch_eff = m.batch_efficiency()
        batch_eff_str = (
            f"[green]{batch_eff:.1f}%[/]"
            if batch_eff > 80
            else f"[yellow]{batch_eff:.1f}%[/]"
            if batch_eff > 50
            else f"[red]{batch_eff:.1f}%[/]"
        )

        queue_depth = m.queue_depth_p50()

        tuner_str = ""
        if tuner_state:
            warmup = "✓" if tuner_state.get("warmup_complete") else "..."
            cooldown = tuner_state.get("cooldown_remaining_s", 0)
            tuner_str = f"\n  Auto-tune:    [cyan]warmup {warmup}[/] cooldown {cooldown:.1f}s"

        content = f"""[bold]Pipeline[/]
  Inflight:     [cyan]{inflight}[/]
  Batch Size:   [cyan]{batch_size}[/]
  Batch Fill:   {batch_eff_str}
  Queue (p50):  [yellow]{queue_depth}[/]
  Pending:      [yellow]{pending}[/]
  Done:         [green]{done}[/] / {total}
  Progress:     [cyan]{progress_pct:.1f}%[/] ETA {eta_str}{tuner_str}
        """
        return Panel(content.strip(), border_style="yellow", title="⚙️  Pipeline")

    def _latency_panel(self, m: RunMetrics) -> Panel:
        if not m.lat.samples_ms:
            return Panel("No latency data yet", border_style="blue", title="📈 Latency")

        buckets = histogram_buckets(m.lat.samples_ms, bins=30)
        chart = render_histogram(buckets, max_width=40)

        p50 = m.lat.percentile(50)
        p95 = m.lat.percentile(95)
        p99 = m.lat.percentile(99)
        mean = m.lat.mean()

        ttft_str = ""
        if m.ttft.samples_ms:
            ttft_p50 = m.ttft.percentile(50)
            ttft_p95 = m.ttft.percentile(95)
            if not math.isnan(ttft_p50):
                ttft_str = f"  TTFT:  P50 [cyan]{ttft_p50:.1f}ms[/] P95 [cyan]{ttft_p95:.1f}ms[/]\n"

        content = f"""[bold]Latency Distribution (ms)[/]

{chart}

  Total: Mean [cyan]{mean:.1f}ms[/] P50 [cyan]{p50:.1f}ms[/]
          P95 [cyan]{p95:.1f}ms[/] P99 [cyan]{p99:.1f}ms[/]
{ttft_str}"""
        return Panel(content.strip(), border_style="blue", title="📈 Latency")
