"""Custom widgets for the unified pipeline TUI dashboard."""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime
from typing import Any

from textual.reactive import reactive
from textual.widgets import DataTable, Log, Static

from vit_curator.tui.themes import (
    PROGRESS_COLORS,
    get_log_style,
    get_status_style,
    get_temp_style,
)


class GPUMeter(Static):
    """Display GPU utilization, VRAM, and temperature."""

    util = reactive(0.0)
    vram_used = reactive(0)
    vram_total = reactive(0)
    temp = reactive(0)
    gpu_visible = reactive(True)

    DEFAULT_CSS = """
    GPUMeter {
        height: auto;
        padding: 1;
        border: solid $primary;
    }
    """

    def __init__(
        self,
        *,
        util: float = 0.0,
        vram_used: int = 0,
        vram_total: int = 0,
        temp: int = 0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.util = util
        self.vram_used = vram_used
        self.vram_total = max(vram_total, 1)
        self.temp = temp

    def watch_util(self) -> None:
        self.update_display()

    def watch_vram_used(self) -> None:
        self.update_display()

    def watch_temp(self) -> None:
        self.update_display()

    def update_display(self) -> None:
        if not self.vram_total:
            self.update("[dim]No GPU detected[/]")
            return

        util_pct = min(100, max(0, self.util))
        vram_pct = (self.vram_used / self.vram_total) * 100 if self.vram_total else 0
        temp_color = get_temp_style(self.temp)
        util_color = "green" if util_pct < 70 else "yellow" if util_pct < 90 else "red"
        vram_color = "green" if vram_pct < 80 else "yellow" if vram_pct < 95 else "red"
        temp_emoji = "🟢" if self.temp < 70 else "🟡" if self.temp < 80 else "🔴"

        vram_used_gb = self.vram_used / 1024
        vram_total_gb = self.vram_total / 1024
        lines = [
            "[bold]GPU Metrics[/]",
            "",
            f"  Util:     [{util_color}]{util_pct:5.1f}%[/] {self._bar(util_pct, 20)}",
            f"  VRAM:     [{vram_color}]{vram_used_gb:.1f}[/]/[dim]{vram_total_gb:.0f}[/] GB",
            f"  Temp:     [{temp_color}]{self.temp}°C[/] {temp_emoji}",
        ]
        self.update("\n".join(lines))

    @staticmethod
    def _bar(pct: float, width: int) -> str:
        filled = int((pct / 100) * width)
        return f"[{'█' * filled}{'░' * (width - filled)}]"


class ProgressMeter(Static):
    """Progress bar with percentage and ETA."""

    total = reactive(0)
    done = reactive(0)
    throughput = reactive(0.0)
    start_time: float | None = None

    DEFAULT_CSS = """
    ProgressMeter {
        height: auto;
        padding: 1;
        border: solid $primary;
    }
    """

    def __init__(self, *, total: int = 0, done: int = 0, throughput: float = 0.0, **kwargs: Any):
        super().__init__(**kwargs)
        self.total = total
        self.done = done
        self.throughput = throughput
        self.start_time = time.perf_counter()

    def watch_total(self) -> None:
        self.update_display()

    def watch_done(self) -> None:
        self.update_display()

    def watch_throughput(self) -> None:
        self.update_display()

    def update_display(self) -> None:
        if self.total <= 0:
            self.update("[dim]No tasks[/]")
            return

        pct = min(100, max(0, (self.done / self.total) * 100))
        remaining = self.total - self.done

        eta_str = "--:--"
        if self.throughput > 0 and remaining > 0:
            eta_s = remaining / self.throughput
            if eta_s < 60:
                eta_str = f"{eta_s:.0f}s"
            elif eta_s < 3600:
                eta_str = f"{eta_s / 60:.1f}m"
            else:
                eta_str = f"{eta_s / 3600:.1f}h"

        color = PROGRESS_COLORS["fill"]

        width = 40
        filled = int((pct / 100) * width)
        bar = f"[{'█' * filled}{'░' * (width - filled)}]"

        lines = [
            "[bold]Progress[/]",
            "",
            f"  {bar} [{color}]{pct:.1f}%[/]",
            f"  ETA: [cyan]{eta_str}[/] | [cyan]{self.throughput:.1f}[/] img/s",
            f"  {self.done:,} / {self.total:,} tasks",
        ]
        self.update("\n".join(lines))


class Sparkline(Static):
    """Mini line chart for time series data."""

    data = reactive(list)
    max_points = reactive(60)
    label = reactive("")

    DEFAULT_CSS = """
    Sparkline {
        height: auto;
        padding: 1;
        border: solid $primary;
    }
    """

    TICKS = " ▁▂▃▄▅▆▇█"

    def __init__(
        self,
        *,
        data: list[float] | None = None,
        max_points: int = 60,
        label: str = "",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.data = data or []
        self.max_points = max_points
        self.label = label

    def watch_data(self) -> None:
        self.update_display()

    def add_point(self, value: float) -> None:
        self.data.append(value)
        if len(self.data) > self.max_points:
            self.data = self.data[-self.max_points :]
        self.update_display()

    def update_display(self) -> None:
        if not self.data:
            self.update(f"[dim]{self.label or 'Sparkline'}: no data[/]")
            return

        recent = self.data[-60:]
        mn, mx = min(recent), max(recent)
        rng = max(1e-9, mx - mn)
        chars = [self.TICKS[int(((v - mn) / rng) * (len(self.TICKS) - 1))] for v in recent]
        spark = "".join(chars)
        current = recent[-1] if recent else 0

        label_str = f"[bold]{self.label}[/]\n\n" if self.label else ""
        lines = [
            label_str,
            f"  [cyan]{spark}[/]",
            f"  Current: [cyan]{current:.2f}[/]  Min: [dim]{mn:.2f}[/]  Max: [dim]{mx:.2f}[/]",
        ]
        self.update("\n".join(lines))


class LatencyHistogram(Static):
    """ASCII histogram for latency distribution."""

    samples = reactive(list)
    bins = reactive(20)
    max_width = reactive(30)

    DEFAULT_CSS = """
    LatencyHistogram {
        height: auto;
        padding: 1;
        border: solid $primary;
    }
    """

    def __init__(self, *, samples: list[float] | None = None, bins: int = 20, **kwargs: Any):
        super().__init__(**kwargs)
        self.samples = samples or []
        self.bins = bins

    def watch_samples(self) -> None:
        self.update_display()

    def add_sample(self, value: float) -> None:
        self.samples.append(value)
        if len(self.samples) > 10000:
            self.samples = self.samples[-10000:]
        self.update_display()

    def update_display(self) -> None:
        if not self.samples:
            self.update("[dim]No latency data[/]")
            return

        mn, mx = min(self.samples), max(self.samples)
        if mn == mx:
            buckets = [(mn, len(self.samples))]
        else:
            width = (mx - mn) / self.bins
            counts = [0] * self.bins
            for s in self.samples:
                idx = min(self.bins - 1, int((s - mn) / max(1e-9, width)))
                counts[idx] += 1
            buckets = [(mn + i * width, counts[i]) for i in range(self.bins)]

        max_count = max(c for _, c in buckets) if buckets else 1
        if max_count == 0:
            max_count = 1

        lines = ["[bold]Latency Distribution (ms)[/]", ""]
        for ms, count in buckets[-12:]:
            bar_len = int((count / max_count) * self.max_width)
            bar = "█" * bar_len
            pct = (count / len(self.samples)) * 100
            lines.append(f"  {ms:6.0f} | {bar} {count} ({pct:.1f}%)")

        if self.samples:
            sorted_samples = sorted(self.samples)
            n = len(sorted_samples)
            p50 = sorted_samples[n // 2]
            p95 = sorted_samples[int(0.95 * n)] if n > 20 else sorted_samples[-1]
            mean = sum(sorted_samples) / n
            lines.extend(
                [
                    "",
                    (
                        f"  Mean: [cyan]{mean:.1f}[/]  "
                        f"P50: [cyan]{p50:.1f}[/]  "
                        f"P95: [cyan]{p95:.1f}[/]"
                    ),
                ]
            )

        self.update("\n".join(lines))


class ActivityLog(Log):
    """Scrollable log widget with color-coded severity levels."""

    DEFAULT_CSS = """
    ActivityLog {
        height: 1fr;
        border: solid $primary;
        padding: 0;
    }
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._entries: deque[tuple[str, str, str]] = deque(maxlen=1000)

    def log_event(
        self, message: str, level: str = "info", timestamp: datetime | None = None
    ) -> None:
        if timestamp is None:
            timestamp = datetime.now()
        ts_str = timestamp.strftime("%H:%M:%S")
        color = get_log_style(level)
        entry = f"[{color}]{ts_str}[/{color}] [{color}]{level.upper():8}[/{color}] {message}"
        self._entries.append((ts_str, level, message))
        self.write_line(entry)

    def log_info(self, message: str) -> None:
        self.log_event(message, "info")

    def log_success(self, message: str) -> None:
        self.log_event(message, "success")

    def log_warning(self, message: str) -> None:
        self.log_event(message, "warning")

    def log_error(self, message: str) -> None:
        self.log_event(message, "error")


class PipelineStatus(Static):
    """Display pipeline status metrics."""

    inflight = reactive(0)
    pending = reactive(0)
    done = reactive(0)
    errors = reactive(0)
    batch_size = reactive(0)

    DEFAULT_CSS = """
    PipelineStatus {
        height: auto;
        padding: 1;
        border: solid $primary;
    }
    """

    def __init__(
        self,
        *,
        inflight: int = 0,
        pending: int = 0,
        done: int = 0,
        errors: int = 0,
        batch_size: int = 0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.inflight = inflight
        self.pending = pending
        self.done = done
        self.errors = errors
        self.batch_size = batch_size

    def watch_inflight(self) -> None:
        self.update_display()

    def watch_pending(self) -> None:
        self.update_display()

    def watch_done(self) -> None:
        self.update_display()

    def watch_errors(self) -> None:
        self.update_display()

    def watch_batch_size(self) -> None:
        self.update_display()

    def update_display(self) -> None:
        total = self.pending + self.done + self.inflight + self.errors
        lines = [
            "[bold]Pipeline Status[/]",
            "",
            f"  Inflight:  [cyan]{self.inflight:,}[/]",
            f"  Pending:   [yellow]{self.pending:,}[/]",
            f"  Done:      [green]{self.done:,}[/]",
            f"  Errors:    [{'red' if self.errors > 0 else 'dim'}]{self.errors:,}[/]",
            f"  Batch:     [dim]{self.batch_size}[/]",
            "",
            f"  Total:     {total:,}",
        ]
        self.update("\n".join(lines))


class RunList(DataTable):
    """DataTable for displaying runs."""

    DEFAULT_CSS = """
    RunList {
        height: 1fr;
        border: solid $primary;
    }
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.cursor_type = "row"
        self.add_columns("Run ID", "Started", "Model", "Status", "Progress")

    def add_run(
        self, run_id: str, started_at: str, model: str, status: str, progress_pct: float
    ) -> None:
        status_color = get_status_style(status)
        status_str = f"[{status_color}]{status}[/{status_color}]"
        progress_str = f"{progress_pct:.1f}%"
        self.add_row(run_id[:16] + "...", started_at[:19], model[:30], status_str, progress_str)


class AssetTable(DataTable):
    """DataTable for displaying files with status."""

    DEFAULT_CSS = """
    AssetTable {
        height: 1fr;
        border: solid $primary;
    }
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.cursor_type = "row"
        self.add_columns("ID", "Path", "Status", "Labels")

    def add_asset(
        self, *, file_pk: int, path: str, status: str, labels: list[int] | None = None
    ) -> None:
        status_color = get_status_style(status)
        status_str = f"[{status_color}]{status}[/{status_color}]"
        labels_str = ""
        if labels:
            labels_str = ", ".join(str(lbl) for lbl in labels[:5])
            if len(labels) > 5:
                labels_str += f" (+{len(labels) - 5})"
        self.add_row(str(file_pk), path[-50:] if len(path) > 50 else path, status_str, labels_str)
