"""Main Textual application for the ViT-Curator TUI."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.reactive import reactive
from textual.widgets import (
    Button,
    ContentSwitcher,
    Footer,
    Header,
    Label,
    ListItem,
    ListView,
    Static,
)

from vit_curator.label.metrics import sample_gpu_info
from vit_curator.label.store import (
    connect_label_db,
    get_last_run,
    summarize,
)
from vit_curator.tui.screens import (
    AssetsScreen,
    ConfirmationDialog,
    DashboardScreen,
    RunsScreen,
    SettingsScreen,
    StatsScreen,
)
from vit_curator.tui.widgets import (
    ActivityLog,
    AssetTable,
    GPUMeter,
    LatencyHistogram,
    PipelineStatus,
    ProgressMeter,
    RunList,
)


class Sidebar(Container):
    """Sidebar navigation widget."""

    DEFAULT_CSS = """
    Sidebar {
        width: 20;
        height: 100%;
        background: $surface;
        border-right: solid $primary;
        padding: 1;
    }

    Sidebar ListView {
        height: auto;
        border: none;
        background: transparent;
    }

    Sidebar ListItem {
        padding: 1;
    }

    Sidebar ListItem.--highlight {
        background: $primary 30%;
    }

    Sidebar #sidebar-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("Pipeline Control", id="sidebar-title")
        yield ListView(
            ListItem(Label("● Dashboard")),
            ListItem(Label("  Runs")),
            ListItem(Label("  Assets")),
            ListItem(Label("  Stats")),
            ListItem(Label("  Settings")),
            id="nav-list",
        )


class PipelineApp(App[None]):
    """ViT-Curator Control Center TUI Application."""

    CSS = """
    Screen {
        align: center middle;
    }

    #main-container {
        width: 100%;
        height: 100%;
    }

    #content-area {
        width: 1fr;
        height: 100%;
    }

    .title {
        text-style: bold;
        text-align: center;
    }
    """

    BINDINGS: ClassVar = [
        Binding("q", "quit", "Quit", show=True),
        Binding("1", "switch_screen('dashboard')", "Dashboard", show=True),
        Binding("2", "switch_screen('runs')", "Runs", show=True),
        Binding("3", "switch_screen('assets')", "Assets", show=True),
        Binding("4", "switch_screen('stats')", "Stats", show=True),
        Binding("5", "switch_screen('settings')", "Settings", show=True),
        Binding("r", "refresh_all", "Refresh", show=True),
    ]

    # Reactive state
    current_run_id: reactive[str | None] = reactive(None)
    gpu_info: reactive[dict[str, Any]] = reactive({})
    pipeline_stats: reactive[dict[str, int]] = reactive({})
    throughput_history: reactive[list[float]] = reactive([])
    latency_history: reactive[list[float]] = reactive([])

    def __init__(
        self,
        db: Path,
        refresh_rate: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.db_path = db
        self.refresh_rate = refresh_rate
        self._conn: Any = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._last_gpu_sample = 0.0
        self._gpu_sample_interval = 2.0

    def on_mount(self) -> None:
        """Initialize the app on mount."""
        try:
            self._conn = connect_label_db(self.db_path)
            self._log("Connected to database", "success")
        except Exception as e:
            self._log(f"Database error: {e}", "error")
            return

        self._refresh_current_run()

        self.set_interval(self.refresh_rate, self._refresh_metrics)
        self.set_interval(self.refresh_rate * 5, self._refresh_pipeline_stats)

        self._refresh_metrics()
        self._refresh_pipeline_stats()

        self.action_switch_screen("dashboard")

    def compose(self) -> ComposeResult:
        """Compose the main app layout."""
        yield Header(show_clock=True)

        with Horizontal(id="main-container"):
            yield Sidebar()

            with ContentSwitcher(id="content-area", initial="dashboard"):
                yield DashboardScreen(id="dashboard")
                yield RunsScreen(id="runs")
                yield AssetsScreen(id="assets")
                yield StatsScreen(id="stats")
                yield SettingsScreen(id="settings")

        yield Footer()

    def action_switch_screen(self, screen_name: str) -> None:
        """Switch to a different screen."""
        switcher = self.query_one("#content-area", ContentSwitcher)
        switcher.current = screen_name

        sidebar = self.query_one("#nav-list", ListView)
        index_map = {
            "dashboard": 0,
            "runs": 1,
            "assets": 2,
            "stats": 3,
            "settings": 4,
        }
        if screen_name in index_map:
            sidebar.index = index_map[screen_name]

    def action_refresh_all(self) -> None:
        """Refresh all data."""
        self._refresh_metrics()
        self._refresh_pipeline_stats()
        self.refresh_runs()
        self.refresh_assets()
        self._log("Refreshed all data", "info")

    # --- Data refresh methods ---

    def _refresh_current_run(self) -> None:
        """Get the most recent run."""
        if self._conn is None:
            return

        try:
            last_run = get_last_run(self._conn)
            if last_run:
                self.current_run_id = last_run.get("run_id")
        except Exception as e:
            self._log(f"Error loading run: {e}", "error")

    def _refresh_metrics(self) -> None:
        """Refresh live GPU metrics."""
        now = time.time()
        if now - self._last_gpu_sample >= self._gpu_sample_interval:
            try:
                gpu = sample_gpu_info()
                if gpu:
                    self.gpu_info = gpu
                    self._last_gpu_sample = now
            except Exception:
                pass

        try:
            gpu_meter = self.query_one("#gpu-meter", GPUMeter)
            if self.gpu_info:
                gpu_meter.util = float(self.gpu_info.get("util", 0))
                gpu_meter.vram_used = int(self.gpu_info.get("mem_used_mib", 0))
                gpu_meter.vram_total = int(self.gpu_info.get("mem_total_mib", 0))
                gpu_meter.temp = int(self.gpu_info.get("temp_c", 0))
        except Exception:
            pass

    def _refresh_pipeline_stats(self) -> None:
        """Refresh pipeline statistics from database."""
        if self._conn is None or not self.current_run_id:
            return

        try:
            stats = summarize(self._conn, run_id=self.current_run_id)
            self.pipeline_stats = stats

            total = sum(stats.values())
            try:
                progress = self.query_one("#progress", ProgressMeter)
                progress.total = total
                progress.done = stats.get("done", 0)

                if len(self.throughput_history) > 1:
                    avg_throughput = sum(self.throughput_history[-10:]) / min(
                        10, len(self.throughput_history[-10:])
                    )
                    progress.throughput = avg_throughput
            except Exception:
                pass

            try:
                status = self.query_one("#pipeline-status", PipelineStatus)
                status.pending = stats.get("pending", 0)
                status.done = stats.get("done", 0)
                status.errors = stats.get("error", 0)
                status.inflight = stats.get("processing", 0)
            except Exception:
                pass

        except Exception as e:
            self._log(f"Error refreshing stats: {e}", "error")

    def refresh_runs(self) -> None:
        """Refresh the runs list."""
        if self._conn is None:
            return

        try:
            runs_list = self.query_one("#runs-list", RunList)
            runs_list.clear()

            rows = self._conn.execute(
                """
                SELECT run_id, started_at, model, prompt_version
                FROM runs
                ORDER BY started_at DESC
                LIMIT 50
                """
            ).fetchall()

            for row in rows:
                run_id, started_at, model, _ = row
                stats = summarize(self._conn, run_id=str(run_id))
                total = sum(stats.values())
                done = stats.get("done", 0)
                progress = (done / total * 100) if total > 0 else 0

                if stats.get("processing", 0) > 0:
                    status = "processing"
                elif stats.get("pending", 0) > 0:
                    status = "pending"
                elif stats.get("error", 0) > 0:
                    status = "error"
                else:
                    status = "done"

                runs_list.add_run(
                    run_id=str(run_id),
                    started_at=str(started_at),
                    model=str(model),
                    status=status,
                    progress_pct=progress,
                )

            self._log(f"Loaded {len(rows)} runs", "info")
        except Exception as e:
            self._log(f"Error loading runs: {e}", "error")

    def refresh_assets(self, status_filter: str | None = None) -> None:
        """Refresh the files list."""
        if self._conn is None or not self.current_run_id:
            return

        try:
            assets_table = self.query_one("#assets-table", AssetTable)
            assets_table.clear()

            query = """
                SELECT t.file_pk, os.fsdecode(f.rel_path_blob), t.status, p.labels
                FROM tasks t
                JOIN files f ON f.file_pk = t.file_pk
                LEFT JOIN predictions p ON p.file_pk = t.file_pk AND p.run_id = t.run_id
                WHERE t.run_id = ?
            """
            params: list[Any] = [self.current_run_id]

            if status_filter and status_filter != "all":
                query += " AND t.status = ?"
                params.append(status_filter)

            query += " ORDER BY f.rel_path_blob LIMIT 1000"

            rows = self._conn.execute(query, params).fetchall()

            for row in rows:
                file_pk, path, status, labels = row
                label_list = list(labels) if labels else None
                assets_table.add_asset(
                    file_pk=int(file_pk),
                    path=str(path),
                    status=str(status),
                    labels=label_list,
                )

            self._log(f"Loaded {len(rows)} files", "info")
        except Exception as e:
            self._log(f"Error loading files: {e}", "error")

    def filter_assets(self, status: str | None) -> None:
        """Filter files by status."""
        self.refresh_assets(status_filter=status)

    def refresh_stats(self) -> None:
        """Refresh latency statistics."""
        if self._conn is None or not self.current_run_id:
            return

        try:
            rows = self._conn.execute(
                """
                SELECT latency_ms
                FROM tasks
                WHERE run_id = ? AND latency_ms IS NOT NULL
                ORDER BY finished_at DESC
                LIMIT 1000
                """,
                [self.current_run_id],
            ).fetchall()

            latencies = [float(r[0]) for r in rows if r[0] is not None]

            try:
                hist = self.query_one("#latency-hist", LatencyHistogram)
                hist.samples = latencies
            except Exception:
                pass

            self._log(f"Loaded {len(latencies)} latency samples", "info")
        except Exception as e:
            self._log(f"Error loading stats: {e}", "error")

    def refresh_settings(self) -> None:
        """Refresh settings display."""
        if self._conn is None:
            return

        try:
            if self.current_run_id:
                row = self._conn.execute(
                    """
                    SELECT settings_json
                    FROM runs
                    WHERE run_id = ?
                    """,
                    [self.current_run_id],
                ).fetchone()

                if row and row[0]:
                    settings_str = str(row[0])
                    lines = ["Current Run Settings:", ""]
                    try:
                        import json  # noqa: PLC0415

                        settings = json.loads(settings_str)
                        for key, value in sorted(settings.items()):
                            lines.append(f"  {key}: {value}")
                    except Exception:
                        lines.append(settings_str[:500])

                    try:
                        display = self.query_one("#settings-display", Static)
                        display.update("\n".join(lines))
                    except Exception:
                        pass
        except Exception as e:
            self._log(f"Error loading settings: {e}", "error")

    def select_run(self, row_index: int) -> None:
        """Select a run from the runs list."""
        self._log(f"Selected run at row {row_index}", "info")

    def _log(self, message: str, level: str = "info") -> None:
        """Log an activity message."""
        try:
            log = self.query_one("#activity-log", ActivityLog)
            log.log_event(message, level)
        except Exception:
            pass

    # --- Event handlers ---

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle sidebar navigation."""
        index_map = {
            0: "dashboard",
            1: "runs",
            2: "assets",
            3: "stats",
            4: "settings",
        }
        if event.list_view.index in index_map:
            self.action_switch_screen(index_map[event.list_view.index])

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "btn-refresh":
            self.action_refresh_all()
        elif button_id == "btn-export":
            self._log("Export not yet implemented", "warning")
        elif button_id == "btn-delete":
            self.push_screen(
                ConfirmationDialog(
                    "Delete Run",
                    "Are you sure you want to delete this run?",
                    "confirm_delete",
                )
            )

    async def on_confirm_delete(self) -> None:
        """Handle run deletion confirmation."""
        self._log("Run deletion confirmed", "warning")

    def watch_current_run_id(self) -> None:
        """React to run ID changes."""
        if self.current_run_id:
            self._log(f"Active run: {self.current_run_id[:16]}...", "info")

    def on_unmount(self) -> None:
        """Clean up on unmount."""
        if self._executor:
            self._executor.shutdown(wait=False)
