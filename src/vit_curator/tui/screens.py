"""Screen definitions for the ViT-Curator TUI."""

from __future__ import annotations

from typing import Any, ClassVar

from textual.app import ComposeResult
from textual.containers import Container, Grid, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    Input,
    Label,
    Rule,
    Select,
    Static,
)

from vit_curator.tui.widgets import (
    ActivityLog,
    AssetTable,
    GPUMeter,
    LatencyHistogram,
    PipelineStatus,
    ProgressMeter,
    RunList,
    Sparkline,
)


class DashboardScreen(Screen):
    """Main dashboard with live metrics overview."""

    DEFAULT_CSS = """
    DashboardScreen {
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr 1fr;
        grid-rows: auto auto auto;
        padding: 1;
    }

    DashboardScreen #progress-section {
        column-span: 2;
        height: auto;
    }

    DashboardScreen #metrics-section {
        column-span: 2;
        height: auto;
    }

    DashboardScreen #activity-section {
        column-span: 2;
        height: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the dashboard layout."""
        with Container(id="progress-section"):
            yield ProgressMeter(id="progress", total=100, done=0)

        with Horizontal(id="metrics-section"):
            with Vertical():
                yield GPUMeter(id="gpu-meter")
            with Vertical():
                yield PipelineStatus(id="pipeline-status")

        with Horizontal():
            yield Sparkline(id="throughput-spark", label="Throughput (img/s)", max_points=60)
            yield Sparkline(id="latency-spark", label="Latency P50 (ms)", max_points=60)

        with Container(id="activity-section"):
            yield ActivityLog(id="activity-log")


class RunsScreen(Screen):
    """Screen for managing runs."""

    DEFAULT_CSS = """
    RunsScreen {
        layout: vertical;
        padding: 1;
    }

    RunsScreen #runs-toolbar {
        height: auto;
        margin-bottom: 1;
    }

    RunsScreen #runs-list {
        height: 1fr;
    }

    RunsScreen #run-details {
        height: auto;
        border: solid $primary;
        padding: 1;
        margin-top: 1;
    }
    """

    BINDINGS: ClassVar = [
        ("r", "refresh", "Refresh"),
        ("enter", "select_run", "Select Run"),
        ("e", "export_run", "Export"),
    ]

    def compose(self) -> ComposeResult:
        """Compose the runs screen."""
        with Horizontal(id="runs-toolbar"):
            yield Button("Refresh", id="btn-refresh", variant="primary")
            yield Button("Export", id="btn-export", variant="success")
            yield Button("Delete", id="btn-delete", variant="error")
            yield Input(placeholder="Search runs...", id="runs-search")

        yield RunList(id="runs-list")

        with Container(id="run-details"):
            yield Static("Select a run to view details", id="run-details-content")

    def on_mount(self) -> None:
        """Load runs on mount."""
        self.app.refresh_runs()  # type: ignore[attr-defined]

    def action_refresh(self) -> None:
        """Refresh the runs list."""
        self.app.refresh_runs()  # type: ignore[attr-defined]

    def action_select_run(self) -> None:
        """Select the current run."""
        table = self.query_one("#runs-list", RunList)
        if table.cursor_row is not None:
            self.app.select_run(table.cursor_row)  # type: ignore[attr-defined]


class AssetsScreen(Screen):
    """Screen for browsing files."""

    DEFAULT_CSS = """
    AssetsScreen {
        layout: vertical;
        padding: 1;
    }

    AssetsScreen #assets-toolbar {
        height: auto;
        margin-bottom: 1;
    }

    AssetsScreen #assets-table {
        height: 1fr;
    }
    """

    BINDINGS: ClassVar = [
        ("r", "refresh", "Refresh"),
        ("f", "filter", "Filter"),
        ("s", "search", "Search"),
    ]

    def compose(self) -> ComposeResult:
        """Compose the assets screen."""
        with Horizontal(id="assets-toolbar"):
            yield Button("Refresh", id="btn-refresh", variant="primary")
            yield Select(
                [
                    ("All", "all"),
                    ("Pending", "pending"),
                    ("Processing", "processing"),
                    ("Done", "done"),
                    ("Error", "error"),
                ],
                prompt="Filter by status",
                id="status-filter",
            )
            yield Input(placeholder="Search by path...", id="assets-search")

        yield AssetTable(id="assets-table")

    def on_mount(self) -> None:
        """Load assets on mount."""
        self.app.refresh_assets()  # type: ignore[attr-defined]

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle status filter change."""
        if event.select.id == "status-filter":
            self.app.filter_assets(event.value)  # type: ignore[attr-defined]

    def action_refresh(self) -> None:
        """Refresh the assets list."""
        self.app.refresh_assets()  # type: ignore[attr-defined]


class StatsScreen(Screen):
    """Screen for detailed statistics and charts."""

    DEFAULT_CSS = """
    StatsScreen {
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr 1fr;
        padding: 1;
    }

    StatsScreen #stats-header {
        column-span: 2;
        height: auto;
        text-align: center;
        text-style: bold;
    }

    StatsScreen #latency-panel {
        height: 1fr;
    }

    StatsScreen #throughput-panel {
        height: 1fr;
    }

    StatsScreen #gpu-history-panel {
        height: 1fr;
    }

    StatsScreen #error-rate-panel {
        height: 1fr;
    }
    """

    BINDINGS: ClassVar = [
        ("r", "refresh", "Refresh"),
        ("c", "clear", "Clear History"),
    ]

    def compose(self) -> ComposeResult:
        """Compose the stats screen."""
        yield Static("Statistics Dashboard", id="stats-header")

        with Container(id="latency-panel"):
            yield LatencyHistogram(id="latency-hist", bins=20)

        with Container(id="throughput-panel"):
            yield Sparkline(id="throughput-history", label="Throughput History", max_points=120)

        with Container(id="gpu-history-panel"):
            yield Sparkline(id="gpu-util-history", label="GPU Utilization %", max_points=120)

        with Container(id="error-rate-panel"):
            yield Sparkline(id="error-rate-history", label="Error Rate %", max_points=120)

    def on_mount(self) -> None:
        """Initialize stats."""
        self.app.refresh_stats()  # type: ignore[attr-defined]


class SettingsScreen(Screen):
    """Screen for viewing and editing settings."""

    DEFAULT_CSS = """
    SettingsScreen {
        layout: vertical;
        padding: 1;
    }

    SettingsScreen #settings-content {
        height: 1fr;
        border: solid $primary;
        padding: 1;
    }

    SettingsScreen #settings-footer {
        height: auto;
        margin-top: 1;
    }
    """

    BINDINGS: ClassVar = [
        ("s", "save", "Save"),
    ]

    def compose(self) -> ComposeResult:
        """Compose the settings screen."""
        yield Label("Current Configuration", classes="title")
        yield Rule()

        with Container(id="settings-content"):
            yield Static("Loading settings...", id="settings-display")

        with Horizontal(id="settings-footer"):
            yield Button("Refresh", id="btn-refresh", variant="primary")
            yield Button("Edit", id="btn-edit", variant="warning")

    def on_mount(self) -> None:
        """Load settings on mount."""
        self.app.refresh_settings()  # type: ignore[attr-defined]


class ConfirmationDialog(Screen):
    """Modal dialog for confirmations."""

    DEFAULT_CSS = """
    ConfirmationDialog {
        align: center middle;
    }

    ConfirmationDialog > Grid {
        width: 60;
        height: auto;
        border: solid $primary;
        padding: 1;
        background: $surface;
    }

    ConfirmationDialog #dialog-title {
        column-span: 2;
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    ConfirmationDialog #dialog-message {
        column-span: 2;
        margin-bottom: 1;
    }
    """

    def __init__(self, title: str, message: str, on_confirm: str, **kwargs: Any):
        super().__init__(**kwargs)
        self.title = title
        self.message = message
        self.on_confirm = on_confirm

    def compose(self) -> ComposeResult:
        """Compose the dialog."""
        with Grid():
            yield Label(self.title, id="dialog-title")
            yield Label(self.message, id="dialog-message")
            yield Button("Cancel", id="btn-cancel", variant="primary")
            yield Button("Confirm", id="btn-confirm", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-confirm":
            self.app.post_message(self.on_confirm)  # type: ignore[arg-type]
        self.dismiss()
