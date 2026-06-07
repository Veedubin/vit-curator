"""Dashboard: Textual TUI for monitoring pipeline runs."""

from __future__ import annotations

from vit_curator.tui.app import PipelineApp
from vit_curator.tui.screens import (
    AssetsScreen,
    ConfirmationDialog,
    DashboardScreen,
    RunsScreen,
    SettingsScreen,
    StatsScreen,
)
from vit_curator.tui.themes import DARK_THEME, LIGHT_THEME, MONO_THEME
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

__all__ = [
    "DARK_THEME",
    "LIGHT_THEME",
    "MONO_THEME",
    "ActivityLog",
    "AssetTable",
    "AssetsScreen",
    "ConfirmationDialog",
    "DashboardScreen",
    "GPUMeter",
    "LatencyHistogram",
    "PipelineApp",
    "PipelineStatus",
    "ProgressMeter",
    "RunList",
    "RunsScreen",
    "SettingsScreen",
    "Sparkline",
    "StatsScreen",
]
