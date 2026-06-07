"""Color themes and styling for the TUI."""

from textual.theme import Theme

# Dark theme (default)
DARK_THEME = Theme(
    name="pipeline_dark",
    primary="#00d4aa",
    secondary="#6b7280",
    accent="#f59e0b",
    foreground="#e5e7eb",
    background="#0f0f0f",
    surface="#1f1f1f",
    panel="#262626",
    dark=True,
    variables={
        "footer-foreground": "#e5e7eb",
        "footer-background": "#1f1f1f",
    },
)

# Light theme
LIGHT_THEME = Theme(
    name="pipeline_light",
    primary="#059669",
    secondary="#6b7280",
    accent="#d97706",
    foreground="#1f2937",
    background="#fafafa",
    surface="#ffffff",
    panel="#f3f4f6",
    dark=False,
)

# Monochrome theme for accessibility
MONO_THEME = Theme(
    name="pipeline_mono",
    primary="#ffffff",
    secondary="#a0a0a0",
    accent="#808080",
    foreground="#e0e0e0",
    background="#000000",
    surface="#1a1a1a",
    panel="#262626",
    dark=True,
)

# Status colors
STATUS_COLORS = {
    "pending": "#6b7280",
    "processing": "#f59e0b",
    "done": "#10b981",
    "error": "#ef4444",
    "unknown": "#9ca3af",
}

# GPU temperature colors
TEMP_COLORS = {
    "normal": "#10b981",
    "warm": "#f59e0b",
    "hot": "#ef4444",
}

# Log level colors
LOG_COLORS = {
    "info": "#3b82f6",
    "success": "#10b981",
    "warning": "#f59e0b",
    "error": "#ef4444",
    "debug": "#8b5cf6",
}

# Progress bar colors
PROGRESS_COLORS = {
    "background": "#262626",
    "fill": "#00d4aa",
    "fill_warning": "#f59e0b",
    "fill_error": "#ef4444",
}

# Sparkline colors
SPARKLINE_COLORS = ["#00d4aa", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6"]


def get_status_style(status: str) -> str:
    """Get color style for a status string."""
    return STATUS_COLORS.get(status.lower(), STATUS_COLORS["unknown"])


def get_temp_style(temp_c: int) -> str:
    """Get color style for GPU temperature."""
    if temp_c >= 80:
        return TEMP_COLORS["hot"]
    if temp_c >= 70:
        return TEMP_COLORS["warm"]
    return TEMP_COLORS["normal"]


def get_log_style(level: str) -> str:
    """Get color style for log level."""
    return LOG_COLORS.get(level.lower(), LOG_COLORS["info"])
