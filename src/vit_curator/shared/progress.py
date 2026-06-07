"""Rich progress bar utilities for pipeline stage reporting."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def make_progress(
    *,
    description: str = "Working",
    show_speed: bool = True,
) -> Progress:
    """Create a standard Rich Progress instance with sensible defaults.

    Uses SpinnerColumn + TextColumn + BarColumn + TimeElapsed + TimeRemaining.
    """
    columns = [
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ]
    if show_speed:
        columns.append(TextColumn("{task.fields[speed]}"))
    columns.extend(
        [
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ]
    )
    return Progress(*columns, transient=True)


@contextmanager
def progress_stage(
    description: str,
    total: int | None = None,
) -> Generator[tuple[Progress, TaskID], None, None]:
    """Context manager that creates a Rich Progress bar for a pipeline stage.

    Args:
        description: Label for the progress task.
        total: Total number of items (None for indeterminate).

    Yields:
        Tuple of (Progress instance, TaskID) for updating.
    """
    progress = make_progress()
    task_id = progress.add_task(description, total=total, speed="")
    with progress:
        yield progress, task_id
