"""Drain-safe signal handling for graceful shutdown.

Salvaged from file-helper-pipeline and adapted for vit_curator.
Provides context managers that catch SIGINT/SIGTERM and transition
gracefully from running → draining → stopped.
"""

from __future__ import annotations

import signal
import threading
from collections.abc import Generator
from contextlib import contextmanager


class DrainController:
    """Manages drain/stop events for graceful shutdown.

    Usage:
        ctrl = DrainController()
        # In worker threads:
        if ctrl.draining.is_set():
            break  # stop accepting new jobs
        if ctrl.stopped.is_set():
            break  # stop immediately
        # In signal handler:
        ctrl.request_drain()  # first SIGINT: drain
        ctrl.request_stop()    # second SIGINT: stop
    """

    def __init__(self) -> None:
        self.draining = threading.Event()
        self.stopped = threading.Event()

    def request_drain(self) -> None:
        """Signal workers to stop accepting new jobs and drain current work."""
        self.draining.set()

    def request_stop(self) -> None:
        """Signal workers to stop immediately."""
        self.draining.set()
        self.stopped.set()

    @property
    def is_draining(self) -> bool:
        return self.draining.is_set()

    @property
    def is_stopped(self) -> bool:
        return self.stopped.is_set()


@contextmanager
def drain_signal_handler(
    controller: DrainController,
) -> Generator[DrainController, None, None]:
    """Install a SIGINT/SIGTERM handler that transitions DrainController.

    First SIGINT: sets draining (workers finish current jobs, stop accepting new).
    Second SIGINT: sets stopped (workers abort immediately).
    SIGTERM: always sets stopped immediately.
    """
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)
    hit_count = 0

    def _handler(signum: int, frame: object) -> None:
        nonlocal hit_count
        hit_count += 1
        if signum == signal.SIGTERM or hit_count >= 2:
            controller.request_stop()
        else:
            controller.request_drain()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    try:
        yield controller
    finally:
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
