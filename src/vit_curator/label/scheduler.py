"""Auto-tune and dynamic concurrency for VLM dispatch."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Ema:
    halflife_s: float
    value: float | None = None
    t_last: float | None = None

    def update(self, x: float) -> float:
        t = time.perf_counter()
        if self.value is None:
            self.value = x
            self.t_last = t
            return x
        assert self.t_last is not None
        dt = max(1e-9, t - self.t_last)
        alpha = 1.0 - 0.5 ** (dt / max(1e-9, self.halflife_s))
        self.value = (1 - alpha) * self.value + alpha * x
        self.t_last = t
        assert self.value is not None
        return self.value


@dataclass
class DynamicConcurrency:
    min_inflight: int
    max_inflight: int
    target_p95_ms: float = 2500.0
    ema_halflife_s: float = 30.0

    def __post_init__(self) -> None:
        self.p95_ema = Ema(self.ema_halflife_s)

    def suggest(self, current: int, p95_ms: float) -> int:
        p95 = self.p95_ema.update(p95_ms)
        nxt = current
        if p95 < self.target_p95_ms * 0.85:
            nxt = current + 4
        elif p95 > self.target_p95_ms * 1.10:
            nxt = current - 4
        return max(self.min_inflight, min(self.max_inflight, nxt))


@dataclass
class AutoTune:
    min_inflight: int
    max_inflight: int
    min_batch_size: int
    max_batch_size: int
    inflight_step: int = 4
    batch_step: int = 16
    target_p95_ms: float = 2500.0
    target_ttft_ms: float | None = None
    min_tok_s: float | None = None
    max_err_rate: float = 5.0
    warmup_batches: int = 2
    cooldown_s: float = 2.0

    batches_seen: int = 0
    last_tune: float | None = None
    decision_log: list[dict[str, str | float]] | None = None
    max_log_entries: int = 100

    def __post_init__(self) -> None:
        if self.decision_log is None:
            self.decision_log = []

    def note_batch(self) -> None:
        self.batches_seen += 1

    def get_state(self) -> dict[str, bool | float | int]:
        """Expose current tuner state for dashboard."""
        now = time.perf_counter()
        cooldown_remaining = 0.0
        if self.last_tune is not None:
            cooldown_remaining = max(0.0, self.cooldown_s - (now - self.last_tune))

        return {
            "batches_seen": self.batches_seen,
            "warmup_complete": self.batches_seen >= self.warmup_batches,
            "last_tune": self.last_tune or 0.0,
            "cooldown_remaining_s": cooldown_remaining,
        }

    def _log_decision(self, action: str, reason: str) -> None:
        entry = {
            "time": time.time(),
            "action": action,
            "reason": reason,
        }
        self.decision_log.append(entry)
        if self.decision_log is not None and len(self.decision_log) > self.max_log_entries:
            self.decision_log = self.decision_log[-self.max_log_entries :]

    def suggest(
        self,
        *,
        current_inflight: int,
        current_batch: int,
        p95_ms: float,
        ttft_p95_ms: float | None,
        tok_s: float | None,
        err_rate: float,
    ) -> tuple[int, int]:
        now = time.perf_counter()
        if self.batches_seen < self.warmup_batches:
            self._log_decision(
                "warmup",
                f"waiting for warmup ({self.batches_seen}/{self.warmup_batches})",
            )
            return current_inflight, current_batch
        if self.last_tune is not None and now - self.last_tune < self.cooldown_s:
            remaining = self.cooldown_s - (now - self.last_tune)
            self._log_decision("cooldown", f"cooldown active ({remaining:.1f}s remaining)")
            return current_inflight, current_batch

        inflight = current_inflight
        batch_size = current_batch
        too_slow = p95_ms > self.target_p95_ms * 1.10
        if self.target_ttft_ms is not None and ttft_p95_ms is not None:
            too_slow = too_slow or ttft_p95_ms > self.target_ttft_ms * 1.10

        if err_rate > self.max_err_rate:
            inflight -= self.inflight_step
            batch_size -= self.batch_step
            self._log_decision(
                "scale_down",
                f"err_rate {err_rate:.1f}% > {self.max_err_rate}%",
            )
        elif too_slow:
            inflight -= self.inflight_step
            batch_size -= self.batch_step
            reason = f"p95 {p95_ms:.1f}ms > target {self.target_p95_ms:.1f}ms"
            if self.target_ttft_ms and ttft_p95_ms and ttft_p95_ms > self.target_ttft_ms * 1.10:
                reason += f" or ttft_p95 {ttft_p95_ms:.1f}ms > target {self.target_ttft_ms:.1f}ms"
            self._log_decision("scale_down", reason)
        elif (
            p95_ms < self.target_p95_ms * 0.85
            and err_rate <= self.max_err_rate
            and (
                self.target_ttft_ms is None
                or ttft_p95_ms is None
                or ttft_p95_ms < self.target_ttft_ms * 0.85
            )
            and (self.min_tok_s is None or tok_s is None or tok_s >= self.min_tok_s)
        ):
            inflight += self.inflight_step
            batch_size += self.batch_step
            self._log_decision(
                "scale_up",
                f"p95 {p95_ms:.1f}ms < target {self.target_p95_ms * 0.85:.1f}ms, "
                f"err_rate {err_rate:.1f}% OK",
            )
        else:
            self._log_decision("hold", f"metrics within acceptable range (p95 {p95_ms:.1f}ms)")

        inflight = max(self.min_inflight, min(self.max_inflight, inflight))
        batch_size = max(self.min_batch_size, min(self.max_batch_size, batch_size))
        batch_size = max(batch_size, inflight)

        if inflight != current_inflight or batch_size != current_batch:
            self.last_tune = now

        return inflight, batch_size
