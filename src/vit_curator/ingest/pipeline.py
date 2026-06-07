"""Stage 0: Download, extract, and sort archives.

Ported from data_janitor.ingest.pipeline and file-helper-pipeline.workers.
Handles URL downloading, archive extraction, and bucket-based sorting
using threaded workers connected by queues.
"""

from __future__ import annotations

import hashlib
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from vit_curator.config import IngestConfig
from vit_curator.shared.errors import IngestError

from .archive import extract_archive, is_archive
from .fsops import ensure_dir, link_or_copy, move_into, safe_relpath
from .sorters import BucketLayout, choose_bucket, collision_safe_target
from .state import IngestState
from .urls import iter_urls

try:
    import requests  # type: ignore[import-untyped]
except Exception:
    requests = None  # type: ignore[assignment]


@dataclass(frozen=True)
class WorkLayout:
    dest_dir: Path

    @property
    def work_root(self) -> Path:
        return self.dest_dir / "_dj_work"

    @property
    def downloads_dir(self) -> Path:
        return self.work_root / "downloads"

    @property
    def extract_root(self) -> Path:
        return self.work_root / "extract"

    @property
    def state_db(self) -> Path:
        return self.work_root / "ingest_state.sqlite"

    @property
    def sorted_root(self) -> Path:
        return self.dest_dir / "sorted"


@dataclass(frozen=True)
class FileTask:
    kind: str
    src: str
    local_path: Path
    preserve_local: bool


@dataclass
class IngestMetrics:
    total_download: int = 0
    total_unarchive: int = 0
    download_done: int = 0
    download_err: int = 0
    unarchive_done: int = 0
    unarchive_err: int = 0
    sort_done: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def inc_download(self, ok: bool) -> None:
        with self._lock:
            if ok:
                self.download_done += 1
            else:
                self.download_err += 1

    def inc_unarchive(self, ok: bool) -> None:
        with self._lock:
            if ok:
                self.unarchive_done += 1
            else:
                self.unarchive_err += 1

    def inc_sort(self) -> None:
        with self._lock:
            self.sort_done += 1

    def snapshot(self) -> tuple[int, int, int, int, int, int, int]:
        with self._lock:
            return (
                self.total_download,
                self.total_unarchive,
                self.download_done,
                self.download_err,
                self.unarchive_done,
                self.unarchive_err,
                self.sort_done,
            )


def run_ingest(cfg: IngestConfig) -> Path:
    """Execute download/unarchive/sort and return the sorted root."""

    if cfg.download_urls_file and requests is None:
        raise IngestError("requests is required for --download mode")

    layout = WorkLayout(cfg.dest_dir)
    ensure_dir(layout.dest_dir)
    ensure_dir(layout.work_root)
    ensure_dir(layout.downloads_dir)
    ensure_dir(layout.extract_root)
    ensure_dir(layout.sorted_root)

    state = IngestState(layout.state_db)
    conn = state.open()

    bucket_layout = BucketLayout(sorted_root=layout.sorted_root)

    q_download: queue.Queue[object] = queue.Queue(maxsize=10_000)
    q_files: queue.Queue[object] = queue.Queue(maxsize=10_000)
    q_sort: queue.Queue[object] = queue.Queue(maxsize=50_000)

    stop = object()

    total_download = 0
    total_unarchive = 0

    if cfg.download_urls_file:
        for item in iter_urls(cfg.download_urls_file):
            state.upsert(conn, "url", item.url, "queued")
            q_download.put(item.url)
            total_download += 1
            total_unarchive += 1

    if cfg.unarchive_source_dir:
        src_root = cfg.unarchive_source_dir
        for p in src_root.rglob("*"):
            if not p.is_file():
                continue
            if is_archive(p) or cfg.include_non_archives_in_unarchive_mode:
                state.upsert(conn, "file", str(p), "queued")
                q_files.put(FileTask(kind="file", src=str(p), local_path=p, preserve_local=True))
                total_unarchive += 1

    for _ in range(cfg.download_workers):
        q_download.put(stop)

    metrics = IngestMetrics(total_download=total_download, total_unarchive=total_unarchive)

    dl_threads: list[threading.Thread] = []
    if cfg.download_urls_file:
        for i in range(cfg.download_workers):
            t = threading.Thread(
                target=_download_worker,
                name=f"up-ingest-download-{i}",
                args=(cfg, layout, state, metrics, q_download, q_files, stop),
                daemon=True,
            )
            t.start()
            dl_threads.append(t)

    un_threads: list[threading.Thread] = []
    for i in range(cfg.unarchive_workers):
        t = threading.Thread(
            target=_unarchive_worker,
            name=f"up-ingest-unarchive-{i}",
            args=(layout, state, metrics, q_files, q_sort, stop),
            daemon=True,
        )
        t.start()
        un_threads.append(t)

    sort_threads: list[threading.Thread] = []
    for i in range(cfg.sort_workers):
        t = threading.Thread(
            target=_sort_worker,
            name=f"up-ingest-sort-{i}",
            args=(bucket_layout, layout, metrics, q_sort, stop),
            daemon=True,
        )
        t.start()
        sort_threads.append(t)

    dl_stop_sent = False
    un_stop_sent = False

    if not dl_threads:
        for _ in range(cfg.unarchive_workers):
            q_files.put(stop)
        dl_stop_sent = True

    if metrics.total_download > 0 or metrics.total_unarchive > 0:
        console = Console()
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TextColumn("{task.fields[stats]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=5,
        )

        with progress:
            download_task = None
            if metrics.total_download > 0:
                download_task = progress.add_task(
                    "download",
                    total=metrics.total_download,
                    stats=f"0/{metrics.total_download}",
                )

            unarchive_task = None
            if metrics.total_unarchive > 0:
                unarchive_task = progress.add_task(
                    "unarchive",
                    total=metrics.total_unarchive,
                    stats=f"0/{metrics.total_unarchive}",
                )

            sort_task = progress.add_task("sort", total=None, stats="0")

            all_threads = dl_threads + un_threads + sort_threads
            dl_done = not dl_threads
            un_done = not un_threads
            while True:
                (
                    _total_dl,
                    _total_un,
                    done_dl,
                    err_dl,
                    done_un,
                    err_un,
                    done_sort,
                ) = metrics.snapshot()

                if download_task is not None:
                    progress.update(
                        download_task,
                        completed=done_dl + err_dl,
                        description=f"download (err {err_dl})",
                        stats=f"{done_dl + err_dl}/{metrics.total_download}",
                    )
                if unarchive_task is not None:
                    progress.update(
                        unarchive_task,
                        completed=done_un + err_un,
                        description=f"unarchive (err {err_un})",
                        stats=f"{done_un + err_un}/{metrics.total_unarchive}",
                    )

                progress.update(
                    sort_task,
                    completed=done_sort,
                    description="sort",
                    stats=f"{done_sort:,} processed",
                )

                if not dl_done and not any(t.is_alive() for t in dl_threads):
                    for _ in range(cfg.unarchive_workers):
                        q_files.put(stop)
                    dl_done = True
                    dl_stop_sent = True

                if dl_done and not un_done and not any(t.is_alive() for t in un_threads):
                    for _ in range(cfg.sort_workers):
                        q_sort.put(stop)
                    un_done = True
                    un_stop_sent = True

                if not any(t.is_alive() for t in all_threads):
                    break
                time.sleep(0.2)

    for t in dl_threads:
        t.join()

    if dl_threads and not dl_stop_sent:
        for _ in range(cfg.unarchive_workers):
            q_files.put(stop)

    for t in un_threads:
        t.join()

    if un_threads and not un_stop_sent:
        for _ in range(cfg.sort_workers):
            q_sort.put(stop)

    for t in sort_threads:
        t.join()

    conn.close()
    return layout.sorted_root


def _download_worker(
    cfg: IngestConfig,
    layout: WorkLayout,
    state: IngestState,
    metrics: IngestMetrics,
    q_download: queue.Queue[object],
    q_files: queue.Queue[object],
    stop: object,
) -> None:
    if requests is None:
        raise IngestError("requests is required for --download mode")

    conn = state.open()
    try:
        while True:
            item = q_download.get()
            try:
                if item is stop:
                    return
                url = str(item)

                base = (
                    url.split("?", maxsplit=1)[0].rstrip("/").split("/")[-1].strip()
                    or "download.bin"
                )
                h = hashlib.sha1(url.encode("utf-8", errors="ignore")).hexdigest()[:10]
                out = layout.downloads_dir / f"{base}__{h}"

                if out.exists() and out.stat().st_size > 0:
                    state.upsert(conn, "url", url, "downloaded", local_path=str(out))
                    q_files.put(FileTask(kind="url", src=url, local_path=out, preserve_local=True))
                    metrics.inc_download(True)
                    continue

                ok = False
                last_err: str | None = None
                for attempt in range(max(1, cfg.retries)):
                    try:
                        with requests.get(url, stream=True, timeout=cfg.timeout_s) as r:  # type: ignore[union-attr]
                            r.raise_for_status()
                            ensure_dir(out.parent)
                            tmp = out.with_suffix(out.suffix + ".part")
                            with tmp.open("wb") as f:
                                for chunk in r.iter_content(chunk_size=1024 * 1024):
                                    if chunk:
                                        f.write(chunk)
                            tmp.replace(out)
                        ok = True
                        break
                    except Exception as e:
                        last_err = str(e)
                        time.sleep(min(2**attempt, 10))

                if not ok:
                    state.upsert(conn, "url", url, "error", err=last_err)
                    metrics.inc_download(False)
                    continue

                state.upsert(conn, "url", url, "downloaded", local_path=str(out))
                q_files.put(FileTask(kind="url", src=url, local_path=out, preserve_local=True))
                metrics.inc_download(True)
            finally:
                q_download.task_done()
    finally:
        conn.close()


def _unarchive_worker(
    layout: WorkLayout,
    state: IngestState,
    metrics: IngestMetrics,
    q_files: queue.Queue[object],
    q_sort: queue.Queue[object],
    stop: object,
) -> None:
    conn = state.open()
    try:
        while True:
            item = q_files.get()
            try:
                if item is stop:
                    return
                task: FileTask = item  # type: ignore[assignment]
                p = task.local_path

                if not p.exists() or not p.is_file():
                    state.upsert(
                        conn, task.kind, task.src, "error", local_path=str(p), err="missing file"
                    )
                    metrics.inc_unarchive(False)
                    continue

                if is_archive(p):
                    job_id = hashlib.sha1(
                        (task.src + "|" + str(p)).encode("utf-8", errors="ignore")
                    ).hexdigest()[:12]
                    out_dir = layout.extract_root / job_id
                    marker = out_dir / ".extracted.ok"

                    if marker.exists():
                        for f in out_dir.rglob("*"):
                            if f.is_file() and f.name != marker.name:
                                q_sort.put((f, False, task.src))
                        state.upsert(conn, task.kind, task.src, "extracted", local_path=str(p))
                        metrics.inc_unarchive(True)
                        continue

                    ensure_dir(out_dir)
                    try:
                        extract_archive(p, out_dir)
                        marker.write_text("ok\n", encoding="utf-8")
                        for f in out_dir.rglob("*"):
                            if f.is_file() and f.name != marker.name:
                                q_sort.put((f, False, task.src))
                        state.upsert(conn, task.kind, task.src, "extracted", local_path=str(p))
                        metrics.inc_unarchive(True)
                    except Exception as e:
                        state.upsert(
                            conn, task.kind, task.src, "error", local_path=str(p), err=str(e)
                        )
                        metrics.inc_unarchive(False)
                else:
                    q_sort.put((p, task.preserve_local, task.src))
                    state.upsert(conn, task.kind, task.src, "sorted", local_path=str(p))
                    metrics.inc_unarchive(True)
            finally:
                q_files.task_done()
    finally:
        conn.close()


def _sort_worker(
    bucket_layout: BucketLayout,
    layout: WorkLayout,
    metrics: IngestMetrics,
    q_sort: queue.Queue[object],
    stop: object,
) -> None:
    while True:
        item = q_sort.get()
        try:
            if item is stop:
                return
            src_path, preserve, src_hint = item
            p = Path(src_path)
            if not p.exists() or not p.is_file():
                continue

            bucket = choose_bucket(p, bucket_layout)
            bucket_dir = bucket_layout.bucket_dir(bucket)
            ensure_dir(bucket_dir)

            try:
                rel = safe_relpath(layout.extract_root, p)
            except Exception:
                rel = Path(p.name)

            dst = collision_safe_target(bucket_dir, rel, src_hint=str(src_hint))

            if preserve:
                link_or_copy(p, dst)
            else:
                move_into(p, dst)
            metrics.inc_sort()
        finally:
            q_sort.task_done()
