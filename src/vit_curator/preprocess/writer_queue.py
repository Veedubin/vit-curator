"""Bounded async writer queue for derivative image files.

Workers perform filesystem operations (link/copy or encode+write) and emit
results. DB updates are intentionally *not* performed in worker threads.
"""

from __future__ import annotations

import io
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Literal

import torch
from PIL import Image
from torchvision.io import encode_jpeg, encode_png

from vit_curator.config import LinkMode
from vit_curator.shared.errors import ERR_WRITE

WriteKind = Literal["link", "encode"]


@dataclass(frozen=True)
class WriteJob:
    """A single write job submitted to the queue."""

    kind: WriteKind
    dst_path: str
    file_pk: int

    # For derivatives (kind=="encode")
    deriv_pk: int | None = None
    preset_id: int | None = None

    # For kind=="link"
    src_path: str | None = None
    link_mode: LinkMode | None = None

    # For kind=="encode"
    img_u8_chw: torch.Tensor | None = None
    fmt: str | None = None
    jpeg_quality: int | None = None


@dataclass(frozen=True)
class WriteResult:
    """Result of a completed write job."""

    kind: WriteKind
    file_pk: int
    deriv_pk: int | None
    preset_id: int | None
    ok: bool
    err_code: int | None
    err_msg: str | None
    dst_path: str


def ensure_dir(p: Path) -> None:
    """Ensure directory exists."""
    p.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: LinkMode) -> None:
    """Create a file at dst as a link or copy of src."""
    if dst.exists():
        return

    ensure_dir(dst.parent)

    if mode.mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass

    if mode.mode == "symlink":
        try:
            os.symlink(src, dst)
            return
        except OSError:
            pass

    import shutil  # noqa: PLC0415

    shutil.copy2(src, dst)


class WriterQueue:
    """Bounded async writer queue.

    Worker threads perform filesystem operations (link/copy or encode+write)
    and emit results. DB updates are intentionally *not* performed in worker
    threads.
    """

    def __init__(self, *, num_workers: int, max_jobs: int) -> None:
        self._jobs: Queue[WriteJob | None] = Queue(maxsize=max(1, int(max_jobs)))
        self._results: Queue[WriteResult] = Queue()
        self._workers: list[Thread] = []
        self._closed = False

        for i in range(max(1, int(num_workers))):
            t = Thread(target=self._worker_loop, name=f"writer-{i}", daemon=True)
            t.start()
            self._workers.append(t)

    def backlog(self) -> int:
        return self._jobs.qsize()

    def submit(self, job: WriteJob) -> None:
        if self._closed:
            raise RuntimeError("WriterQueue is closed")
        self._jobs.put(job, block=True)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for _ in self._workers:
            self._jobs.put(None, block=True)

    def join(self) -> None:
        self._jobs.join()
        for t in self._workers:
            t.join(timeout=60)

    def drain_results(self, *, max_items: int = 10_000) -> list[WriteResult]:
        out: list[WriteResult] = []
        for _ in range(max_items):
            try:
                out.append(self._results.get_nowait())
            except Empty:
                break
        return out

    def _emit(self, res: WriteResult) -> None:
        self._results.put(res)

    def _worker_loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                self._jobs.task_done()
                return

            try:
                if job.kind == "link":
                    self._do_link(job)
                    self._emit(
                        WriteResult(
                            kind=job.kind,
                            file_pk=job.file_pk,
                            deriv_pk=job.deriv_pk,
                            preset_id=job.preset_id,
                            ok=True,
                            err_code=None,
                            err_msg=None,
                            dst_path=job.dst_path,
                        )
                    )
                elif job.kind == "encode":
                    self._do_encode(job)
                    self._emit(
                        WriteResult(
                            kind=job.kind,
                            file_pk=job.file_pk,
                            deriv_pk=job.deriv_pk,
                            preset_id=job.preset_id,
                            ok=True,
                            err_code=None,
                            err_msg=None,
                            dst_path=job.dst_path,
                        )
                    )
                else:
                    raise ValueError(f"Unknown kind: {job.kind}")
            except Exception as e:
                self._emit(
                    WriteResult(
                        kind=job.kind,
                        file_pk=job.file_pk,
                        deriv_pk=job.deriv_pk,
                        preset_id=job.preset_id,
                        ok=False,
                        err_code=int(ERR_WRITE),
                        err_msg=f"{type(e).__name__}: {e}",
                        dst_path=job.dst_path,
                    )
                )
            finally:
                self._jobs.task_done()

    def _do_link(self, job: WriteJob) -> None:
        if job.src_path is None or job.link_mode is None:
            raise ValueError("link job missing src_path/link_mode")
        link_or_copy(Path(job.src_path), Path(job.dst_path), job.link_mode)

    @staticmethod
    def _tensor_to_pil(img_u8_chw: torch.Tensor) -> Image.Image:
        img = img_u8_chw
        if img.dtype != torch.uint8:
            raise ValueError("img_u8_chw must be uint8")
        if img.device.type != "cpu":
            img = img.to("cpu")

        if img.ndim != 3:
            raise ValueError("img_u8_chw must be CHW")

        c = int(img.shape[0])
        arr = img.permute(1, 2, 0).contiguous().numpy()  # HWC

        if c == 1:
            return Image.fromarray(arr[:, :, 0], mode="L")
        if c == 3:
            return Image.fromarray(arr, mode="RGB")
        if c == 4:
            return Image.fromarray(arr, mode="RGBA")

        # Fallback: take first channel
        return Image.fromarray(arr[:, :, 0], mode="L")

    def _do_encode(self, job: WriteJob) -> None:
        if job.img_u8_chw is None or job.fmt is None:
            raise ValueError("encode job missing img_u8_chw/fmt")

        dst = Path(job.dst_path)
        if dst.exists():
            return

        ensure_dir(dst.parent)

        img = job.img_u8_chw
        fmt = str(job.fmt).lower()

        data: bytes
        if fmt in ("jpeg", "jpg"):
            # torchvision encode_jpeg supports 1- or 3-channel CHW uint8
            x = img
            if x.dtype != torch.uint8:
                raise ValueError("img_u8_chw must be uint8")
            if x.device.type != "cpu":
                x = x.to("cpu")
            quality = int(job.jpeg_quality or 80)
            blob = encode_jpeg(x.contiguous(), quality=quality)
            data = bytes(blob)
        elif fmt == "png":
            x = img
            if x.dtype != torch.uint8:
                raise ValueError("img_u8_chw must be uint8")
            if x.device.type != "cpu":
                x = x.to("cpu")
            blob = encode_png(x.contiguous())
            data = bytes(blob)
        elif fmt in ("webp", "tiff", "tif"):
            # Preserve-source formats are slower and go through Pillow.
            pil = self._tensor_to_pil(img)
            buf = io.BytesIO()
            save_kwargs: dict = {}
            if fmt == "webp":
                save_kwargs["quality"] = int(job.jpeg_quality or 80)
            pil.save(buf, format=fmt.upper(), **save_kwargs)
            data = buf.getvalue()
        else:
            raise ValueError(f"Unsupported fmt: {fmt}")

        tmp = dst.with_suffix(dst.suffix + f".tmp.{os.getpid()}.{threading.get_ident()}")
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, dst)


def out_name(bucket_pos: int, file_pk: int, ext: str) -> str:
    """Generate output filename for a file.

    Format: {bucket_pos:06d}_{file_pk:012d}{ext}

    Args:
        bucket_pos: Position within bucket.
        file_pk: File primary key.
        ext: File extension.

    Returns:
        Formatted filename string.
    """
    return f"{bucket_pos:06d}_{file_pk:012d}{ext}"
