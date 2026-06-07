"""Image decode backends — CPU (PIL+torch) and optional DALI (GPU).

CPU backend is always available. DALI backend requires the nvidia-dali-cuda120
optional extra and is only imported on demand.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------------------
# CPU decode
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecodedImage:
    """Result of CPU image decode."""

    img_u8_chw: torch.Tensor
    width: int
    height: int


def decode_rgb_u8_chw(path: Path) -> DecodedImage:
    """Decode an image file into an RGB uint8 CHW torch Tensor (CPU).

    Raises:
      - UnidentifiedImageError for non-images
      - Other exceptions for I/O, truncated images, etc.
    """
    with Image.open(path) as img:
        try:
            if getattr(img, "n_frames", 1) > 1:
                img.seek(0)
        except Exception:
            pass
        img_rgb = img.convert("RGB")
        w, h = img_rgb.size
        arr = np.array(img_rgb, dtype=np.uint8, copy=True)  # HWC writable copy

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise UnidentifiedImageError(f"Unexpected decoded shape: {arr.shape}")

    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW uint8
    return DecodedImage(img_u8_chw=t, width=int(w), height=int(h))


# ---------------------------------------------------------------------------
# Decode backend protocol + batch result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecodedBatch:
    """Batch decode result from a DecodeBackend."""

    labels: list[int]
    imgs_u8_chw: list[torch.Tensor]


class DecodeBackend:
    """Protocol: decode backend interface.

    Backends may implement only a subset of formats; callers should
    provide fallbacks.
    """

    def decode_resize_jpeg_rgb_u8_chw(
        self,
        paths: Sequence[Path],
        labels: Sequence[int],
        *,
        out_w: int,
        out_h: int,
    ) -> DecodedBatch: ...


# ---------------------------------------------------------------------------
# DALI decode (optional)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DaliBatchResult:
    """DALI batch decode+resize result."""

    labels: list[int]
    imgs_u8_chw: list[torch.Tensor]


@dataclass
class DaliDerivativeGenerator:
    """Generate derivative images using NVIDIA DALI for GPU-accelerated decode+resize."""

    batch_size: int
    device: str
    threads: int = 4
    preserve_color: bool = True

    def run(self, paths: Sequence[str], presets: Sequence[object]) -> list[list[torch.Tensor]]:
        path_list = list(paths)
        preset_list = list(presets)
        if not path_list or not preset_list:
            return [[] for _ in range(len(path_list))]

        results: list[list[torch.Tensor]] = [[] for _ in range(len(path_list))]
        chunk_size = max(1, int(self.batch_size))
        device_id = self._device_id()

        for start in range(0, len(path_list), chunk_size):
            chunk_paths = path_list[start : start + chunk_size]
            encoded = [
                np.frombuffer(Path(p).read_bytes(), dtype=np.uint8).copy() for p in chunk_paths
            ]
            labels = list(range(len(encoded)))

            per_preset: list[list[torch.Tensor]] = []
            for preset in preset_list:
                res = dali_decode_resize_jpeg_rgb_u8_chw(
                    encoded,
                    labels,
                    out_w=int(preset.width),
                    out_h=int(preset.height),
                    device_id=device_id,
                    num_threads=int(self.threads),
                )
                imgs = res.imgs_u8_chw
                if not self.preserve_color:
                    imgs = [self._to_gray(img) for img in imgs]
                per_preset.append(imgs)

            for i in range(len(chunk_paths)):
                results[start + i] = [per_preset[j][i] for j in range(len(per_preset))]

        return results

    def _device_id(self) -> int:
        if str(self.device).lower() in ("cuda", "gpu"):
            return 0
        return 0

    @staticmethod
    def _to_gray(img_u8_chw: torch.Tensor) -> torch.Tensor:
        if img_u8_chw.ndim != 3:
            return img_u8_chw
        c = int(img_u8_chw.shape[0])
        if c == 1:
            return img_u8_chw
        if c < 3:
            return img_u8_chw[:1]
        r = img_u8_chw[0].to(dtype=torch.float32)
        g = img_u8_chw[1].to(dtype=torch.float32)
        b = img_u8_chw[2].to(dtype=torch.float32)
        y = (0.299 * r + 0.587 * g + 0.114 * b).round().clamp(0.0, 255.0).to(torch.uint8)
        return y.unsqueeze(0).contiguous()


def _require_dali() -> None:
    """Import DALI and raise RuntimeError if not available."""
    try:
        from nvidia.dali import (  # noqa: PLC0415
            fn,  # noqa: F401
            types,  # noqa: F401
        )
        from nvidia.dali.pipeline import Pipeline  # noqa: F401, PLC0415
    except Exception as e:
        raise RuntimeError(
            "DALI decode backend requested but NVIDIA DALI is not importable. "
            "Install the appropriate CUDA-specific DALI wheel (e.g., `uv sync --extra dali`)."
        ) from e


def dali_decode_resize_jpeg_rgb_u8_chw(
    encoded_jpegs: list[np.ndarray],
    labels: list[int],
    *,
    out_w: int,
    out_h: int,
    device_id: int = 0,
    num_threads: int = 2,
) -> DaliBatchResult:
    """Decode+resize a batch of JPEG byte arrays on GPU via DALI."""
    _require_dali()

    from nvidia.dali import fn, types  # noqa: PLC0415
    from nvidia.dali.pipeline import Pipeline  # noqa: PLC0415

    if len(encoded_jpegs) != len(labels):
        raise ValueError("encoded_jpegs and labels must have same length")

    bs = len(encoded_jpegs)

    class _Pipe(Pipeline):
        def __init__(self) -> None:
            super().__init__(batch_size=bs, num_threads=num_threads, device_id=device_id)

        def define_graph(self) -> None:
            self._encoded = fn.external_source(name="encoded", device="cpu")
            self._labels = fn.external_source(name="labels", device="cpu")
            imgs = fn.decoders.image(self._encoded, device="mixed", output_type=types.RGB)
            imgs = fn.resize(imgs, resize_x=int(out_w), resize_y=int(out_h))
            return imgs, self._labels

    pipe = _Pipe()
    pipe.build()

    pipe.feed_input("encoded", encoded_jpegs)
    pipe.feed_input("labels", np.asarray(labels, dtype=np.int32))

    out_imgs, out_labels = pipe.run()

    # Convert to CPU numpy for handoff to torch encode/write.
    imgs_hwc = out_imgs.as_cpu().as_array()  # (N,H,W,3) uint8
    lbls = out_labels.as_array().tolist()

    t = torch.from_numpy(imgs_hwc).permute(0, 3, 1, 2).contiguous()
    return DaliBatchResult(
        labels=[int(x) for x in lbls], imgs_u8_chw=[t[i] for i in range(t.shape[0])]
    )
