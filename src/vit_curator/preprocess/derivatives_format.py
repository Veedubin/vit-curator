"""Format and image-transformation helpers for derivative generation.

Extracted from derivatives.py for clarity. These are pure functions with no
DB or pipeline coupling.
"""

from __future__ import annotations

import os

import torch

from vit_curator.config import RunConfig
from vit_curator.shared.db import Preset

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def resize_u8_chw(img_u8_chw: torch.Tensor, *, out_w: int, out_h: int, device: str) -> torch.Tensor:
    """Resize a CHW uint8 tensor to (out_h, out_w) using bilinear interpolation."""
    import torch.nn.functional as F  # noqa: PLC0415

    x = img_u8_chw.to(device=device)
    x = x.unsqueeze(0).float() / 255.0
    y = F.interpolate(x, size=(int(out_h), int(out_w)), mode="bilinear", align_corners=False)
    y = (y.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    return y.squeeze(0).to("cpu")


def maybe_grayscale_u8_chw(img_u8_chw: torch.Tensor, *, preserve_color: bool) -> torch.Tensor:
    """Optionally convert CHW uint8 tensor to grayscale."""
    if preserve_color:
        return img_u8_chw

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


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------


def ext_for_fmt(fmt: str) -> str:
    f = fmt.lower()
    if f in ("jpeg", "jpg"):
        return ".jpg"
    if f == "png":
        return ".png"
    if f == "webp":
        return ".webp"
    if f in ("tif", "tiff"):
        return ".tif"
    raise ValueError(f"Unsupported fmt: {fmt}")


def fmt_from_ext(ext: str) -> str:
    e = ext.lower()
    if e in (".jpg", ".jpeg"):
        return "jpeg"
    if e == ".png":
        return "png"
    if e == ".webp":
        return "webp"
    if e in (".tif", ".tiff"):
        return "tiff"
    return "jpeg"


def select_out_fmt_and_ext(cfg: RunConfig, ext_blob: bytes, preset: Preset) -> tuple[str, str]:
    """Choose output format and extension for a derivative."""
    preset_fmt = str(preset.fmt).lower()
    preset_ext = ext_for_fmt(preset_fmt)

    if not cfg.preserve_source:
        return preset_fmt, preset_ext

    src_ext_raw = os.fsdecode(ext_blob)
    src_ext = src_ext_raw.lower()
    if src_ext in (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"):
        return fmt_from_ext(src_ext), src_ext_raw

    return preset_fmt, preset_ext
