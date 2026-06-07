from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from PIL import Image

BgMode = Literal["auto", "white", "black"]


@dataclass(frozen=True)
class TransformSettings:
    crop: bool = False
    deskew: bool = False
    preview_long_edge: int = 1024
    bg_mode: BgMode = "auto"
    white_bg_thresh: int = 245
    black_bg_thresh: int = 10
    crop_padding_px: int = 8
    max_crop_margin_ratio: float = 0.25
    min_retained_area_ratio: float = 0.60
    min_box_px: int = 256
    deskew_max_angle_deg: float = 2.0
    deskew_step_deg: float = 0.5
    deskew_min_conf: float = 0.15


@dataclass(frozen=True)
class TransformResult:
    bg: Literal["white", "black"]
    crop_box_xyxy: tuple[int, int, int, int] | None
    crop_clamped: bool
    deskew_angle_deg: float | None
    deskew_confidence: float | None
    preview_w: int
    preview_h: int
    analysis_ms: float


def _tensor_chw_to_pil_rgb(img_u8_chw: torch.Tensor) -> Image.Image:
    if img_u8_chw.dtype != torch.uint8:
        raise ValueError("img_u8_chw must be uint8")
    if img_u8_chw.ndim != 3 or int(img_u8_chw.shape[0]) not in (1, 3, 4):
        raise ValueError("img_u8_chw must be CHW with 1/3/4 channels")

    x = img_u8_chw
    if x.device.type != "cpu":
        x = x.to("cpu")

    c = int(x.shape[0])
    arr = x.permute(1, 2, 0).contiguous().numpy()

    if c == 1:
        return Image.fromarray(arr[:, :, 0], mode="L").convert("RGB")
    if c == 3:
        return Image.fromarray(arr, mode="RGB")
    return Image.fromarray(arr, mode="RGBA").convert("RGB")


def _pil_rgb_to_tensor_chw_u8(im: Image.Image) -> torch.Tensor:
    im = im.convert("RGB")
    arr = np.asarray(im, dtype=np.uint8)
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return t


def _make_preview(im_rgb: Image.Image, long_edge: int) -> Image.Image:
    w, h = im_rgb.size
    le = max(w, h)
    if le <= long_edge:
        return im_rgb
    scale = float(long_edge) / float(le)
    nw = max(1, round(w * scale))
    nh = max(1, round(h * scale))
    return im_rgb.resize((nw, nh), resample=Image.Resampling.BILINEAR)


def _infer_bg_mode(gray_u8: np.ndarray) -> Literal["white", "black"]:
    h, w = gray_u8.shape
    b = max(1, min(8, h // 32, w // 32))
    top = gray_u8[:b, :]
    bot = gray_u8[-b:, :]
    left = gray_u8[:, :b]
    right = gray_u8[:, -b:]
    border = np.concatenate([top.ravel(), bot.ravel(), left.ravel(), right.ravel()])
    m = float(border.mean()) if border.size else 255.0
    return "white" if m >= 128.0 else "black"


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _apply_padding(
    box: tuple[int, int, int, int], w: int, h: int, pad: int
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return (max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad))


def _clamp_crop_box(
    *,
    src_w: int,
    src_h: int,
    box: tuple[int, int, int, int],
    settings: TransformSettings,
) -> tuple[tuple[int, int, int, int] | None, bool]:
    x0, y0, x1, y1 = box
    bw = max(0, x1 - x0)
    bh = max(0, y1 - y0)

    if bw < settings.min_box_px or bh < settings.min_box_px:
        return None, True

    area = float(bw) * float(bh)
    if area / max(1.0, float(src_w) * float(src_h)) < settings.min_retained_area_ratio:
        return None, True

    left = float(x0) / float(src_w)
    top = float(y0) / float(src_h)
    right = float(src_w - x1) / float(src_w)
    bottom = float(src_h - y1) / float(src_h)
    m = settings.max_crop_margin_ratio
    if left > m or top > m or right > m or bottom > m:
        return None, True

    return box, False


def _deskew_score(gray_u8: np.ndarray) -> float:
    proj = gray_u8.astype(np.float32).sum(axis=1)
    d = np.diff(proj)
    return float(np.mean(np.abs(d)))


def _estimate_deskew_angle(
    gray_u8: np.ndarray,
    *,
    bg: Literal["white", "black"],
    settings: TransformSettings,
) -> tuple[float | None, float | None]:
    if not settings.deskew:
        return None, None

    max_a = float(settings.deskew_max_angle_deg)
    step = float(settings.deskew_step_deg)
    if max_a <= 0 or step <= 0:
        return None, None

    base = Image.fromarray(gray_u8, mode="L")
    fill = 255 if bg == "white" else 0

    best_a: float | None = None
    best_s = -1.0
    second_s = -1.0

    a = -max_a
    while a <= (max_a + 1e-9):
        imr = base.rotate(a, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=fill)
        arr = np.asarray(imr, dtype=np.uint8)
        s = _deskew_score(arr)
        if s > best_s:
            second_s = best_s
            best_s = s
            best_a = a
        elif s > second_s:
            second_s = s
        a += step

    if best_a is None or best_s <= 0:
        return None, None

    conf = 0.0
    if second_s > 0:
        conf = float((best_s - second_s) / max(1e-9, best_s))

    if abs(best_a) < 1e-6:
        return 0.0, conf

    return float(best_a), conf


def analyze_transform(
    img_u8_chw: torch.Tensor,
    *,
    src_w: int,
    src_h: int,
    settings: TransformSettings,
) -> TransformResult:
    t0 = time.perf_counter()

    im = _tensor_chw_to_pil_rgb(img_u8_chw)
    prev = _make_preview(im, int(settings.preview_long_edge))
    pw, ph = prev.size

    gray = np.asarray(prev.convert("L"), dtype=np.uint8)

    if settings.bg_mode == "white":
        bg = "white"
    elif settings.bg_mode == "black":
        bg = "black"
    else:
        bg = _infer_bg_mode(gray)

    crop_box_full: tuple[int, int, int, int] | None = None
    crop_clamped = False

    if settings.crop:
        if bg == "white":
            fg = gray < np.uint8(settings.white_bg_thresh)
        else:
            fg = gray > np.uint8(settings.black_bg_thresh)

        bbox = _bbox_from_mask(fg)
        if bbox is not None:
            bbox = _apply_padding(bbox, pw, ph, int(settings.crop_padding_px))

            sx = float(src_w) / float(pw)
            sy = float(src_h) / float(ph)
            x0 = round(bbox[0] * sx)
            y0 = round(bbox[1] * sy)
            x1 = round(bbox[2] * sx)
            y1 = round(bbox[3] * sy)
            x0, x1 = max(0, min(src_w, x0)), max(0, min(src_w, x1))
            y0, y1 = max(0, min(src_h, y0)), max(0, min(src_h, y1))
            if x1 > x0 and y1 > y0:
                box = (x0, y0, x1, y1)
                box2, clamped = _clamp_crop_box(
                    src_w=src_w, src_h=src_h, box=box, settings=settings
                )
                crop_box_full = box2
                crop_clamped = clamped

    deskew_angle, deskew_conf = _estimate_deskew_angle(gray, bg=bg, settings=settings)

    ms = (time.perf_counter() - t0) * 1000.0
    return TransformResult(
        bg=bg,
        crop_box_xyxy=crop_box_full,
        crop_clamped=bool(crop_clamped),
        deskew_angle_deg=deskew_angle,
        deskew_confidence=deskew_conf,
        preview_w=int(pw),
        preview_h=int(ph),
        analysis_ms=float(ms),
    )


def apply_transform(
    img_u8_chw: torch.Tensor,
    *,
    result: TransformResult,
    settings: TransformSettings,
) -> torch.Tensor:
    x = img_u8_chw
    if x.device.type != "cpu":
        x = x.to("cpu")

    if settings.crop and result.crop_box_xyxy is not None:
        x0, y0, x1, y1 = result.crop_box_xyxy
        x = x[:, int(y0) : int(y1), int(x0) : int(x1)].contiguous()

    if settings.deskew and result.deskew_angle_deg is not None:
        ang = float(result.deskew_angle_deg)
        conf = float(result.deskew_confidence or 0.0)
        if abs(ang) > 1e-6 and conf >= float(settings.deskew_min_conf):
            im = _tensor_chw_to_pil_rgb(x)
            fill = (255, 255, 255) if result.bg == "white" else (0, 0, 0)
            im2 = im.rotate(ang, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=fill)
            x = _pil_rgb_to_tensor_chw_u8(im2)

    return x
