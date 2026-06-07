"""Derivative image generation — CPU and optional DALI paths.

This module merges the main pipeline orchestration from data_janitor:
  - pipeline_common.py (resize, grayscale helpers, format helpers)
  - pipeline_db.py (pump_results, upsert_derivative_pending, mark_derivative_error)
  - pipeline_cpu.py (run_derivatives_cpu)
  - pipeline_dali.py (run_derivatives_dali_then_cpu, run_derivatives_dali)
  - pipeline.py (run_pipeline main orchestrator)

All DB operations use vit_curator.shared.db functions and the unified
schema (file_pk, not asset_id).
"""

from __future__ import annotations

import json
import os
import time

import duckdb
import torch
import xxhash
from PIL import UnidentifiedImageError
from rich.console import Console

from vit_curator.config import RunConfig
from vit_curator.preprocess.bucket import iter_bucket_assignments
from vit_curator.preprocess.decode import DaliDerivativeGenerator, decode_rgb_u8_chw
from vit_curator.preprocess.dedupe import hash_and_mark_dupes
from vit_curator.preprocess.scan import scan_into_duckdb
from vit_curator.preprocess.transform import (
    TransformResult,
    TransformSettings,
    apply_transform,
)
from vit_curator.preprocess.writer_queue import (
    WriteJob,
    WriteResult,
    WriterQueue,
    out_name,
)
from vit_curator.shared.db import (
    Preset,
    connect,
    ensure_preset_rows,
    get_or_create_transform_cfg,
    load_presets,
    next_deriv_pk,
    next_file_pk,
    next_transform_run_pk,
)
from vit_curator.shared.errors import ERR_DECODE

# ---------------------------------------------------------------------------
# Format helpers (from pipeline_common.py)
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


# ---------------------------------------------------------------------------
# DB helpers (from pipeline_db.py)
# ---------------------------------------------------------------------------


def pump_results(
    con: duckdb.DuckDBPyConnection,
    wq: WriterQueue,
    *,
    console: Console | None = None,
    max_items: int = 50_000,
) -> int:
    """Drain writer queue results and update DB status."""
    results = wq.drain_results(max_items=max_items)
    if not results:
        return 0

    now_ns = time.time_ns()
    deriv_updates: list[tuple[int, int | None, str | None, int]] = []
    link_errs: list[WriteResult] = []

    for r in results:
        if r.kind == "encode" and r.deriv_pk is not None:
            if r.ok:
                deriv_updates.append((1, None, None, int(r.deriv_pk)))
            else:
                deriv_updates.append(
                    (2, int(r.err_code or 0), str(r.err_msg or ""), int(r.deriv_pk))
                )
        elif r.kind == "link" and not r.ok:
            link_errs.append(r)

    if deriv_updates:
        con.executemany(
            "UPDATE image_derivatives SET status=?, err_code=?, err_msg=?, created_at_ns=? "
            "WHERE deriv_pk=?;",
            [(st, ec, em, now_ns, pk) for (st, ec, em, pk) in deriv_updates],
        )

    if link_errs and console is not None:
        for r in link_errs[:5]:
            console.print(
                f"[yellow]orig[/yellow] write failed {r.dst_path}: {r.err_msg}",
                highlight=False,
            )
        if len(link_errs) > 5:
            console.print(
                f"[yellow]orig[/yellow] write failed total={len(link_errs)}",
                highlight=False,
            )

    return len(results)


def _update_transform_ok(
    con: duckdb.DuckDBPyConnection,
    run_id: int,
    now_ns: int,
    res: TransformResult,
) -> None:
    con.execute(
        "UPDATE file_transform_runs SET status=1, err_code=NULL, err_msg=NULL, bg=?, "
        "crop_x0=?, crop_y0=?, crop_x1=?, crop_y1=?, crop_clamped=?, "
        "deskew_angle_deg=?, deskew_confidence=?, preview_w=?, preview_h=?, "
        "analysis_ms=?, updated_at_ns=? "
        "WHERE run_id=?;",
        [
            str(res.bg),
            (int(res.crop_box_xyxy[0]) if res.crop_box_xyxy else None),
            (int(res.crop_box_xyxy[1]) if res.crop_box_xyxy else None),
            (int(res.crop_box_xyxy[2]) if res.crop_box_xyxy else None),
            (int(res.crop_box_xyxy[3]) if res.crop_box_xyxy else None),
            bool(res.crop_clamped),
            (float(res.deskew_angle_deg) if res.deskew_angle_deg is not None else 0.0),
            (float(res.deskew_confidence) if res.deskew_confidence is not None else 0.0),
            int(res.preview_w),
            int(res.preview_h),
            float(res.analysis_ms),
            now_ns,
            int(run_id),
        ],
    )


def _update_transform_err(
    con: duckdb.DuckDBPyConnection, run_id: int, now_ns: int, err: Exception
) -> None:
    con.execute(
        "UPDATE file_transform_runs SET status=2, err_code=?, err_msg=?, "
        "updated_at_ns=? WHERE run_id=?;",
        [int(ERR_DECODE), str(err)[:1000], now_ns, int(run_id)],
    )


def get_or_compute_transform_run(
    con: duckdb.DuckDBPyConnection,
    *,
    file_pk: int,
    transform_cfg_id: int,
    img_u8_chw: torch.Tensor,
    src_w: int,
    src_h: int,
    tsettings: TransformSettings,
) -> tuple[int, TransformResult]:
    """Return existing transform run or compute and store a new one."""
    row = con.execute(
        "SELECT run_id, status, err_code, err_msg, bg, crop_x0, crop_y0, crop_x1, crop_y1, "
        "crop_clamped, deskew_angle_deg, deskew_confidence, preview_w, preview_h, analysis_ms "
        "FROM file_transform_runs WHERE file_pk=? AND transform_cfg_id=?;",
        [int(file_pk), int(transform_cfg_id)],
    ).fetchone()

    now_ns = int(time.time_ns())
    if row and row[0] is not None:
        run_id = int(row[0])
        status = int(row[1] or 0)
        if status == 1:
            bg = str(row[4] or "white")
            crop_box = None
            if (
                row[5] is not None
                and row[6] is not None
                and row[7] is not None
                and row[8] is not None
            ):
                crop_box = (int(row[5]), int(row[6]), int(row[7]), int(row[8]))
            res = TransformResult(
                bg=("black" if bg.lower().startswith("b") else "white"),
                crop_box_xyxy=crop_box,
                crop_clamped=bool(row[9] or False),
                deskew_angle_deg=(float(row[10]) if row[10] is not None else 0.0),
                deskew_confidence=(float(row[11]) if row[11] is not None else 0.0),
                preview_w=int(row[12] or 0),
                preview_h=int(row[13] or 0),
                analysis_ms=float(row[14] or 0.0),
            )
            return run_id, res

        con.execute(
            "UPDATE file_transform_runs SET status=0, err_code=NULL, err_msg=NULL, "
            "updated_at_ns=? WHERE run_id=?;",
            [now_ns, run_id],
        )
    else:
        run_id = next_transform_run_pk(con)
        con.execute(
            "INSERT INTO file_transform_runs "
            "(run_id, file_pk, transform_cfg_id, status, created_at_ns, updated_at_ns) "
            "VALUES (?, ?, ?, 0, ?, ?);",
            [int(run_id), int(file_pk), int(transform_cfg_id), now_ns, now_ns],
        )

    try:
        res = apply_transform(img_u8_chw, src_w=src_w, src_h=src_h, settings=tsettings)
        _update_transform_ok(con, int(run_id), now_ns, res)
        return int(run_id), res
    except Exception as e:
        _update_transform_err(con, int(run_id), now_ns, e)
        raise


def upsert_derivative_pending(
    con: duckdb.DuckDBPyConnection,
    *,
    deriv_pk: int,
    file_pk: int,
    preset: Preset,
    transform_cfg_id: int,
    run_id: int | None,
    out_rel_blob: bytes,
    out_fmt: str,
) -> int:
    """Insert or update a derivative row as pending."""
    row = con.execute(
        "SELECT deriv_pk FROM image_derivatives "
        "WHERE file_pk=? AND preset_id=? AND transform_cfg_id=?;",
        [int(file_pk), int(preset.preset_id), int(transform_cfg_id)],
    ).fetchone()

    if row and row[0] is not None:
        pk = int(row[0])
        con.execute(
            "UPDATE image_derivatives "
            "SET run_id=?, out_rel_path=?, width=?, height=?, fmt=?, "
            "status=0, err_code=NULL, err_msg=NULL "
            "WHERE deriv_pk=?;",
            [
                run_id,
                out_rel_blob,
                int(preset.width),
                int(preset.height),
                str(out_fmt),
                pk,
            ],
        )
        return pk

    con.execute(
        "INSERT INTO image_derivatives "
        "(deriv_pk, file_pk, preset_id, transform_cfg_id, run_id, out_rel_path, "
        "width, height, fmt, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0);",
        [
            int(deriv_pk),
            int(file_pk),
            int(preset.preset_id),
            int(transform_cfg_id),
            run_id,
            out_rel_blob,
            int(preset.width),
            int(preset.height),
            str(out_fmt),
        ],
    )
    return int(deriv_pk)


def mark_derivative_error(
    con: duckdb.DuckDBPyConnection,
    *,
    deriv_pk: int,
    file_pk: int,
    preset: Preset,
    transform_cfg_id: int,
    run_id: int | None,
    out_rel_blob: bytes,
    out_fmt: str,
    err_msg: str,
) -> int:
    """Insert a derivative as errored."""
    pk = upsert_derivative_pending(
        con,
        deriv_pk=deriv_pk,
        file_pk=file_pk,
        preset=preset,
        transform_cfg_id=transform_cfg_id,
        run_id=run_id,
        out_rel_blob=out_rel_blob,
        out_fmt=out_fmt,
    )
    con.execute(
        "UPDATE image_derivatives SET status=2, err_code=?, err_msg=?, "
        "created_at_ns=? WHERE deriv_pk=?;",
        [int(ERR_DECODE), str(err_msg)[:1000], int(time.time_ns()), int(pk)],
    )
    return pk


# ---------------------------------------------------------------------------
# CPU derivative pipeline (from pipeline_cpu.py)
# ---------------------------------------------------------------------------


def run_derivatives_cpu(
    con: duckdb.DuckDBPyConnection,
    cfg: RunConfig,
    presets: list[Preset],
    wq: WriterQueue,
    *,
    transform_cfg_id: int,
    tsettings: TransformSettings,
    console: Console,
) -> None:
    """Generate derivatives using CPU decode+resize."""
    deriv_pk = next_deriv_pk(con)
    num_presets = len(presets)
    eff_jq = 95 if cfg.preserve_quality else int(cfg.jpeg_quality)

    last_pk: int | None = None
    processed = 0
    t0 = time.time()
    last_print = t0

    join_clause = (
        "LEFT JOIN image_derivatives d ON d.file_pk = f.file_pk AND d.transform_cfg_id = ?"
    )

    while True:
        if last_pk is None:
            rows = con.execute(
                "SELECT f.file_pk, f.rel_path_blob, f.ext_blob, f.bucket_id, f.bucket_pos, "
                "       LIST(d.preset_id) FILTER (WHERE d.status=1) AS ok_presets "
                "FROM files f "
                f"{join_clause} "
                "WHERE f.status=1 AND f.dupe_of_file_pk IS NULL AND f.ok_index IS NOT NULL "
                "GROUP BY f.file_pk, f.rel_path_blob, f.ext_blob, f.bucket_id, f.bucket_pos "
                "HAVING COUNT(DISTINCT d.preset_id) FILTER (WHERE d.status=1) < ? "
                "ORDER BY f.file_pk LIMIT ?;",
                [int(transform_cfg_id), int(num_presets), int(cfg.decode_batch)],
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT f.file_pk, f.rel_path_blob, f.ext_blob, f.bucket_id, f.bucket_pos, "
                "       LIST(d.preset_id) FILTER (WHERE d.status=1) AS ok_presets "
                "FROM files f "
                f"{join_clause} "
                "WHERE f.status=1 AND f.dupe_of_file_pk IS NULL AND f.ok_index IS NOT NULL "
                "  AND f.file_pk > ? "
                "GROUP BY f.file_pk, f.rel_path_blob, f.ext_blob, f.bucket_id, f.bucket_pos "
                "HAVING COUNT(DISTINCT d.preset_id) FILTER (WHERE d.status=1) < ? "
                "ORDER BY f.file_pk LIMIT ?;",
                [
                    int(transform_cfg_id),
                    int(last_pk),
                    int(num_presets),
                    int(cfg.decode_batch),
                ],
            ).fetchall()

        if not rows:
            break

        for file_pk, rel_blob, ext_blob, bucket_id, bucket_pos, ok_presets in rows:
            file_pk_i = int(file_pk)
            last_pk = file_pk_i

            ok_set = {int(x) for x in (ok_presets or [])}
            missing = [p for p in presets if int(p.preset_id) not in ok_set]
            if not missing:
                continue

            rel = os.fsdecode(rel_blob)
            src = cfg.src_root / rel

            try:
                decoded = decode_rgb_u8_chw(src)
                con.execute(
                    "UPDATE files SET decode_status=1, decode_err_code=NULL, "
                    "decode_err_msg=NULL, orig_w=?, orig_h=? WHERE file_pk=?;",
                    [int(decoded.width), int(decoded.height), file_pk_i],
                )

                img = decoded.img_u8_chw
                run_id: int | None = None

                if transform_cfg_id != 0 and (tsettings.crop or tsettings.deskew):
                    run_id_i, tres = get_or_compute_transform_run(
                        con,
                        file_pk=file_pk_i,
                        transform_cfg_id=int(transform_cfg_id),
                        img_u8_chw=img,
                        src_w=int(decoded.width),
                        src_h=int(decoded.height),
                        tsettings=tsettings,
                    )
                    run_id = int(run_id_i)
                    img = apply_transform(img, result=tres, settings=tsettings)

                for p in missing:
                    out_fmt, out_ext = select_out_fmt_and_ext(cfg, ext_blob, p)
                    out_img = resize_u8_chw(
                        img, out_w=int(p.width), out_h=int(p.height), device=cfg.device
                    )
                    out_img = maybe_grayscale_u8_chw(out_img, preserve_color=cfg.preserve_color)

                    tseg = "" if int(transform_cfg_id) == 0 else f"t{int(transform_cfg_id):06d}/"
                    out_rel = (
                        f"deriv/{p.name}/"
                        + tseg
                        + f"b{int(bucket_id):06d}/"
                        + out_name(int(bucket_pos), file_pk_i, out_ext)
                    )
                    out_path = cfg.out_root / out_rel
                    out_rel_blob = os.fsencode(out_rel)

                    pk = upsert_derivative_pending(
                        con,
                        deriv_pk=deriv_pk,
                        file_pk=file_pk_i,
                        preset=p,
                        transform_cfg_id=int(transform_cfg_id),
                        run_id=run_id,
                        out_rel_blob=out_rel_blob,
                        out_fmt=out_fmt,
                    )
                    if pk == int(deriv_pk):
                        deriv_pk += 1

                    jq = eff_jq if out_fmt.lower() in ("jpeg", "jpg", "webp") else None
                    wq.submit(
                        WriteJob(
                            kind="encode",
                            dst_path=str(out_path),
                            file_pk=file_pk_i,
                            deriv_pk=int(pk),
                            preset_id=int(p.preset_id),
                            img_u8_chw=out_img,
                            fmt=str(out_fmt),
                            jpeg_quality=jq,
                        )
                    )

                processed += 1

            except UnidentifiedImageError as e:
                con.execute(
                    "UPDATE files SET decode_status=3, decode_err_code=?, "
                    "decode_err_msg=? WHERE file_pk=?;",
                    [int(ERR_DECODE), str(e)[:1000], file_pk_i],
                )

                for p in missing:
                    out_fmt, out_ext = select_out_fmt_and_ext(cfg, ext_blob, p)
                    tseg = "" if int(transform_cfg_id) == 0 else f"t{int(transform_cfg_id):06d}/"
                    out_rel = (
                        f"deriv/{p.name}/"
                        + tseg
                        + f"b{int(bucket_id):06d}/"
                        + out_name(int(bucket_pos), file_pk_i, out_ext)
                    )
                    out_rel_blob = os.fsencode(out_rel)

                    pk = mark_derivative_error(
                        con,
                        deriv_pk=deriv_pk,
                        file_pk=file_pk_i,
                        preset=p,
                        transform_cfg_id=int(transform_cfg_id),
                        run_id=None,
                        out_rel_blob=out_rel_blob,
                        out_fmt=out_fmt,
                        err_msg=str(e),
                    )
                    if pk == int(deriv_pk):
                        deriv_pk += 1

            except Exception as e:
                con.execute(
                    "UPDATE files SET decode_status=2, decode_err_code=?, "
                    "decode_err_msg=? WHERE file_pk=?;",
                    [int(ERR_DECODE), str(e)[:1000], file_pk_i],
                )

                for p in missing:
                    out_fmt, out_ext = select_out_fmt_and_ext(cfg, ext_blob, p)
                    tseg = "" if int(transform_cfg_id) == 0 else f"t{int(transform_cfg_id):06d}/"
                    out_rel = (
                        f"deriv/{p.name}/"
                        + tseg
                        + f"b{int(bucket_id):06d}/"
                        + out_name(int(bucket_pos), file_pk_i, out_ext)
                    )
                    out_rel_blob = os.fsencode(out_rel)
                    pk = mark_derivative_error(
                        con,
                        deriv_pk=deriv_pk,
                        file_pk=file_pk_i,
                        preset=p,
                        transform_cfg_id=int(transform_cfg_id),
                        run_id=None,
                        out_rel_blob=out_rel_blob,
                        out_fmt=out_fmt,
                        err_msg=str(e),
                    )
                    if pk == int(deriv_pk):
                        deriv_pk += 1

            if processed % 100 == 0:
                pump_results(con, wq, console=console)

            now = time.time()
            if now - last_print >= cfg.metrics_every_s:
                last_print = now
                done = processed
                rate = done / max(1e-9, now - t0)
                console.print(
                    f"[green]deriv(cpu)[/green] processed={done:,} "
                    f"rate={rate:,.2f}/s writer_backlog={wq.backlog():,}",
                    highlight=False,
                )

        pump_results(con, wq, console=console)


def run_derivatives_cpu_batch_fallback(
    con: duckdb.DuckDBPyConnection,
    cfg: RunConfig,
    presets: list[Preset],
    wq: WriterQueue,
    *,
    transform_cfg_id: int,
    tsettings: TransformSettings,
    console: Console,
) -> None:
    """Fallback for DALI path that delegates to CPU."""
    run_derivatives_cpu(
        con,
        cfg,
        presets,
        wq,
        transform_cfg_id=int(transform_cfg_id),
        tsettings=tsettings,
        console=console,
    )


# ---------------------------------------------------------------------------
# DALI derivative pipeline (from pipeline_dali.py)
# ---------------------------------------------------------------------------


def run_derivatives_dali_then_cpu(
    con: duckdb.DuckDBPyConnection,
    cfg: RunConfig,
    presets: list[Preset],
    wq: WriterQueue,
    *,
    transform_cfg_id: int,
    tsettings: TransformSettings,
    console: Console,
) -> None:
    """Run DALI derivatives first, then fallback to CPU for remaining."""
    if transform_cfg_id != 0:
        raise ValueError(
            "DALI derivatives only supported with identity transform (transform_cfg_id=0)"
        )

    run_derivatives_dali(con, cfg, presets, wq, transform_cfg_id=0, console=console)

    run_derivatives_cpu_batch_fallback(
        con,
        cfg,
        presets,
        wq,
        transform_cfg_id=0,
        tsettings=tsettings,
        console=console,
    )


def run_derivatives_dali(
    con: duckdb.DuckDBPyConnection,
    cfg: RunConfig,
    presets: list[Preset],
    wq: WriterQueue,
    *,
    transform_cfg_id: int,
    console: Console,
) -> None:
    """Generate derivatives using DALI GPU-accelerated decode+resize."""
    if int(transform_cfg_id) != 0:
        raise ValueError(
            "DALI derivatives only supported with identity transform (transform_cfg_id=0)"
        )

    worker = DaliDerivativeGenerator(
        batch_size=int(cfg.decode_batch) * int(cfg.dali_batch_multiplier),
        device=str(cfg.device),
        threads=4,
        preserve_color=bool(cfg.preserve_color),
    )

    deriv_pk = next_deriv_pk(con)
    eff_jq = 95 if cfg.preserve_quality else int(cfg.jpeg_quality)

    last_pk: int | None = None
    processed = 0
    t0 = time.time()
    last_print = t0

    jpg_exts = [b".jpg", b".jpeg"]

    while True:
        if last_pk is None:
            rows = con.execute(
                "SELECT f.file_pk, f.rel_path_blob, f.ext_blob, f.bucket_id, f.bucket_pos "
                "FROM files f "
                "LEFT JOIN image_derivatives d ON d.file_pk = f.file_pk "
                "AND d.transform_cfg_id=0 "
                "WHERE f.status=1 AND f.dupe_of_file_pk IS NULL AND f.ok_index IS NOT NULL "
                "  AND f.ext_blob = ANY(?) "
                "GROUP BY f.file_pk, f.rel_path_blob, f.ext_blob, f.bucket_id, f.bucket_pos "
                "HAVING COUNT(DISTINCT d.preset_id) FILTER (WHERE d.status=1) < ? "
                "ORDER BY f.file_pk LIMIT ?;",
                [
                    jpg_exts,
                    len(presets),
                    int(cfg.decode_batch) * int(cfg.dali_batch_multiplier),
                ],
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT f.file_pk, f.rel_path_blob, f.ext_blob, f.bucket_id, f.bucket_pos "
                "FROM files f "
                "LEFT JOIN image_derivatives d ON d.file_pk = f.file_pk "
                "AND d.transform_cfg_id=0 "
                "WHERE f.status=1 AND f.dupe_of_file_pk IS NULL AND f.ok_index IS NOT NULL "
                "  AND f.file_pk > ? AND f.ext_blob = ANY(?) "
                "GROUP BY f.file_pk, f.rel_path_blob, f.ext_blob, f.bucket_id, f.bucket_pos "
                "HAVING COUNT(DISTINCT d.preset_id) FILTER (WHERE d.status=1) < ? "
                "ORDER BY f.file_pk LIMIT ?;",
                [
                    int(last_pk),
                    jpg_exts,
                    len(presets),
                    int(cfg.decode_batch) * int(cfg.dali_batch_multiplier),
                ],
            ).fetchall()

        if not rows:
            break

        paths: list[str] = []
        meta: list[tuple[int, bytes, bytes, int, int]] = []
        for file_pk, rel_blob, ext_blob, bucket_id, bucket_pos in rows:
            file_pk_i = int(file_pk)
            last_pk = file_pk_i
            rel = os.fsdecode(rel_blob)
            paths.append(str(cfg.src_root / rel))
            meta.append((file_pk_i, rel_blob, ext_blob, int(bucket_id), int(bucket_pos)))

        batch_out = worker.run(paths, presets)

        file_pks = [m[0] for m in meta]
        if file_pks:
            con.execute(
                "UPDATE files SET decode_status=1 WHERE file_pk = ANY(?);",
                [file_pks],
            )

        for i, (file_pk_i, _rel_blob, ext_blob, bucket_id, bucket_pos) in enumerate(meta):
            for j, p in enumerate(presets):
                out_fmt, out_ext = select_out_fmt_and_ext(cfg, ext_blob, p)
                out_img = batch_out[i][j]
                out_img = maybe_grayscale_u8_chw(out_img, preserve_color=cfg.preserve_color)

                out_rel = (
                    f"deriv/{p.name}/"
                    + f"b{int(bucket_id):06d}/"
                    + out_name(int(bucket_pos), int(file_pk_i), out_ext)
                )
                out_path = cfg.out_root / out_rel
                out_rel_blob = os.fsencode(out_rel)

                pk = upsert_derivative_pending(
                    con,
                    deriv_pk=deriv_pk,
                    file_pk=int(file_pk_i),
                    preset=p,
                    transform_cfg_id=0,
                    run_id=None,
                    out_rel_blob=out_rel_blob,
                    out_fmt=out_fmt,
                )
                if pk == int(deriv_pk):
                    deriv_pk += 1

                jq = eff_jq if out_fmt.lower() in ("jpeg", "jpg", "webp") else None
                wq.submit(
                    WriteJob(
                        kind="encode",
                        dst_path=str(out_path),
                        file_pk=int(file_pk_i),
                        deriv_pk=int(pk),
                        preset_id=int(p.preset_id),
                        img_u8_chw=out_img,
                        fmt=str(out_fmt),
                        jpeg_quality=jq,
                    )
                )

        processed += len(meta)
        pump_results(con, wq, console=console)

        now = time.time()
        if now - last_print >= cfg.metrics_every_s:
            last_print = now
            rate = processed / max(1e-9, now - t0)
            console.print(
                f"[green]deriv(dali)[/green] processed={processed:,} "
                f"rate={rate:,.2f}/s writer_backlog={wq.backlog():,}",
                highlight=False,
            )


# ---------------------------------------------------------------------------
# Main pipeline orchestrator (from pipeline.py)
# ---------------------------------------------------------------------------


def run_pipeline(cfg: RunConfig, *, console: Console | None = None) -> None:
    """Run the full preprocess pipeline: scan → hash → dedupe → bucket → derivatives."""
    console = console or Console()

    out_root = cfg.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    db_path = out_root / "index.duckdb"

    db = connect(db_path)
    con = db.con

    start_pk = next_file_pk(con)
    scan_stats = scan_into_duckdb(
        con,
        cfg.src_root,
        start_file_pk=start_pk,
        allow_exts=None,
        max_files=cfg.max_files,
        insert_batch=cfg.scan_insert_batch,
    )
    console.print(
        f"[blue]scan[/blue] seen={scan_stats.seen:,} inserted={scan_stats.inserted:,} "
        f"skipped={scan_stats.skipped:,}",
        highlight=False,
    )

    ds = hash_and_mark_dupes(
        con,
        cfg.src_root,
        num_workers=cfg.hash_workers,
        metrics_every_s=cfg.metrics_every_s,
        console=console,
    )
    console.print(
        f"[cyan]hash+dedupe[/cyan] candidates={ds.total_candidates:,} "
        f"ok={ds.hashed_ok:,} err={ds.hash_err:,} "
        f"canonicals={ds.uniques:,} dupes={ds.dupes:,}",
        highlight=False,
    )

    from vit_curator.shared.db import parse_presets_arg  # noqa: PLC0415

    presets_parsed = parse_presets_arg(cfg.presets_arg) if cfg.presets_arg.strip() else []
    if presets_parsed:
        eff_jq = 95 if cfg.preserve_quality else int(cfg.jpeg_quality)
        ensure_preset_rows(con, presets_parsed, cfg.fmt, eff_jq)
    presets = load_presets(con)

    tsettings = TransformSettings(
        crop=bool(cfg.crop),
        deskew=bool(cfg.deskew),
        preview_long_edge=int(cfg.preview_long_edge),
        bg_mode=str(cfg.crop_bg),
        white_bg_thresh=int(cfg.crop_white_bg_thresh),
        black_bg_thresh=int(cfg.crop_black_bg_thresh),
        crop_padding_px=int(cfg.crop_padding_px),
        max_crop_margin_ratio=float(cfg.max_crop_margin_ratio),
        min_retained_area_ratio=float(cfg.min_retained_area_ratio),
        deskew_max_angle_deg=float(cfg.deskew_max_angle_deg),
        deskew_step_deg=float(cfg.deskew_step_deg),
        deskew_min_conf=float(cfg.deskew_min_conf),
    )

    transform_cfg_id = 0
    if tsettings.crop or tsettings.deskew:
        settings_obj = {
            "crop": tsettings.crop,
            "deskew": tsettings.deskew,
            "preview_long_edge": tsettings.preview_long_edge,
            "bg_mode": tsettings.bg_mode,
            "white_bg_thresh": tsettings.white_bg_thresh,
            "black_bg_thresh": tsettings.black_bg_thresh,
            "crop_padding_px": tsettings.crop_padding_px,
            "max_crop_margin_ratio": tsettings.max_crop_margin_ratio,
            "min_retained_area_ratio": tsettings.min_retained_area_ratio,
            "deskew_max_angle_deg": tsettings.deskew_max_angle_deg,
            "deskew_step_deg": tsettings.deskew_step_deg,
            "deskew_min_conf": tsettings.deskew_min_conf,
        }
        settings_json = json.dumps(settings_obj, sort_keys=True, separators=(",", ":"))
        settings_hash = xxhash.xxh3_128_digest(settings_json.encode("utf-8"))
        transform_cfg_id = get_or_create_transform_cfg(
            con,
            settings_json=settings_json,
            settings_hash=settings_hash,
            algo_version="transform_v1",
        )

    num_presets = max(1, len(presets))
    max_jobs = max(1, int(cfg.decode_batch) * int(cfg.inflight_batches) * min(4, num_presets + 1))
    wq = WriterQueue(num_workers=int(cfg.writer_workers), max_jobs=max_jobs)

    assigned = 0
    for a in iter_bucket_assignments(
        con,
        bucket_size=cfg.bucket_size,
        metrics_every_s=cfg.metrics_every_s,
        console=console,
    ):
        rel = os.fsdecode(a.rel_path_blob)
        src = cfg.src_root / rel
        dst = (
            out_root
            / "orig"
            / f"b{a.bucket_id:06d}"
            / out_name(a.bucket_pos, a.file_pk, os.fsdecode(a.ext_blob))
        )
        wq.submit(
            WriteJob(
                kind="link",
                dst_path=str(dst),
                src_path=str(src),
                link_mode=cfg.link_mode,
                file_pk=a.file_pk,
            )
        )
        assigned += 1
        if assigned % 10_000 == 0:
            pump_results(con, wq, console=console)

    console.print(
        f"[magenta]bucket[/magenta] assigned={assigned:,} writer_backlog={wq.backlog():,}",
        highlight=False,
    )

    if presets:
        if (tsettings.crop or tsettings.deskew) and cfg.decode_backend.lower() == "dali":
            console.print(
                "[yellow]note[/yellow] forcing CPU decode for derivatives "
                "because --crop/--deskew are enabled",
                highlight=False,
            )

        if cfg.decode_backend.lower() == "dali" and not (tsettings.crop or tsettings.deskew):
            run_derivatives_dali_then_cpu(
                con,
                cfg,
                presets,
                wq,
                transform_cfg_id=transform_cfg_id,
                tsettings=tsettings,
                console=console,
            )
        else:
            run_derivatives_cpu(
                con,
                cfg,
                presets,
                wq,
                transform_cfg_id=transform_cfg_id,
                tsettings=tsettings,
                console=console,
            )

    wq.close()
    wq.join()
    pump_results(con, wq, console=console)

    console.print("[bold green]Pipeline complete.[/bold green]")
