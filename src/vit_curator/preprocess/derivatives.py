"""Derivative image generation — CPU and optional DALI paths.

This module merges the main pipeline orchestration from data_janitor:
  - pipeline_common.py (resize, grayscale helpers, format helpers)
  - pipeline_db.py (pump_results, upsert_derivative_pending, mark_derivative_error)
  - pipeline_cpu.py (run_derivatives_cpu)
  - pipeline_dali.py (run_derivatives_dali_then_cpu, run_derivatives_dali)
  - pipeline.py (run_pipeline main orchestrator)

All DB operations use vit_curator.shared.db functions and the unified
schema (file_pk, not asset_id).

Format helpers live in derivatives_format.py; DB helpers in derivatives_db.py.
"""

from __future__ import annotations

import json
import os
import time

import duckdb
import xxhash
from PIL import UnidentifiedImageError
from rich.console import Console

from vit_curator.config import RunConfig
from vit_curator.preprocess.bucket import iter_bucket_assignments
from vit_curator.preprocess.derivatives_db import (
    get_or_compute_transform_run,
    mark_derivative_error,
    pump_results,
    upsert_derivative_pending,
)
from vit_curator.preprocess.derivatives_format import (
    maybe_grayscale_u8_chw,
    resize_u8_chw,
    select_out_fmt_and_ext,
)
from vit_curator.preprocess.transform import TransformSettings
from vit_curator.preprocess.writer_queue import WriteJob, WriterQueue, out_name
from vit_curator.shared.db import (
    Preset,
    connect,
    ensure_preset_rows,
    get_or_create_transform_cfg,
    load_presets,
    next_deriv_pk,
    next_file_pk,
)
from vit_curator.shared.errors import ERR_DECODE

# Re-export for backward compatibility — anything that was previously
# importable from this module remains importable via __getattr__.
__all__ = [
    "get_or_compute_transform_run",
    "mark_derivative_error",
    "maybe_grayscale_u8_chw",
    "pump_results",
    "resize_u8_chw",
    "run_derivatives_cpu",
    "run_derivatives_cpu_batch_fallback",
    "run_derivatives_dali",
    "run_derivatives_dali_then_cpu",
    "run_pipeline",
    "select_out_fmt_and_ext",
    "upsert_derivative_pending",
]

# Lazy re-exports so that `from derivatives import ext_for_fmt` still works
# without importing the heavy dependencies of the submodules at module level.


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    """Re-export helpers from extracted modules for backward compatibility."""
    _format_names = {
        "resize_u8_chw",
        "maybe_grayscale_u8_chw",
        "ext_for_fmt",
        "fmt_from_ext",
        "select_out_fmt_and_ext",
    }
    _db_names = {
        "pump_results",
        "_update_transform_ok",
        "_update_transform_err",
        "get_or_compute_transform_run",
        "upsert_derivative_pending",
        "mark_derivative_error",
    }
    if name in _format_names:
        from vit_curator.preprocess import derivatives_format as _mod  # noqa: PLC0415

        return getattr(_mod, name)
    if name in _db_names:
        from vit_curator.preprocess import derivatives_db as _mod  # noqa: PLC0415

        return getattr(_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# CPU derivative pipeline (from pipeline_cpu.py)
# ---------------------------------------------------------------------------


def _process_cpu_batch(
    con: duckdb.DuckDBPyConnection,
    cfg: RunConfig,
    presets: list[Preset],
    wq: WriterQueue,
    *,
    transform_cfg_id: int,
    tsettings: TransformSettings,
    rows: list[tuple],
    deriv_pk: int,
) -> tuple[int, int]:
    """Process one batch of rows in the CPU derivative pipeline.

    Returns (updated deriv_pk, count_of_processed_files).
    """
    eff_jq = 95 if cfg.preserve_quality else int(cfg.jpeg_quality)
    processed = 0

    # Import here to avoid circular / heavy import at module level.
    from vit_curator.preprocess.decode import decode_rgb_u8_chw  # noqa: PLC0415
    from vit_curator.preprocess.transform import apply_transform  # noqa: PLC0415

    for file_pk, rel_blob, ext_blob, bucket_id, bucket_pos, ok_presets in rows:
        file_pk_i = int(file_pk)

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

    return deriv_pk, processed


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

        deriv_pk, batch_count = _process_cpu_batch(
            con,
            cfg,
            presets,
            wq,
            transform_cfg_id=transform_cfg_id,
            tsettings=tsettings,
            rows=rows,
            deriv_pk=deriv_pk,
        )
        last_pk = int(rows[-1][0])
        processed += batch_count

        if processed % 100 == 0:
            pump_results(con, wq, console=console)

        now = time.time()
        if now - last_print >= cfg.metrics_every_s:
            last_print = now
            rate = processed / max(1e-9, now - t0)
            console.print(
                f"[green]deriv(cpu)[/green] processed={processed:,} "
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


def _process_dali_batch(
    con: duckdb.DuckDBPyConnection,
    cfg: RunConfig,
    presets: list[Preset],
    wq: WriterQueue,
    *,
    deriv_pk: int,
    eff_jq: int,
    meta: list[tuple[int, bytes, bytes, int, int]],
    batch_out: list,
) -> int:
    """Process one batch of DALI-decoded results.

    Returns the updated deriv_pk.
    """
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

    return deriv_pk


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

    from vit_curator.preprocess.decode import DaliDerivativeGenerator  # noqa: PLC0415

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

        deriv_pk = _process_dali_batch(
            con,
            cfg,
            presets,
            wq,
            deriv_pk=deriv_pk,
            eff_jq=eff_jq,
            meta=meta,
            batch_out=batch_out,
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


def _run_scan_stage(cfg: RunConfig, con: duckdb.DuckDBPyConnection, *, console: Console) -> None:
    """Run the scan stage of the pipeline."""
    from vit_curator.preprocess.scan import scan_into_duckdb  # noqa: PLC0415

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


def _run_dedupe_stage(cfg: RunConfig, con: duckdb.DuckDBPyConnection, *, console: Console) -> None:
    """Run the hash + dedupe stage of the pipeline."""
    from vit_curator.preprocess.dedupe import hash_and_mark_dupes  # noqa: PLC0415

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


def _run_derivatives_stage(
    cfg: RunConfig,
    con: duckdb.DuckDBPyConnection,
    *,
    console: Console,
) -> None:
    """Run the derivative generation stage."""
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
            cfg.out_root
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


def run_pipeline(cfg: RunConfig, *, console: Console | None = None) -> None:
    """Run the full preprocess pipeline: scan → hash → dedupe → bucket → derivatives."""
    console = console or Console()

    out_root = cfg.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    db_path = out_root / "index.duckdb"

    db = connect(db_path)
    con = db.con

    _run_scan_stage(cfg, con, console=console)
    _run_dedupe_stage(cfg, con, console=console)
    _run_derivatives_stage(cfg, con, console=console)

    console.print("[bold green]Pipeline complete.[/bold green]")
