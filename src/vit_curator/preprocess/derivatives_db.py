"""Database helper functions for derivative generation.

Extracted from derivatives.py for clarity. These encapsulate all direct
SQL operations for transform runs and derivative rows.
"""

from __future__ import annotations

import time

import duckdb
import torch
from rich.console import Console

from vit_curator.preprocess.transform import (
    TransformResult,
    TransformSettings,
    apply_transform,
)
from vit_curator.preprocess.writer_queue import (
    WriteResult,
    WriterQueue,
)
from vit_curator.shared.db import (
    Preset,
    next_transform_run_pk,
)
from vit_curator.shared.errors import ERR_DECODE

# ---------------------------------------------------------------------------
# Writer-queue result pump
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


# ---------------------------------------------------------------------------
# Transform-run helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Derivative row helpers
# ---------------------------------------------------------------------------


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
