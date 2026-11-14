"""Module C_pub: public constraints may ship; C_priv does not live here."""

from __future__ import annotations

import pandas as pd

from synthfraud.log import log
from synthfraud.schema import Schema, certificate_coverage, constraint_mask


def sample_with_c_pub(
    model,
    n: int,
    schema: Schema,
    *,
    max_draw: int | None = None,
    batch_size: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Rejection-sample until `n` C_pub-valid rows, or `max_draw` candidates."""
    max_draw = n * 30 if max_draw is None else int(max_draw)
    batch_size = n if batch_size is None else int(batch_size)
    chunks: list[pd.DataFrame] = []
    kept = 0
    drawn = 0
    while kept < n and drawn < max_draw:
        remaining = n - kept
        take = min(batch_size, max_draw - drawn, max(remaining, 1))
        batch = model.sample(take)
        drawn += len(batch)
        ok = batch.loc[constraint_mask(schema, batch)]
        chunks.append(ok)
        kept += len(ok)
        log(f"  C_pub draw {drawn:,}  kept {kept:,}/{n:,}")
    if not chunks:
        raise RuntimeError("C_pub filter produced no rows")
    out = pd.concat(chunks, ignore_index=True).head(n)
    if len(out) < n:
        raise RuntimeError(
            f"C_pub kept {len(out)}/{n} rows after drawing {drawn} (max_draw={max_draw})"
        )
    return out, {
        "n_drawn": drawn,
        "n_kept": int(len(out)),
        "reject_rate": 1.0 - (kept / max(drawn, 1)),
        "certificate_coverage": certificate_coverage(schema, out),
        "uncertified_fraud": int(
            certificate_coverage(schema, out).get("uncertified_fraud", 0)
        ),
    }


def describe_c_pub(schema: Schema, frame: pd.DataFrame, *, n_drawn: int | None = None) -> dict:
    mask = constraint_mask(schema, frame)
    n = len(frame)
    n_bad = int((~mask).sum())
    report = {
        "n_kept": int(mask.sum()),
        "constraint_violation_rate": n_bad / max(n, 1),
        "certificate_coverage": certificate_coverage(schema, frame),
        "uncertified_fraud": int(
            certificate_coverage(schema, frame).get("uncertified_fraud", 0)
        ),
    }
    if n_drawn is not None:
        report["n_drawn"] = int(n_drawn)
        report["reject_rate"] = 1.0 - (n / max(n_drawn, 1))
    return report
