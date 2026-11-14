"""Module R: release controller. C_priv (bank probe) never ships."""

from __future__ import annotations

import numpy as np
import pandas as pd

from synthfraud.c_pub import describe_c_pub, sample_with_c_pub
from synthfraud.detect import predict_proba
from synthfraud.log import log
from synthfraud.metrics.utility import honest_utility
from synthfraud.schema import Schema


def sample_with_r(
    model,
    n: int,
    schema: Schema,
    *,
    bank,
    val: pd.DataFrame,
    train_fraud_rate: float,
    seed: int,
    pool_mult: float = 12.0,
    target_fraud_rate: float | None = None,
    rate_grid: list | None = None,
    row_select: str = "val_tables",
    min_margin: float = 0.15,
    n_val_candidates: int = 6,
    lambda_ua: float = 1.0,
    elite_mult: float = 4.0,
    c_pub: bool = True,
    max_draw: int | None = None,
    also_rate_only: bool = False,
) -> tuple[pd.DataFrame, dict, pd.DataFrame | None]:
    """Draw a pool from G, filter with C_pub / C_priv, release n rows.

    C_priv is the bank detector plus the real val split. Neither is serialized.
    ``row_select='topk'`` takes per-class top rows by (U_h - λ U_a) — no 13-row
    table lottery. ``rate_grid`` lets val choose a declared fraud budget.
    """
    row_select = str(row_select)
    if row_select not in {"topk", "val_tables"}:
        raise ValueError(f"row_select must be topk or val_tables, got {row_select!r}")
    grid = list(rate_grid) if rate_grid else []
    if grid:
        row_select = "topk"

    scored = _draw_scored_pool(
        model,
        n,
        schema,
        bank=bank,
        pool_mult=pool_mult,
        lambda_ua=lambda_ua,
        min_margin=min_margin,
        c_pub=c_pub,
        max_draw=max_draw,
        extra_log=(f"  rate_grid={len(grid)} budgets" if grid else ""),
        row_select=row_select,
    )
    pool = scored["pool"]
    c_stats = scored["c_stats"]
    score = scored["score"]
    keep_margin = scored["keep_margin"]
    u_h = scored["u_h"]
    u_a = scored["u_a"]
    margin = scored["margin"]

    grid_log: list[dict] = []
    if grid:
        released, rate, val_pr, grid_log = _select_rate_on_val(
            pool,
            schema,
            val=val,
            train_fraud_rate=train_fraud_rate,
            n=n,
            keep_margin=keep_margin,
            score=score,
            rate_grid=grid,
            seed=seed,
        )
        n1 = int(released[schema.label].sum())
        n0 = n - n1
    else:
        rate = float(train_fraud_rate if target_fraud_rate is None else target_fraud_rate)
        rate = min(max(rate, 0.0), 1.0)
        n1, n0 = _class_counts(n, rate)
        log(f"R target n_fraud={n1}  (rate={rate:.4%})")
        if row_select == "topk":
            released = _topk_release(pool, schema, keep_margin=keep_margin, score=score, n0=n0, n1=n1)
            val_pr = float(honest_utility(released, val, schema, seed=seed)["pr_auc"])
            log(f"  R topk val PR-AUC={val_pr:.3f}")
        else:
            elite = _elite_pool(
                pool,
                schema,
                keep_margin=keep_margin,
                score=score,
                n0=n0,
                n1=n1,
                elite_mult=elite_mult,
                seed=seed + 23,
            )
            released, val_pr = _select_on_val(
                elite,
                schema,
                val=val,
                n0=n0,
                n1=n1,
                n_candidates=int(n_val_candidates),
                seed=seed,
            )

    rate_only = None
    if also_rate_only:
        rate_only = _stratified_take(pool, schema, n0=n0, n1=n1, seed=seed + 17)

    stats = _r_stats(
        schema,
        bank=bank,
        pool=pool,
        released=released,
        c_pub=c_pub,
        row_select=row_select,
        rate=rate,
        train_fraud_rate=train_fraud_rate,
        grid_log=grid_log,
        min_margin=min_margin,
        lambda_ua=lambda_ua,
        n_val_candidates=n_val_candidates,
        keep_margin=keep_margin,
        u_h=u_h,
        u_a=u_a,
        margin=margin,
        val_pr=val_pr,
        c_stats=c_stats,
        declared=False,
    )
    log(
        f"R release fraud={stats['n_release_fraud']}/{len(released)}  "
        f"rate={rate:.4%}  val_PR-AUC={val_pr:.3f}  row_select={row_select}"
    )
    return released[schema.all_cols], stats, rate_only


KNOWN_CONTROLLERS = ("topk", "rate_only", "margin_random", "type_topk")

# Columns a T1 product may withhold (not C_pub). Jitter skips time-like ints.
PRIVATE_NUMERIC = {
    "paysim": ("oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"),
    "toy": ("n_tx_24h",),
    "ieee_cis": ("C1", "C2", "C13", "D1", "V307", "V310", "V314", "V317", "V320"),
}
JITTER_SKIP = frozenset({"step", "hour", "TransactionDT"})


def apply_sanitizer(
    table: pd.DataFrame,
    schema: Schema,
    *,
    kind: str,
    seed: int,
    rel: float | None = None,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, Schema, dict]:
    """Post-process a rate-only table. Does not rank with C_priv.

    ``jitter``: additive Gaussian noise, σ = rel × column std. Keeps the schema.
    ``withhold``: drop private-looking numeric columns. Honest party and
    surrogate see only what ships; the bank scorer stays on the full schema.
    """
    kind = str(kind)
    if kind == "jitter":
        if rel is None:
            raise ValueError("jitter sanitizer needs rel")
        out, scales = _jitter_numeric(table, schema, rel=float(rel), seed=seed)
        log(f"  R sanitizer jitter rel={rel:g}  cols={list(scales)}")
        meta = {"sanitizer": "jitter", "rel": float(rel), "noise_std": scales}
        return out, schema, meta
    if kind == "withhold":
        drop = tuple(columns) if columns else PRIVATE_NUMERIC.get(schema.name, ())
        if not drop:
            raise ValueError(f"withhold sanitizer: no private numeric columns for {schema.name}")
        eval_schema = schema.drop_numeric(drop)
        missing = [c for c in drop if c not in table.columns]
        if missing:
            raise KeyError(f"withhold missing columns {missing}")
        out = table[eval_schema.all_cols].copy()
        log(f"  R sanitizer withhold drop={list(drop)}  keep={eval_schema.feature_cols}")
        meta = {
            "sanitizer": "withhold",
            "dropped": list(drop),
            "release_features": eval_schema.feature_cols,
        }
        return out, eval_schema, meta
    raise ValueError(f"unknown sanitizer {kind!r}; choose jitter or withhold")


def _jitter_numeric(
    table: pd.DataFrame, schema: Schema, *, rel: float, seed: int
) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(seed)
    out = table.copy()
    scales: dict[str, float] = {}
    for col in schema.numeric:
        if col in JITTER_SKIP:
            continue
        x = out[col].to_numpy(dtype=float)
        std = float(np.std(x))
        sigma = float(rel) * std if std > 0 else float(rel)
        scales[col] = sigma
        y = x + rng.normal(0.0, sigma, size=len(x))
        if col in {"amount", "n_tx_24h"}:
            y = np.clip(y, 0.0, None)
        out[col] = y
    return out, scales


def sample_declared_rates(
    model,
    n: int,
    schema: Schema,
    *,
    bank,
    val: pd.DataFrame,
    train_fraud_rate: float,
    seed: int,
    rates: list[float],
    pool_mult: float = 12.0,
    min_margin: float = 0.15,
    lambda_ua: float = 1.0,
    c_pub: bool = True,
    max_draw: int | None = None,
) -> list[tuple[float, pd.DataFrame, dict, pd.DataFrame]]:
    """One G pool; a top-k and a rate-only table at each declared fraud rate.

    Val is scored as a probe only. It does not choose the rate.
    """
    releases = sample_declared_controllers(
        model,
        n,
        schema,
        bank=bank,
        val=val,
        train_fraud_rate=train_fraud_rate,
        seed=seed,
        rates=rates,
        controllers=("topk", "rate_only"),
        pool_mult=pool_mult,
        min_margin=min_margin,
        lambda_ua=lambda_ua,
        c_pub=c_pub,
        max_draw=max_draw,
    )
    out = []
    for rate, tables in releases:
        topk, stats = tables["topk"]
        rate_only, _ = tables["rate_only"]
        out.append((rate, topk, stats, rate_only))
    return out


def sample_declared_controllers(
    model,
    n: int,
    schema: Schema,
    *,
    bank,
    val: pd.DataFrame,
    train_fraud_rate: float,
    seed: int,
    rates: list[float],
    controllers: tuple[str, ...] | list[str],
    pool_mult: float = 12.0,
    min_margin: float = 0.15,
    lambda_ua: float = 1.0,
    c_pub: bool = True,
    max_draw: int | None = None,
) -> list[tuple[float, dict[str, tuple[pd.DataFrame, dict]]]]:
    """One G pool; named controllers at each declared fraud rate.

    Controllers (v4 isolation of why unconstrained top-k failed):
    - ``rate_only``: class-stratified random from the C_pub pool (default T1).
    - ``topk``: per-class C_priv ranking after the margin filter (known type wreck).
    - ``margin_random``: drop thin-margin rows, then class-stratified random (no rank).
    - ``type_topk``: same ranking, but type quotas match the pool's class-conditional
      type mix so ranking cannot concentrate ``type``.
    Val is a probe only. It does not choose the rate.
    """
    names = tuple(controllers)
    unknown = [c for c in names if c not in KNOWN_CONTROLLERS]
    if unknown:
        raise ValueError(f"unknown R controllers {unknown}; choose from {KNOWN_CONTROLLERS}")
    if not names:
        raise ValueError("controllers must be non-empty")
    scored = _draw_scored_pool(
        model,
        n,
        schema,
        bank=bank,
        pool_mult=pool_mult,
        lambda_ua=lambda_ua,
        min_margin=min_margin,
        c_pub=c_pub,
        max_draw=max_draw,
        extra_log=f"  declared_rates={len(rates)}  controllers={list(names)}",
        row_select="+".join(names),
    )
    out: list[tuple[float, dict[str, tuple[pd.DataFrame, dict]]]] = []
    for raw in rates:
        rate = min(max(float(raw), 0.0), 1.0)
        n1, n0 = _class_counts(n, rate)
        log(f"R declared n_fraud={n1}  (rate={rate:.4%})  val does not select")
        tables: dict[str, tuple[pd.DataFrame, dict]] = {}
        for name in names:
            released = _apply_controller(
                name,
                scored["pool"],
                schema,
                keep_margin=scored["keep_margin"],
                score=scored["score"],
                n0=n0,
                n1=n1,
                seed=seed,
            )
            val_pr = float(honest_utility(released, val, schema, seed=seed)["pr_auc"])
            log(f"  R {name} val PR-AUC={val_pr:.3f} (probe, not a selector)")
            stats = _r_stats(
                schema,
                bank=bank,
                pool=scored["pool"],
                released=released,
                c_pub=c_pub,
                row_select=name,
                rate=rate,
                train_fraud_rate=train_fraud_rate,
                grid_log=[],
                min_margin=min_margin,
                lambda_ua=lambda_ua,
                n_val_candidates=0,
                keep_margin=scored["keep_margin"],
                u_h=scored["u_h"],
                u_a=scored["u_a"],
                margin=scored["margin"],
                val_pr=val_pr,
                c_stats=scored["c_stats"],
                declared=True,
            )
            tables[name] = (released, stats)
        out.append((rate, tables))
    return out


def _draw_scored_pool(
    model,
    n: int,
    schema: Schema,
    *,
    bank,
    pool_mult: float,
    lambda_ua: float,
    min_margin: float,
    c_pub: bool,
    max_draw: int | None,
    extra_log: str,
    row_select: str,
) -> dict:
    pool_n = max(int(np.ceil(pool_mult * n)), n)
    log(
        f"R pool target={pool_n:,}  release n={n:,}  row_select={row_select}  "
        f"min_margin={min_margin:g}{extra_log}"
    )
    if c_pub:
        pool, c_stats = sample_with_c_pub(
            model, pool_n, schema, max_draw=max_draw or pool_n * 30
        )
    else:
        pool = model.sample(pool_n)
        c_stats = {"enabled": False, **describe_c_pub(schema, pool)}
    y = pool[schema.label].astype(int).to_numpy()
    p = predict_proba(bank, pool, schema)
    margin = np.abs(p - 0.5)
    u_h = np.where(y == 1, p, 1.0 - p)
    u_a = 1.0 - 2.0 * margin
    score = u_h - float(lambda_ua) * u_a
    keep_margin = margin >= float(min_margin)
    return {
        "pool": pool,
        "c_stats": c_stats,
        "score": score,
        "keep_margin": keep_margin,
        "u_h": u_h,
        "u_a": u_a,
        "margin": margin,
    }


def _r_stats(
    schema: Schema,
    *,
    bank,
    pool: pd.DataFrame,
    released: pd.DataFrame,
    c_pub: bool,
    row_select: str,
    rate: float,
    train_fraud_rate: float,
    grid_log: list,
    min_margin: float,
    lambda_ua: float,
    n_val_candidates: int,
    keep_margin: np.ndarray,
    u_h: np.ndarray,
    u_a: np.ndarray,
    margin: np.ndarray,
    val_pr: float,
    c_stats: dict,
    declared: bool = False,
) -> dict:
    return {
        "enabled": True,
        "c_pub": bool(c_pub),
        "row_select": row_select,
        "n_pool": int(len(pool)),
        "n_release": int(len(released)),
        "n_release_fraud": int(released[schema.label].sum()),
        "target_fraud_rate": rate,
        "train_fraud_rate": float(train_fraud_rate),
        "rate_grid": grid_log,
        "val_selects_rate": bool(grid_log) and not declared,
        "declared_rate": declared,
        "min_margin": float(min_margin),
        "lambda_ua": float(lambda_ua),
        "controller": row_select,
        "n_val_candidates": 0 if row_select in {"topk", "margin_random", "type_topk", "rate_only"} else int(n_val_candidates),
        "n_margin_kept": int(keep_margin.sum()),
        "mean_u_h_row": float(u_h.mean()),
        "mean_u_a_row": float(u_a.mean()),
        "mean_margin_pool": float(margin.mean()),
        "mean_margin_release": float(
            np.abs(predict_proba(bank, released, schema) - 0.5).mean()
        ),
        "val_pr_auc_selected": val_pr,
        "c_pub_stats": c_stats,
        "note": (
            "C_priv is the train-fitted bank detector; val is the U_h probe. "
            "Neither ships. declared_rate: val does not choose the budget. "
            "topk is unconstrained C_priv ranking; margin_random drops thin-margin then "
            "samples at random; type_topk ranks within type cells whose quotas match the "
            "pool mix; val_tables is the 13-row lottery (legacy)."
        ),
    }


def _class_counts(n: int, rate: float) -> tuple[int, int]:
    n1 = int(round(n * rate))
    n1 = min(max(n1, 0), n)
    return n1, n - n1


def _apply_controller(
    name: str,
    pool: pd.DataFrame,
    schema: Schema,
    *,
    keep_margin: np.ndarray,
    score: np.ndarray,
    n0: int,
    n1: int,
    seed: int,
) -> pd.DataFrame:
    if name == "topk":
        return _topk_release(pool, schema, keep_margin=keep_margin, score=score, n0=n0, n1=n1)
    if name == "rate_only":
        return _stratified_take(pool, schema, n0=n0, n1=n1, seed=seed + 17)
    if name == "margin_random":
        return _margin_random_release(pool, schema, keep_margin=keep_margin, n0=n0, n1=n1, seed=seed + 31)
    if name == "type_topk":
        return _type_preserving_release(
            pool,
            schema,
            keep_margin=keep_margin,
            score=score,
            n0=n0,
            n1=n1,
            seed=seed + 43,
            rank=True,
        )
    raise ValueError(f"unknown R controller {name!r}")


def _margin_random_release(
    pool: pd.DataFrame,
    schema: Schema,
    *,
    keep_margin: np.ndarray,
    n0: int,
    n1: int,
    seed: int,
) -> pd.DataFrame:
    """Class-stratified random after dropping thin-margin rows. No C_priv ranking."""
    y = pool[schema.label].astype(int).to_numpy()
    rng = np.random.default_rng(seed)
    parts = []
    for cls, need in ((0, n0), (1, n1)):
        if need <= 0:
            continue
        idx = np.where((y == cls) & keep_margin)[0]
        if len(idx) < need:
            idx = np.where(y == cls)[0]
            log(
                f"  R class {cls}: margin filter too tight; "
                f"margin-random uses unfiltered class pool ({len(idx)} rows)"
            )
        if len(idx) == 0:
            raise RuntimeError(f"R pool has no class-{cls} rows")
        if len(idx) < need:
            log(f"  R class {cls}: need {need}, have {len(idx)}; sampling with replacement")
            pick = rng.choice(idx, size=need, replace=True)
        else:
            pick = rng.choice(idx, size=need, replace=False)
        parts.append(pool.iloc[pick])
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)[schema.all_cols]


def _type_keys(schema: Schema, pool: pd.DataFrame) -> np.ndarray:
    cols = list(schema.categorical)
    if not cols:
        return np.full(len(pool), "_", dtype=object)
    if len(cols) == 1:
        return pool[cols[0]].astype(str).to_numpy()
    return pool[cols].astype(str).agg("|".join, axis=1).to_numpy()


def _largest_remainder(labels: np.ndarray, n: int) -> dict[str, int]:
    uniq, counts = np.unique(labels, return_counts=True)
    if n <= 0 or len(uniq) == 0:
        return {str(u): 0 for u in uniq}
    raw = counts * (n / counts.sum())
    floors = np.floor(raw).astype(int)
    rem = int(n - int(floors.sum()))
    order = np.argsort(-(raw - floors), kind="stable")
    floors[order[:rem]] += 1
    return {str(u): int(k) for u, k in zip(uniq.tolist(), floors.tolist())}


def _select_indices(
    idx: np.ndarray, score: np.ndarray, n: int, rng: np.random.Generator, *, rank: bool
) -> np.ndarray:
    if n <= 0 or len(idx) == 0:
        return np.array([], dtype=int)
    if len(idx) <= n:
        return idx
    if rank:
        return idx[np.argsort(-score[idx])[:n]]
    return rng.choice(idx, size=n, replace=False)


def _type_preserving_release(
    pool: pd.DataFrame,
    schema: Schema,
    *,
    keep_margin: np.ndarray,
    score: np.ndarray,
    n0: int,
    n1: int,
    seed: int,
    rank: bool,
) -> pd.DataFrame:
    """Per-class type quotas from the pool mix; rank or sample inside each cell."""
    types = _type_keys(schema, pool)
    y = pool[schema.label].astype(int).to_numpy()
    rng = np.random.default_rng(seed)
    parts = []
    for cls, need in ((0, n0), (1, n1)):
        if need <= 0:
            continue
        class_idx = np.where(y == cls)[0]
        if len(class_idx) == 0:
            raise RuntimeError(f"R pool has no class-{cls} rows")
        quotas = _largest_remainder(types[class_idx], need)
        picked: list[np.ndarray] = []
        unused: list[np.ndarray] = []
        for typ, q in quotas.items():
            if q <= 0:
                continue
            cell = class_idx[types[class_idx] == typ]
            margin_cell = cell[keep_margin[cell]]
            source = margin_cell if len(margin_cell) >= q else cell
            take = _select_indices(source, score, q, rng, rank=rank)
            if len(take) < q:
                rest = np.setdiff1d(cell, take, assume_unique=False)
                take = np.concatenate(
                    [take, _select_indices(rest, score, q - len(take), rng, rank=rank)]
                )
            picked.append(take)
            leftover = np.setdiff1d(cell, take, assume_unique=False)
            if len(leftover):
                unused.append(leftover)
        got = np.concatenate(picked) if picked else np.array([], dtype=int)
        short = need - len(got)
        if short > 0:
            leftover_all = (
                np.concatenate(unused) if unused else class_idx[~np.isin(class_idx, got)]
            )
            leftover_all = leftover_all[~np.isin(leftover_all, got)]
            if len(leftover_all) < short:
                log(
                    f"  R class {cls}: type quotas short {short}; "
                    "sampling remaining class rows with replacement"
                )
                fill = rng.choice(class_idx, size=short, replace=True)
            else:
                fill = _select_indices(leftover_all, score, short, rng, rank=rank)
            got = np.concatenate([got, fill])
        parts.append(pool.iloc[got])
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)[schema.all_cols]


def _topk_release(
    pool: pd.DataFrame,
    schema: Schema,
    *,
    keep_margin: np.ndarray,
    score: np.ndarray,
    n0: int,
    n1: int,
) -> pd.DataFrame:
    y = pool[schema.label].astype(int).to_numpy()
    parts = []
    for cls, need in ((0, n0), (1, n1)):
        if need <= 0:
            continue
        idx = np.where((y == cls) & keep_margin)[0]
        if len(idx) < need:
            idx = np.where(y == cls)[0]
            log(f"  R class {cls}: margin filter too tight; using unfiltered class pool ({len(idx)} rows)")
        if len(idx) == 0:
            raise RuntimeError(f"R pool has no class-{cls} rows")
        ranked = idx[np.argsort(-score[idx])]
        if len(idx) < need:
            log(f"  R class {cls}: need {need}, have {len(idx)}; sampling with replacement")
            extra = np.random.default_rng(cls).choice(idx, size=need - len(idx), replace=True)
            take = np.concatenate([ranked, extra])
        else:
            take = ranked[:need]
        parts.append(pool.iloc[take])
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=0).reset_index(drop=True)[schema.all_cols]


def _select_rate_on_val(
    pool: pd.DataFrame,
    schema: Schema,
    *,
    val: pd.DataFrame,
    train_fraud_rate: float,
    n: int,
    keep_margin: np.ndarray,
    score: np.ndarray,
    rate_grid: list,
    seed: int,
) -> tuple[pd.DataFrame, float, float, list[dict]]:
    """Val chooses a declared fraud budget; rows at that budget are C_priv top-k."""
    best_table = None
    best_pr = -1.0
    best_rate = float(train_fraud_rate)
    grid_log: list[dict] = []
    for raw in rate_grid:
        rate = float(train_fraud_rate if raw is None else raw)
        rate = min(max(rate, 0.0), 1.0)
        n1, n0 = _class_counts(n, rate)
        table = _topk_release(pool, schema, keep_margin=keep_margin, score=score, n0=n0, n1=n1)
        uh = honest_utility(table, val, schema, seed=seed)
        pr = float(uh["pr_auc"])
        rec = float(uh["recall_at_1pct_fpr"])
        log(f"  R rate grid {rate:.4%}  n_fraud={int(table[schema.label].sum())}  val PR-AUC={pr:.3f}")
        grid_log.append(
            {
                "target_fraud_rate": rate,
                "n_fraud": int(table[schema.label].sum()),
                "val_pr_auc": pr,
                "val_recall_at_1pct_fpr": rec,
            }
        )
        if pr > best_pr:
            best_pr = pr
            best_table = table
            best_rate = rate
    assert best_table is not None
    return best_table, best_rate, best_pr, grid_log


def _elite_pool(
    pool: pd.DataFrame,
    schema: Schema,
    *,
    keep_margin: np.ndarray,
    score: np.ndarray,
    n0: int,
    n1: int,
    elite_mult: float,
    seed: int,
) -> pd.DataFrame:
    """Per-class top rows by (U_h - λ U_a), after dropping thin-margin rows."""
    y = pool[schema.label].astype(int).to_numpy()
    parts = []
    for cls, need in ((0, n0), (1, n1)):
        if need <= 0:
            continue
        cap = max(need, int(np.ceil(need * elite_mult)))
        idx = np.where((y == cls) & keep_margin)[0]
        if len(idx) < need:
            idx = np.where(y == cls)[0]
            log(f"  R class {cls}: margin filter too tight ({len(idx)} rows); using unfiltered class pool")
        if len(idx) == 0:
            raise RuntimeError(f"R pool has no class-{cls} rows")
        ranked = idx[np.argsort(-score[idx])]
        take = ranked[: min(cap, len(ranked))]
        parts.append(pool.iloc[take])
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _select_on_val(
    elite: pd.DataFrame,
    schema: Schema,
    *,
    val: pd.DataFrame,
    n0: int,
    n1: int,
    n_candidates: int,
    seed: int,
) -> tuple[pd.DataFrame, float]:
    n_candidates = max(int(n_candidates), 1)
    best_table = None
    best_pr = -1.0
    for i in range(n_candidates):
        cand = _stratified_take(elite, schema, n0=n0, n1=n1, seed=seed + 100 + i)
        uh = honest_utility(cand, val, schema, seed=seed)
        pr = float(uh["pr_auc"])
        log(f"  R val candidate {i + 1}/{n_candidates}  PR-AUC={pr:.3f}")
        if pr > best_pr:
            best_pr = pr
            best_table = cand
    assert best_table is not None
    return best_table, best_pr


def _stratified_take(
    frame: pd.DataFrame, schema: Schema, *, n0: int, n1: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = frame[schema.label].astype(int)
    parts = []
    for cls, need in ((0, n0), (1, n1)):
        if need <= 0:
            continue
        sub = frame[y == cls]
        if len(sub) == 0:
            raise RuntimeError(f"R stratified take: no class-{cls} rows")
        if len(sub) < need:
            log(f"  R class {cls}: need {need}, have {len(sub)}; sampling with replacement")
            pick = rng.choice(len(sub), size=need, replace=True)
        else:
            pick = rng.choice(len(sub), size=need, replace=False)
        parts.append(sub.iloc[pick])
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)[schema.all_cols]
