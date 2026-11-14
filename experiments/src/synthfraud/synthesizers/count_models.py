"""Count-block joints that are not a row copy of (C1, C2, C13, ProductCD)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from synthfraud.schema import IEEE_CIS_COUNT_BLOCK, IEEE_CIS_COUNT_CAT, IEEE_CIS_COUNT_NUMERIC, Schema

_COUNT_NUM = list(IEEE_CIS_COUNT_NUMERIC)
_EPS = 1e-6


def _require_count_schema(schema: Schema) -> None:
    missing = [c for c in IEEE_CIS_COUNT_BLOCK if c not in schema.all_cols]
    if missing:
        raise ValueError(f"count block {missing} not in schema")


def _rest_cols(schema: Schema) -> list[str]:
    joint = set(IEEE_CIS_COUNT_BLOCK)
    return [c for c in schema.feature_cols if c not in joint]


class _GaussianCopula:
    """Empirical-marginal Gaussian copula on the three count numerics."""

    def __init__(self, x: np.ndarray, corr_keep: float = 1.0) -> None:
        if not 0.0 <= corr_keep <= 1.0:
            raise ValueError(f"corr_keep must be in [0, 1], got {corr_keep}")
        self.n, self.d = x.shape
        self.corr_keep = float(corr_keep)
        self.sorted = [np.sort(x[:, j]) for j in range(self.d)]
        self.constant = [float(self.sorted[j][0]) if self.sorted[j][-1] <= self.sorted[j][0] else None for j in range(self.d)]
        z = np.zeros_like(x, dtype=float)
        active: list[int] = []
        for j in range(self.d):
            if self.constant[j] is not None:
                continue
            ranks = rankdata(x[:, j], method="average") / (self.n + 1.0)
            ranks = np.clip(ranks, _EPS, 1.0 - _EPS)
            z[:, j] = norm.ppf(ranks)
            active.append(j)
        self.active = active
        self.chol: np.ndarray | None = None
        if len(active) >= 2:
            corr = np.corrcoef(z[:, active], rowvar=False)
            if np.ndim(corr) == 0:
                corr = np.array([[1.0]])
            corr = np.atleast_2d(np.nan_to_num(corr, nan=0.0))
            np.fill_diagonal(corr, 1.0)
            corr = 0.5 * (corr + corr.T)
            dim = corr.shape[0]
            corr = self.corr_keep * corr + (1.0 - self.corr_keep) * np.eye(dim)
            w, v = np.linalg.eigh(corr)
            w = np.clip(w, 1e-8, None)
            recon = (v * w) @ v.T
            self.chol = np.linalg.cholesky(recon + 1e-10 * np.eye(len(active)))

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        out = np.zeros((n, self.d), dtype=float)
        u = rng.random((n, self.d))
        if self.chol is not None and self.active:
            g = rng.standard_normal((n, len(self.active))) @ self.chol.T
            u[:, self.active] = norm.cdf(g)
        for j in range(self.d):
            if self.constant[j] is not None:
                out[:, j] = self.constant[j]
                continue
            grid = (np.arange(1, self.n + 1, dtype=float)) / (self.n + 1.0)
            out[:, j] = np.interp(np.clip(u[:, j], grid[0], grid[-1]), grid, self.sorted[j])
        return out


def _fit_copulas(src: pd.DataFrame, ycol: str, corr_keep: float) -> tuple[dict, dict, dict]:
    """class copula, (class, product) copula, product categorical given class."""
    class_c: dict[int, _GaussianCopula] = {}
    cell_c: dict[tuple[int, object], _GaussianCopula] = {}
    prod: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for cls in (0, 1):
        slice_y = src.loc[src[ycol].eq(cls)]
        x = slice_y[_COUNT_NUM].to_numpy(dtype=float)
        class_c[cls] = _GaussianCopula(x, corr_keep=corr_keep)
        vals, counts = np.unique(slice_y[IEEE_CIS_COUNT_CAT].to_numpy(), return_counts=True)
        prod[cls] = (vals, counts.astype(float) / counts.sum())
        for val in vals:
            part = slice_y.loc[slice_y[IEEE_CIS_COUNT_CAT].eq(val)]
            if len(part) < 30:
                continue
            cell_c[(cls, val)] = _GaussianCopula(
                part[_COUNT_NUM].to_numpy(dtype=float), corr_keep=corr_keep
            )
    return class_c, cell_c, prod


class ClassConditionalCountCopulaSynthesizer:
    """Gaussian copula on C1/C2/C13 given (y, ProductCD); independent columns elsewhere.

    Not a privacy mechanism. Tests whether modeled count joints match the row copy.
    """

    def __init__(self, seed: int = 0, schema: Schema | None = None, corr_keep: float = 1.0) -> None:
        self.seed = seed
        self.schema = schema
        self.corr_keep = float(corr_keep)
        self._train: pd.DataFrame | None = None
        self._class_c: dict[int, _GaussianCopula] = {}
        self._cell_c: dict[tuple[int, object], _GaussianCopula] = {}
        self._prod: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, train: pd.DataFrame) -> None:
        if self.schema is None:
            raise RuntimeError("schema is required")
        _require_count_schema(self.schema)
        self._train = train.reset_index(drop=True)[list(self.schema.all_cols)]
        self._class_c, self._cell_c, self._prod = _fit_copulas(
            self._train, self.schema.label, self.corr_keep
        )

    def sample(self, n: int) -> pd.DataFrame:
        if self._train is None or self.schema is None:
            raise RuntimeError("call fit() with a schema before sample()")
        rng = np.random.default_rng(self.seed)
        return _sample_count_model(
            rng=rng,
            n=n,
            train=self._train,
            schema=self.schema,
            draw_block=lambda rng, src, take, cls: self._draw_block(rng, take, cls),
        )

    def _draw_block(self, rng: np.random.Generator, take: int, cls: int) -> dict[str, np.ndarray]:
        vals, p = self._prod[cls]
        product = rng.choice(vals, size=take, p=p)
        counts = np.zeros((take, 3), dtype=float)
        for val in np.unique(product):
            mask = product == val
            model = self._cell_c.get((cls, val), self._class_c[cls])
            counts[mask] = model.sample(rng, int(mask.sum()))
        out = {col: counts[:, j] for j, col in enumerate(_COUNT_NUM)}
        out[IEEE_CIS_COUNT_CAT] = product
        return out


@dataclass
class _HistClass:
    edges: list[np.ndarray]
    keys: np.ndarray
    p: np.ndarray
    products: np.ndarray
    bins: np.ndarray


def _fit_hist(src: pd.DataFrame, ycol: str, n_bins: int) -> dict[int, _HistClass]:
    out: dict[int, _HistClass] = {}
    for cls in (0, 1):
        slice_y = src.loc[src[ycol].eq(cls)]
        x = slice_y[_COUNT_NUM].to_numpy(dtype=float)
        edges = []
        codes = np.zeros((len(slice_y), 3), dtype=np.int16)
        for j, col in enumerate(_COUNT_NUM):
            e = np.unique(np.quantile(x[:, j], np.linspace(0.0, 1.0, n_bins + 1)))
            if len(e) < 2:
                e = np.array([float(e[0]), float(e[0]) + 1.0], dtype=float)
            edges.append(e)
            if len(e) == 2:
                codes[:, j] = 0
            else:
                codes[:, j] = np.clip(np.digitize(x[:, j], e[1:-1], right=True), 0, len(e) - 2)
        product = slice_y[IEEE_CIS_COUNT_CAT].to_numpy()
        frame = pd.DataFrame(codes, columns=_COUNT_NUM)
        frame[IEEE_CIS_COUNT_CAT] = product
        grouped = frame.value_counts()
        products = np.array([k[-1] for k in grouped.index], dtype=object)
        bins = np.array([k[:-1] for k in grouped.index], dtype=np.int16)
        p = grouped.to_numpy(dtype=float)
        p = p / p.sum()
        out[cls] = _HistClass(edges=edges, keys=np.arange(len(p)), p=p, products=products, bins=bins)
    return out


class ClassConditionalCountHistSynthesizer:
    """Quantile histogram on C1/C2/C13 × ProductCD given y; uniform inside bins.

    Coarsens the count block so released tuples are not training rows.
    Not a privacy mechanism (no DP accounting).
    """

    def __init__(self, seed: int = 0, schema: Schema | None = None, n_bins: int = 8) -> None:
        self.seed = seed
        self.schema = schema
        self.n_bins = n_bins
        self._train: pd.DataFrame | None = None
        self._hist: dict[int, _HistClass] = {}

    def fit(self, train: pd.DataFrame) -> None:
        if self.schema is None:
            raise RuntimeError("schema is required")
        _require_count_schema(self.schema)
        self._train = train.reset_index(drop=True)[list(self.schema.all_cols)]
        self._hist = _fit_hist(self._train, self.schema.label, self.n_bins)

    def sample(self, n: int) -> pd.DataFrame:
        if self._train is None or self.schema is None:
            raise RuntimeError("call fit() with a schema before sample()")
        rng = np.random.default_rng(self.seed)
        return _sample_count_model(
            rng=rng,
            n=n,
            train=self._train,
            schema=self.schema,
            draw_block=lambda rng, src, take, cls: self._draw_block(rng, take, cls),
        )

    def _draw_block(self, rng: np.random.Generator, take: int, cls: int) -> dict[str, np.ndarray]:
        hist = self._hist[cls]
        pick = rng.choice(len(hist.p), size=take, p=hist.p)
        product = hist.products[pick]
        counts = np.zeros((take, 3), dtype=float)
        for j, edges in enumerate(hist.edges):
            b = hist.bins[pick, j]
            lo = edges[b]
            hi = edges[np.minimum(b + 1, len(edges) - 1)]
            span = hi - lo
            counts[:, j] = np.where(span <= 0, lo, lo + rng.random(take) * span)
        out = {col: counts[:, j] for j, col in enumerate(_COUNT_NUM)}
        out[IEEE_CIS_COUNT_CAT] = product
        return out


def _sample_count_model(*, rng, n: int, train: pd.DataFrame, schema: Schema, draw_block) -> pd.DataFrame:
    ycol = schema.label
    y = rng.choice(train[ycol].to_numpy(), size=n, replace=True)
    rest = _rest_cols(schema)
    chunks: list[pd.DataFrame] = []
    for cls in (0, 1):
        take = int((y == cls).sum())
        if take == 0:
            continue
        src = train.loc[train[ycol].eq(cls)]
        if src.empty:
            raise RuntimeError(f"no training rows with {ycol}={cls}")
        part = draw_block(rng, src, take, cls)
        for col in rest:
            part[col] = rng.choice(src[col].to_numpy(), size=take, replace=True)
        part[ycol] = np.full(take, cls, dtype=src[ycol].dtype)
        chunks.append(pd.DataFrame(part))
    frame = pd.concat(chunks, ignore_index=True)[schema.all_cols]
    return frame.sample(frac=1.0, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)
