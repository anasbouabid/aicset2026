"""Independent column sampling (IndHist-style). Low fidelity on correlations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from synthfraud.schema import IEEE_CIS_COUNT_BLOCK, Schema


class IndependentSynthesizer:
    def __init__(self, seed: int = 0, schema: Schema | None = None) -> None:
        self.seed = seed
        self.schema = schema
        self._train: pd.DataFrame | None = None

    def fit(self, train: pd.DataFrame) -> None:
        self._train = train.reset_index(drop=True)

    def sample(self, n: int) -> pd.DataFrame:
        if self._train is None or self.schema is None:
            raise RuntimeError("call fit() with a schema before sample()")
        rng = np.random.default_rng(self.seed)
        train = self._train
        out: dict[str, np.ndarray] = {}
        for col in self.schema.all_cols:
            out[col] = rng.choice(train[col].to_numpy(), size=n, replace=True)
        return pd.DataFrame(out)


class ClassConditionalIndependentSynthesizer:
    """Per-class independent columns: transfers P(x_j | y), not P(x | y) joints.

    Used to test whether copying count *marginals given label* recovers U_h
    without a GAN. Not a privacy mechanism.
    """

    def __init__(self, seed: int = 0, schema: Schema | None = None) -> None:
        self.seed = seed
        self.schema = schema
        self._train: pd.DataFrame | None = None

    def fit(self, train: pd.DataFrame) -> None:
        if self.schema is None:
            raise RuntimeError("schema is required")
        self._train = train.reset_index(drop=True)[list(self.schema.all_cols)]

    def sample(self, n: int) -> pd.DataFrame:
        if self._train is None or self.schema is None:
            raise RuntimeError("call fit() with a schema before sample()")
        rng = np.random.default_rng(self.seed)
        train = self._train
        ycol = self.schema.label
        y = rng.choice(train[ycol].to_numpy(), size=n, replace=True)
        chunks: list[pd.DataFrame] = []
        for cls in (0, 1):
            take = int((y == cls).sum())
            if take == 0:
                continue
            src = train.loc[train[ycol].eq(cls)]
            if src.empty:
                raise RuntimeError(f"no training rows with {ycol}={cls}")
            part = {col: rng.choice(src[col].to_numpy(), size=take, replace=True) for col in self.schema.feature_cols}
            part[ycol] = np.full(take, cls, dtype=src[ycol].dtype)
            chunks.append(pd.DataFrame(part))
        frame = pd.concat(chunks, ignore_index=True)[self.schema.all_cols]
        return frame.sample(frac=1.0, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)


class ClassConditionalCountBlockSynthesizer:
    """Per-class row bootstrap on the count block; independent columns elsewhere.

    Tests whether C1/C2/C13/ProductCD *joints* given y recover the real counts
    HGB (~0.41), which independent P(x_j | y) does not. Not a privacy mechanism.
    """

    def __init__(self, seed: int = 0, schema: Schema | None = None) -> None:
        self.seed = seed
        self.schema = schema
        self._train: pd.DataFrame | None = None

    def fit(self, train: pd.DataFrame) -> None:
        if self.schema is None:
            raise RuntimeError("schema is required")
        missing = [c for c in IEEE_CIS_COUNT_BLOCK if c not in self.schema.all_cols]
        if missing:
            raise ValueError(f"count block {missing} not in schema")
        self._train = train.reset_index(drop=True)[list(self.schema.all_cols)]

    def sample(self, n: int) -> pd.DataFrame:
        if self._train is None or self.schema is None:
            raise RuntimeError("call fit() with a schema before sample()")
        rng = np.random.default_rng(self.seed)
        train = self._train
        ycol = self.schema.label
        y = rng.choice(train[ycol].to_numpy(), size=n, replace=True)
        joint = list(IEEE_CIS_COUNT_BLOCK)
        rest = [c for c in self.schema.feature_cols if c not in joint]
        chunks: list[pd.DataFrame] = []
        for cls in (0, 1):
            take = int((y == cls).sum())
            if take == 0:
                continue
            src = train.loc[train[ycol].eq(cls)]
            if src.empty:
                raise RuntimeError(f"no training rows with {ycol}={cls}")
            idx = rng.integers(0, len(src), size=take)
            block = src.iloc[idx][joint].reset_index(drop=True)
            part = {col: block[col].to_numpy() for col in joint}
            for col in rest:
                part[col] = rng.choice(src[col].to_numpy(), size=take, replace=True)
            part[ycol] = np.full(take, cls, dtype=src[ycol].dtype)
            chunks.append(pd.DataFrame(part))
        frame = pd.concat(chunks, ignore_index=True)[self.schema.all_cols]
        return frame.sample(frac=1.0, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)
