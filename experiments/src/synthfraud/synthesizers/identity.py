"""Resample the training table — utility ceiling, zero privacy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from synthfraud.schema import Schema


class IdentitySynthesizer:
    def __init__(self, seed: int = 0, schema: Schema | None = None) -> None:
        self.seed = seed
        self.schema = schema
        self._train: pd.DataFrame | None = None

    def fit(self, train: pd.DataFrame) -> None:
        self._train = train.reset_index(drop=True)

    def sample(self, n: int) -> pd.DataFrame:
        if self._train is None:
            raise RuntimeError("call fit() before sample()")
        rng = np.random.default_rng(self.seed)
        idx = rng.integers(0, len(self._train), size=n)
        return self._train.iloc[idx].reset_index(drop=True)
