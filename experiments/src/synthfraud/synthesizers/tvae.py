"""TVAE baseline (Xu et al., NeurIPS 2019, same paper as CTGAN). Optional extra: pip install ctgan."""

from __future__ import annotations

import pandas as pd

from synthfraud.schema import Schema
from synthfraud.log import log


class TVAESynthesizer:
    def __init__(self, seed: int, schema: Schema, epochs: int = 50) -> None:
        self.seed = seed
        self.schema = schema
        self.epochs = epochs
        self._model = None

    def fit(self, train: pd.DataFrame) -> None:
        try:
            from ctgan import TVAE
        except ImportError as exc:
            raise ImportError("TVAE baseline needs `pip install ctgan`") from exc
        discrete = list(self.schema.categorical) + [self.schema.label]
        model = TVAE(epochs=self.epochs, verbose=True)
        model.set_random_state(self.seed)
        log(
            f"TVAE fit: {len(train):,} rows, {self.epochs} epochs "
            f"({'same epoch budget as CTGAN T1' if self.epochs == 50 else 'library default is 300'})."
        )
        model.fit(train[self.schema.all_cols], discrete_columns=discrete)
        self._model = model
        log("TVAE fit finished")

    def sample(self, n: int) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("call fit() before sample()")
        return self._model.sample(n)
