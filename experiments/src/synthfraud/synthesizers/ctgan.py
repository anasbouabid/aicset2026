"""CTGAN baseline (Xu et al., NeurIPS 2019). Optional extra: pip install ctgan."""

from __future__ import annotations

import pandas as pd

from synthfraud.schema import Schema
from synthfraud.log import log


class CTGANSynthesizer:
    def __init__(self, seed: int, schema: Schema, epochs: int = 50) -> None:
        self.seed = seed
        self.schema = schema
        self.epochs = epochs
        self._model = None

    def fit(self, train: pd.DataFrame) -> None:
        try:
            from ctgan import CTGAN
        except ImportError as exc:
            raise ImportError("CTGAN baseline needs `pip install ctgan`") from exc
        discrete = list(self.schema.categorical) + [self.schema.label]
        model = CTGAN(epochs=self.epochs, verbose=True, pac=1)
        model.set_random_state(self.seed)
        log(
            f"CTGAN fit: {len(train):,} rows, {self.epochs} epochs. "
            "A transformer/GMM step runs first (no bar); then an epoch bar."
        )
        model.fit(train[self.schema.all_cols], discrete_columns=discrete)
        self._model = model
        log("CTGAN fit finished")

    def sample(self, n: int) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("call fit() before sample()")
        return self._model.sample(n)
