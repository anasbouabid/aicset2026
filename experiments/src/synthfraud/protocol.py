"""Train / ref / val / test protocol."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from synthfraud.schema import Schema


@dataclass(frozen=True)
class Splits:
    train: pd.DataFrame
    ref: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def split_table(
    frame: pd.DataFrame,
    schema: Schema,
    *,
    train: float,
    ref: float,
    val: float,
    test: float,
    seed: int,
) -> Splits:
    total = train + ref + val + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1, got {total}")
    label = schema.label
    rest_frac = 1.0 - train
    train_df, rest = train_test_split(
        frame, test_size=rest_frac, stratify=frame[label], random_state=seed
    )
    ref_share = ref / rest_frac
    ref_df, rest2 = train_test_split(
        rest, test_size=1.0 - ref_share, stratify=rest[label], random_state=seed + 1
    )
    val_share = val / (val + test)
    val_df, test_df = train_test_split(
        rest2, test_size=1.0 - val_share, stratify=rest2[label], random_state=seed + 2
    )
    return Splits(
        train=train_df.reset_index(drop=True),
        ref=ref_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
    )
