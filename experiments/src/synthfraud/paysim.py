"""Load the Lopez-Rojas PaySim dump (stand-in private bank)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from synthfraud.schema import PAYSIM

ALIASES = {
    "isfraud": "isFraud",
    "is_fraud": "isFraud",
    "oldbalanceorig": "oldbalanceOrg",
    "oldbalanceorg": "oldbalanceOrg",
    "newbalanceorig": "newbalanceOrig",
    "oldbalancedest": "oldbalanceDest",
    "newbalancedest": "newbalanceDest",
}

# Full PaySim fraud rate is about 0.13%. Enriched != paper protocol.
PROTOCOLS = ("enriched", "natural")


def load_paysim(
    path: Path,
    *,
    seed: int,
    protocol: str = "enriched",
    n_legit: int = 40000,
    all_fraud: bool = True,
    n_rows: int = 500_000,
) -> pd.DataFrame:
    """Read parquet/csv, drop IDs, subsample.

    ``enriched``: every fraud row plus ``n_legit`` legitimate rows (harness default).
    ``natural``: stratified draw that matches the dump's fraud rate; ``n_rows=-1`` is the
    full table. Both are kept; do not overwrite one with the other.
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown PaySim protocol {protocol!r}; use {PROTOCOLS}")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"PaySim dump not found at {path}. Run experiments/scripts/fetch_paysim.py"
        )
    if path.suffix == ".parquet":
        raw = pd.read_parquet(path)
    else:
        raw = pd.read_csv(path)
    raw = _normalize_columns(raw)
    keep = list(PAYSIM.all_cols)
    missing = [c for c in keep if c not in raw.columns]
    if missing:
        raise ValueError(f"PaySim file missing columns {missing}; got {list(raw.columns)}")
    frame = raw[keep].copy()
    frame["type"] = frame["type"].astype(str)
    frame[PAYSIM.label] = frame[PAYSIM.label].astype(int)
    if protocol == "natural":
        return _natural(frame, seed=seed, n_rows=n_rows)
    return _enriched(frame, seed=seed, n_legit=n_legit, all_fraud=all_fraud)


def _enriched(frame: pd.DataFrame, *, seed: int, n_legit: int, all_fraud: bool) -> pd.DataFrame:
    if n_legit < 0:
        return frame.reset_index(drop=True)
    label = PAYSIM.label
    fraud = frame[frame[label] == 1]
    legit = frame[frame[label] == 0]
    rng = np.random.default_rng(seed)
    n = min(n_legit, len(legit))
    legit_s = legit.iloc[rng.choice(len(legit), size=n, replace=False)]
    out = pd.concat([fraud, legit_s], ignore_index=True) if all_fraud else legit_s
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _natural(frame: pd.DataFrame, *, seed: int, n_rows: int) -> pd.DataFrame:
    if n_rows < 0 or n_rows >= len(frame):
        return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    label = PAYSIM.label
    fraud = frame[frame[label] == 1]
    legit = frame[frame[label] == 0]
    rate = len(fraud) / len(frame)
    n_fraud = int(round(n_rows * rate))
    n_fraud = min(max(n_fraud, 1), len(fraud))
    n_legit = min(n_rows - n_fraud, len(legit))
    rng = np.random.default_rng(seed)
    fraud_s = fraud.iloc[rng.choice(len(fraud), size=n_fraud, replace=False)]
    legit_s = legit.iloc[rng.choice(len(legit), size=n_legit, replace=False)]
    out = pd.concat([fraud_s, legit_s], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for col in frame.columns:
        key = col.replace(" ", "").lower()
        if key in ALIASES:
            mapping[col] = ALIASES[key]
    return frame.rename(columns=mapping)
