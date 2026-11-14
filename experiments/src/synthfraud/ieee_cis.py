"""Load the IEEE-CIS / Vesta labeled train tables (C2 corpus)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from synthfraud.schema import IEEE_CIS

PROTOCOLS = ("natural",)
TX_NAME = "train_transaction.csv"
ID_NAME = "train_identity.csv"
PARQUET_NAME = "ieee_cis.parquet"


def load_ieee_cis(
    path: Path,
    *,
    seed: int,
    protocol: str = "natural",
    n_rows: int = 500_000,
) -> pd.DataFrame:
    """Read transaction+identity, keep IEEE_CIS columns, stratified subsample.

    ``natural``: stratified draw at the dump fraud rate; ``n_rows=-1`` is the full
    labeled train table. IDs are dropped. Categorical NA is the public token
    ``missing`` (not a private imputation).
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown IEEE-CIS protocol {protocol!r}; use {PROTOCOLS}")
    path = Path(path)
    frame = _read_table(path)
    keep = list(IEEE_CIS.all_cols)
    missing = [c for c in keep if c not in frame.columns]
    if missing:
        raise ValueError(f"IEEE-CIS table missing columns {missing}; got {list(frame.columns)}")
    out = frame[keep].copy()
    for col in IEEE_CIS.categorical:
        out[col] = out[col].astype("string").fillna("missing").astype(str)
    for col in IEEE_CIS.numeric:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if out[col].isna().any():
            out[col] = out[col].fillna(float(out[col].median()))
    out[IEEE_CIS.label] = out[IEEE_CIS.label].astype(int)
    if protocol == "natural":
        return _natural(out, seed=seed, n_rows=n_rows)
    raise ValueError(protocol)


def _read_table(path: Path) -> pd.DataFrame:
    if path.is_file() and path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.is_dir():
        parquet = path / PARQUET_NAME
        if parquet.is_file():
            return pd.read_parquet(parquet)
        tx = path / TX_NAME
        ident = path / ID_NAME
        if not tx.is_file():
            raise FileNotFoundError(
                f"IEEE-CIS dump not found at {tx}. Run experiments/scripts/fetch_ieee_cis.py"
            )
        raw = pd.read_csv(tx)
        if ident.is_file():
            extra = pd.read_csv(ident, usecols=lambda c: c in {"TransactionID", "DeviceType"})
            raw = raw.merge(extra, on="TransactionID", how="left")
        keep = [c for c in raw.columns if c in set(IEEE_CIS.all_cols) | {"TransactionID"}]
        slim = raw[keep].copy()
        slim.to_parquet(parquet, index=False)
        return slim
    raise FileNotFoundError(
        f"IEEE-CIS dump not found at {path}. Run experiments/scripts/fetch_ieee_cis.py"
    )


def _natural(frame: pd.DataFrame, *, seed: int, n_rows: int) -> pd.DataFrame:
    if n_rows < 0 or n_rows >= len(frame):
        return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    label = IEEE_CIS.label
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
