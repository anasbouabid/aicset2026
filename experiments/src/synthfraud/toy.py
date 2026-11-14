"""Simulated private-bank table for harness development (not PaySim)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from synthfraud.schema import CHANNELS, COUNTRIES, MERCHANT_CATS, TOY


def make_toy_transactions(
    n_rows: int,
    fraud_rate: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n_fraud = int(round(n_rows * fraud_rate))
    n_legit = n_rows - n_fraud
    legit = _draw_legit(n_legit, rng)
    fraud = _draw_fraud(n_fraud, rng)
    frame = pd.concat([legit, fraud], ignore_index=True)
    return frame.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(
        drop=True
    )


def _draw_legit(n: int, rng: np.random.Generator) -> pd.DataFrame:
    hour = rng.integers(7, 22, size=n)
    return pd.DataFrame(
        {
            "amount": np.clip(rng.lognormal(mean=3.2, sigma=0.6, size=n), 1.0, 2_000.0),
            "hour": hour,
            "n_tx_24h": rng.integers(1, 8, size=n),
            "merchant_cat": rng.choice(MERCHANT_CATS, size=n, p=[0.4, 0.25, 0.15, 0.1, 0.1]),
            "country": rng.choice(COUNTRIES[:3], size=n),
            "channel": rng.choice(CHANNELS, size=n, p=[0.7, 0.15, 0.15]),
            TOY.label: 0,
        }
    )


def _draw_fraud(n: int, rng: np.random.Generator) -> pd.DataFrame:
    hour = rng.choice(np.array([0, 1, 2, 3, 4, 23]), size=n)
    return pd.DataFrame(
        {
            "amount": np.clip(rng.lognormal(mean=5.5, sigma=0.7, size=n), 50.0, 15_000.0),
            "hour": hour,
            "n_tx_24h": rng.integers(4, 20, size=n),
            "merchant_cat": rng.choice(MERCHANT_CATS, size=n, p=[0.1, 0.1, 0.35, 0.3, 0.15]),
            "country": rng.choice(COUNTRIES, size=n, p=[0.1, 0.1, 0.1, 0.35, 0.35]),
            "channel": rng.choice(CHANNELS, size=n, p=[0.15, 0.7, 0.15]),
            TOY.label: 1,
        }
    )
