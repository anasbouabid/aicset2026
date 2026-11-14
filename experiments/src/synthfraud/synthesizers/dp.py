"""DP independent histograms (Laplace). Fast DP-only floor; not PATE-GAN.

Public numeric bounds spend no budget (PaySim simulator domain). Category
domains are public (`PAYSIM_TYPES`, binary label). Sequential composition
splits ε across columns. δ = 0 (pure ε-DP).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from synthfraud.log import log
from synthfraud.schema import PAYSIM_TYPES, Schema

# Public PaySim domain (Lopez-Rojas simulator). Not estimated from train.
PAYSIM_PUBLIC_BOUNDS = {
    "step": (1.0, 744.0),
    "amount": (0.0, 1.0e8),
    "oldbalanceOrg": (0.0, 1.0e8),
    "newbalanceOrig": (0.0, 1.0e8),
    "oldbalanceDest": (0.0, 1.0e8),
    "newbalanceDest": (0.0, 1.0e8),
}


class DPIndependentSynthesizer:
    """Per-column DP histograms. Cite as DP-only / IndHist+Laplace, not as PATE-GAN."""

    def __init__(
        self,
        seed: int,
        schema: Schema,
        epsilon: float = 10.0,
        n_bins: int = 32,
        skip_label: bool = False,
    ) -> None:
        self.seed = seed
        self.schema = schema
        self.epsilon = float(epsilon)
        self.n_bins = int(n_bins)
        self.skip_label = bool(skip_label)
        self._cols: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.accounting: dict | None = None

    def fit(self, train: pd.DataFrame) -> None:
        rng = np.random.default_rng(self.seed)
        cols = list(self.schema.all_cols)
        if self.skip_label:
            cols = [c for c in cols if c != self.schema.label]
        eps_col = self.epsilon / max(len(cols), 1)
        log(f"DP-independent fit ε={self.epsilon:g}  per-column ε={eps_col:g}  n={len(train):,}")
        for col in cols:
            self._cols[col] = _noisy_histogram(
                train[col],
                name=col,
                schema=self.schema,
                eps=eps_col,
                n_bins=self.n_bins,
                rng=rng,
            )
        self.accounting = {
            "mechanism": "laplace_histogram",
            "epsilon": self.epsilon,
            "delta": 0.0,
            "composition": "sequential_equal_split",
            "n_columns": len(cols),
            "epsilon_per_column": eps_col,
            "n_bins": self.n_bins,
            "bounds": "public_paysim_domain",
        }

    def sample(self, n: int) -> pd.DataFrame:
        if not self._cols:
            raise RuntimeError("call fit() before sample()")
        rng = np.random.default_rng(self.seed + 1)
        out: dict[str, np.ndarray] = {}
        for col, (values, probs) in self._cols.items():
            out[col] = rng.choice(values, size=n, p=probs)
        frame = pd.DataFrame(out)
        if self.schema.name == "paysim":
            if self.schema.label in frame.columns:
                frame[self.schema.label] = frame[self.schema.label].round().astype(int)
            if "step" in frame.columns:
                frame["step"] = frame["step"].round().clip(lower=1).astype(int)
            if "type" in frame.columns:
                frame["type"] = frame["type"].astype(str)
        return frame


class DPClassConditionalSynthesizer:
    """Two DP independents (legit / fraud) plus a DP label rate.

    Spends more of ε on the fraud slice. Label split is *not* parallel
    composition (the label is sensitive); we report sequential ε_total.
    """

    def __init__(
        self,
        seed: int,
        schema: Schema,
        epsilon: float = 10.0,
        epsilon_fraud_share: float = 0.6,
        epsilon_rate: float = 1.0,
        n_bins: int = 32,
    ) -> None:
        self.seed = seed
        self.schema = schema
        self.epsilon = float(epsilon)
        self.epsilon_rate = float(epsilon_rate)
        rest = self.epsilon - self.epsilon_rate
        if rest <= 0:
            raise ValueError("epsilon_rate must be < epsilon")
        self.epsilon_fraud = rest * float(epsilon_fraud_share)
        self.epsilon_legit = rest * (1.0 - float(epsilon_fraud_share))
        self.n_bins = int(n_bins)
        self._p_fraud = 0.0
        self._g0: DPIndependentSynthesizer | None = None
        self._g1: DPIndependentSynthesizer | None = None
        self.accounting: dict | None = None

    def fit(self, train: pd.DataFrame) -> None:
        rng = np.random.default_rng(self.seed)
        y = train[self.schema.label].astype(int).to_numpy()
        n1 = int((y == 1).sum())
        # One Laplace query on the fraud count; n0 = n - n1 is determined.
        noisy1 = n1 + rng.laplace(0.0, 1.0 / self.epsilon_rate)
        self._p_fraud = float(np.clip(noisy1 / max(len(train), 1), 0.0, 1.0))
        log(
            f"DP class-conditional: p_fraud_hat={self._p_fraud:.4%}  "
            f"ε_rate={self.epsilon_rate:g} ε_fraud={self.epsilon_fraud:g} "
            f"ε_legit={self.epsilon_legit:g} (sequential total {self.epsilon:g})"
        )
        self._g0 = DPIndependentSynthesizer(
            seed=self.seed + 10,
            schema=self.schema,
            epsilon=self.epsilon_legit,
            n_bins=self.n_bins,
            skip_label=True,
        )
        self._g1 = DPIndependentSynthesizer(
            seed=self.seed + 20,
            schema=self.schema,
            epsilon=self.epsilon_fraud,
            n_bins=self.n_bins,
            skip_label=True,
        )
        legit = train[y == 0]
        fraud = train[y == 1]
        if len(legit) == 0 or len(fraud) == 0:
            raise RuntimeError("class-conditional DP needs both classes in train")
        self._g0.fit(legit)
        self._g1.fit(fraud)
        self.accounting = {
            "mechanism": "class_conditional_laplace_histogram",
            "epsilon": self.epsilon,
            "delta": 0.0,
            "composition": "sequential_rate_plus_both_classes",
            "epsilon_rate": self.epsilon_rate,
            "epsilon_fraud": self.epsilon_fraud,
            "epsilon_legit": self.epsilon_legit,
            "p_fraud_hat": self._p_fraud,
            "note": "Partition by label is not parallel composition; ε_total is sequential.",
        }

    def sample(self, n: int) -> pd.DataFrame:
        if self._g0 is None or self._g1 is None:
            raise RuntimeError("call fit() before sample()")
        rng = np.random.default_rng(self.seed + 3)
        n1 = int(rng.binomial(n, min(max(self._p_fraud, 0.0), 1.0)))
        n0 = n - n1
        parts = []
        if n0:
            s0 = self._g0.sample(n0)
            s0[self.schema.label] = 0
            parts.append(s0)
        if n1:
            s1 = self._g1.sample(n1)
            s1[self.schema.label] = 1
            parts.append(s1)
        out = pd.concat(parts, ignore_index=True)
        out = out.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
        return out[self.schema.all_cols]


def _noisy_histogram(
    series: pd.Series,
    *,
    name: str,
    schema: Schema,
    eps: float,
    n_bins: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if name in schema.categorical or name == schema.label:
        if name == "type" and schema.name == "paysim":
            domain = list(PAYSIM_TYPES)
        elif name == schema.label:
            domain = [0, 1]
        else:
            domain = sorted(series.astype(str).unique().tolist())
        raw = series.astype(str) if name == "type" else series
        counts = np.array([(raw == v).sum() for v in domain], dtype=float)
        noisy = counts + rng.laplace(0.0, 1.0 / eps, size=len(counts))
        noisy = np.maximum(noisy, 1e-12)
        probs = noisy / noisy.sum()
        values = np.array(domain, dtype=object if name == "type" else float)
        return values, probs

    lo, hi = PAYSIM_PUBLIC_BOUNDS.get(name, (float(series.min()), float(series.max())))
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, n_bins + 1)
    clipped = np.clip(series.astype(float).to_numpy(), lo, hi)
    counts, _ = np.histogram(clipped, bins=edges)
    noisy = counts.astype(float) + rng.laplace(0.0, 1.0 / eps, size=len(counts))
    noisy = np.maximum(noisy, 1e-12)
    probs = noisy / noisy.sum()
    # Sample bin then midpoint (deterministic given bin) — enough for a DP floor.
    mids = 0.5 * (edges[:-1] + edges[1:])
    return mids, probs
