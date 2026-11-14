from __future__ import annotations

import pandas as pd
from scipy.stats import wasserstein_distance

from synthfraud.schema import Schema, constraint_violations


def fidelity(real: pd.DataFrame, synth: pd.DataFrame, schema: Schema) -> dict:
    w1 = {
        col: float(wasserstein_distance(real[col].to_numpy(), synth[col].to_numpy()))
        for col in schema.numeric
    }
    tvd = {col: _tvd(real[col], synth[col]) for col in schema.categorical}
    n = len(synth)
    n_bad = sum(1 for rec in synth.to_dict(orient="records") if constraint_violations(schema, rec))
    return {
        "wasserstein1": w1,
        "total_variation": tvd,
        "constraint_violation_rate": n_bad / max(n, 1),
    }


def _tvd(a: pd.Series, b: pd.Series) -> float:
    pa = a.value_counts(normalize=True)
    pb = b.value_counts(normalize=True)
    keys = sorted(set(pa.index) | set(pb.index))
    return 0.5 * float(sum(abs(pa.get(k, 0.0) - pb.get(k, 0.0)) for k in keys))
