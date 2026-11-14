from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_curve

from synthfraud.detect import encode_fit_detector, predict_proba
from synthfraud.schema import Schema


def honest_utility_slices(
    synth: pd.DataFrame, real_test: pd.DataFrame, slices: dict[str, Schema], *, seed: int
) -> dict[str, dict]:
    """U_h on feature subsets of the same release (does G transfer counts?)."""
    out: dict[str, dict] = {}
    for name, schema in slices.items():
        missing = [c for c in schema.all_cols if c not in synth.columns or c not in real_test.columns]
        if missing:
            out[name] = {"skipped": True, "missing": missing}
            continue
        stats = honest_utility(synth, real_test, schema, seed=seed)
        stats["n_features"] = len(schema.feature_cols)
        out[name] = stats
    return out


def honest_utility(
    synth: pd.DataFrame, real_test: pd.DataFrame, schema: Schema, *, seed: int
) -> dict:
    model = encode_fit_detector(synth, schema, seed=seed, kind="hgb")
    scores = predict_proba(model, real_test, schema)
    y = real_test[schema.label].to_numpy().astype(int)
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "recall_at_1pct_fpr": _recall_at_fpr(y, scores, 0.01),
        "n_test": int(len(real_test)),
        "n_test_fraud": int(y.sum()),
    }


def _recall_at_fpr(y: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y, scores)
    ok = np.where(fpr <= target_fpr)[0]
    if len(ok) == 0:
        return 0.0
    return float(tpr[ok[-1]])
