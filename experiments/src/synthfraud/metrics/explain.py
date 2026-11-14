"""C4: public certificates plus explanation preservation on real held-out rows."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from synthfraud.schema import Schema


def map_transformed_to_original(feature_names: np.ndarray | list[str], schema: Schema) -> list[str]:
    """Map ColumnTransformer names (num__C1, cat__ProductCD_W) back to schema columns."""
    cats = list(schema.categorical)
    mapped: list[str] = []
    for raw in feature_names:
        name = str(raw)
        if name.startswith("num__"):
            mapped.append(name[5:])
            continue
        if name.startswith("cat__"):
            rest = name[5:]
            hit = None
            for col in sorted(cats, key=len, reverse=True):
                if rest == col or rest.startswith(col + "_"):
                    hit = col
                    break
            mapped.append(hit if hit is not None else rest)
            continue
        mapped.append(name)
    return mapped


def mean_abs_shap_by_feature(model, frame, schema: Schema, *, seed: int, max_rows: int | None = None) -> dict[str, float]:
    """Mean |SHAP| on *frame* (callers pass real test fraud), grouped by original column.

    Explains the HGB on the transformed matrix. Not a privacy claim and not SHAP-on-synth.
    """
    import shap

    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    x = frame[schema.feature_cols]
    if max_rows is not None and len(x) > max_rows:
        x = x.sample(n=max_rows, random_state=seed)
    xt = np.asarray(prep.transform(x), dtype=float)
    explainer = shap.TreeExplainer(clf)
    values = explainer.shap_values(xt)
    if isinstance(values, list):
        values = values[-1]
    values = np.abs(np.asarray(values, dtype=float))
    if values.ndim == 3:
        values = values[:, :, -1]
    names = map_transformed_to_original(prep.get_feature_names_out(), schema)
    acc = {c: 0.0 for c in schema.feature_cols}
    for j, col in enumerate(names):
        if col not in acc:
            continue
        acc[col] += float(values[:, j].mean())
    return acc


def rank_correlation(a: dict[str, float], b: dict[str, float], cols: list[str]) -> dict:
    va = np.array([a[c] for c in cols], dtype=float)
    vb = np.array([b[c] for c in cols], dtype=float)
    rho, pval = spearmanr(va, vb)
    ra = np.argsort(-va)
    rb = np.argsort(-vb)
    top3_a = [cols[i] for i in ra[:3]]
    top3_b = [cols[i] for i in rb[:3]]
    return {
        "spearman_rho": float(rho) if rho == rho else None,
        "spearman_p": float(pval) if pval == pval else None,
        "top3_bank": top3_a,
        "top3_synth": top3_b,
        "top3_overlap": sorted(set(top3_a) & set(top3_b)),
    }
