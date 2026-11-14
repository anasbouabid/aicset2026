"""Risk B: surrogate extraction + a public-constraint-legal perturbation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from synthfraud.detect import encode_fit_detector, predict_proba
from synthfraud.schema import Schema, public_evasion


def adversarial_utility(
    *,
    bank_model,
    synth: pd.DataFrame,
    real_test: pd.DataFrame,
    schema: Schema,
    seed: int,
    bank_schema: Schema | None = None,
) -> dict:
    """Surrogate is trained on the released schema. Bank stays on bank_schema.

    When they differ (column withholding), agreement is the Risk B question:
    can the released columns copy the production scorer.
    """
    bank_schema = bank_schema or schema
    surrogate = encode_fit_detector(synth, schema, seed=seed, kind="linear")
    bank_p = predict_proba(bank_model, real_test, bank_schema)
    sur_p = predict_proba(surrogate, real_test, schema)
    bank_hat = (bank_p >= 0.5).astype(int)
    sur_hat = (sur_p >= 0.5).astype(int)
    agreement = float((bank_hat == sur_hat).mean())

    fraud = real_test[real_test[schema.label] == 1].copy()
    if len(fraud) == 0:
        asr = float("nan")
        sur_asr = float("nan")
    else:
        attacked = public_evasion(schema, fraud)
        before = (predict_proba(bank_model, fraud, bank_schema) >= 0.5).astype(int)
        after = (predict_proba(bank_model, attacked, bank_schema) >= 0.5).astype(int)
        asr = float(((before == 1) & (after == 0)).mean())
        sur_before = (predict_proba(surrogate, fraud, schema) >= 0.5).astype(int)
        sur_after = (predict_proba(surrogate, attacked, schema) >= 0.5).astype(int)
        sur_asr = float(((sur_before == 1) & (sur_after == 0)).mean())

    margin = np.abs(bank_p - 0.5)
    return {
        "surrogate_label_agreement": agreement,
        "public_evasion_asr": asr,
        "surrogate_evasion_asr": sur_asr,
        "mean_abs_margin_on_test": float(margin.mean()),
        "release_feature_cols": list(schema.feature_cols),
        "bank_feature_cols": list(bank_schema.feature_cols),
        "note": (
            "public_evasion_asr is scored on the real-trained bank detector "
            "(does not move with the synthesizer). surrogate_evasion_asr uses "
            "the synth-trained surrogate on the same public perturbation. "
            "If release_feature_cols is a subset of bank_feature_cols, the "
            "surrogate is the T1 adversary who only has the released columns."
        ),
    }
