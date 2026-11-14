"""IEEE-CIS public-vs-private signal diagnostic (real train/test, no synthesizer)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score

from synthfraud.detect import encode_fit_detector, predict_proba
from synthfraud.log import log
from synthfraud.ieee_cis import load_ieee_cis
from synthfraud.protocol import split_table
from synthfraud.schema import IEEE_CIS, Schema

ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT.parent / "data" / "raw" / "ieee_cis"


def slice_schema(name: str, numeric: tuple[str, ...], categorical: tuple[str, ...]) -> Schema:
    return Schema(name=name, numeric=numeric, categorical=categorical, label=IEEE_CIS.label)


def main() -> int:
    seed = 0
    log("load IEEE-CIS natural 500k seed=0")
    table = load_ieee_cis(DUMP, seed=seed, protocol="natural", n_rows=500_000)
    log(
        f"table n={len(table):,} fraud={int(table[IEEE_CIS.label].sum()):,} "
        f"rate={float(table[IEEE_CIS.label].mean()):.4%}"
    )
    splits = split_table(
        table, IEEE_CIS, seed=seed, train=0.50, ref=0.20, val=0.15, test=0.15
    )
    y = splits.test[IEEE_CIS.label].to_numpy().astype(int)
    log(f"test n={len(splits.test):,} fraud={int(y.sum())}")

    public_cat = ("ProductCD", "card4", "card6", "P_emaildomain", "DeviceType")
    public_num = ("TransactionAmt", "TransactionDT")
    count_num = tuple(c for c in IEEE_CIS.numeric if c.startswith("C"))
    delay_num = tuple(c for c in IEEE_CIS.numeric if c.startswith("D"))
    v_num = tuple(c for c in IEEE_CIS.numeric if c.startswith("V"))

    slices = {
        "full": IEEE_CIS,
        "public": slice_schema("public", public_num, public_cat),
        "counts": slice_schema("counts", count_num, ("ProductCD",)),
        "delays": slice_schema("delays", delay_num, ("ProductCD",)),
        "v": slice_schema("v", v_num, ("ProductCD",)),
        "amt_product": slice_schema("amt_product", ("TransactionAmt",), ("ProductCD",)),
        "no_counts": IEEE_CIS.drop_numeric(count_num),
        "public_plus_v": slice_schema("public_plus_v", public_num + v_num, public_cat),
        "public_plus_v_d1": slice_schema(
            "public_plus_v_d1", public_num + v_num + delay_num, public_cat
        ),
        "counts_plus_public": slice_schema(
            "counts_plus_public", public_num + count_num, public_cat
        ),
    }

    rows = []
    for name, schema in slices.items():
        log(f"fit HGB on {name} n_feat={len(schema.feature_cols)}")
        model = encode_fit_detector(splits.train, schema, seed=seed, kind="hgb")
        p = predict_proba(model, splits.test, schema)
        rows.append(
            {
                "slice": name,
                "n_features": len(schema.feature_cols),
                "features": schema.feature_cols,
                "pr_auc": float(average_precision_score(y, p)),
            }
        )
        log(f"  {name} PR-AUC={rows[-1]['pr_auc']:.3f}")

    # Public majority ProductCD rule is too weak; score ProductCD=='C' which is
    # the elevated-fraud product in this dump (computed below on the table).
    prod = splits.test["ProductCD"].astype(str)
    fraud_by_prod = (
        table.groupby(table["ProductCD"].astype(str))[IEEE_CIS.label].mean().to_dict()
    )
    hot = max(fraud_by_prod, key=fraud_by_prod.get)
    rule = prod.eq(hot).to_numpy(dtype=float)
    rows.append(
        {
            "slice": f"public_product_{hot}",
            "n_features": 1,
            "features": ["ProductCD"],
            "pr_auc": float(average_precision_score(y, rule)),
            "fraud_rate_by_product": {str(k): float(v) for k, v in fraud_by_prod.items()},
        }
    )
    log(f"  public_product_{hot} PR-AUC={rows[-1]['pr_auc']:.3f}")

    report = {
        "run": {"utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), "python": sys.version.split()[0]},
        "protocol": "natural",
        "n_rows": int(len(table)),
        "n_fraud": int(table[IEEE_CIS.label].sum()),
        "fraud_rate": float(table[IEEE_CIS.label].mean()),
        "seed": seed,
        "n_test_fraud": int(y.sum()),
        "missing_frac": {
            c: float(table[c].isna().mean()) for c in IEEE_CIS.feature_cols
        },
        "slices": rows,
        "note": (
            "Real train/test only; no synthesizer. Same split fractions as PaySim T1. "
            "no_counts = compact schema without C1/C2/C13. If no_counts PR-AUC is near "
            "public, counts are the honest residual and G-on-C is a PaySim-style dead end."
        ),
    }
    out = ROOT / "results" / "ieee_cis_natural_feature_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
