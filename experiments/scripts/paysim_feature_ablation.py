"""PaySim feature ablation: is the bank scorer just public empty-origin physics?"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from synthfraud.detect import encode_fit_detector, predict_proba
from synthfraud.log import log
from synthfraud.paysim import load_paysim
from synthfraud.protocol import split_table
from synthfraud.schema import PAYSIM, Schema

ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT.parent / "data" / "raw" / "paysim.parquet"


def slice_schema(name: str, numeric: tuple[str, ...]) -> Schema:
    return Schema(name="paysim", numeric=numeric, categorical=("type",), label=PAYSIM.label)


SLICES = {
    "full": PAYSIM,
    "origin": slice_schema("origin", ("amount", "oldbalanceOrg", "newbalanceOrig")),
    "no_dest": slice_schema(
        "no_dest", ("step", "amount", "oldbalanceOrg", "newbalanceOrig")
    ),
    "type_amount": slice_schema("type_amount", ("amount",)),
    "dest": slice_schema("dest", ("oldbalanceDest", "newbalanceDest")),
    "step_type_amount": slice_schema("step_type_amount", ("step", "amount")),
}


def public_empty_origin_score(frame) -> np.ndarray:
    """Public PaySim fraud physics: TRANSFER/CASH_OUT that empty the origin."""
    types = frame["type"].astype(str).isin(("TRANSFER", "CASH_OUT"))
    empty = frame["newbalanceOrig"].astype(float).eq(0.0)
    match_amt = (frame["oldbalanceOrg"].astype(float) - frame["amount"].astype(float)).abs() < 1e-6
    return (types & empty & match_amt).to_numpy(dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(description="PaySim public-physics feature ablation")
    parser.add_argument("--n-rows", type=int, default=500_000, help="-1 is the full dump")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    seed = args.seed
    n_rows = args.n_rows
    tag = "full dump" if n_rows < 0 else f"{n_rows:,} rows"
    log(f"load natural-rate PaySim {tag} seed={seed}")
    table = load_paysim(DUMP, seed=seed, protocol="natural", n_rows=n_rows)
    splits = split_table(
        table, PAYSIM, seed=seed, train=0.50, ref=0.20, val=0.15, test=0.15
    )
    y = splits.test[PAYSIM.label].to_numpy().astype(int)
    log(f"test n={len(splits.test):,} fraud={int(y.sum())}")
    bank = encode_fit_detector(splits.train, PAYSIM, seed=seed, kind="hgb")
    bank_p = predict_proba(bank, splits.test, PAYSIM)
    bank_hat = (bank_p >= 0.5).astype(int)
    rows = []
    for name, schema in SLICES.items():
        log(f"fit HGB on {name} {schema.feature_cols}")
        model = encode_fit_detector(splits.train, schema, seed=seed, kind="hgb")
        p = predict_proba(model, splits.test, schema)
        hat = (p >= 0.5).astype(int)
        rows.append(
            {
                "slice": name,
                "features": schema.feature_cols,
                "pr_auc": float(average_precision_score(y, p)),
                "agreement_with_full_bank": float((hat == bank_hat).mean()),
            }
        )
        log(f"  {name} PR-AUC={rows[-1]['pr_auc']:.3f}  agr={rows[-1]['agreement_with_full_bank']:.3f}")
    rule = public_empty_origin_score(splits.test)
    rule_hat = (rule >= 0.5).astype(int)
    tp = int(((rule_hat == 1) & (y == 1)).sum())
    fp = int(((rule_hat == 1) & (y == 0)).sum())
    fn = int(((rule_hat == 0) & (y == 1)).sum())
    tn = int(((rule_hat == 0) & (y == 0)).sum())
    rows.append(
        {
            "slice": "public_empty_origin_rule",
            "features": ["type", "amount", "oldbalanceOrg", "newbalanceOrig"],
            "pr_auc": float(average_precision_score(y, rule)),
            "agreement_with_full_bank": float((rule_hat == bank_hat).mean()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
            "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        }
    )
    log(
        f"  public_empty_origin_rule PR-AUC={rows[-1]['pr_auc']:.3f}  "
        f"P={rows[-1]['precision']:.3f} R={rows[-1]['recall']:.3f}  "
        f"tp={tp} fp={fp} fn={fn}"
    )

    # Table-wide physics (same draw as the HGB slices, not just test).
    fraud = table[table[PAYSIM.label] == 1]
    legit = table[table[PAYSIM.label] == 0]
    types = fraud["type"].astype(str).isin(("TRANSFER", "CASH_OUT"))
    empty = fraud["newbalanceOrig"].astype(float).eq(0.0)
    match_amt = (
        fraud["oldbalanceOrg"].astype(float) - fraud["amount"].astype(float)
    ).abs() < 1e-6
    physics = {
        "n_table": int(len(table)),
        "n_fraud": int(len(fraud)),
        "fraud_empty_origin": float(empty.mean()),
        "fraud_oldbalance_eq_amount": float(match_amt.mean()),
        "fraud_public_identity": float((types & empty & match_amt).mean()),
        "legit_public_identity": float(
            (
                legit["type"].astype(str).isin(("TRANSFER", "CASH_OUT"))
                & legit["newbalanceOrig"].astype(float).eq(0.0)
                & (
                    (
                        legit["oldbalanceOrg"].astype(float)
                        - legit["amount"].astype(float)
                    ).abs()
                    < 1e-6
                )
            ).mean()
        ),
    }
    log(
        f"table physics: fraud identity={physics['fraud_public_identity']:.3f}  "
        f"legit identity={physics['legit_public_identity']:.4f}"
    )
    report = {
        "run": {"utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), "python": sys.version.split()[0]},
        "protocol": "natural",
        "n_rows": int(len(table)),
        "seed": seed,
        "n_test_fraud": int(y.sum()),
        "full_bank_pr_auc": float(average_precision_score(y, bank_p)),
        "slices": rows,
        "physics": physics,
        "note": (
            "Real train/test only; no synthesizer. 0.5-threshold agreement is near 1.0 "
            "because both the bank HGB and most slices predict almost all negatives at "
            "this prevalence; use PR-AUC and the rule confusion matrix instead."
        ),
    }
    if n_rows < 0:
        stem = "paysim_full_feature_ablation"
    elif n_rows == 500_000:
        stem = "paysim_natural_feature_ablation"
    else:
        stem = f"paysim_natural_feature_ablation_n{n_rows}"
    out = ROOT / "results" / f"{stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
