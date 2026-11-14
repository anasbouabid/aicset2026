"""IEEE-CIS C4: C_pub certificates + SHAP rank correlation on real test fraud.

T1 product vs unfiltered CTGAN. Does not search a new G. Regenerates the seed-0
tables because releases were not persisted. Do not run on ULB PCA.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from sklearn.metrics import average_precision_score

from synthfraud.c_pub import describe_c_pub
from synthfraud.detect import encode_fit_detector, predict_proba
from synthfraud.ieee_cis import load_ieee_cis
from synthfraud.log import log
from synthfraud.metrics.explain import mean_abs_shap_by_feature, rank_correlation
from synthfraud.protocol import split_table
from synthfraud.r import sample_declared_controllers
from synthfraud.schema import IEEE_CIS, certificate_coverage
from synthfraud.synthesizers import build_synthesizer

ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT.parent / "data" / "raw" / "ieee_cis"
CACHE = ROOT / "results" / "c4_cache"


def _cert_stats(frame) -> dict:
    cov = certificate_coverage(IEEE_CIS, frame)
    n_fraud = int(frame[IEEE_CIS.label].sum())
    uncert = int(cov.get("uncertified_fraud", 0))
    typ = {k: int(v) for k, v in cov.items() if str(k).startswith("typology")}
    return {
        "n_rows": int(len(frame)),
        "n_fraud": n_fraud,
        "uncertified_fraud": uncert,
        "typology_coverage": (n_fraud - uncert) / n_fraud if n_fraud else None,
        "certificate_coverage": {k: int(v) for k, v in cov.items()},
        "typology_counts": typ,
        "constraint_violation_rate": describe_c_pub(IEEE_CIS, frame)["constraint_violation_rate"],
    }


def _write(report: dict, path: Path, utc: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    archive = path.parent / "archive" / f"{path.stem}_{utc}{path.suffix}"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_text(path.read_text())
    log(f"wrote {path} and {archive}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IEEE-CIS C4 (real-test SHAP + certificates)")
    parser.add_argument("--synthesizer", default="ctgan")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-synth", type=int, default=10_000)
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args(argv)

    seed = args.seed
    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log(f"C4 IEEE-CIS synthesizer={args.synthesizer} seed={seed}")
    table = load_ieee_cis(DUMP, seed=seed, protocol="natural", n_rows=500_000)
    splits = split_table(table, IEEE_CIS, seed=seed, train=0.50, ref=0.20, val=0.15, test=0.15)
    test_fraud = splits.test.loc[splits.test[IEEE_CIS.label].eq(1)].copy()
    log(f"test fraud rows for SHAP={len(test_fraud):,}")

    CACHE.mkdir(parents=True, exist_ok=True)
    unf_path = CACHE / f"{args.synthesizer}_seed{seed}_unfiltered.parquet"
    r05_path = CACHE / f"{args.synthesizer}_seed{seed}_r05_rateonly.parquet"

    if args.reuse_cache and unf_path.is_file() and r05_path.is_file():
        log("reuse cached synth tables")
        import pandas as pd

        unfiltered = pd.read_parquet(unf_path)
        rateonly = pd.read_parquet(r05_path)
    else:
        extra = {"epochs": args.epochs} if args.synthesizer == "ctgan" else {}
        model = build_synthesizer(args.synthesizer, seed=seed, schema=IEEE_CIS, **extra)
        log("fit synthesizer on real train (same protocol as T1; not a new G search)")
        model.fit(splits.train)
        log(f"sample unfiltered n={args.n_synth:,}")
        unfiltered = model.sample(args.n_synth)
        log("fit bank detector for R")
        bank = encode_fit_detector(splits.train, IEEE_CIS, seed=seed, kind="hgb")
        log("sample declared 5% rate-only")
        packed = sample_declared_controllers(
            model,
            args.n_synth,
            IEEE_CIS,
            bank=bank,
            val=splits.val,
            train_fraud_rate=float(splits.train[IEEE_CIS.label].mean()),
            seed=seed,
            rates=[0.05],
            controllers=("rate_only",),
            pool_mult=12.0,
            min_margin=0.15,
            c_pub=True,
        )
        rateonly = packed[0][1]["rate_only"][0]
        unfiltered.to_parquet(unf_path, index=False)
        rateonly.to_parquet(r05_path, index=False)
        log(f"cached {unf_path.name} and {r05_path.name}")

    log("fit real-train bank HGB")
    bank = encode_fit_detector(splits.train, IEEE_CIS, seed=seed, kind="hgb")
    arms = {"unfiltered": unfiltered, "rate_only_5pct": rateonly}
    y_test = splits.test[IEEE_CIS.label].to_numpy().astype(int)
    bank_pr = float(average_precision_score(y_test, predict_proba(bank, splits.test, IEEE_CIS)))
    log(f"bank real-train PR-AUC on real test={bank_pr:.3f}")

    log("SHAP on real test fraud (bank)")
    bank_shap = mean_abs_shap_by_feature(bank, test_fraud, IEEE_CIS, seed=seed)
    report_arms = {}
    for name, synth in arms.items():
        log(f"arm {name}")
        synth_model = encode_fit_detector(synth[IEEE_CIS.all_cols], IEEE_CIS, seed=seed, kind="hgb")
        uh = float(average_precision_score(y_test, predict_proba(synth_model, splits.test, IEEE_CIS)))
        log(f"  U_h PR-AUC={uh:.3f}")
        log("  SHAP on real test fraud (synth-trained)")
        synth_shap = mean_abs_shap_by_feature(synth_model, test_fraud, IEEE_CIS, seed=seed)
        ranks = rank_correlation(bank_shap, synth_shap, IEEE_CIS.feature_cols)
        log(f"  Spearman rho={ranks['spearman_rho']:.3f}  top3 overlap={ranks['top3_overlap']}")
        synth_fraud = synth.loc[synth[IEEE_CIS.label].eq(1)]
        synth_test_shap = None
        if len(synth_fraud) >= 50:
            log("  SHAP on synth fraud rows (contrast only; not the C4 metric)")
            synth_test_shap = mean_abs_shap_by_feature(synth_model, synth_fraud, IEEE_CIS, seed=seed)
        report_arms[name] = {
            "honest_utility_pr_auc": uh,
            "certificates": _cert_stats(synth),
            "shap_real_test_fraud": synth_shap,
            "rank_vs_bank_real_test": ranks,
            "shap_on_synth_fraud_contrast": synth_test_shap,
            "rank_bank_vs_synth_test_contrast": (
                rank_correlation(bank_shap, synth_test_shap, IEEE_CIS.feature_cols)
                if synth_test_shap is not None
                else None
            ),
        }

    report = {
        "protocol": "ieee_cis_c4",
        "seed": seed,
        "synthesizer": args.synthesizer,
        "n_test_fraud": int(len(test_fraud)),
        "bank_pr_auc_real_test": bank_pr,
        "bank_shap_real_test_fraud": bank_shap,
        "note": (
            "Primary C4 metric is rank_vs_bank_real_test on real held-out fraud. "
            "shap_on_synth_fraud_contrast is logged to show why SHAP-on-synth is not the claim. "
            "Certificates are C_pub ProductCD typologies, not count columns."
        ),
        "arms": report_arms,
    }
    if args.synthesizer == "ctgan":
        stem = "ieee_cis_c4" if seed == 0 else f"ieee_cis_c4_seed{seed}"
    else:
        stem = f"ieee_cis_c4_{args.synthesizer}_seed{seed}"
    _write(report, ROOT / "results" / f"{stem}.json", utc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
