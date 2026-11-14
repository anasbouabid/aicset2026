"""PaySim C3 diagnostic: distance MIA vs DOMIAS-style density-ratio MIA.

Does not overwrite T1 JSON. Identity n_train is the leaky control.
CTGAN 5% rate-only is the product cell (--ctgan). Do not mark C3 verified
from identity/independent alone. See notes/c3-protocol.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from synthfraud.detect import encode_fit_detector
from synthfraud.log import log
from synthfraud.metrics.privacy import density_ratio_mia, privacy
from synthfraud.paysim import load_paysim
from synthfraud.protocol import split_table
from synthfraud.r import sample_declared_controllers
from synthfraud.schema import PAYSIM
from synthfraud.synthesizers import build_synthesizer

ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT.parent / "data" / "raw" / "paysim.parquet"


def _write(report: dict, path: Path, utc: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    archive = path.parent / "archive" / f"{path.stem}_{utc}{path.suffix}"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_text(path.read_text())
    log(f"wrote {path} and {archive}")


def _arm(name: str, synth, train, ref) -> dict:
    log(f"privacy {name} n_synth={len(synth):,}")
    dist = privacy(train, ref, synth, PAYSIM)
    ratio = density_ratio_mia(train, ref, synth, PAYSIM, seed=0)
    return {
        "name": name,
        "n_synth": int(len(synth)),
        "n_synth_fraud": int(synth[PAYSIM.label].sum()),
        "synth_fraud_rate": float(synth[PAYSIM.label].mean()),
        "privacy": dist,
        "density_ratio": ratio,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PaySim C3 density-ratio MIA")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-rows", type=int, default=500_000)
    parser.add_argument("--n-synth", type=int, default=10_000)
    parser.add_argument("--ctgan", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args(argv)

    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log(f"C3 PaySim seed={args.seed} n_rows={args.n_rows}")
    table = load_paysim(DUMP, seed=args.seed, protocol="natural", n_rows=args.n_rows)
    splits = split_table(
        table, PAYSIM, seed=args.seed, train=0.50, ref=0.20, val=0.15, test=0.15
    )
    train = splits.train[PAYSIM.all_cols]
    ref = splits.ref[PAYSIM.all_cols]
    log(f"train={len(train):,} ref={len(ref):,} train fraud={int(train[PAYSIM.label].sum())}")

    arms = []
    ident = build_synthesizer("identity", args.seed, PAYSIM)
    ident.fit(train)
    arms.append(_arm("identity_10k", ident.sample(args.n_synth), train, ref))
    arms.append(_arm("identity_ntrain", ident.sample(len(train)), train, ref))

    indep = build_synthesizer("independent", args.seed, PAYSIM)
    indep.fit(train)
    arms.append(_arm("independent_10k", indep.sample(args.n_synth), train, ref))

    if args.ctgan:
        log(f"fit CTGAN epochs={args.epochs}")
        ctgan = build_synthesizer("ctgan", args.seed, PAYSIM, epochs=args.epochs)
        ctgan.fit(train)
        raw = ctgan.sample(args.n_synth)
        arms.append(_arm("ctgan_unfiltered", raw, train, ref))
        log("fit bank detector for 5% rate-only")
        bank = encode_fit_detector(train, PAYSIM, seed=args.seed, kind="hgb")
        packed = sample_declared_controllers(
            ctgan,
            args.n_synth,
            PAYSIM,
            bank=bank,
            val=splits.val,
            train_fraud_rate=float(train[PAYSIM.label].mean()),
            seed=args.seed,
            rates=[0.05],
            controllers=("rate_only",),
            pool_mult=12.0,
            c_pub=True,
            max_draw=args.n_synth * 30,
        )
        r05 = packed[0][1]["rate_only"][0]
        arms.append(_arm("ctgan_r05_rateonly", r05, train, ref))

    report = {
        "protocol": "paysim_c3_mia",
        "seed": args.seed,
        "n_rows": int(len(table)),
        "n_train": int(len(train)),
        "n_ref": int(len(ref)),
        "n_train_fraud": int(train[PAYSIM.label].sum()),
        "n_ref_fraud": int(ref[PAYSIM.label].sum()),
        "ctgan": bool(args.ctgan),
        "epochs": args.epochs if args.ctgan else None,
        "status": "product cell scored; C3 membership unverified (CTGAN k-NN ~0.50)",
        "arms": arms,
    }
    out = ROOT / "results" / ("paysim_c3_mia_ctgan.json" if args.ctgan else "paysim_c3_mia.json")
    _write(report, out, utc)
    for arm in arms:
        dr = arm["density_ratio"]
        log(
            f"{arm['name']}: dist={arm['privacy']['mia_distance_auc']:.3f} "
            f"knn={dr['mia_knn_ratio_auc']:.3f} clf={dr['mia_clf_ratio_auc']:.3f} "
            f"knn_fraud={dr.get('mia_knn_ratio_auc_fraud')} "
            f"clf_fraud={dr.get('mia_clf_ratio_auc_fraud')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
