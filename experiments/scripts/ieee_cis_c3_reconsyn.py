"""IEEE-CIS T2 SampleAttack (lab-only). Query G; do not ship G.

This is Ganev & De Cristofaro ReconSyn *SampleAttack* without the vendor
similarity-metric oracle. Full ReconSyn SearchAttack queries IMS/DCR/NNDR
against train; that oracle is not in the T1 product and is not this cell.

Attack: black-box samples from a fitted generator at increasing query budgets.
The reconstructed set is the concatenation of those samples. Scoring reuses
``table_reconstruction`` (train recall vs ref specificity). IEEE-CIS compact
is the recon corpus (PaySim empty-origin exact-match is a typology collision).

Do not overwrite ieee_cis_c4*.json, ieee_cis_c3_mia.json, or ieee_cis_c3_recon.json.
See notes/c3-protocol.md.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

from synthfraud.ieee_cis import load_ieee_cis
from synthfraud.log import log
from synthfraud.metrics.privacy import RECON_THRESHOLDS, _matrix, table_reconstruction
from synthfraud.protocol import split_table
from synthfraud.schema import IEEE_CIS
from synthfraud.synthesizers import build_synthesizer

ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT.parent / "data" / "raw" / "ieee_cis"
CACHE = ROOT / "results" / "c4_cache"

DEFAULT_BUDGETS = (10_000, 50_000, 100_000, 250_000)


def _write(report: dict, path: Path, utc: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    archive = path.parent / "archive" / f"{path.stem}_{utc}{path.suffix}"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_text(path.read_text())
    log(f"wrote {path} and {archive}")


def _isolated_idx(x: np.ndarray, n_out: int, seed: int) -> np.ndarray:
    n_out = min(n_out, len(x))
    iso = IsolationForest(n_estimators=100, random_state=seed, n_jobs=-1)
    scores = -iso.fit(x).score_samples(x)
    return np.argsort(-scores)[:n_out]


def _outlier_slice(synth, *, train_x, ref_x, mem_i, non_i, fitted) -> dict:
    synth_x, _ = _matrix(synth, IEEE_CIS, fitted=fitted)
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(synth_x)
    d_mem = nn.kneighbors(train_x[mem_i], return_distance=True)[0].ravel()
    d_non = nn.kneighbors(ref_x[non_i], return_distance=True)[0].ravel()
    out: dict = {
        "n_outliers": int(min(len(mem_i), len(non_i))),
        "mean_nn_train_outliers": float(d_mem.mean()) if len(d_mem) else None,
        "mean_nn_ref_outliers": float(d_non.mean()) if len(d_non) else None,
        "nn_recall": {},
    }
    for eps in RECON_THRESHOLDS:
        rec_tr = float((d_mem <= eps).mean()) if len(d_mem) else None
        rec_rf = float((d_non <= eps).mean()) if len(d_non) else None
        out["nn_recall"][f"eps_{eps}"] = {
            "recall_train_outliers": rec_tr,
            "recall_ref_outliers": rec_rf,
            "advantage_outliers": (
                None if rec_tr is None or rec_rf is None else rec_tr - rec_rf
            ),
        }
    return out


def _score(name: str, synth, train, ref, ctx: dict) -> dict:
    log(f"T2 recon {name} n={len(synth):,}")
    recon = table_reconstruction(train, ref, synth, IEEE_CIS)
    eps0 = recon["nn_recall"]["eps_0.0"]
    log(
        f"  exact train={recon['exact_match_rate_train']:.4f} "
        f"ref={recon['exact_match_rate_ref']:.4f} "
        f"adv={recon['exact_match_advantage']:.4f} "
        f"nn0 train={eps0['recall_train']:.4f} ref={eps0['recall_ref']:.4f} "
        f"fraud_exact={recon.get('exact_match_rate_train_fraud')}"
    )
    outliers = _outlier_slice(
        synth,
        train_x=ctx["train_x"],
        ref_x=ctx["ref_x"],
        mem_i=ctx["mem_i"],
        non_i=ctx["non_i"],
        fitted=ctx["fitted"],
    )
    o0 = outliers["nn_recall"]["eps_0.0"]
    log(
        f"  outliers n={outliers['n_outliers']} nn0 train={o0['recall_train_outliers']:.4f} "
        f"ref={o0['recall_ref_outliers']:.4f} "
        f"adv={o0['advantage_outliers']:.4f}"
    )
    return {
        "name": name,
        "n_synth": int(len(synth)),
        "n_synth_fraud": int(synth[IEEE_CIS.label].sum()),
        "synth_fraud_rate": float(synth[IEEE_CIS.label].mean()),
        "reconstruction": recon,
        "outliers": outliers,
    }


def _sample_curve(model, budgets: tuple[int, ...], train, ref, ctx: dict, prefix: str) -> list[dict]:
    chunks: list[pd.DataFrame] = []
    drawn = 0
    arms = []
    for n in budgets:
        need = n - drawn
        if need > 0:
            log(f"{prefix} sample +{need:,} (total {n:,})")
            chunks.append(model.sample(need))
            drawn = n
        pool = pd.concat(chunks, ignore_index=True).head(n)
        arms.append(_score(f"{prefix}_n{n}", pool, train, ref, ctx))
    return arms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IEEE-CIS T2 SampleAttack (lab-only)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n-outliers", type=int, default=200)
    parser.add_argument(
        "--budgets",
        default="10000,50000,100000,250000",
        help="cumulative query sizes from G",
    )
    args = parser.parse_args(argv)
    budgets = tuple(int(x.strip()) for x in args.budgets.split(",") if x.strip())
    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    seed = args.seed
    log(f"T2 SampleAttack IEEE-CIS seed={seed} epochs={args.epochs} budgets={list(budgets)}")
    log("Not SearchAttack: no IMS/DCR/NNDR train oracle. Lab-only; G does not ship.")
    table = load_ieee_cis(DUMP, seed=seed, protocol="natural", n_rows=500_000)
    splits = split_table(
        table, IEEE_CIS, seed=seed, train=0.50, ref=0.20, val=0.15, test=0.15
    )
    train = splits.train[IEEE_CIS.all_cols]
    ref = splits.ref[IEEE_CIS.all_cols]
    log(f"train={len(train):,} ref={len(ref):,} train fraud={int(train[IEEE_CIS.label].sum())}")
    log(f"precompute {args.n_outliers} isolated rows (once)")
    train_x, fitted = _matrix(train, IEEE_CIS)
    ref_x, _ = _matrix(ref, IEEE_CIS, fitted=fitted)
    ctx = {
        "train_x": train_x,
        "ref_x": ref_x,
        "fitted": fitted,
        "mem_i": _isolated_idx(train_x, args.n_outliers, seed),
        "non_i": _isolated_idx(ref_x, args.n_outliers, seed + 1),
    }

    arms: list[dict] = []
    ident = build_synthesizer("identity", seed, IEEE_CIS)
    ident.fit(train)
    arms.extend(_sample_curve(ident, budgets, train, ref, ctx, "identity"))

    indep = build_synthesizer("independent", seed, IEEE_CIS)
    indep.fit(train)
    arms.extend(_sample_curve(indep, (budgets[0], budgets[-1]), train, ref, ctx, "independent"))

    t1_path = CACHE / f"ctgan_seed{seed}_r05_rateonly.parquet"
    if t1_path.is_file():
        log(f"T1 contrast from C4 cache {t1_path.name} (not a T2 query)")
        arms.append(_score("t1_ctgan_r05_rateonly_cache", pd.read_parquet(t1_path), train, ref, ctx))
    else:
        log("no C4 5% cache; skip T1 contrast")

    log(f"fit CTGAN epochs={args.epochs} (T2 G; do not overwrite C4 JSON)")
    ctgan = build_synthesizer("ctgan", seed, IEEE_CIS, epochs=args.epochs)
    ctgan.fit(train)
    arms.extend(_sample_curve(ctgan, budgets, train, ref, ctx, "ctgan"))

    report = {
        "protocol": "ieee_cis_c3_reconsyn_sampleattack",
        "status": (
            "T2 lab-only SampleAttack (query G). Not SearchAttack. "
            "Not the T1 product. Do not overwrite c4/c3_mia/c3_recon JSON."
        ),
        "seed": seed,
        "epochs": args.epochs,
        "budgets": list(budgets),
        "n_outliers": args.n_outliers,
        "n_train": int(len(train)),
        "n_ref": int(len(ref)),
        "n_train_fraud": int(train[IEEE_CIS.label].sum()),
        "n_ref_fraud": int(ref[IEEE_CIS.label].sum()),
        "cite": "ganev2025inadequacy",
        "arms": arms,
    }
    _write(report, ROOT / "results" / "ieee_cis_c3_reconsyn.json", utc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
