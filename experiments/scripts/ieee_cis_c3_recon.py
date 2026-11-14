"""IEEE-CIS T1 table-as-reconstruction (no-aux attack, ref is a score control).

The released table is the reconstructed set. Not ReconSyn (T2 multi-draw from G).
Reuses C4 cached CTGAN tables. Does not overwrite ieee_cis_c4*.json.
Identity n_train is seed 0 only (positive control). See notes/c3-protocol.md.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from synthfraud.ieee_cis import load_ieee_cis
from synthfraud.log import log
from synthfraud.metrics.privacy import table_reconstruction
from synthfraud.protocol import split_table
from synthfraud.schema import IEEE_CIS
from synthfraud.synthesizers import build_synthesizer

ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT.parent / "data" / "raw" / "ieee_cis"
CACHE = ROOT / "results" / "c4_cache"


def _write(report: dict, path: Path, utc: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    archive = path.parent / "archive" / f"{path.stem}_{utc}{path.suffix}"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_text(path.read_text())
    log(f"wrote {path} and {archive}")


def _arm(name: str, synth, train, ref) -> dict:
    log(f"recon {name} n_synth={len(synth):,}")
    recon = table_reconstruction(train, ref, synth, IEEE_CIS)
    eps0 = recon["nn_recall"]["eps_0.0"]
    log(
        f"  exact train={recon['exact_match_rate_train']:.4f} "
        f"ref={recon['exact_match_rate_ref']:.4f} "
        f"adv={recon['exact_match_advantage']:.4f} "
        f"nn0 train={eps0['recall_train']:.4f} ref={eps0['recall_ref']:.4f} "
        f"fraud_exact={recon.get('exact_match_rate_train_fraud')}"
    )
    return {
        "name": name,
        "n_synth": int(len(synth)),
        "n_synth_fraud": int(synth[IEEE_CIS.label].sum()),
        "synth_fraud_rate": float(synth[IEEE_CIS.label].mean()),
        "reconstruction": recon,
    }


def _seed_run(seed: int, n_synth: int, *, ntrain_control: bool) -> dict:
    log(f"C3 recon IEEE-CIS seed={seed}")
    table = load_ieee_cis(DUMP, seed=seed, protocol="natural", n_rows=500_000)
    splits = split_table(
        table, IEEE_CIS, seed=seed, train=0.50, ref=0.20, val=0.15, test=0.15
    )
    train = splits.train[IEEE_CIS.all_cols]
    ref = splits.ref[IEEE_CIS.all_cols]
    log(f"train={len(train):,} ref={len(ref):,} train fraud={int(train[IEEE_CIS.label].sum())}")
    arms = []
    ident = build_synthesizer("identity", seed, IEEE_CIS)
    ident.fit(train)
    arms.append(_arm("identity_10k", ident.sample(n_synth), train, ref))
    if ntrain_control:
        arms.append(_arm("identity_ntrain", ident.sample(len(train)), train, ref))
    indep = build_synthesizer("independent", seed, IEEE_CIS)
    indep.fit(train)
    arms.append(_arm("independent_10k", indep.sample(n_synth), train, ref))
    unf_path = CACHE / f"ctgan_seed{seed}_unfiltered.parquet"
    r05_path = CACHE / f"ctgan_seed{seed}_r05_rateonly.parquet"
    cached = unf_path.is_file() and r05_path.is_file()
    if cached:
        log(f"reuse C4 cache seed={seed}")
        arms.append(_arm("ctgan_unfiltered", pd.read_parquet(unf_path), train, ref))
        arms.append(_arm("ctgan_r05_rateonly", pd.read_parquet(r05_path), train, ref))
    else:
        log(f"no C4 cache for seed {seed}; skip CTGAN")
    return {
        "seed": seed,
        "n_train": int(len(train)),
        "n_ref": int(len(ref)),
        "n_train_fraud": int(train[IEEE_CIS.label].sum()),
        "n_ref_fraud": int(ref[IEEE_CIS.label].sum()),
        "identity_ntrain": ntrain_control,
        "ctgan_from_c4_cache": cached,
        "arms": arms,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IEEE-CIS T1 table reconstruction")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--n-synth", type=int, default=10_000)
    args = parser.parse_args(argv)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runs = [_seed_run(seed, args.n_synth, ntrain_control=(seed == seeds[0])) for seed in seeds]
    report = {
        "protocol": "ieee_cis_c3_recon",
        "seeds": seeds,
        "status": "T1 table-as-reconstruction; not ReconSyn; do not overwrite c4 JSON",
        "runs": runs,
    }
    _write(report, ROOT / "results" / "ieee_cis_c3_recon.json", utc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
