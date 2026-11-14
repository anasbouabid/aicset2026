"""IEEE-CIS C3: Stadler-style attribute inference on compact T1 tables.

Reuses C4 cached CTGAN tables. Does not overwrite ieee_cis_c4*.json,
ieee_cis_c3_mia.json, or ieee_cis_c3_recon.json. SearchAttack is off.
See notes/c3-protocol.md.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from synthfraud.ieee_cis import load_ieee_cis
from synthfraud.log import log
from synthfraud.metrics.privacy import DEFAULT_AIA_TARGETS, attribute_inference
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


def _arm(name: str, synth, train, ref, seed: int) -> dict:
    log(f"AIA {name} n_synth={len(synth):,}")
    aia = attribute_inference(train, ref, synth, IEEE_CIS, seed=seed)
    row = {
        "name": name,
        "n_synth": int(len(synth)),
        "n_synth_fraud": int(synth[IEEE_CIS.label].sum()),
        "synth_fraud_rate": float(synth[IEEE_CIS.label].mean()),
        "attribute_inference": aia,
    }
    for target, cell in aia["targets"].items():
        log(
            f"  {target}: acc_tr={cell['acc_train']:.3f} acc_ref={cell['acc_ref']:.3f} "
            f"adv={cell['advantage']:.3f} lift={cell['lift_vs_majority']:.3f} "
            f"maj={cell['majority_acc_ref']:.3f}"
        )
    return row


def _seed_run(seed: int, n_synth: int) -> dict:
    log(f"C3 AIA IEEE-CIS seed={seed}")
    table = load_ieee_cis(DUMP, seed=seed, protocol="natural", n_rows=500_000)
    splits = split_table(
        table, IEEE_CIS, seed=seed, train=0.50, ref=0.20, val=0.15, test=0.15
    )
    train = splits.train[IEEE_CIS.all_cols]
    ref = splits.ref[IEEE_CIS.all_cols]
    arms = []
    ident = build_synthesizer("identity", seed, IEEE_CIS)
    ident.fit(train)
    arms.append(_arm("identity_10k", ident.sample(n_synth), train, ref, seed))
    arms.append(_arm("identity_ntrain", ident.sample(len(train)), train, ref, seed))
    indep = build_synthesizer("independent", seed, IEEE_CIS)
    indep.fit(train)
    arms.append(_arm("independent_10k", indep.sample(n_synth), train, ref, seed))

    unf_path = CACHE / f"ctgan_seed{seed}_unfiltered.parquet"
    r05_path = CACHE / f"ctgan_seed{seed}_r05_rateonly.parquet"
    cached = unf_path.is_file() and r05_path.is_file()
    if cached:
        log(f"reuse C4 cache {unf_path.name} / {r05_path.name}")
        arms.append(_arm("ctgan_unfiltered", pd.read_parquet(unf_path), train, ref, seed))
        arms.append(_arm("ctgan_r05_rateonly", pd.read_parquet(r05_path), train, ref, seed))
    else:
        log(f"no C4 cache for seed {seed}; skip CTGAN arms")
    return {
        "seed": seed,
        "n_train": int(len(train)),
        "n_ref": int(len(ref)),
        "ctgan_from_c4_cache": cached,
        "arms": arms,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IEEE-CIS C3 attribute inference")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--n-synth", type=int, default=10_000)
    args = parser.parse_args(argv)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runs = [_seed_run(seed, args.n_synth) for seed in seeds]
    report = {
        "protocol": "ieee_cis_c3_aia",
        "targets": list(DEFAULT_AIA_TARGETS),
        "seeds": seeds,
        "status": "attribute inference scored; do not overwrite c4/c3 mia/recon JSON",
        "cite": "stadler2022synthetic",
        "runs": runs,
    }
    _write(report, ROOT / "results" / "ieee_cis_c3_aia.json", utc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
