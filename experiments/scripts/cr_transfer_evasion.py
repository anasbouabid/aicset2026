"""Camera-ready (AICSET-107 reviews): formal U_a, transfer evasion, baselines, bootstrap CIs.

Answers Reviewer 1 / Reviewer 3:
- class-aware surrogate measures instead of raw label agreement,
- a zero-query transfer-evasion attack under public constraints, scored against the
  bank detector at a bank-query budget k,
- attacker baselines: public-only (no data), uninformative release (independent
  marginals), auxiliary real data (ref split), white-box oracle,
- rebalanced real-data detector baselines for U_h,
- bootstrap confidence intervals over real test rows / attacked fraud rows.

Writes results/cr_transfer_<corpus>_seed<k>.json. Does not overwrite earlier JSON.
PaySim release tables are cached in results/cr_cache; IEEE-CIS reuses results/c4_cache.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, cohen_kappa_score

from synthfraud.detect import encode_fit_detector, predict_proba
from synthfraud.ieee_cis import load_ieee_cis
from synthfraud.log import log
from synthfraud.paysim import load_paysim
from synthfraud.protocol import split_table
from synthfraud.r import apply_sanitizer, sample_declared_controllers
from synthfraud.schema import IEEE_CIS, IEEE_CIS_PRODUCTS, PAYSIM, constraint_mask
from synthfraud.synthesizers import build_synthesizer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parent / "data" / "raw"
C4_CACHE = ROOT / "results" / "c4_cache"
CR_CACHE = ROOT / "results" / "cr_cache"

THRESHOLD = 0.5
BANK_RATE = 0.05  # bank/aux detectors: real fraud oversampled to this training rate
BUDGETS = (1, 5, 20)
N_BOOT = 1000


# ---------------------------------------------------------------- data helpers


def oversample(frame: pd.DataFrame, label: str, rate: float, seed: int) -> pd.DataFrame:
    """Random oversampling of real fraud rows up to a training fraud rate."""
    fraud = frame[frame[label] == 1]
    legit = frame[frame[label] == 0]
    n1 = int(round(len(legit) * rate / (1.0 - rate)))
    if n1 <= len(fraud):
        return frame
    boosted = fraud.sample(n=n1, replace=True, random_state=seed)
    return pd.concat([legit, boosted]).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def real_subsample(frame: pd.DataFrame, label: str, n: int, rate: float, seed: int) -> pd.DataFrame:
    """n real rows at a given fraud rate (same size / rate as the release)."""
    n1 = int(round(n * rate))
    fraud = frame[frame[label] == 1]
    legit = frame[frame[label] == 0]
    return pd.concat(
        [
            legit.sample(n=n - n1, random_state=seed),
            fraud.sample(n=n1, replace=len(fraud) < n1, random_state=seed),
        ]
    ).reset_index(drop=True)


def load_corpus(corpus: str, seed: int):
    if corpus == "paysim":
        schema = PAYSIM
        table = load_paysim(RAW / "paysim.parquet", seed=seed, protocol="natural", n_rows=500_000)
    else:
        schema = IEEE_CIS
        table = load_ieee_cis(RAW / "ieee_cis", seed=seed, protocol="natural", n_rows=500_000)
    splits = split_table(table, schema, seed=seed, train=0.50, ref=0.20, val=0.15, test=0.15)
    return schema, splits


def releases(corpus: str, schema, splits, seed: int, n_synth: int, epochs: int) -> dict[str, pd.DataFrame]:
    """Unfiltered CTGAN table and the declared 5% rate-only table (same fit)."""
    if corpus == "ieee":
        unf = pd.read_parquet(C4_CACHE / f"ctgan_seed{seed}_unfiltered.parquet")
        cap = pd.read_parquet(C4_CACHE / f"ctgan_seed{seed}_r05_rateonly.parquet")
        return {"unfiltered": unf, "cap5": cap}
    CR_CACHE.mkdir(parents=True, exist_ok=True)
    unf_path = CR_CACHE / f"paysim_ctgan_seed{seed}_unfiltered.parquet"
    cap_path = CR_CACHE / f"paysim_ctgan_seed{seed}_r05_rateonly.parquet"
    if unf_path.is_file() and cap_path.is_file():
        log("reuse cached PaySim release tables")
        return {"unfiltered": pd.read_parquet(unf_path), "cap5": pd.read_parquet(cap_path)}
    model = build_synthesizer("ctgan", seed=seed, schema=schema, epochs=epochs)
    log("fit CTGAN on real train (same protocol as the submitted T1 runs)")
    model.fit(splits.train)
    unf = model.sample(n_synth)
    legacy_bank = encode_fit_detector(splits.train, schema, seed=seed, kind="hgb")
    packed = sample_declared_controllers(
        model,
        n_synth,
        schema,
        bank=legacy_bank,
        val=splits.val,
        train_fraud_rate=float(splits.train[schema.label].mean()),
        seed=seed,
        rates=[0.05],
        controllers=("rate_only",),
        pool_mult=12.0,
        min_margin=0.15,
        c_pub=True,
    )
    cap = packed[0][1]["rate_only"][0]
    unf.to_parquet(unf_path, index=False)
    cap.to_parquet(cap_path, index=False)
    return {"unfiltered": unf, "cap5": cap}


# ------------------------------------------------------------- attack surface


def candidates(corpus: str, fraud: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Public-constraint-legal edits of attacker-controlled fields.

    Returns (candidate rows, target index per row, retained-amount fraction per row).
    Candidate 0 of every target is the unmodified transaction.
    """
    n = len(fraud)
    if corpus == "paysim":
        grid = list(
            itertools.product((1.0, 0.9, 0.75, 0.5, 0.25, 0.1), (None, "TRANSFER", "CASH_OUT"), (0, 6, 12, 18))
        )
        grid = [g for g in grid if g[1] is not None or g == (1.0, None, 0)]
    else:
        grid = list(
            itertools.product(
                (1.0, 0.75, 0.5, 0.25),
                (None,) + tuple(IEEE_CIS_PRODUCTS),
                (None, "desktop", "mobile"),
                (None, "gmail.com", "yahoo.com", "hotmail.com", "anonymous.com"),
                (0, 6 * 3600, 12 * 3600, 18 * 3600),
            )
        )
        grid = [
            g
            for g in grid
            if (g[1] is not None and g[2] is not None and g[3] is not None) or g == (1.0, None, None, None, 0)
        ]
    grid.sort(key=lambda g: 0 if all(v in (1.0, None, 0) for v in g) else 1)  # original first
    parts, kept = [], []
    for g in grid:
        c = fraud.copy()
        if corpus == "paysim":
            scale, kind, shift = g
            new_amount = c["amount"].astype(float) * scale
            delta = c["amount"].astype(float) - new_amount
            c["amount"] = new_amount
            # accounting identities: the origin keeps what is not taken; the destination receives less
            c["newbalanceOrig"] = c["newbalanceOrig"].astype(float) + delta
            c["newbalanceDest"] = (c["newbalanceDest"].astype(float) - delta).clip(lower=0.0)
            if kind is not None:
                c["type"] = kind
            c["step"] = c["step"].astype(int) + shift
        else:
            scale, prod, dev, mail, shift = g
            c["TransactionAmt"] = c["TransactionAmt"].astype(float) * scale
            if prod is not None:
                c["ProductCD"] = prod
            if dev is not None:
                c["DeviceType"] = dev
            if mail is not None:
                c["P_emaildomain"] = mail
            c["TransactionDT"] = c["TransactionDT"].astype(float) + shift
        parts.append(c)
        kept.append(np.full(n, g[0]))
    cand = pd.concat(parts, ignore_index=True)
    target = np.tile(np.arange(n), len(grid))
    return cand, target, np.concatenate(kept)


def evasion_matrix(bank, schema, cand, target, kept, n_targets):
    """bank_evades[t, c] and kept[t, c] for target t and candidate c."""
    n_c = len(cand) // n_targets
    p = predict_proba(bank, cand, schema)
    ev = (p < THRESHOLD).reshape(n_c, n_targets).T
    kp = kept.reshape(n_c, n_targets).T
    return ev, kp


def attack_orders(score: np.ndarray, kept: np.ndarray) -> np.ndarray:
    """Candidate order per target: lowest surrogate score first, larger amount on ties."""
    return np.lexsort((-kept, score), axis=1)


def per_target_outcomes(order: np.ndarray, ev: np.ndarray, kp: np.ndarray) -> dict[str, np.ndarray]:
    rows = np.arange(len(order))[:, None]
    ev_sorted = ev[rows, order]
    kp_sorted = kp[rows, order]
    out = {}
    for k in BUDGETS:
        out[f"asr@{k}"] = ev_sorted[:, :k].any(axis=1).astype(float)
    out["value@1"] = ev_sorted[:, 0] * kp_sorted[:, 0]
    return out


def boot_ci(x: np.ndarray, rng, n_boot: int = N_BOOT) -> list[float]:
    if len(x) == 0:
        return [float("nan"), float("nan")]
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize(outcomes: dict[str, np.ndarray], rng) -> dict:
    return {
        name: {"mean": float(x.mean()) if len(x) else float("nan"), "ci95": boot_ci(x, rng)}
        for name, x in outcomes.items()
    }


# ------------------------------------------------------------------- metrics


def pr_auc_boot(y: np.ndarray, scores: dict[str, np.ndarray], rng, n_boot: int = N_BOOT):
    """Shared test-row bootstrap for every scorer, so differences are paired."""
    n = len(y)
    draws = {k: np.empty(n_boot) for k in scores}
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if yb.sum() == 0:
            for k in scores:
                draws[k][b] = np.nan
            continue
        for k, s in scores.items():
            draws[k][b] = average_precision_score(yb, s[idx])
    return draws


def fidelity(bank_p, sur_p, y) -> dict:
    """Class-aware agreement between a surrogate and the bank detector on real test rows."""
    bank_hat = (bank_p >= THRESHOLD).astype(int)
    sur_hat = (sur_p >= THRESHOLD).astype(int)
    fraud = y == 1
    flagged = bank_hat == 1
    return {
        "label_agreement_all": float((bank_hat == sur_hat).mean()),
        "label_agreement_fraud": float((bank_hat[fraud] == sur_hat[fraud]).mean()),
        "label_agreement_legit": float((bank_hat[~fraud] == sur_hat[~fraud]).mean()),
        "cohen_kappa": float(cohen_kappa_score(bank_hat, sur_hat)),
        "recall_of_bank_alerts": float(sur_hat[flagged].mean()) if flagged.any() else float("nan"),
        "surrogate_alert_rate": float(sur_hat.mean()),
        "bank_alert_rate": float(bank_hat.mean()),
        "score_spearman_all": float(spearmanr(bank_p, sur_p).statistic),
        "score_spearman_fraud": float(spearmanr(bank_p[fraud], sur_p[fraud]).statistic),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=("paysim", "ieee"), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-synth", type=int, default=10_000)
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args(argv)
    seed, corpus = args.seed, args.corpus
    rng = np.random.default_rng(1000 + seed)
    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    schema, splits = load_corpus(corpus, seed)
    label = schema.label
    test = splits.test
    y = test[label].to_numpy().astype(int)
    log(f"{corpus} seed={seed} test rows={len(test):,} test fraud={int(y.sum())}")

    rel = releases(corpus, schema, splits, seed, args.n_synth, args.epochs)
    indep = build_synthesizer("independent", seed=seed, schema=schema)
    indep.fit(splits.train)
    rel["independent"] = indep.sample(args.n_synth)

    # Detectors. Bank and auxiliary attacker use real rows with fraud oversampled to 5%.
    log("fit detectors")
    bank = encode_fit_detector(oversample(splits.train, label, BANK_RATE, seed), schema, seed=seed, kind="hgb")
    models = {
        "bank_real_rebalanced": bank,
        "real_unweighted": encode_fit_detector(splits.train, schema, seed=seed, kind="hgb"),
        "real_10k_5pct": encode_fit_detector(
            real_subsample(splits.train, label, args.n_synth, 0.05, seed), schema, seed=seed, kind="hgb"
        ),
        "aux_real_ref": encode_fit_detector(
            oversample(splits.ref, label, BANK_RATE, seed), schema, seed=seed, kind="hgb"
        ),
    }
    for name, table in rel.items():
        models[f"release_{name}"] = encode_fit_detector(table[schema.all_cols], schema, seed=seed, kind="hgb")
    # Sanitized variants of the capped table (same parameters as the submitted ablation).
    model_schema = {name: schema for name in models}
    for i, (tag, kind, relv) in enumerate((("jitter01", "jitter", 0.1), ("jitter05", "jitter", 0.5), ("withhold", "withhold", None))):
        cleaned, eval_schema, _ = apply_sanitizer(rel["cap5"], schema, kind=kind, seed=seed + 200 + i, rel=relv)
        models[f"release_cap5_{tag}"] = encode_fit_detector(cleaned[eval_schema.all_cols], eval_schema, seed=seed, kind="hgb")
        model_schema[f"release_cap5_{tag}"] = eval_schema
    legacy_lr = {
        name: encode_fit_detector(table[schema.all_cols], schema, seed=seed, kind="linear")
        for name, table in rel.items()
    }

    scores = {name: predict_proba(m, test, model_schema[name]) for name, m in models.items()}
    bank_p = scores["bank_real_rebalanced"]

    log("U_h bootstrap")
    draws = pr_auc_boot(y, scores, rng)
    uh = {
        name: {
            "pr_auc": float(average_precision_score(y, s)),
            "ci95": [float(np.nanquantile(draws[name], 0.025)), float(np.nanquantile(draws[name], 0.975))],
        }
        for name, s in scores.items()
    }
    diff = draws["release_cap5"] - draws["release_unfiltered"]
    uh_diff = {
        "cap5_minus_unfiltered": float(uh["release_cap5"]["pr_auc"] - uh["release_unfiltered"]["pr_auc"]),
        "ci95": [float(np.nanquantile(diff, 0.025)), float(np.nanquantile(diff, 0.975))],
    }

    fid = {name: fidelity(bank_p, s, y) for name, s in scores.items() if name != "bank_real_rebalanced"}
    legacy = {}
    legacy_bank_hat = (scores["real_unweighted"] >= THRESHOLD).astype(int)
    for name, m in legacy_lr.items():
        hat = (predict_proba(m, test, schema) >= THRESHOLD).astype(int)
        legacy[name] = {
            "label_agreement_legacy": float((hat == legacy_bank_hat).mean()),
            "surrogate_alert_rate": float(hat.mean()),
            "legacy_bank_alert_rate": float(legacy_bank_hat.mean()),
        }

    # Transfer evasion: real held-out fraud the bank currently flags.
    targets = test[(y == 1) & (bank_p >= THRESHOLD)].reset_index(drop=True)
    n_t = len(targets)
    log(f"evasion targets (bank-flagged real test fraud) = {n_t}")
    cand, target_idx, kept = candidates(corpus, targets)
    cpub_ok = float(constraint_mask(schema, cand).mean())
    ev, kp = evasion_matrix(bank, schema, cand, target_idx, kept, n_t)
    n_c = ev.shape[1]

    attackers = {
        "oracle_bank": bank,
        "aux_real_ref": models["aux_real_ref"],
        "release_unfiltered": models["release_unfiltered"],
        "release_cap5": models["release_cap5"],
        "release_independent": models["release_independent"],
        "release_cap5_jitter01": models["release_cap5_jitter01"],
        "release_cap5_jitter05": models["release_cap5_jitter05"],
        "release_cap5_withhold": models["release_cap5_withhold"],
    }
    evasion, raw = {}, {}
    for name, model in attackers.items():
        s = predict_proba(model, cand, model_schema.get(name, schema)).reshape(n_c, n_t).T
        raw[name] = per_target_outcomes(attack_orders(s, kp), ev, kp)
        evasion[name] = summarize(raw[name], rng)
    # Public-only attacker: knows schema + C_pub, has no data; tries random legal edits.
    perm_out = {k: np.zeros(n_t) for k in [f"asr@{b}" for b in BUDGETS] + ["value@1"]}
    n_perm = 200
    for _ in range(n_perm):
        order = np.argsort(rng.random((n_t, n_c)), axis=1)
        for k, v in per_target_outcomes(order, ev, kp).items():
            perm_out[k] += v / n_perm
    raw["public_only_random"] = perm_out
    evasion["public_only_random"] = summarize(perm_out, rng)

    paired = {}
    for a, b in (("release_cap5", "release_unfiltered"), ("release_cap5", "public_only_random"),
                 ("release_cap5", "aux_real_ref")):
        paired[f"{a}_minus_{b}"] = {
            k: {"mean": float((raw[a][k] - raw[b][k]).mean()), "ci95": boot_ci(raw[a][k] - raw[b][k], rng)}
            for k in raw[a]
        }

    report = {
        "protocol": "cr_transfer_evasion",
        "corpus": corpus,
        "seed": seed,
        "run": {"utc": utc, "python": sys.version.split()[0]},
        "definitions": {
            "threshold": THRESHOLD,
            "bank": "HGB (max_depth 4, 80 iterations) on the real train split, fraud oversampled to 5%",
            "surrogate": "same HGB configuration trained on the attacker's table",
            "targets": "real held-out fraud rows with bank score >= threshold",
            "attack": "rank public-constraint-legal edits by surrogate score (ascending; larger amount on ties), "
            "submit the first k to the bank; success if any has bank score < threshold",
            "value@1": "retained amount fraction of the first submitted edit if it evades, else 0",
            "n_boot": N_BOOT,
        },
        "n_test": int(len(test)),
        "n_test_fraud": int(y.sum()),
        "n_targets": int(n_t),
        "n_candidates_per_target": int(n_c),
        "candidate_cpub_valid_rate": cpub_ok,
        "candidate_evading_fraction": float(ev.mean()) if n_t else float("nan"),
        "targets_with_any_evading_edit": float(ev.any(axis=1).mean()) if n_t else float("nan"),
        "release_fraud_rate": {k: float(v[label].mean()) for k, v in rel.items()},
        "honest_utility": uh,
        "honest_utility_diff": uh_diff,
        "surrogate_fidelity": fid,
        "legacy_agreement": legacy,
        "transfer_evasion": evasion,
        "transfer_evasion_paired": paired,
    }
    out = ROOT / "results" / f"cr_transfer_{corpus}_seed{seed}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    archive = out.parent / "archive" / f"{out.stem}_{utc}.json"
    archive.write_text(out.read_text())
    # Per-target outcomes and bootstrap draws, so seeds can be pooled into one interval.
    arrays = {f"ev__{a}__{k}": v for a, d in raw.items() for k, v in d.items()}
    arrays.update({f"uh__{k}": v for k, v in draws.items()})
    np.savez_compressed(out.with_suffix(".npz"), **arrays)
    log(f"wrote {out}")
    for name, row in evasion.items():
        log(f"  {name:22s} " + "  ".join(f"{k}={v['mean']:.3f}" for k, v in row.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
