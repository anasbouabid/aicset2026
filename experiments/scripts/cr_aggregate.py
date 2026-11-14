"""Pool cr_transfer_<corpus>_seed*.json/.npz across seeds for the camera-ready tables.

Intervals: per-target outcomes are pooled over seeds and bootstrapped (evasion);
U_h uses the mean over seeds of the per-seed test-row bootstrap draws.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "results"
ATTACKERS = (
    "public_only_random",
    "release_independent",
    "release_unfiltered",
    "release_cap5",
    "release_cap5_jitter01",
    "release_cap5_jitter05",
    "release_cap5_withhold",
    "aux_real_ref",
    "oracle_bank",
)
METRICS = ("asr@1", "asr@5", "asr@20", "value@1")


def ci(x: np.ndarray, rng, n_boot: int = 2000) -> tuple[float, float]:
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main(corpus: str) -> dict:
    rng = np.random.default_rng(7)
    seeds = [s for s in (0, 1, 2) if (ROOT / f"cr_transfer_{corpus}_seed{s}.npz").is_file()]
    js = {s: json.loads((ROOT / f"cr_transfer_{corpus}_seed{s}.json").read_text()) for s in seeds}
    npz = {s: np.load(ROOT / f"cr_transfer_{corpus}_seed{s}.npz") for s in seeds}
    out: dict = {"corpus": corpus, "seeds": seeds}
    print(f"== {corpus} seeds={seeds} targets={[js[s]['n_targets'] for s in seeds]} "
          f"test fraud={[js[s]['n_test_fraud'] for s in seeds]} cand/target={js[seeds[0]]['n_candidates_per_target']} "
          f"evading cand frac={[round(js[s]['candidate_evading_fraction'], 3) for s in seeds]} "
          f"targets with any evading edit={[round(js[s]['targets_with_any_evading_edit'], 3) for s in seeds]}")

    print("-- U_h (PR-AUC, real test): per seed | mean [95% CI of the seed mean]")
    out["uh"] = {}
    for name in js[seeds[0]]["honest_utility"]:
        per = [js[s]["honest_utility"][name]["pr_auc"] for s in seeds]
        d = np.mean([npz[s][f"uh__{name}"] for s in seeds], axis=0)
        lo, hi = np.nanquantile(d, [0.025, 0.975])
        out["uh"][name] = {"per_seed": per, "mean": float(np.mean(per)), "ci95": [float(lo), float(hi)]}
        print(f"  {name:24s} " + " ".join(f"{v:.3f}" for v in per) + f" | {np.mean(per):.3f} [{lo:.3f}, {hi:.3f}]")
    d = np.mean([npz[s]["uh__release_cap5"] - npz[s]["uh__release_unfiltered"] for s in seeds], axis=0)
    lo, hi = np.nanquantile(d, [0.025, 0.975])
    per = [js[s]["honest_utility_diff"]["cap5_minus_unfiltered"] for s in seeds]
    out["uh_diff"] = {"per_seed": per, "mean": float(np.mean(per)), "ci95": [float(lo), float(hi)]}
    print(f"  cap5 - unfiltered        " + " ".join(f"{v:+.3f}" for v in per) + f" | {np.mean(per):+.3f} [{lo:+.3f}, {hi:+.3f}]")

    print("-- surrogate fidelity vs bank (mean over seeds)")
    out["fidelity"] = {}
    keys = ("label_agreement_all", "label_agreement_fraud", "cohen_kappa", "recall_of_bank_alerts",
            "surrogate_alert_rate", "score_spearman_all", "score_spearman_fraud")
    for name in js[seeds[0]]["surrogate_fidelity"]:
        row = {k: float(np.mean([js[s]["surrogate_fidelity"][name][k] for s in seeds])) for k in keys}
        out["fidelity"][name] = row
        print(f"  {name:24s} " + " ".join(f"{k.split('_', 1)[-1][:14]}={v:.3f}" for k, v in row.items()))
    print("  bank alert rate", np.mean([js[s]["surrogate_fidelity"]["release_cap5"]["bank_alert_rate"] for s in seeds]))
    print("-- legacy agreement (balanced LR vs unweighted bank)")
    for name in js[seeds[0]]["legacy_agreement"]:
        print(f"  {name:12s} " + " ".join(f"{js[s]['legacy_agreement'][name]['label_agreement_legacy']:.3f}" for s in seeds),
              "alert", " ".join(f"{js[s]['legacy_agreement'][name]['surrogate_alert_rate']:.3f}" for s in seeds))

    print("-- transfer evasion: pooled over seeds, mean [95% CI]; per-seed ASR@1")
    out["evasion"] = {}
    for a in ATTACKERS:
        row = {}
        for m in METRICS:
            x = np.concatenate([npz[s][f"ev__{a}__{m}"] for s in seeds])
            lo, hi = ci(x, rng)
            row[m] = {"mean": float(x.mean()), "ci95": [lo, hi]}
        row["asr@1_per_seed"] = [float(npz[s][f"ev__{a}__asr@1"].mean()) for s in seeds]
        out["evasion"][a] = row
        print(f"  {a:22s} " + "  ".join(f"{m}={row[m]['mean']:.3f} [{row[m]['ci95'][0]:.3f},{row[m]['ci95'][1]:.3f}]" for m in METRICS)
              + "  seeds " + " ".join(f"{v:.3f}" for v in row["asr@1_per_seed"]))
    print("-- paired differences (pooled targets)")
    out["paired"] = {}
    for a, b in (("release_cap5", "release_unfiltered"), ("release_cap5", "public_only_random"),
                 ("release_unfiltered", "public_only_random"), ("release_cap5", "aux_real_ref"),
                 ("release_cap5", "release_independent")):
        row = {}
        for m in METRICS:
            x = np.concatenate([npz[s][f"ev__{a}__{m}"] - npz[s][f"ev__{b}__{m}"] for s in seeds])
            lo, hi = ci(x, rng)
            row[m] = {"mean": float(x.mean()), "ci95": [lo, hi]}
        out["paired"][f"{a}-{b}"] = row
        print(f"  {a} - {b}: " + "  ".join(f"{m}={row[m]['mean']:+.3f} [{row[m]['ci95'][0]:+.3f},{row[m]['ci95'][1]:+.3f}]" for m in METRICS))
    (ROOT / f"cr_summary_{corpus}.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    for c in sys.argv[1:] or ("ieee", "paysim"):
        main(c)
