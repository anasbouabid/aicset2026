from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from synthfraud.c_pub import describe_c_pub, sample_with_c_pub
from synthfraud.detect import encode_fit_detector
from synthfraud.log import log
from synthfraud.metrics.adversarial import adversarial_utility
from synthfraud.metrics.fidelity import fidelity
from synthfraud.metrics.privacy import privacy
from synthfraud.metrics.utility import honest_utility, honest_utility_slices
from synthfraud.ieee_cis import load_ieee_cis
from synthfraud.paysim import load_paysim
from synthfraud.protocol import split_table
from synthfraud.r import apply_sanitizer, sample_declared_controllers, sample_declared_rates, sample_with_r
from synthfraud.schema import IEEE_CIS, PAYSIM, TOY, ieee_signal_slices
from synthfraud.synthesizers import build_synthesizer
from synthfraud.toy import make_toy_transactions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T1 synthetic-fraud evaluation harness")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    log(f"config {args.config}  synthesizer={cfg.get('synthesizer')}  dataset={cfg.get('dataset')}")
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)
    dataset = cfg.get("dataset", "toy")
    if dataset == "paysim":
        schema = PAYSIM
        pc = cfg.get("paysim", {})
        log(
            f"load PaySim protocol={pc.get('protocol', 'enriched')} "
            f"n_rows={pc.get('n_rows', 500_000)}"
        )
        table = load_paysim(
            Path(pc.get("path", "../data/raw/paysim.parquet")),
            seed=seed,
            protocol=str(pc.get("protocol", "enriched")),
            n_legit=int(pc.get("n_legit", 40000)),
            all_fraud=bool(pc.get("all_fraud", True)),
            n_rows=int(pc.get("n_rows", 500_000)),
        )
    elif dataset == "ieee_cis":
        schema = IEEE_CIS
        ic = cfg.get("ieee_cis", {})
        log(
            f"load IEEE-CIS protocol={ic.get('protocol', 'natural')} "
            f"n_rows={ic.get('n_rows', 500_000)}"
        )
        table = load_ieee_cis(
            Path(ic.get("path", "../data/raw/ieee_cis")),
            seed=seed,
            protocol=str(ic.get("protocol", "natural")),
            n_rows=int(ic.get("n_rows", 500_000)),
        )
    elif dataset == "toy":
        schema = TOY
        table = make_toy_transactions(int(cfg["n_rows"]), float(cfg["fraud_rate"]), rng)
    else:
        raise ValueError(f"unknown dataset {dataset!r}")

    log(f"table {len(table):,} rows  fraud={int(table[schema.label].sum()):,}  rate={float(table[schema.label].mean()):.4%}")
    splits = split_table(table, schema, seed=seed, **cfg["splits"])
    log(
        f"splits train={len(splits.train):,} ref={len(splits.ref):,} "
        f"val={len(splits.val):,} test={len(splits.test):,}"
    )
    synth_model = build_synthesizer(
        cfg["synthesizer"], seed=seed, schema=schema, **(cfg.get("synth") or {})
    )
    log(f"fit synthesizer {cfg['synthesizer']!r} on train")
    synth_model.fit(splits.train)
    n_synth = int(cfg["n_synth"])
    log("fit bank detector on real train")
    bank = encode_fit_detector(splits.train, schema, seed=seed, kind="hgb")
    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ctx = {
        "cfg": cfg,
        "utc": utc,
        "dataset": dataset,
        "table": table,
        "splits": splits,
        "schema": schema,
        "seed": seed,
        "bank": bank,
        "synth_model": synth_model,
    }

    unfiltered_out = cfg.get("output_unfiltered")
    if unfiltered_out:
        log(f"sample unfiltered n={n_synth:,}")
        raw = synth_model.sample(n_synth)
        log("evaluate unfiltered release")
        _write(
            _report(raw, ctx, c_pub={"enabled": False, **describe_c_pub(schema, raw)}),
            Path(unfiltered_out),
            utc,
        )

    r_cfg = cfg.get("r") or {}
    r_outputs = cfg.get("r_outputs")
    r_stats = None
    if cfg.get("r_filter") and r_outputs:
        log("sample with R at declared rates (val does not choose the budget)")
        controller_maps = [_controller_outputs(spec) for spec in r_outputs]
        named = []
        for mapping in controller_maps:
            named.extend(mapping.keys())
        controllers = tuple(dict.fromkeys(named))
        c_pub_on = bool(r_cfg.get("c_pub", True))
        draw_kw = dict(
            bank=bank,
            val=splits.val,
            train_fraud_rate=float(splits.train[schema.label].mean()),
            seed=seed,
            rates=[float(spec["rate"]) for spec in r_outputs],
            pool_mult=float(r_cfg.get("pool_mult", 12)),
            min_margin=float(r_cfg.get("min_margin", 0.15)),
            lambda_ua=float(r_cfg.get("lambda_ua", 1.0)),
            c_pub=c_pub_on,
            max_draw=int(cfg.get("c_pub_max_draw", n_synth * 30 * float(r_cfg.get("pool_mult", 12)))),
        )
        if any("controllers" in spec for spec in r_outputs):
            releases = sample_declared_controllers(
                synth_model, n_synth, schema, controllers=controllers, **draw_kw
            )
            for spec, (rate, tables) in zip(r_outputs, releases, strict=True):
                for name, path in _controller_outputs(spec).items():
                    table, stats = tables[name]
                    log(f"evaluate declared {name} rate={rate:.4%}")
                    _write(
                        _report(
                            table,
                            ctx,
                            c_pub={"enabled": c_pub_on, **describe_c_pub(schema, table)},
                            r=stats,
                        ),
                        Path(path),
                        utc,
                    )
                base = tables.get("rate_only")
                for i, san in enumerate(spec.get("sanitizers") or []):
                    if base is None:
                        raise ValueError("r sanitizers require a rate_only controller on the same rate")
                    raw, base_stats = base
                    cleaned, eval_schema, meta = apply_sanitizer(
                        raw,
                        schema,
                        kind=str(san["name"]),
                        seed=seed + 200 + i,
                        rel=san.get("rel"),
                        columns=san.get("columns"),
                    )
                    log(f"evaluate sanitizer {meta['sanitizer']} rate={rate:.4%}")
                    r_meta = {**base_stats, **meta, "controller": meta["sanitizer"]}
                    _write(
                        _report(
                            cleaned,
                            ctx,
                            c_pub={"enabled": c_pub_on, **describe_c_pub(eval_schema, cleaned)},
                            r=r_meta,
                            eval_schema=eval_schema,
                        ),
                        Path(san["output"]),
                        utc,
                    )
        else:
            releases = sample_declared_rates(synth_model, n_synth, schema, **draw_kw)
            for spec, (rate, topk, stats, rate_only) in zip(r_outputs, releases, strict=True):
                log(f"evaluate declared top-k rate={rate:.4%}")
                _write(
                    _report(
                        topk,
                        ctx,
                        c_pub={"enabled": c_pub_on, **describe_c_pub(schema, topk)},
                        r=stats,
                    ),
                    Path(spec["output"]),
                    utc,
                )
                if spec.get("output_rate_only"):
                    log(f"evaluate rate-only ablation rate={rate:.4%}")
                    _write(
                        _report(
                            rate_only,
                            ctx,
                            c_pub={"enabled": c_pub_on, **describe_c_pub(schema, rate_only)},
                            r={
                                "enabled": True,
                                "ablation": "rate_only",
                                "target_fraud_rate": rate,
                                "declared_rate": True,
                            },
                        ),
                        Path(spec["output_rate_only"]),
                        utc,
                    )
        log("done")
        return 0
    if cfg.get("r_filter"):
        log("sample with R (C_pub + C_priv + prevalence + C_priv ranking)")
        rate_only_out = cfg.get("output_rate_only")
        synth, r_stats, rate_only = sample_with_r(
            synth_model,
            n_synth,
            schema,
            bank=bank,
            val=splits.val,
            train_fraud_rate=float(splits.train[schema.label].mean()),
            seed=seed,
            pool_mult=float(r_cfg.get("pool_mult", 12)),
            target_fraud_rate=r_cfg.get("target_fraud_rate"),
            rate_grid=r_cfg.get("rate_grid"),
            row_select=str(r_cfg.get("row_select", "val_tables")),
            min_margin=float(r_cfg.get("min_margin", 0.15)),
            n_val_candidates=int(r_cfg.get("n_val_candidates", 6)),
            lambda_ua=float(r_cfg.get("lambda_ua", 1.0)),
            elite_mult=float(r_cfg.get("elite_mult", 4.0)),
            c_pub=bool(r_cfg.get("c_pub", True)),
            max_draw=int(cfg.get("c_pub_max_draw", n_synth * 30 * float(r_cfg.get("pool_mult", 12)))),
            also_rate_only=bool(rate_only_out),
        )
        c_pub = {"enabled": bool(r_cfg.get("c_pub", True)), **describe_c_pub(schema, synth)}
        if rate_only_out and rate_only is not None:
            log("evaluate rate-only ablation (same pool, no U_h/U_a ranking)")
            _write(
                _report(
                    rate_only,
                    ctx,
                    c_pub={"enabled": bool(r_cfg.get("c_pub", True)), **describe_c_pub(schema, rate_only)},
                    r={"enabled": True, "ablation": "rate_only", "target_fraud_rate": r_stats["target_fraud_rate"]},
                ),
                Path(rate_only_out),
                utc,
            )
    elif cfg.get("c_pub_filter"):
        log("sample with C_pub rejection filter")
        synth, c_stats = sample_with_c_pub(
            synth_model,
            n_synth,
            schema,
            max_draw=int(cfg.get("c_pub_max_draw", n_synth * 30)),
        )
        c_pub = {"enabled": True, **c_stats}
    else:
        log(f"sample n={n_synth:,}")
        synth = synth_model.sample(n_synth)
        c_pub = {"enabled": False, **describe_c_pub(schema, synth)}

    log("evaluate release")
    _write(_report(synth, ctx, c_pub=c_pub, r=r_stats), Path(cfg["output"]), utc)
    log("done")
    return 0


def _controller_outputs(spec: dict) -> dict[str, str]:
    """Map controller name → output path from a v3 or v4 r_outputs entry."""
    mapping = dict(spec.get("controllers") or {})
    if spec.get("output"):
        mapping.setdefault("topk", spec["output"])
    if spec.get("output_rate_only"):
        mapping.setdefault("rate_only", spec["output_rate_only"])
    return mapping


def _report(synth, ctx: dict, *, c_pub: dict, r: dict | None = None, eval_schema=None) -> dict:
    cfg = ctx["cfg"]
    bank_schema = ctx["schema"]
    schema = eval_schema or bank_schema
    table = ctx["table"]
    splits = ctx["splits"]
    train = splits.train[schema.all_cols]
    ref = splits.ref[schema.all_cols]
    test = splits.test
    report = {
        "config": {k: v for k, v in cfg.items() if not str(k).startswith("output")},
        "run": {"utc": ctx["utc"], "python": sys.version.split()[0]},
        "dataset": ctx["dataset"],
        "n_rows": int(len(table)),
        "n_fraud": int(table[bank_schema.label].sum()),
        "n_train": int(len(splits.train)),
        "n_ref": int(len(splits.ref)),
        "n_val": int(len(splits.val)),
        "n_test": int(len(splits.test)),
        "n_synth": int(len(synth)),
        "n_synth_fraud": int(synth[schema.label].sum()),
        "fraud_rate": float(table[bank_schema.label].mean()),
        "synth_fraud_rate": float(synth[schema.label].mean()),
        "release_schema": {
            "numeric": list(schema.numeric),
            "categorical": list(schema.categorical),
            "label": schema.label,
        },
        "c_pub": c_pub,
        "r": r,
        "dp": getattr(ctx.get("synth_model"), "accounting", None),
        "fidelity": _metric("fidelity", lambda: fidelity(train, synth[schema.all_cols], schema)),
        "honest_utility": _metric(
            "honest utility U_h",
            lambda: honest_utility(synth[schema.all_cols], test[schema.all_cols], schema, seed=ctx["seed"]),
        ),
        "privacy": _metric(
            "privacy (distance MIA)",
            lambda: privacy(train, ref, synth[schema.all_cols], schema),
        ),
        "adversarial_utility": _metric(
            "adversarial utility U_a",
            lambda: adversarial_utility(
                bank_model=ctx["bank"],
                synth=synth[schema.all_cols],
                real_test=test,
                schema=schema,
                seed=ctx["seed"],
                bank_schema=bank_schema,
            ),
        ),
    }
    if bank_schema.name == "ieee_cis":
        report["honest_utility_slices"] = _metric(
            "U_h slices (counts / no_counts / public)",
            lambda: honest_utility_slices(
                synth, test, ieee_signal_slices(), seed=ctx["seed"]
            ),
        )
    uh = report["honest_utility"]
    ua = report["adversarial_utility"]
    extra = ""
    slices = report.get("honest_utility_slices") or {}
    parts = [
        f"{name}={row['pr_auc']:.3f}"
        for name, row in slices.items()
        if isinstance(row, dict) and "pr_auc" in row
    ]
    if parts:
        extra = "  slices " + " ".join(parts)
    log(
        f"summary PR-AUC={uh['pr_auc']:.3f}  synth_fraud_rate={report['synth_fraud_rate']:.4%}  "
        f"MIA={report['privacy']['mia_distance_auc']:.3f}  "
        f"surr={ua['surrogate_label_agreement']:.3f}{extra}"
    )
    return report


def _metric(name: str, fn):
    log(f"  metric {name} …")
    return fn()


def _json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write(report: dict, out: Path, utc: str) -> None:
    text = json.dumps(report, indent=2, default=_json_default)
    out.parent.mkdir(parents=True, exist_ok=True)
    archive = out.parent / "archive" / f"{out.stem}_{utc}{out.suffix}"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_text(text + "\n")
    out.write_text(text + "\n")
    log(f"wrote {out} and {archive}")


if __name__ == "__main__":
    raise SystemExit(main())
