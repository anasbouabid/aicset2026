# Asymmetric-utility synthetic fraud data: experiments

Code and archived results for *Asymmetric-Utility Synthetic Fraud Data: Prevalence Control Does Not Isolate Adversarial Utility* (AICSET 2026).

A CTGAN table is filtered by public typology constraints and capped at a declared fraud rate. The harness scores honest utility (a detector trained on the table, tested on real data), record privacy (membership, reconstruction, attribute inference) and adversarial utility (transfer evasion against a bank detector) on PaySim and IEEE-CIS.

## Layout

| Path | Contents |
| --- | --- |
| `experiments/src/synthfraud/` | Generators, public constraints, release cap, detectors, metrics |
| `experiments/configs/` | One YAML per run (`paysim_natural_*`, `ieee_cis_natural_*`, `paysim_full_*`) |
| `experiments/scripts/` | Evaluation scripts: transfer evasion, membership/reconstruction/attribute inference, SHAP agreement, data fetchers |
| `experiments/results/` | Archived JSON reports (`results/archive/` keeps every run) and per-target outcome arrays (`cr_transfer_*.npz`) |

Data are not included. The code expects `data/raw/paysim.parquet` and `data/raw/ieee_cis/` at the repository root.

## Setup

```bash
cd experiments
conda env create -f environment.yml
conda activate synthfraud
pip install -e '.[ctgan,xai,huggingface]'      # '.[dp]' adds PATE-CTGAN
python scripts/fetch_paysim.py                 # PaySim parquet (Hugging Face)
python scripts/fetch_ieee_cis.py               # needs a Kaggle token and accepted competition rules
```

## Reproducing the main experiments

Each run writes JSON to `experiments/results/` and an immutable copy to `results/archive/`.

```bash
cd experiments

# Honest utility, privacy and surrogate metrics per release (example: PaySim, seed 0)
python -u -m synthfraud --config configs/paysim_natural_ctgan_r_v5.yaml

# Transfer evasion with baselines and bootstrap intervals (the adversarial-utility experiment)
python scripts/ieee_cis_c4.py --seed 0         # fits CTGAN, caches the IEEE-CIS release tables
python scripts/cr_transfer_evasion.py --corpus ieee --seed 0
python scripts/cr_transfer_evasion.py --corpus paysim --seed 0   # refits CTGAN (about 35 min per seed on Apple silicon)
python scripts/cr_aggregate.py ieee paysim     # pools seeds 0-2 into results/cr_summary_*.json
```

Repeat with `--seed 1` and `--seed 2` before aggregating. See `experiments/README.md` for the full list of configs.

## Notes

- CTGAN training is not bit-reproducible: refitting with the same seed gives a different table and values that differ in the second decimal. The paper pools all fits of a cell and states how many there are.
- Splits are train 50% / reference 20% / validation 15% / test 15%, stratified by label.
- Cached release tables (`*.parquet`) and model weights are not tracked; the JSON and `.npz` results are.
