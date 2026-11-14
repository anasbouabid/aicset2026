# Evaluation harness

Phase 1 of `notes/research-plan.md`: four metric families **before** building module G.

## Python environment (Conda)

Default is a named **conda-forge** env `synthfraud` (Python 3.11). File: `environment.yml`.

```bash
cd experiments
conda env create -f environment.yml    # first time
conda activate synthfraud
# later: conda env update -n synthfraud -f environment.yml --prune
# or:    ./scripts/setup_env.sh
```

If `conda` is missing from a new terminal after Miniforge install:

```bash
conda init zsh
# restart the shell, or:
eval "$(conda shell.zsh hook)"
```

Always run the harness **inside** `synthfraud`:

```bash
conda activate synthfraud
python -m synthfraud --config configs/default.yaml
```

Do not use the leftover `experiments/.venv` (an earlier uv/venv attempt).

## Watching progress

Agent-run jobs in chat do not stream well. Run the harness **in your own terminal** (Cursor: Terminal → New Terminal):

```bash
conda activate synthfraud
cd experiments
python -u -m synthfraud --config configs/paysim_natural_ctgan_cpub.yaml
```

`-u` is unbuffered. Stages print as `[synthfraud HH:MM:SS] …` on stderr. CTGAN then shows a tqdm epoch bar (`1/50`, `2/50`, …). The quiet gap *before* that bar is the data transformer/GMM — it can take several minutes on 250k rows; that is expected.

JSON reports are written to `results/` at the end; they are no longer dumped into the live log.

## Protocol

| Split | Role |
| --- | --- |
| `train` | Only input to the synthesizer (members for MIA) |
| `ref` | Non-members for MIA; not used to train G |
| `val` | Honest-utility probe for the release controller R (never the reported test $U_h$) |
| `test` | Forever held-out **real** test for `U_h` |

Default product (T1): evaluate a released table, not generator weights.

## Metric families

1. **Fidelity** — Wasserstein-1 / TVD; public-constraint violation rate.
2. **Honest utility `U_h`** — train on synth, test on real (PR-AUC, recall at 1% FPR).
3. **Privacy (Risk A)** — distance-based MIA AUC. Nearest-neighbour distance is logged, not a privacy claim.
4. **Adversarial utility `U_a` (Risk B)** — surrogate agreement + public-constraint-legal evasion ASR.

## Commands

```bash
conda activate synthfraud
python -m synthfraud --config configs/default.yaml
python -m synthfraud --config configs/paysim_identity.yaml
python -m synthfraud --config configs/paysim_independent.yaml
python -m synthfraud --config configs/paysim_ctgan.yaml
python -m synthfraud --config configs/paysim_natural_identity.yaml
python -m synthfraud --config configs/paysim_natural_identity_ntrain.yaml
python -m synthfraud --config configs/paysim_natural_independent.yaml
python -m synthfraud --config configs/paysim_natural_ctgan.yaml
python -m synthfraud --config configs/paysim_natural_independent_cpub.yaml
python -m synthfraud --config configs/paysim_natural_ctgan_cpub.yaml  # same G: unfiltered + C_pub
python -m synthfraud --config configs/paysim_natural_dpind_eps1.yaml
python -m synthfraud --config configs/paysim_natural_dpind_eps10.yaml
python -m synthfraud --config configs/paysim_natural_dpclasscond_eps10.yaml
# PATE-CTGAN (CPU; loop stops when ε is spent, can take hours):
pip install -e '.[dp]'
python -u -m synthfraud --config configs/paysim_natural_patectgan_eps10.yaml
python -u -m synthfraud --config configs/paysim_natural_independent_r.yaml
python -u -m synthfraud --config configs/paysim_natural_ctgan_r.yaml  # unfiltered + rate-only + full R, one G
python -u -m synthfraud --config configs/paysim_natural_ctgan_r_seed1.yaml
python -u -m synthfraud --config configs/paysim_natural_ctgan_r_seed2.yaml
# R v3 declared 2%/5% (do not overwrite): paysim_natural_ctgan_r_v3*.yaml
# R v4 margin vs type-preserving ranking (do not overwrite v3):
python -u -m synthfraud --config configs/paysim_natural_ctgan_r_v4.yaml
python -u -m synthfraud --config configs/paysim_natural_ctgan_r_v4_seed1.yaml
python -u -m synthfraud --config configs/paysim_natural_ctgan_r_v4_seed2.yaml
# R v5 sanitize 5% rate-only (jitter / withhold balances; no C_priv ranking):
python -u -m synthfraud --config configs/paysim_natural_ctgan_r_v5.yaml
python -u -m synthfraud --config configs/paysim_natural_ctgan_r_v5_seed1.yaml
python -u -m synthfraud --config configs/paysim_natural_ctgan_r_v5_seed2.yaml
# Full dump (do not overwrite 500k JSON):
python -u scripts/paysim_feature_ablation.py --n-rows -1
python -u -m synthfraud --config configs/paysim_full_ctgan.yaml
```

PaySim dump: `data/raw/paysim.parquet` (gitignored). Two protocols (keep both):

- **enriched** (`configs/paysim_*.yaml`): all 8,213 fraud rows + 40k legit.
- **natural** (`configs/paysim_natural_*.yaml`): stratified 500k-row draw at the dump rate (~0.13%). `n_rows: -1` is the full table (`configs/paysim_full_ctgan.yaml`; do not overwrite 500k JSON).

## Keeping results

Venue is not locked. Every run writes:

- `results/<name>.json` — latest pointer
- `results/archive/<name>_<UTC>.json` — immutable copy

Do not delete old protocol variants. Catalog numbers in `notes/findings.md`. JSON is tracked in git; model weights and parquet dumps are not.

