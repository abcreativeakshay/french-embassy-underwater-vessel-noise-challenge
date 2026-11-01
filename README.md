# Underwater Vessel Noise Challenge — Zindi (French Embassy in South Africa)

Goal: **rank 1**. Prize $5,000. Metric: **1 − RMSE** on `share_added_vessel` (higher is better).
Competition closes ~13 Dec 2026 (per main-chat summary; confirm on the competition page).

## Key facts (verified 2026-11-01 from zindi.world)

- **Training**: 1,715,779 real, *unlabelled* acoustic records from 10 SA coastal sites, ~1-min
  resolution. One Parquet per site (`TOL_<freq>` band levels), plus `stations.csv`,
  `recorders.csv`, `DATA_DICTIONARY.md`. No Train.csv, no labels.
- **Test**: 1,500 rows × 375 features. Each row = 15-min window × 25 third-octave bands
  (31.6 Hz – 7.94 kHz). Columns like `m01_TOL_31.6` (minute position × band).
  Test = real background + controlled vessel-passage additions. Target
  `share_added_vessel` = fraction of total energy from the added vessel.
- **dB handling**: levels are in dB — do NOT add/subtract in dB. Convert with E = 10^(dB/10).
  Blank cells = NaN (missing), **not** silence.
- **Submissions**: 2/day, 200 total. Select **2 submissions for the private leaderboard**
  before close (default: your 2 best public). 30% public / 70% private split.
  Submit finite [0,1] values for `share_added_vessel` and `score_added_vessel` per ID.
- **Rules that can disqualify you**:
  - Open-source tools only. **AutoML banned.** Pretrained models OK if openly available.
  - Test data is **inference only**: no fitting, fine-tuning, calibration, pseudo-labeling,
    or building training examples from it. Fixed per-window transformations are allowed.
  - Training may use released SA data + **declared** public corpora. Organisers explicitly
    name **Ocean Networks Canada (ONC)** and **HearMyShip** — declare source URLs,
    versions, preprocessing, access. They supply **no synthetic training set**.
  - Top 10 → code review within 48h. Top 3 assign copyright to Zindi. Tiebreak = earlier submission.
  - Reproducibility: set seeds, declare splits/preprocessing, ship runnable inference code
    + trained weights + dependency versions; report training hardware/time, inference
    runtime, peak memory, model size.

## Repo layout

- `data/raw/` — Zindi downloads (SA acoustics, Test.csv, starter notebook) — needs login
- `data/external/` — ONC / HearMyShip vessel audio, declared per rules
- `data/processed/` — synthetic training sets we build ourselves
- `notebooks/` — EDA, baseline, synthetic-training experiments
- `src/` — `features.py`, `synthetic.py`, `model.py`, `submit.py`
- `submissions/` — every submission CSV, dated and logged
- `experiments/` — experiment log (model, validation RMSE, public LB score, submission id)

## Plan

See `PLAN.md` for the modeling approach: energy-domain heuristic baseline → synthetic
training from SA backgrounds + HearMyShip/ONC passages → gradient-boosted regressor →
iterate within the 2/day submission budget.

## What is built (2026-11-01)

Pipeline in `src/` (reproducible, seeds fixed):
- `src/common.py` — dB↔energy conversion, test loading (NaN = missing, never
  filled), Stage B analytic share, diagnostic detection score, submission checks.
- `src/baseline_analytic.py` — Stage B: per-window/per-band background = 10th
  percentile over minutes in energy domain; share = excess/total energy.
  → `submissions/baseline_analytic.csv`
- `src/make_synthetic.py` — Stage C data: contiguous 15-min background windows
  from SA parquets (per site/recorder, gaps ≤ 180 s) + **parametric** vessel
  templates (small-craft / large-ship spectral bumps + passage envelopes) mixed
  in the energy domain at exact ground-truth shares (25% zeros).
  → `data/processed/synthetic_{train,val}.parquet`
- `src/train_lgbm.py` — 30 energy-domain features (analytic share, per-band
  excess fractions, transient stats, excess spectral centroid) → LightGBM
  regressor. Validation on **held-out sites** (Fish Hoek, Sea Point).
  → `models/lgbm_synth.txt`, `submissions/synth_lgbm.csv`

Validation (held-out sites, 543 windows): **RMSE 0.1374 → score 1−RMSE = 0.8626**.
Stage B analytic baseline on the same val: RMSE 0.3585 (score 0.6415) — the
heuristic over-attributes natural background variation to vessels. Model bias:
slight over-prediction at share≈0 (+0.10), slight under-prediction at high
shares (−0.07). Top features: peak/median minute-energy ratio, high-band
(6–8 kHz) excess fractions, transient fraction.

Caveats: vessel templates are parametric, not real HearMyShip/ONC recordings —
real-test performance will be lower than synthetic val. Next steps: replace
templates with real passages, tune the share sampling to match the test prior,
ensemble seeds.

## Blockers (waiting on user)

1. Repo destination: GitHub? What name? (local scaffold will be pushed/renamed accordingly)
2. Nothing submitted to Zindi yet — first submission needs a go-ahead.
