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

## External vessel data (declared per challenge rules)

**HearMyShip** — "A Database of Underwater Radiated Noise from Small Vessels in
the Coastal Area" (Shipton, Obradović, Ferreira, Mišković, Bulat, Cukrov,
Diamant; recorded in the Šibenik canal, Croatia; open access).
- Source URL: https://hearmyship.fer.hr/ (accessed 2026-11-01)
- Paper record: https://repozitorij.fer.unizg.hr/islandora/object/fer:12025
  (urn:nbn:hr:168:030404)
- What we used: 322 twenty-second 48 kHz mono WAV passages at closest point of
  approach (stratified: Motorboat/Yacht/Tour boat/Ferry/Fishing/Cargo/Aux;
  Sailboat clip URLs on the site were stale → excluded from the WAV pool) +
  published per-vessel third-octave CPA spectra for 1,100 vessels (30 bands,
  31.5 Hz–20 kHz, mapped onto our 25 bands).
- Preprocessing (`src/vessel_signatures.py`): per-second third-octave band
  energies via rfft on 1-s Hann windows (25 challenge bands, 31.6 Hz–7.94 kHz);
  per-band ambient floor (5th percentile over the clip) subtracted so the
  signature is the *vessel-added* component; signatures normalised and rescaled
  to exact target shares in the energy domain at mixing time (hydrophone
  calibration differences cancel). Licence: open access per the repository
  record; cite Shipton et al. when publishing.
- ONC (Ocean Networks Canada, data.oceannetworks.ca) was **not** used — it
  requires an API token/account; HearMyShip alone was sufficient for this
  iteration.

## What is built (2026-11-01, real-vessel iteration)

Replaced the parametric vessel templates with real HearMyShip passages:
- `src/vessel_signatures.py` — WAV → (20 s × 25 band) energy signatures
  → `data/external/vessel_signatures.npz` (322 clips); CPA spectra →
  `data/external/vessel_cpa_templates.npz` (1,100 templates).
- `src/make_synthetic.py` — now samples real clips (65%) / CPA templates
  (35%); clips embedded as 20-s band totals in 1–2 minute bins, templates as
  1–3 min raised-cosine passages; exact-share energy-domain mixing; vessel
  clips split **disjointly** between train/val (no passage leakage).
- `src/train_lgbm.py` — same 30 features; new flags `--out-name`,
  `--model-name`. → `models/lgbm_real.txt`,
  `submissions/synth_lgbm_real.csv`

Validation (held-out **sites** Fish Hoek + Sea Point **and** held-out vessel
clips, 543 windows): **RMSE 0.0807 → score 1−RMSE = 0.9193**
(previous parametric: 0.8626). Bias now balanced: +0.034 at share≈0, −0.034 at
high shares. Top features: transient_frac, peak_median_ratio, excess_centroid,
high-band excess fractions. Test prediction prior: mean share 0.26, median
0.21 — plausible vs the over-attributing analytic baseline (0.72).

Caveats: synthetic val still overstates real-test performance (our passages
≠ organisers' passages; SA backgrounds differ from test backgrounds). Next:
seed ensemble, share-prior tuning, error analysis by vessel category.
