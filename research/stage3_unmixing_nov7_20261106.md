# Direction #3 — two-stage background + spectral unmixing: GATE PASS
Date: 2026-11-06. Candidate for Nov 7 slots.

## What was built
- `src/features_v3.py` — 68 scale-invariant features (NO absolute levels):
  * 0–29: original 30 features (p10 background), unchanged.
  * 30–57: same 30 features recomputed with a TWO-STAGE background:
    detect passage minutes (minute broadband energy > 1.5x window median),
    background per band = median over clean minutes only.
  * 58–67: NNLS spectral-unmixing features. Total window spectrum
    S ~ a*bg_template + b1*CPA-mean + b2*clip-mean (HearMyShip templates,
    already declared); share = (b1+b2)/(a+b1+b2). Plus clip-only / CPA-only
    variants, per-minute closed-form unmixing profile (max, std,
    frac>0.1), excess-spectrum fit, fit residual, clean-minute fraction.
- `src/train_v3.py` — LightGBM, 10 seeds [7..41], 600 rounds, same params
  as the real-only baseline, hms_clip only (5,797 train / 378 val).
  Per-seed checkpoints (`submissions/v3_ckpt_s<seed>.npz`), models
  `models/v3_s<seed>.txt`.

## Diagnostics (synthetic val, 378 hms_clip rows, true bg known)
- Background level error (mean |log10(bg_est/bg_true)|): p10 0.1625 ->
  two-stage median-of-clean 0.0493 (3.3x better). Oracle true-bg: 0.0000.
- Analytic share 1-RMSE: p10 0.6201 -> two-stage 0.6845 -> oracle 0.7104.
- Passage-minute detector (broadband ratio): precision low (0.26 @ th=1.3)
  but recall 0.99 — background fluctuations trigger constantly, yet the
  median-of-clean background is robust to the false flags.
- NNLS unmixing standalone 1-RMSE: 0.8348 (bg2 + CPA-mean + clip-mean),
  corr 0.861 with truth; per-minute closed-form 0.8017. Far above the
  analytic share (0.62) but below LightGBM — an independent estimator,
  ideal as a feature / blend component.
- 5-template (per-category CPA) unmixing was WORSE (0.7796) — not used.

## Honest validation (held-out sites Fish Hoek + Sea Point, disjoint clips)
| model | val 1-RMSE | Fish Hoek | Sea Point |
|---|---|---|---|
| real-only baseline (10 seeds) | 0.9229 | 0.890* | 0.935* |
| **v3 ensemble (10 seeds)** | **0.9277** | **0.9141** | **0.9488** |
| 50/50 v3 + current best (ro+v1) | 0.9281 | 0.9152 | 0.9478 |
| 70/30 v3 + current best | 0.9287 | 0.9156 | 0.9488 |

*baseline site split from the VTUAD report (FH 1-RMSE 0.890, SP 0.935).
Every single v3 seed beats the baseline ensemble (0.9254–0.9275).
Fish Hoek MAE 0.0542 (was ~0.064–0.068), Sea Point MAE 0.0244.
By share band (v3): zero 0.0785 RMSE (bias +0.032), low 0.0433,
mid 0.0785, high 0.0640, vhigh 0.0733 — the mid-share weakness is reduced
and the zero-share over-prediction bias is cut (+0.032 vs +0.034 for v1,
but on a much lower RMSE base).

GATE: v3 0.9277 >= 0.9229 — PASS (+0.0048). Blend gate also PASS.
Feature importance (seed 7): unmix_share is #1 by gain (1939.8, ~3.7x
the #2 feature) — the unmixing estimator is the dominant new signal.

## Submission files (verified: 1,500 rows, ID order = Test.csv, finite [0,1])
- `submissions/nov7_v3.csv` — standalone v3, test mean 0.2558, max 0.7895.
- `submissions/nov7_v3_blend.csv` — 50/50 with the current best blend
  (nov4_slot2_ro_v1_avg.csv, public 0.79519); mean 0.2568, corr 0.9614
  vs best, mean|diff| 0.0404. Blend == 0.5*v3 + 0.5*best verified exactly.
- `submissions/nov7_v3_blend70.csv` — 70% v3 + 30% current best; mean
  0.2564, max 0.7837, corr 0.9807 vs best. Val 1-RMSE 0.9287 (best of the
  three val blends; 50/50 is 0.9281, standalone v3 0.9277). Built after the
  main run from the verified v3 + best files, same verification.

## Declaration
No new declaration needed. All templates are HearMyShip-derived
(CPA templates + clip spectra); HearMyShip and ONC are already declared.
No VTUAD data or spectra used anywhere in this candidate.

## Caveats (v2 lesson applies)
Validation does not predict public: v2 passed val (0.9245) and failed
publicly (0.78965/0.78088). The v3 val gain (+0.0048) is real and
consistent across all 10 seeds and both sites, and the mechanism
(background-noise reduction + an independent physics estimator) is the
validated failure mode — but the public score is still the judge.
Recommended slot order: slot 1 = blend (safe, val 0.9281), slot 2 = v3
standalone (upside, val 0.9277).
