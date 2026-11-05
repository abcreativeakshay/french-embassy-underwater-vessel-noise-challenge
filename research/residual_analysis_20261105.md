# Vessel residual analysis — v1 ensemble on synthetic val (543 rows)
Date: 2026-11-05. Model: ens_v1 (5 seeds). Overall 1-RMSE 0.9188.

## Where the model fails (abs error)
- By true share: zero 0.034 | low 0.054 | mid (0.2-0.4) 0.065 | high 0.062 | vhigh 0.049
  → Mid-share windows are hardest (vessel energy ≈ background energy = ambiguous).
- Bias: zero-share OVER-predicted (+0.034), high/vhigh UNDER-predicted (-0.031/-0.037).
  → Classic shrinkage + false positives on pure background.
- By vessel type: uniform (0.042-0.056). No single type dominates.
- By passage length: 1-min 0.045 | 2-min 0.070 | 3-min 0.047.
  → 2-minute passages are distinctly worst (temporal ambiguity).
- By site: Fish Hoek 0.064 vs Sea Point 0.035 — NEARLY 2x WORSE.
  → Fish Hoek mean level 69.9 dB vs Sea Point 59.2 dB (~10 dB louder background).

## The key gap: NO absolute-level features
All 30 current features are RELATIVE (excess over per-band background, ratios, fractions).
None capture the ABSOLUTE background level or total window energy. The model cannot tell
"loud site, no vessel" from "quiet site, vessel" — so at loud Fish Hoek it hallucinates
vessels from background fluctuations (worst cases: true 0.0 → predicted 0.27-0.41, mostly
Fish Hoek). Conversely it underestimates strong vessels (true 0.76-0.82 → 0.36-0.47).

## Concrete new-feature candidates (for tomorrow)
1. Per-band absolute background level (bg_dB, 25 values) + total window energy.
   Gives the model site/background context it currently lacks.
2. Background VARIANCE per band (not just 10th percentile) — loud sites have noisier
   backgrounds; variance lets the model discount fluctuations at loud sites.
3. Site-adaptive excess threshold: the analytic share uses a fixed +3 dB cut; at loud
   sites the background variance is higher, so a fixed cut over-triggers. Scale the
   threshold by background variance.
4. Temporal features: minute-by-minute excess profile (which minutes, rise/fall shape).
   Current features collapse all 15 minutes into aggregates; 2-min passages (worst)
   suggest temporal shape matters.
5. Low-frequency emphasis: vessel noise is low-frequency dominant; per-band excess is
   already there, but a dedicated low-band (31-250 Hz) vs high-band energy ratio
   feature could sharpen the vessel signature vs broadband background.

## What this rules OUT
- More seeds (proven: 20 < 10 on public). More same-family averaging. Same 30 features.
