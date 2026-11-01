# Technical plan — Underwater Vessel Noise Challenge

Score = 1 − RMSE on `share_added_vessel` ∈ [0,1]. `score_added_vessel` is diagnostic only
(not scored) — any reasonable detection score is fine.

## 1. What we're actually predicting

share_added_vessel = E_added_vessel / E_total over the 15-min × 25-band window, in the
**energy domain** (E = 10^(dB/10)). Organisers built test by adding recorded vessel
passages onto real SA background minutes; backgrounds used for test are excluded from
the training release. Zero target = no controlled addition (background may still contain
vessels — do NOT treat zero as "vessel-free").

**Open question for when data arrives**: exact aggregation of the share (sum over all
minutes × bands, presumably). Check `DATA_DICTIONARY.md` and the starter notebook's
baseline definition before finalizing the training target.

## 2. Modeling strategy

### Stage A — get on the board (day 1)
- Run the starter notebook's constant baseline, submit. Establishes a floor and verifies
  the submission pipeline.
- Check `DATA_DICTIONARY.md`: band coverage per recorder, quality flags, `dur` semantics.

### Stage B — analytic energy baseline (no training data needed)
Per window, estimate background per band from the window itself (vessel passages are
transient, backgrounds are persistent):
- bg_b = low percentile (e.g. 10th) or min over the 15 minute-positions, in energy domain
- share ≈ Σ_b Σ_m max(E_obs − bg_b, 0) / Σ_b Σ_m E_obs
- Handle NaNs by masking (never fill with 0 — blank ≠ silence).
- This respects the "fixed per-window transformation" rule and uses zero training.

### Stage C — synthetic training (the real model)
Organisers give no synthetic training set; we build our own:
1. **Backgrounds**: sample 15-min windows from released SA acoustics (real site minutes,
   but from periods/sites disjoint from anything test-like).
2. **Vessel passages**: audio from HearMyShip (>1000 small-vessel passages, Croatia;
   hearmyship.fer.hr) and ONC hydrophones (data.oceannetworks.ca) → convert to the same
   1-min × 25-band TOL representation (third-octave, 31.6 Hz–7.94 kHz, dB convention of
   the exports; document units/time-alignment/missingness handling).
3. **Mixing**: add passage energy onto background energy at controlled shares across the
   full [0,1] range, including exact 0s; vary passage type (small craft vs large ship),
   duration within the 15-min window, and level. Keep per-mix ground truth share.
4. **Model**: gradient boosting (LightGBM, open-source) or small MLP on features —
   per-band energy excess over window background estimate, temporal deltas, spectral
   shape stats. Validate on held-out sites AND held-out passages (not random splits —
   leakage via similar backgrounds/passages will lie to us).
5. Calibrate predictions to [0,1] (clip), keep raw values (rules: don't round/threshold
   to game the LB).

### Stage D — iterate
- Error analysis by share level, band region, passage type.
- Ensemble of diverse seeds/models; watch public LB for overfit (30% public / 70%
  private — don't chase the public board).
- Ablate: does Stage B's analytic background estimate as a feature dominate? Keep the
  model honest with a pure-heuristic fallback.

### score_added_vessel
Diagnostic only. Use a simple detection score, e.g. max over minutes of broadband
energy excess vs background, scaled to [0,1]. Low effort.

## 3. Submission discipline (2/day, 200 total)

- Log EVERY submission in `experiments/`: date, model, validation RMSE, public score,
  submission id, selected-for-private? (y/n).
- Budget: early weeks explore (Stage A–C); reserve late submissions for the best
  2–3 candidates. **Before close, explicitly select the 2 private-LB submissions**
  (default = 2 best public, which may not be the best private choice).
- Tiebreak is submission time — when two candidates tie on public, prefer the earlier.

## 4. Reproducibility (required for top 10)

- Set all seeds; record them. Declare train/validation splits and preprocessing.
- Keep `requirements.txt` pinned; inference script must reproduce submitted predictions
  from `data/raw` + weights alone.
- Report: training hardware/time, inference runtime, peak memory, model size.

## 5. Data acquisition checklist

- [ ] Zindi login → download SA acoustics zip, Test.csv, SampleSubmission.csv, starter notebook
- [ ] HearMyShip access (hearmyship.fer.hr) — check licence, declare version/URL
- [ ] ONC access (data.oceannetworks.ca) — hydrophone data, check licence/access terms
- [ ] Optional: external AIS (timestamps must be checked against acoustic export clock;
      absence of AIS ≠ no vessel — treat as weak signal only, or skip)
