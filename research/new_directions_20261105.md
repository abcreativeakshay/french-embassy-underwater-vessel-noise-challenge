# Vessel Noise Challenge — New Directions Research Report
Date: 2026-11-05. Current best public: 0.7952 (10-seed real-only + v1 blend).
Plateau: last 4 submissions in a 0.0005 band (0.7947–0.7952).

## 0. Grounding: what we already do (so this is not repeated)

- Input: test windows are 15 minutes × 25 third-octave bands (31.6 Hz–7.94 kHz), dB.
- Current 30 features, ALL relative / scale-invariant:
  analytic_share, 25× per-band excess fractions, transient_frac,
  cells_above_3dB, peak_median_ratio, excess_centroid.
  Background = 10th percentile over the 15 minutes, per band, in energy domain.
- Training: synthetic mixtures — real SA backgrounds + HearMyShip vessel passages
  (322 WAV clips, 20 s at CPA, embedded in 1–2 min) or CPA spectral templates
  (1,100, raised-cosine 1–3 min). Passages span only 1–3 of 15 minutes.
- Models: LightGBM, 600 rounds. Best = real-only (5,797 hms_clip mixtures) + v1 blend.
- Tried and exhausted: seed averaging (20 seeds < 10 on public), isotonic
  calibration on val (v2 lesson: overfits synthetic), same-family averaging.

## 1. Key empirical findings (measured today, not hypothesized)

### 1a. TEST IS ~16 dB LOUDER THAN ALL TRAINING DATA (the dominant gap)
- Synthetic val mixed windows: median 66.5 dB (Fish Hoek 68.5, Sea Point 58.0).
- Synthetic val background-only: median 66.0 dB.
- TEST windows: median 82.4 dB; 82.8% of test windows > 65 dB; 70.9% > 70 dB.
- Vessel additions explain only ~1.3 dB of this (mean predicted share 0.26 →
  total = bg/0.74 = +1.3 dB). The remaining ~15 dB is background level or
  recorder calibration. Either way: test lives FAR outside the training level range.
- Consequence: the model has never seen backgrounds this loud. LightGBM cannot
  extrapolate; every loud test window lands in the top leaf of any level split.

### 1b. THE BACKGROUND ESTIMATOR IS EXTREMELY NOISY (calibrated on synthetic val)
- Duration estimator test: with NO vessel (share=0), ~7 of 15 minutes exceed the
  10th-percentile background by >30% in energy. The estimator cannot distinguish
  1-min from 3-min passages (estimates 7.16 / 7.46 / 7.78 — nearly identical).
- The 10th percentile over 15 samples is between the 1st and 2nd lowest value:
  high variance, and it is the foundation of ALL 30 features. Every feature
  inherits this noise.

### 1c. RESIDUAL ANALYSIS (parent, v1 ensemble on synthetic val, 543 rows)
- Fish Hoek (mean 69.9 dB) error 0.064 vs Sea Point (59.2 dB) error 0.035 — 2× worse.
- Worst false positives: true 0.0 → predicted 0.27–0.41, mostly Fish Hoek.
  The model hallucinates vessels from background fluctuations at loud sites.
- Mid-share (0.2–0.4) hardest (0.065); high shares under-predicted (-0.031/-0.037).
- 2-min passages worst (0.070) vs 1-min (0.045) — temporal ambiguity is real.
- NO absolute-level feature exists: the model cannot tell "loud site, no vessel"
  from "quiet site, vessel".

### 1d. TEST PREDICTION PATTERN MATCHES THE FAILURE MODE
- Mean predicted share by test loudness tercile: quiet 0.245, mid 0.233, LOUD 0.295.
  The model systematically predicts MORE vessel at loud windows — the same
  hallucination seen at Fish Hoek, now applied to a test set that is 71% loud.
  This is the plateau mechanism: the bias, not the variance, is what is stuck.

### 1e. TEST MISSINGNESS STRUCTURE (not exploited, and a train/test mismatch)
- Overall NaN 3.26%, but 48% of rows have some NaN.
- Minute m01 valid only 58.4% of the time; m15 93.3%; minutes m02–m14 ~100%.
- Synthetic training windows are COMPLETE (mask all True). The model has never
  seen a window with a missing first minute. Features are NaN-aware, but the
  model was never trained on incomplete inputs.

## 2. Research questions — evidence

### Q1. What do top published approaches use?
- Vessel CLASSIFICATION (not quantification) is well studied: ShipsEar, DeepShip,
  Oceanship, VTUAD. Best methods: compact CNNs on STFT/Mel/CQT spectrograms;
  CAPSE (coherently averaged power spectrum) for machinery tonals + ViT/CNN
  (CAPSE-ViT: 97.98% DeepShip, 99.19% ShipsEar); LOFAR grams + DEMON grams
  (cyclostationary envelope analysis) for propeller/machinery signatures.
  Sources: https://arxiv.org/pdf/2609.29179 (deployable vessel classification);
  http://journals.pan.pl/Content/135220/aoa.2025.153662.pdf (CAPSE-ViT);
  https://arxiv.org/html/2512.11545v1/ (UATR-GTransformer).
- CRITICAL CAVEAT: all of these need RAW AUDIO or spectrograms. Our test data
  is 15×25 third-octave band levels only — no waveform, no fine spectrogram.
  Audio-domain models CANNOT be applied to test. What transfers is the FEATURE
  IDEAS: low-band dominance of vessel noise (ship band 40–315 Hz per Baldwin et
  al.), tonal/machinery structure, temporal passage shape.
- QUANTIFICATION (energy fraction) has no published equivalent — this challenge
  is novel in asking for a fraction rather than a class. The closest literature
  is spectral unmixing / source separation and soundscape decomposition.

### Q2. Domain adaptation synthetic→real: what actually works?
- A 4-corpus UDA benchmark (Oceanship, QiandaoEar22, VTUAD, ShipsEar; 344 runs)
  finds: CORAL (moment matching) is the MOST STABLE method; MMD can cause
  significant NEGATIVE transfer (-0.147 in one direction); DANN/CDAN show
  direction-dependent collapse. On the hardest direction NO method was usable.
  Source: https://www.preprints.org/manuscript/202609.1124/v3
- A synthetic→real acoustic regression paper (ARS pressure sensing) shows CORAL
  with a CNN feature extractor + regressor generalizes from FEM-simulated
  training to real measurements without labeled real data.
  Source: https://www.frontiersin.org/journals/acoustics/articles/10.3389/facou.2026.1867954/full
- APPLICABILITY TO US: LOW for feature-space CORAL/MMD/DANN, for three reasons:
  (a) we cannot fit anything on test (rule); (b) our synthetic data is ALREADY
  built from real SA backgrounds, so feature distributions are close — the gap
  is vessel TYPE and background LEVEL, not feature covariance; (c) the benchmark
  warns that adaptation can hurt. The PRACTICAL domain adaptation for us is
  TRAINING-DATA adaptation: level augmentation + vessel diversity (see §3),
  which needs no target-domain fitting.

### Q3. How can the 1.7M unlabelled real records be used legitimately?
- They are RELEASED TRAINING data, not test — fitting on them is fully allowed.
  The inference-only rule covers Test.csv only.
- Legitimate uses, ranked by value:
  (a) Learn background spectral templates + variability (PCA of background
      spectra, per-band variance profiles) → background-model features.
  (b) Generate MORE diverse synthetic backgrounds, including loud periods and
      storm/weather conditions (the release spans sites and seasons).
  (c) Estimate background fluctuation statistics → adaptive excess thresholds
      (threshold scaled by local background variance, not fixed +3 dB).
  (d) Self-supervised autoencoder on background windows → reconstruction error
      as a vessel-evidence feature. WEAKER: SA backgrounds themselves contain
      incidental vessel traffic, so "background-only" is not clean; the
      autoencoder would partly learn vessels too.
- Note: we already sample synthetic backgrounds from these records, but we use
  them only as passive carriers — we extract NO learned background statistics
  from them. That is the untapped part.

### Q4. What in the 376 columns are we NOT exploiting?
- ABSOLUTE LEVELS: all 30 features are scale-invariant. Per-band background dB,
  total window energy, and loudness context are entirely absent.
- TEMPORAL STRUCTURE: the 15-minute sequence is collapsed to peak/median.
  Lost: which minutes have excess, rise/fall shape, passage duration, smoothness
  (a vessel hump is smooth; background fluctuation is spiky), autocorrelation.
- BACKGROUND VARIANCE: only the 10th percentile is used. Per-band std/IQR over
  minutes, and minute-to-minute variability, are absent.
- SPECTRAL SHAPE DETAIL: only the centroid. Lost: spectral slope of excess,
  low-band (31–250 Hz) vs high-band ratio, number of bands with significant
  excess, spectral flatness, correlation of the excess spectrum with known
  vessel templates.
- MISSINGNESS: the model never sees incomplete windows in training (see §1e).

### Q5. Physics-informed ideas (target = energy fraction)
- Energy addition is incoherent: E_obs = E_bg + E_vessel (per minute, per band).
  Share = ΣE_vessel / ΣE_obs. The ENTIRE problem reduces to estimating E_bg.
- Spectral unmixing: per window, fit observed spectrum ≈ a·background_template
  + b·vessel_template by non-negative least squares; share ≈ b/(a+b) in energy.
  Templates: background = quiet-minute median spectrum of the same window;
  vessel = mean HearMyShip/VTUAD spectrum. This is a fixed per-window transform
  (allowed) and gives a DIRECT, interpretable share estimate — either as a new
  feature or a standalone estimator to blend.
- Vessel spectra are low-frequency dominant (ship band 40–315 Hz) with a
  characteristic slope; backgrounds are flatter or weather-shaped. A dedicated
  low-band excess ratio is a sharper vessel prior than broadband excess.
- Passage-duration-aware background: if excess minutes are identified first,
  background can be estimated ONLY from clean minutes — unbiased, unlike the
  10th percentile which is contaminated when passages are long. Two-stage
  estimation (detect → mask → re-estimate) is the principled fix for the
  noisy-background problem.

### Q6. Zindi forum / discussion (public, no login used)
- Competitors publicly declare VTUAD (rebuilt) v1.3.2/v1.3.3 and HearMyShip.
  VTUAD is the community's large-vessel source. No approach details leaked
  beyond data declarations; the discussion is mostly declarations + rules Q&A.
- VTUAD (rebuilt): 525 labeled single-vessel scenario clips, ~56 h of 32 kHz
  WAV, 12 classes (passengership, tug, tanker, cargo, pleasurecraft, fishing,
  sailing, pilotvessel, rescue, dredger, other, background), ONC hydrophone
  ICLISTENAF2523, Strait of Georgia, May 2016–May 2017, CC BY 4.0.
  Source: http://zenodo.org/records/23773305 (v1.3.3, Oct 30 2026);
  original VTUAD paper classes incl. cargo/tanker/tug/passenger with F1 > 0.94:
  https://www.mdpi.com/2077-1312/13/8/1455
- Why VTUAD matters for us: HearMyShip is SMALL vessels only (<55 m, Šibenik
  canal). If organisers' test passages include large commercial vessels
  (cargo/tankers — likely in SA coastal waters), their spectra differ
  materially (more low-frequency energy, different slope). Our vessel pool has
  ZERO large ships. VTUAD fills exactly that hole, and it is already declared
  by competitors, so it is unambiguously allowed.
- Practical note: the full VTUAD WAV release is ~25 GB. We do NOT need all of
  it — a subset (e.g., tanker + cargo + tug clips) converted to 25-band
  energies is sufficient, and the per-class ZIPs allow selective download.

## 3. Ranked new directions

### #1. ABSOLUTE-LEVEL + BACKGROUND-VARIANCE FEATURES, WITH dB-OFFSET AUGMENTATION
- What: add to the 30 features: (a) per-band absolute background dB (25),
  (b) total window energy, (c) per-band background std/IQR over minutes,
  (d) low-band (31–250 Hz) vs high-band excess ratio, (e) per-minute excess
  profile summary (duration above threshold, peak minute, rise/fall).
  AND augment synthetic training with random global dB offsets (e.g., uniform
  0 to +25 dB, plus -10 to 0 for symmetry) applied to whole windows.
- WHY it should beat 0.7952 (mechanism): the validated failure is loud-site
  hallucination (Fish Hoek 2× error; test 71% loud; loud tercile over-predicted
  0.295 vs 0.245). The model cannot currently condition on level at all.
  Absolute features let it learn "at high levels, discount this excess pattern".
  The offset augmentation is the KEY enabler: share is scale-invariant, so a
  global dB offset preserves labels EXACTLY — we can manufacture training
  windows at test's level range for free. Without augmentation, absolute
  features would extrapolate (LightGBM cannot); with it, the model LEARNS the
  level-dependence across 57–95 dB. Note: offset augmentation alone does
  NOTHING to the current 30 features (all scale-invariant) — it only matters
  paired with absolute features. The two changes are one package.
- Data/features needed: no new data. New featurize() + regeneration of
  features from stored synthetic parquet arrays (fast) + retrain LightGBM.
  Also add NaN-mask augmentation (random minute/cell masking, so the model
  sees incomplete windows like test's 48%-with-NaN pattern) — same pipeline.
- Effort: 2–4 hours (featurize rewrite ~1h, retrain + validate ~1h, submit ~30min).
- Main risk: if test's level shift is a pure recorder-calibration offset with
  NO behavioral meaning, absolute features could mislead. MITIGATION: validate
  on held-out Fish Hoek first — if Fish Hoek error does not drop toward Sea
  Point's, do not submit the pure model; submit only as a blend. Also the
  blend slot hedges this risk.
- Validation gate (decide before submitting): retrained model must cut Fish
  Hoek val error materially (target: 0.064 → ≤0.045) AND not worsen Sea Point.
  If it fails the gate, fall back to blend-only.

### #2. VTUAD LARGE-VESSEL PASSAGES (diversify the vessel pool)
- What: download VTUAD tanker/cargo/tug/pilotvessel clips (subset of the
  25 GB release), convert to 25-band energies with the existing
  vessel_signatures pipeline, add as a new vessel pool in make_synthetic,
  retrain (real-only + VTUAD mix, or VTUAD-only variant).
- WHY: HearMyShip = small vessels only. SA coastal test passages plausibly
  include large commercial vessels whose spectra differ (low-frequency heavy).
  Our model has never seen a large-ship spectrum; if test contains them, we
  systematically misestimate those windows. Competitors already use VTUAD —
  this is table stakes we are missing, not an exotic bet.
- Data needed: VTUAD Zenodo release (CC BY 4.0, already declared by others).
- Effort: 4–8 hours (selective download, band conversion, synthetic rebuild,
  retrain). Feasible for Nov 7 slots, NOT for tomorrow (Nov 6).
- Main risk: VTUAD vessels are from the Strait of Georgia (different water,
  different hydrophone) — same cross-water caveat as HearMyShip. Also the
  25 GB download is slow. MITIGATION: use as a blend component, not a
  replacement; validate on held-out sites.

### #3. TWO-STAGE BACKGROUND ESTIMATION + SPECTRAL UNMIXING (physics estimator)
- What: (a) detect passage minutes from broadband excess, mask them, and
  re-estimate background ONLY from clean minutes (unbiased vs the noisy 10th
  percentile); (b) NNLS spectral unmixing per window (background template from
  quiet minutes + vessel template from HearMyShip/VTUAD mean spectra) → direct
  share estimate. Use the unmixed share as a new feature AND as a standalone
  blend component.
- WHY: the whole problem is background estimation (§1b: the 10th pct is so
  noisy that 7/15 minutes exceed it by >30% with zero vessel). A cleaner
  background estimate improves EVERY downstream feature, and unmixing adds a
  genuinely independent estimator (different failure mode from LightGBM) —
  ideal blend diversity.
- Data needed: none new (templates from existing HearMyShip spectra + window
  itself). Fixed per-window transform → rule-compliant.
- Effort: 3–5 hours (implement, validate on synthetic val where truth is known,
  blend). Feasible for Nov 7–8.
- Main risk: the passage-minute detector can misfire (masking true background
  minutes, or missing passage minutes), which biases the re-estimate. MITIGATION:
  soft masking (weights, not hard masks); validate the detector on synthetic
  val where passage minutes are known.

### #4. BACKGROUND-MODEL FEATURES FROM THE 1.7M UNLABELLED RECORDS
- What: compute PCA / spectral templates of SA background spectra from the
  released acoustics (92 MB parquet, 10 sites); derive per-window features:
  distance of the window's spectrum from the background manifold, background
  template reconstruction residual, per-band variance profiles. Add as features.
- WHY: we currently use the unlabelled data only as passive carriers. A learned
  background model gives the model a prior for "what does pure background look
  like" — the exact thing it lacks when hallucinating vessels at loud sites.
  This is legitimate (released training data) and complementary to #1.
- Data needed: existing SA acoustics (local).
- Effort: 3–6 hours.
- Main risk: test backgrounds are from EXCLUDED sites, so site-specific
  templates may not transfer; only the GENERAL background structure transfers.
  MITIGATION: use global (all-site) templates, not per-site; validate on
  held-out sites (Fish Hoek/Sea Point already simulate this).

### #5. FULL-MATRIX MODEL (LightGBM on 375 features, or small 1D-CNN)
- What: feed the model the full 15×25 excess matrix (or raw dB + mask) instead
  of 30 summaries — either LightGBM on 375 features or a small 1D-CNN over
  minutes (15 steps × 25 channels). Captures temporal passage shape and
  spectral detail that the summaries discard.
- WHY: the 2-min passage problem (0.070 vs 0.045) and the collapse of temporal
  info suggest shape matters. A full-matrix model can learn passage-shape
  cues (smooth hump vs spike) that no aggregate captures.
- Data needed: none new (same synthetic arrays, different featurization).
- Effort: 4–6 hours.
- Main risk: HIGHEST overfit risk of all directions — more capacity on synthetic
  data whose distribution differs from test, and validation does not predict
  public. MITIGATION: use ONLY as a small-weight blend component (≤0.3);
  never as a standalone submission until it proves itself on public.

## 4. #1 RECOMMENDATION FOR TOMORROW'S (NOV 6) 2 SLOTS

Build Direction #1 today (2–4 h, no new data, same pipeline):
- SLOT 1: 50/50 blend of [new absolute-feature model] + [current best 0.7952
  blend]. The safe slot: blends have never hurt us (both blends beat their
  parents), and this hedges the calibration-offset risk.
- SLOT 2: the new absolute-feature model ALONE. The upside slot: if the level
  fix works as the mechanism predicts, this jumps well above 0.7952; if it
  fails the validation gate (§3 #1), replace slot 2 with the ro0.7 blend
  candidate already built (nov5_candidate_ro0.7.csv).
- DECISION RULE before submitting: retrained model must cut Fish Hoek val
  error materially (0.064 → ≤0.045) without worsening Sea Point. If it passes,
  submit both slots as above. If it fails, submit slot 1 = current-best blend
  re-weightings and keep #1 for further work — do NOT burn a slot on a model
  that fails its own validation gate.
- In parallel (does not block tomorrow): start the VTUAD selective download
  (Direction #2) so it is ready for Nov 7.

## 5. REJECTED IDEAS (do not revisit without new evidence)

- MORE SEEDS / SAME-FAMILY AVERAGING: proven saturated (20-seed 0.7947 < 10-seed
  0.7948 on public). Dead.
- ISOTONIC/AFFINE CALIBRATION FIT ON VAL: proven to overfit synthetic (v2 won
  val 0.9241, lost on public 0.7881). Dead.
- XGBOOST / CATBOOST AS A DROP-IN: same 30 features → same failure mode. Might
  add tiny blend diversity but cannot fix the level problem. Not a direction.
- HYPERPARAMETER TUNING OF LIGHTGBM: the model is feature-limited, not
  capacity-limited; errors are systematic (level bias), not variance. Low value.
- DANN/CDAN ADVERSARIAL ADAPTATION: unstable (CDAN collapse in the 4-corpus
  benchmark); needs a target domain we cannot use (test off-limits). Dead.
- MMD ALIGNMENT: negative-transfer risk in benchmark; same target-domain problem.
- CORAL ON FEATURES: synthetic features already come from real SA backgrounds;
  the gap is vessel type + level, not feature covariance. Low expected value.
- RAW-AUDIO NEURAL NETS (CNN on waveforms/spectrograms, CAPSE-ViT, SSAMBA):
  test provides NO audio — only 15×25 band levels. Cannot be applied to test.
  Dead on data-availability grounds (feature IDEAS transfer; models do not).
- PSEUDO-LABELING / FITTING / CALIBRATING ON TEST: explicitly banned by rules.
- MORE HEARMSHIP CLIPS: pool exhausted (all 322 used). Marginal value zero.
- ONC RAW HYDROPHONE DOWNLOAD (token + API): VTUAD already packages the same
  ONC vessels pre-labeled; prefer VTUAD (Direction #2).
- SIMPLE FEATURE-COUNT INCREASES WITHOUT A MECHANISM (e.g., "add 50 more
  generic stats"): the failure is specific (level bias + background noise),
  not a general feature shortage. Every new feature must target a validated
  failure mode.

## Sources
- Shipton et al. 2025, HearMyShip database, Sci. Data 12:289 —
  https://www.nature.com/articles/s41597-025-04584-x
- Shipton et al. 2025, URN characteristics of small vessels, Mar. Pollut. Bull.
  216:117903 — https://pubmed.ncbi.nlm.nih.gov/40209438/
- Shipton & Diamant 2026, log-linear URN spectra models, Front. Mar. Sci. —
  https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2026.1759355/full
- Soti et al., Towards Deployable Underwater Vessel Classification (ShipsEar/
  DeepShip, compact CNN, recording-level splits) — https://arxiv.org/pdf/2609.29179
- Najamuddin et al. 2025, CAPSE-ViT — http://journals.pan.pl/Content/135220/aoa.2025.153662.pdf
- UATR-GTransformer (Mel-graph) — https://arxiv.org/html/2512.11545v1/
- Cross-water UDA benchmark (CORAL/MMD/DANN/CDAN, 344 runs) —
  https://www.preprints.org/manuscript/202609.1124/v3
- CORAL synthetic→real acoustic regression (ARS) —
  https://www.frontiersin.org/journals/acoustics/articles/10.3389/facou.2026.1867954/full
- AquaSignal (autoencoder novelty detection on ONC/DeepShip) —
  https://arxiv.org/html/2505.14285v1
- VTUAD (rebuilt) v1.3.3, Zenodo, CC BY 4.0 — http://zenodo.org/records/23773305
- VTUAD original (cargo/tanker/tug/passenger F1 > 0.94) —
  https://www.mdpi.com/2077-1312/13/8/1455
- ONC hydrophone tools (Oceans 3.0 API, CC BY 4.0) —
  https://pypi.org/project/onc-hydrophone-data/
- NOAA SanctSound vessel-event detection (passage duration thresholds) —
  https://coastwatch.pfeg.noaa.gov/erddap/griddap/noaaSanctSound_MB01_05_ships.html
- Zindi competition page + discussion (VTUAD/HearMyShip declarations) —
  https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge
