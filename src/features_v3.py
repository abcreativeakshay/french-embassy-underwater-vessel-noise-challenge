"""Direction #3 features: two-stage background + spectral unmixing.

All features are SCALE-INVARIANT (no absolute dB levels — the v2 lesson:
absolute-level features failed publicly). The two-stage background only
changes *how* the background is estimated, and unmixing adds an independent
physics estimator as features.

Two-stage background (fixed per-window transform, rule-compliant):
  1. detect passage minutes: minute broadband energy > 1.5x window median
  2. background per band = median over CLEAN minutes only
     (p10 over 15 samples is between the 1st/2nd lowest value — very noisy;
      median-of-clean cuts bg level error 0.1625 -> 0.0493 in log10, measured
      on synthetic val where true bg is known)

Spectral unmixing (fixed per-window transform):
  total window spectrum S (25,) in energy ~ a*bg_template + b1*CPA_mean
  + b2*clip_mean, solved by NNLS (scipy). Vessel templates are
  HearMyShip-derived (already declared): mean CPA spectrum + mean clip
  spectrum. share_unmix = (b1+b2)/(a+b1+b2).
  Per-minute closed-form 2-template unmixing gives a passage-shape profile.

Feature list (68):
  0-29   original 30 features (p10 background) — unchanged, for continuity
  30-57  two-stage versions: analytic_share2, 25x excess_frac2,
         cells_above_3dB2, excess_centroid2
  58-67  unmixing/detector features:
         unmix_share (bg2+CPA+clip), unmix_share_clip, unmix_share_cpa,
         unmix_share_perminute (clip template), unmix_resid (relative fit
         error), unmix_excess_share (excess-spectrum fit),
         pm_max (max minute vessel frac), pm_frac_gt01, pm_std,
         n_clean_frac
"""
import numpy as np
from scipy.optimize import nnls

from src.common import N_MIN, N_BAND, db_to_energy, analytic_share
from src.train_lgbm import featurize as featurize30

ROOT = '/home/hatch/workspace/zindi-vessel-noise-challenge'
EXT = ROOT + '/data/external'

_TPL = None


def _templates():
    """HearMyShip vessel templates (unit-sum spectra). Loaded once."""
    global _TPL
    if _TPL is not None:
        return _TPL
    t = np.load(EXT + '/vessel_cpa_templates.npz')
    spec = t['spectra'].astype(np.float64)
    cpa = spec.mean(axis=0)
    cpa = cpa / cpa.sum()
    s = np.load(EXT + '/vessel_signatures.npz')
    clip = s['sig'].sum(axis=1).astype(np.float64)
    clip = clip / clip.sum(axis=1, keepdims=True)
    clipm = clip.mean(axis=0)
    clipm = clipm / clipm.sum()
    _TPL = (cpa, clipm)
    return _TPL


def detect_passage_minutes(E, thresh=1.5):
    minute_E = E.sum(axis=2)
    med = np.median(minute_E, axis=1, keepdims=True)
    return minute_E > thresh * med


def twostage_bg(E, mask_pass):
    """Per-band background = median over clean minutes; fall back to p10
    over all minutes when <3 clean minutes."""
    n = E.shape[0]
    bg = np.zeros((n, N_BAND))
    clean = ~mask_pass
    for i in range(n):
        Ec = E[i][clean[i]]
        if len(Ec) < 3:
            bg[i] = np.percentile(E[i], 10, axis=0)
        else:
            bg[i] = np.median(Ec, axis=0)
    return bg


def _featurize30_with_bg(A, mask, bg):
    """The 30-feature set with a supplied background (n,25) energy."""
    n = A.shape[0]
    E = np.where(mask, db_to_energy(A), 0.0)
    bg_full = np.where(mask, np.nan_to_num(bg[:, None, :]), 0.0)
    excess = np.clip(E - bg_full, 0.0, None)
    band_E = E.sum(axis=1)
    band_excess = excess.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        excess_frac = np.divide(band_excess, band_E,
                                out=np.zeros_like(band_excess),
                                where=band_E > 0)
    tot = E.sum(axis=(1, 2))
    exc = excess.sum(axis=(1, 2))
    analytic = np.clip(np.divide(exc, tot, out=np.zeros_like(exc),
                                 where=tot > 0), 0, 1)
    with np.errstate(invalid='ignore'):
        above = (A > (10 * np.log10(np.maximum(bg_full, 1e-12)) + 3.0)) & mask
    cells_above = above.sum(axis=(1, 2)) / np.clip(mask.sum(axis=(1, 2)), 1, None)
    band_idx = np.arange(N_BAND, dtype=np.float64)
    exc_sum = band_excess.sum(axis=1)
    centroid = np.divide((band_excess * band_idx).sum(axis=1), exc_sum,
                         out=np.full(n, N_BAND / 2), where=exc_sum > 0) / N_BAND
    return analytic, excess_frac, cells_above, centroid


def _nnls_unmix(S, bg_tpl, v_tpls):
    """Per-window NNLS: S ~ a*bg + sum b_j*V_j. Returns share and residual."""
    n = S.shape[0]
    share = np.zeros(n)
    resid = np.zeros(n)
    for i in range(n):
        M = np.column_stack([bg_tpl[i]] + v_tpls)
        coef, _ = nnls(M, S[i])
        tot = coef.sum()
        share[i] = coef[1:].sum() / tot if tot > 0 else 0.0
        fit = M @ coef
        resid[i] = np.linalg.norm(S[i] - fit) / max(np.linalg.norm(S[i]), 1e-12)
    return np.clip(share, 0, 1), resid


def _perminute_unmix(E, bg_tpl, v_tpl):
    """Closed-form 2-template NNLS per minute. Returns (n,15) vessel fracs."""
    b = bg_tpl[:, None, :]
    v = v_tpl[None, None, :]
    Gbb = (bg_tpl ** 2).sum(axis=1)
    Gvv = float(v_tpl @ v_tpl)
    Gbv = (bg_tpl @ v_tpl)
    xb = (E * b).sum(axis=2)
    xv = (E * v).sum(axis=2)
    det2 = Gbb[:, None] * Gvv - Gbv[:, None] ** 2
    det2 = np.where(np.abs(det2) < 1e-12, np.nan, det2)
    a = (Gvv * xb - Gbv[:, None] * xv) / det2
    c = (Gbb[:, None] * xv - Gbv[:, None] * xb) / det2
    neg_a = a < 0
    neg_c = c < 0
    a2 = np.where(neg_a, 0.0, a)
    c2 = np.where(neg_c, 0.0, c)
    a2 = np.where(neg_a, xb / np.maximum(Gbb[:, None], 1e-12), a2)
    c2 = np.where(neg_c, xv / max(Gvv, 1e-12), c2)
    a2 = np.where(neg_a & neg_c, 0.0, a2)
    c2 = np.where(neg_a & neg_c, 0.0, c2)
    denom = a2 + c2
    return np.divide(c2, denom, out=np.zeros_like(c2), where=denom > 0)


def featurize_v3(A, mask):
    """A: (n,15,25) dB, mask bool -> (n,68) features."""
    cpa, clipm = _templates()
    n = A.shape[0]
    E = np.where(mask, db_to_energy(A), 0.0)
    En = np.where(mask, db_to_energy(A), np.nan)

    X30 = featurize30(A, mask)

    # --- two-stage background ---
    det = detect_passage_minutes(E, 1.5)
    bg2 = twostage_bg(E, det)
    a2, ef2, ca2, ce2 = _featurize30_with_bg(A, mask, bg2)

    # --- spectral unmixing on total spectrum ---
    S = E.sum(axis=1)                                  # (n,25)
    bg_tpl = bg2 / bg2.sum(axis=1, keepdims=True)
    unmix, resid = _nnls_unmix(S, bg_tpl, [cpa, clipm])
    unmix_clip, _ = _nnls_unmix(S, bg_tpl, [clipm])
    unmix_cpa, _ = _nnls_unmix(S, bg_tpl, [cpa])
    # excess-spectrum fit: S - 15*bg2 against vessel templates only;
    # share = fitted vessel energy / total spectrum energy
    excess_spec = np.clip(S - N_MIN * bg2, 0.0, None)
    ves_e = np.zeros(n)
    for i in range(n):
        M = np.column_stack([cpa, clipm])
        coef, _ = nnls(M, excess_spec[i])
        ves_e[i] = coef.sum()
    tot_S = S.sum(axis=1)
    unmix_excess = np.clip(np.divide(ves_e, tot_S, out=np.zeros_like(ves_e),
                                     where=tot_S > 0), 0, 1)

    # --- per-minute unmixing profile (clip template) ---
    pm = _perminute_unmix(E, bg_tpl, clipm)            # (n,15) vessel fracs
    pm_share = np.clip(pm.mean(axis=1), 0, 1)
    pm_max = pm.max(axis=1)
    pm_std = pm.std(axis=1)
    pm_frac_gt01 = (pm > 0.1).mean(axis=1)
    n_clean_frac = (~det).mean(axis=1)

    Xnew = np.column_stack([
        a2, ef2, ca2, ce2,                            # 30-57 (1+25+1+1)
        unmix, unmix_clip, unmix_cpa, pm_share, resid,
        unmix_excess, pm_max, pm_frac_gt01, pm_std, n_clean_frac,
    ])
    X = np.column_stack([X30, Xnew])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


FEATURE_NAMES_V3 = (
    ['f30_' + f for f in
     (['analytic_share'] + [f'excess_frac_b{b:02d}' for b in range(N_BAND)] +
      ['transient_frac', 'cells_above_3dB', 'peak_median_ratio',
       'excess_centroid'])] +
    ['ts_analytic_share'] + [f'ts_excess_frac_b{b:02d}' for b in range(N_BAND)] +
    ['ts_cells_above_3dB', 'ts_excess_centroid'] +
    ['unmix_share', 'unmix_share_clip', 'unmix_share_cpa',
     'unmix_share_perminute', 'unmix_resid', 'unmix_excess_share',
     'pm_max', 'pm_frac_gt01', 'pm_std', 'n_clean_frac']
)
