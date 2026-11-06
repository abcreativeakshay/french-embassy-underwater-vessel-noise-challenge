"""Direction #3 diagnostics: two-stage background + spectral unmixing.

Fast, no LightGBM: evaluates on the hms_clip synthetic val (378 rows,
Fish Hoek + Sea Point, disjoint clips) where TRUE background (bg_dB) and
true share are known, so the background estimator and the unmixing estimator
can be scored directly against truth.

Compares:
  * background estimators via the analytic-share RMSE they induce
  * passage-minute detector precision/recall vs true passage minutes
    (true vessel energy per minute = mixed_E - bg_E)
  * NNLS spectral-unmixing share (per-window total spectrum, and per-minute
    closed-form 2-template) vs true share, standalone RMSE
"""
import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import nnls

from src.common import N_MIN, N_BAND, db_to_energy

ROOT = '/home/hatch/workspace/zindi-vessel-noise-challenge'
EXT = ROOT + '/data/external'


def load_hms_val():
    t = pq.read_table(ROOT + '/data/processed/synthetic_val.parquet',
                      columns=['vessel_type', 'site', 'mixed_dB', 'bg_dB',
                               'share', 'passage_minutes'])
    df = t.to_pandas()
    df = df[df['vessel_type'] == 'hms_clip'].reset_index(drop=True)
    n = len(df)
    A = np.stack(df['mixed_dB'].to_numpy()).reshape(n, N_MIN, N_BAND)
    B = np.stack(df['bg_dB'].to_numpy()).reshape(n, N_MIN, N_BAND)
    y = df['share'].to_numpy(dtype=np.float64)
    site = df['site'].to_numpy()
    return A, B, y, site, df['passage_minutes'].to_numpy()


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def analytic_from_bg(E, bg):
    """share = sum max(E-bg,0)/sum E, bg (n,25) energy."""
    bg_full = bg[:, None, :]
    excess = np.clip(E - bg_full, 0.0, None)
    tot = E.sum(axis=(1, 2))
    return np.clip(excess.sum(axis=(1, 2)) / np.maximum(tot, 1e-12), 0, 1)


def detect_passage_minutes(E, thresh=1.3):
    """Broadband-energy detector: minute flagged if its total energy exceeds
    thresh x the window median minute energy. Returns bool (n,15)."""
    minute_E = E.sum(axis=2)                       # (n,15)
    med = np.median(minute_E, axis=1, keepdims=True)
    return minute_E > thresh * med


def twostage_bg(E, mask_pass, agg='median'):
    """Background per band from CLEAN minutes only.
    agg: 'median' | 'p25' | 'p10'. Falls back to all-minute p10 when too few
    clean minutes (<3)."""
    n = E.shape[0]
    bg = np.zeros((n, N_BAND))
    clean = ~mask_pass
    for i in range(n):
        Ec = E[i][clean[i]]
        if len(Ec) < 3:
            bg[i] = np.percentile(E[i], 10, axis=0)
        elif agg == 'median':
            bg[i] = np.median(Ec, axis=0)
        elif agg == 'p25':
            bg[i] = np.percentile(Ec, 25, axis=0)
        elif agg == 'p10':
            bg[i] = np.percentile(Ec, 10, axis=0)
    return bg


def main():
    A, B, y, site, kpass = load_hms_val()
    E = db_to_energy(A)
    Eb = db_to_energy(B)
    n = len(y)
    print(f'val hms_clip n={n}  FishHoek={(site=="09_Fish_Hoek").sum()} '
          f'SeaPoint={(site=="10_Sea_Point").sum()}')
    # --- ground-truth passage minutes: vessel energy per minute ---
    V_E = np.clip(E - Eb, 0.0, None)
    v_min = V_E.sum(axis=2)                        # (n,15) vessel energy
    bg_min = Eb.sum(axis=2)
    true_pass = v_min > 0.02 * bg_min              # vessel >2% of bg energy
    print(f'true passage minutes/row: mean={true_pass.sum(1).mean():.2f} '
          f'(k from file: mean={kpass.mean():.2f})')

    # --- detector sweep ---
    print('\n-- passage-minute detector (broadband ratio vs median) --')
    for th in [1.15, 1.3, 1.5, 2.0]:
        det = detect_passage_minutes(E, th)
        tp = (det & true_pass).sum()
        fp = (det & ~true_pass).sum()
        fn = (~det & true_pass).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        print(f'  thresh={th}: flagged/row={det.sum(1).mean():.2f} '
              f'prec={prec:.3f} rec={rec:.3f} F1={f1:.3f}')

    # --- analytic share with different backgrounds ---
    print('\n-- analytic share RMSE by background estimator --')
    bg_p10 = np.percentile(E, 10, axis=1)
    bg_true_med = np.median(Eb, axis=1)            # oracle-ish: true bg median
    res = {}
    res['p10 (current)'] = analytic_from_bg(E, bg_p10)
    res['oracle true-bg median'] = analytic_from_bg(E, bg_true_med)
    for th in [1.3, 1.5]:
        det = detect_passage_minutes(E, th)
        for agg in ['median', 'p25', 'p10']:
            bg2 = twostage_bg(E, det, agg)
            res[f'two-stage th={th} {agg}'] = analytic_from_bg(E, bg2)
    for name, s in res.items():
        fh = site == '09_Fish_Hoek'
        print(f'  {name:28s} 1-RMSE={1-rmse(s,y):.4f} rmse={rmse(s,y):.4f} '
              f'FH={1-rmse(s[fh],y[fh]):.4f} SP={1-rmse(s[~fh],y[~fh]):.4f} '
              f'bias={np.mean(s-y):+.4f}')

    # --- background level error (per-band, vs true bg median) ---
    print('\n-- background level error: mean |log10(bg_est/bg_true)| per band --')
    for name, bg in [('p10', bg_p10), ('oracle', bg_true_med)]:
        err = np.mean(np.abs(np.log10(np.maximum(bg, 1e-12) /
                                      np.maximum(bg_true_med, 1e-12))))
        print(f'  {name}: {err:.4f}')
    det = detect_passage_minutes(E, 1.3)
    for agg in ['median', 'p25']:
        bg2 = twostage_bg(E, det, agg)
        err = np.mean(np.abs(np.log10(np.maximum(bg2, 1e-12) /
                                      np.maximum(bg_true_med, 1e-12))))
        print(f'  two-stage th=1.3 {agg}: {err:.4f}')

    # --- spectral unmixing ---
    print('\n-- spectral unmixing (NNLS on total window spectrum) --')
    t = np.load(EXT + '/vessel_cpa_templates.npz')
    cats = t['cats'].astype(str)
    spec = t['spectra'].astype(np.float64)         # (1100,25) normalised
    tpl_mean = spec.mean(axis=0)
    tpl_mean = tpl_mean / tpl_mean.sum()
    tpl_cats = {}
    for c in np.unique(cats):
        m = spec[cats == c].mean(axis=0)
        tpl_cats[c] = m / m.sum()
    # HearMyShip clip mean spectrum
    s = np.load(EXT + '/vessel_signatures.npz')
    clip_spec = s['sig'].sum(axis=1)               # (322,25) per-clip totals
    clip_spec = clip_spec / clip_spec.sum(axis=1, keepdims=True)
    clip_mean = clip_spec.mean(axis=0)
    clip_mean = clip_mean / clip_mean.sum()
    print('  low-band (31-250Hz) frac: CPA mean %.3f  clip mean %.3f  '
          'per-cat %s' % (tpl_mean[:10].sum(), clip_mean[:10].sum(),
                          {c: round(float(v[:10].sum()), 3)
                           for c, v in tpl_cats.items()}))

    S = E.sum(axis=1)                              # (n,25) total spectrum
    bg2 = twostage_bg(E, det, 'median')
    bg_tpl = bg2 / bg2.sum(axis=1, keepdims=True)

    def nnls_share(bg_tpl, vessel_tpls):
        """Per-window NNLS: S ~ a*bg + sum b_j*V_j; share = sum b/(a+sum b)."""
        out = np.zeros(len(S))
        for i in range(len(S)):
            M = np.column_stack([bg_tpl[i]] + vessel_tpls)
            coef, _ = nnls(M, S[i])
            tot = coef.sum()
            out[i] = coef[1:].sum() / tot if tot > 0 else 0.0
        return np.clip(out, 0, 1)

    variants = {
        'bg + CPA-mean': [tpl_mean],
        'bg + clip-mean': [clip_mean],
        'bg + CPA-mean + clip-mean': [tpl_mean, clip_mean],
        'bg + 5 CPA cats': [tpl_cats[c] for c in sorted(tpl_cats)],
        'bg + 5 CPA cats + clip-mean':
            [tpl_cats[c] for c in sorted(tpl_cats)] + [clip_mean],
    }
    for name, vtpls in variants.items():
        sh = nnls_share(bg_tpl, vtpls)
        fh = site == '09_Fish_Hoek'
        print(f'  {name:32s} 1-RMSE={1-rmse(sh,y):.4f} rmse={rmse(sh,y):.4f} '
              f'FH={1-rmse(sh[fh],y[fh]):.4f} SP={1-rmse(sh[~fh],y[~fh]):.4f} '
              f'bias={np.mean(sh-y):+.4f} corr={np.corrcoef(sh,y)[0,1]:.3f}')

    # sensitivity: same unmixing but with p10 background template
    bg_p10_tpl = bg_p10 / bg_p10.sum(axis=1, keepdims=True)
    sh = nnls_share(bg_p10_tpl, [tpl_mean, clip_mean])
    fh = site == '09_Fish_Hoek'
    print(f'  {"bg(p10) + CPA-mean + clip-mean":32s} 1-RMSE={1-rmse(sh,y):.4f} '
          f'FH={1-rmse(sh[fh],y[fh]):.4f} SP={1-rmse(sh[~fh],y[~fh]):.4f} '
          f'bias={np.mean(sh-y):+.4f} corr={np.corrcoef(sh,y)[0,1]:.3f}')

    # VTUAD large-vessel templates as a DIAGNOSTIC only (would need declaration)
    try:
        vz = np.load(EXT + '/vtuad_signatures.npz')
        print('  vtuad sig keys:', list(vz.keys()))
        for k in vz.keys():
            print('   ', k, vz[k].shape, vz[k].dtype)
    except Exception as e:
        print('  vtuad signatures unavailable:', e)

    # --- per-minute closed-form 2-template unmixing ---
    print('\n-- per-minute 2-template unmixing (closed form) --')
    def per_minute_unmix(bg_tpl, v_tpl):
        """Fit each minute: x ~ a*bg + c*v (unit-sum templates).
        share = sum c / sum(a+c). Vectorised 2x2 NNLS."""
        b = bg_tpl[:, None, :]                     # (n,1,25)
        v = v_tpl[None, None, :]
        Gbb = (bg_tpl ** 2).sum(axis=1)            # (n,) ||bg||^2
        Gvv = float(v_tpl @ v_tpl)
        Gbv = (bg_tpl @ v_tpl)                    # (n,)
        xb = (E * b).sum(axis=2)                   # (n,15)
        xv = (E * v).sum(axis=2)
        det2 = Gbb[:, None] * Gvv - Gbv[:, None] ** 2
        det2 = np.where(np.abs(det2) < 1e-12, np.nan, det2)
        a = (Gvv * xb - Gbv[:, None] * xv) / det2
        c = (Gbb[:, None] * xv - Gbv[:, None] * xb) / det2
        # NNLS boundary cases: refit with the single positive template
        neg_a = a < 0
        neg_c = c < 0
        a2 = np.where(neg_a, 0.0, a)
        c2 = np.where(neg_c, 0.0, c)
        a2 = np.where(neg_a, xb / np.maximum(Gbb[:, None], 1e-12), a2)
        c2 = np.where(neg_c, xv / max(Gvv, 1e-12), c2)
        a2 = np.where(neg_a & neg_c, 0.0, a2)
        c2 = np.where(neg_a & neg_c, 0.0, c2)
        tot_c = c2.sum(axis=1)
        tot = (a2 + c2).sum(axis=1)
        return np.clip(np.divide(tot_c, tot, out=np.zeros_like(tot_c),
                                 where=tot > 0), 0, 1)
    for name, v_tpl in [('clip-mean', clip_mean), ('CPA-mean', tpl_mean)]:
        sh = per_minute_unmix(bg_tpl, v_tpl)
        fh = site == '09_Fish_Hoek'
        print(f'  per-minute {name:10s} 1-RMSE={1-rmse(sh,y):.4f} '
              f'FH={1-rmse(sh[fh],y[fh]):.4f} SP={1-rmse(sh[~fh],y[~fh]):.4f} '
              f'bias={np.mean(sh-y):+.4f} corr={np.corrcoef(sh,y)[0,1]:.3f}')


if __name__ == '__main__':
    main()
