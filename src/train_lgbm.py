"""Stage C model training (PLAN.md).

Features are computed per window in the ENERGY domain from the *observed*
mixed dB only (the model never sees the background or vessel separately):
  1. analytic_share  - Stage B heuristic (10th-pct background estimate)
  2. per-band excess fractions (25): sum_m max(E-bg_b,0) / sum_m E per band
  3. broadband transient fraction (1): (peak_min_E - median_min_E)/total_E
  4. frac_cells_above_bg_3dB (1)
  5. peak_to_median_minute_ratio (1)
  6. excess spectral centroid (1): band index of excess energy centre

Model: LightGBM regressor on synthetic train (8 sites), validated on 2
held-out sites (Fish Hoek, Sea Point). Metric: RMSE on share (score=1-RMSE).

Usage:
    python -m src.train_lgbm [--rounds 500]
Writes models/lgbm_synth.txt and submissions/synth_lgbm.csv
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb

from src.common import (N_MIN, N_BAND, SEED, db_to_energy, load_test,
                        analytic_share, detection_score, check_submission)

ROOT = Path(__file__).resolve().parent.parent
FEATURE_NAMES = (['analytic_share'] +
                 [f'excess_frac_b{b:02d}' for b in range(N_BAND)] +
                 ['transient_frac', 'cells_above_3dB',
                  'peak_median_ratio', 'excess_centroid'])


def featurize(A, mask):
    """A: (n,15,25) dB, mask: (n,15,25) bool -> (n,30) features."""
    n = A.shape[0]
    E = np.where(mask, db_to_energy(A), 0.0)
    # background: 10th pct over minutes per band (energy domain, NaN-aware)
    En = np.where(mask, db_to_energy(A), np.nan)
    bg = np.nanpercentile(En, 10, axis=1)                      # (n,25)
    bg_full = np.where(mask, np.nan_to_num(bg[:, None, :]), 0.0)

    analytic, _ = analytic_share(A, mask)
    excess = np.clip(E - bg_full, 0.0, None)
    band_E = E.sum(axis=1)                                     # (n,25)
    band_excess = excess.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        excess_frac = np.divide(band_excess, band_E,
                                out=np.zeros_like(band_excess),
                                where=band_E > 0)              # (n,25)

    minute_E = E.sum(axis=2)                                   # (n,15)
    minute_valid = mask.any(axis=2)
    tot = E.sum(axis=(1, 2))
    peak = np.array([m[v].max() if v.any() else 0.0
                     for m, v in zip(minute_E, minute_valid)])
    med = np.array([np.median(m[v]) if v.any() else 0.0
                    for m, v in zip(minute_E, minute_valid)])
    transient_frac = np.divide(np.clip(peak - med, 0, None), tot,
                               out=np.zeros_like(tot), where=tot > 0)
    peak_median_ratio = np.divide(peak, med, out=np.ones_like(peak),
                                  where=med > 0)

    with np.errstate(invalid='ignore'):
        above = (A > (10 * np.log10(np.maximum(bg_full, 1e-12)) + 3.0)) & mask
    cells_above = above.sum(axis=(1, 2)) / np.clip(mask.sum(axis=(1, 2)), 1, None)

    band_idx = np.arange(N_BAND, dtype=np.float64)
    exc_sum = band_excess.sum(axis=1)
    centroid = np.divide((band_excess * band_idx).sum(axis=1), exc_sum,
                         out=np.full(n, N_BAND / 2), where=exc_sum > 0) / N_BAND

    X = np.column_stack([analytic, excess_frac, transient_frac,
                         cells_above, peak_median_ratio, centroid])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def load_synthetic(name, suffix=''):
    t = pq.read_table(ROOT / 'data' / 'processed' / f'synthetic_{name}{suffix}.parquet')
    df = t.to_pandas()
    n = len(df)
    A = np.stack(df['mixed_dB'].to_numpy()).reshape(n, N_MIN, N_BAND)
    y = df['share'].to_numpy(dtype=np.float64)
    mask = np.ones_like(A, dtype=bool)   # synthetic windows are complete
    return A, mask, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=600)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--out-name', type=str, default='synth_lgbm.csv',
                    help='submission filename in submissions/')
    ap.add_argument('--model-name', type=str, default='lgbm_synth.txt',
                    help='model filename in models/')
    args = ap.parse_args()

    print('loading synthetic data...', flush=True)
    A_tr, m_tr, y_tr = load_synthetic('train')
    A_va, m_va, y_va = load_synthetic('val')
    print(f'train {A_tr.shape}  val {A_va.shape}', flush=True)

    print('featurizing...', flush=True)
    X_tr, X_va = featurize(A_tr, m_tr), featurize(A_va, m_va)

    dtr = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_NAMES)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr, feature_name=FEATURE_NAMES)
    params = dict(objective='regression', metric='rmse', learning_rate=0.05,
                  num_leaves=31, min_data_in_leaf=40, feature_fraction=0.9,
                  bagging_fraction=0.9, bagging_freq=1, seed=args.seed,
                  verbose=-1)
    print('training LightGBM...', flush=True)
    model = lgb.train(params, dtr, num_boost_round=args.rounds,
                      valid_sets=[dtr, dva], valid_names=['train', 'val'])

    pred_va = np.clip(model.predict(X_va), 0, 1)
    rmse = float(np.sqrt(np.mean((pred_va - y_va) ** 2)))
    print(f'VAL RMSE={rmse:.4f}  (1-RMSE={1 - rmse:.4f})')
    # error by share band
    for lo, hi in [(0, 0.001), (0.001, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0)]:
        m = (y_va >= lo) & (y_va < hi)
        if m.sum():
            r = np.sqrt(np.mean((pred_va[m] - y_va[m]) ** 2))
            print(f'  share[{lo:4.2f},{hi:4.2f}): n={m.sum():4d} rmse={r:.4f} '
                  f'bias={np.mean(pred_va[m] - y_va[m]):+.4f}')

    imp = pd.Series(model.feature_importance(importance_type='gain'),
                    index=FEATURE_NAMES).sort_values(ascending=False)
    print('top features:\n', imp.head(10).to_string())

    models_dir = ROOT / 'models'
    models_dir.mkdir(exist_ok=True)
    model.save_model(models_dir / args.model_name)

    # predict the real test set
    ids, A_te, m_te = load_test(ROOT / 'data' / 'Test.csv')
    X_te = featurize(A_te, m_te)
    share = np.clip(model.predict(X_te), 0, 1)
    score = detection_score(A_te, m_te)
    sub = pd.DataFrame({'ID': ids, 'share_added_vessel': share,
                        'score_added_vessel': score})
    check_submission(sub, len(ids))
    out = ROOT / 'submissions' / args.out_name
    sub.to_csv(out, index=False)
    print(f'wrote {out}')
    print(f'test share: mean={share.mean():.4f} p50={np.median(share):.4f} '
          f'max={share.max():.4f}')


if __name__ == '__main__':
    main()
