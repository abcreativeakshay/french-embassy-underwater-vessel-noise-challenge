"""Direction #1 (Nov 6 slots): absolute-level + background-variance features
with dB-offset augmentation. Targets the validated failure: test is ~16 dB
louder than training, and all 30 current features are scale-invariant, so the
model cannot condition on level and hallucinates vessels at loud sites.

New features (on top of the existing 30 from train_lgbm.featurize):
  - bg_dB (25): per-band background level in dB (absolute, not relative)
  - tot_E_log (1): log10 total window energy
  - bg_std_dB (25): per-band background std across minutes (dB domain)
  - low_high_ratio (1): low-band (31-250 Hz) excess / high-band excess
  - minute excess profile summaries (3): max minute excess frac, std of
    minute excess fracs, argmax minute / 15
  - nan_frac (1): fraction of missing cells

Augmentation (training only; labels preserved exactly):
  - Global dB offset: uniform(-10, +25) dB added to all cells. Share is
    scale-invariant, so labels are unchanged. Manufactures training windows
    at test's level range (57-95 dB) for free.
  - NaN masking: mimic test missingness — earlier minutes more likely
    missing (m01 ~42% missing → m15 ~7% missing), plus random cell drops.

Validation gate (must pass before submitting):
  - Fish Hoek val abs error <= 0.045 (was 0.064) WITHOUT worsening Sea Point.

Usage:
    python -m src.train_v2 [--rounds 600] [--seeds 7 11 13 17 19]
Writes: models/v2_s<seed>.txt, submissions/nov6_v2.csv (standalone),
        submissions/nov6_v2_blend.csv (50/50 with current best blend),
        submissions/nov6_v2_results.txt
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb

from src.common import (N_MIN, N_BAND, db_to_energy, load_test,
                        analytic_share, detection_score, check_submission)
from src.train_lgbm import featurize, FEATURE_NAMES

ROOT = Path(__file__).resolve().parent.parent
LOW_BANDS = list(range(0, 10))   # 31.6 - 250 Hz approx (bands 0-9)
V2_NAMES = ([f'bg_dB_b{b:02d}' for b in range(N_BAND)] +
            ['tot_E_log'] +
            [f'bg_std_dB_b{b:02d}' for b in range(N_BAND)] +
            ['low_high_ratio', 'minute_excess_max', 'minute_excess_std',
             'minute_excess_argmax', 'nan_frac'])
ALL_NAMES = FEATURE_NAMES + V2_NAMES
REAL_VTYPE = 'hms_clip'


def featurize_v2(A, mask):
    """A: (n,15,25) dB, mask: (n,15,25) bool -> (n,86) features."""
    X_old = featurize(A, mask)
    n = A.shape[0]
    A_masked = np.where(mask, A, np.nan)
    # per-band background level (dB) and variability
    with np.errstate(all='ignore'):
        bg_dB = np.nanpercentile(A_masked, 10, axis=1)          # (n,25)
        bg_std = np.nanstd(A_masked, axis=1)                    # (n,25)
    bg_dB = np.nan_to_num(bg_dB, nan=0.0)
    bg_std = np.nan_to_num(bg_std, nan=0.0)
    # total window energy (log)
    E = np.where(mask, db_to_energy(A), 0.0)
    tot = E.sum(axis=(1, 2))
    tot_E_log = np.log10(np.maximum(tot, 1e-12))
    # low vs high band excess ratio
    En = np.where(mask, db_to_energy(A), np.nan)
    bg_E = np.nanpercentile(En, 10, axis=1)                      # (n,25)
    bg_full = np.where(mask, np.nan_to_num(bg_E[:, None, :]), 0.0)
    excess = np.clip(E - bg_full, 0.0, None)
    low_exc = excess[:, :, LOW_BANDS].sum(axis=(1, 2))
    high_exc = excess[:, :, 10:].sum(axis=(1, 2))
    low_high = np.divide(low_exc, high_exc, out=np.zeros(n),
                         where=high_exc > 0)
    # minute excess profile summaries
    minute_exc = excess.sum(axis=2)                              # (n,15)
    minute_tot = E.sum(axis=2)
    with np.errstate(invalid='ignore', divide='ignore'):
        minute_frac = np.divide(minute_exc, minute_tot,
                                out=np.zeros_like(minute_exc),
                                where=minute_tot > 0)
    minute_max = minute_frac.max(axis=1)
    minute_std = minute_frac.std(axis=1)
    minute_argmax = minute_frac.argmax(axis=1) / N_MIN
    nan_frac = (~mask).mean(axis=(1, 2))
    X_new = np.column_stack([bg_dB, tot_E_log, bg_std, low_high,
                             minute_max, minute_std, minute_argmax,
                             nan_frac])
    X_new = np.nan_to_num(X_new, nan=0.0, posinf=0.0, neginf=0.0)
    return np.hstack([X_old, X_new])


def augment(A, mask, rng, n_copies=2):
    """Return augmented copies: dB offsets + NaN masking. Labels unchanged.
    NaN pattern matches test exactly: m01 dropped entirely in 41.6% of rows,
    m15 in 6.7%, plus scattered cell drops (~0.2%)."""
    outs_A, outs_m = [], []
    n = A.shape[0]
    # per-minute drop probs: m01 0.416, m15 0.067, others ~0.002
    miss_p = np.full(N_MIN, 0.002)
    miss_p[0] = 0.416
    miss_p[-1] = 0.067
    for _ in range(n_copies):
        offsets = rng.uniform(-10, 25, size=(n, 1, 1))
        A2 = A + offsets
        m2 = mask.copy()
        drop_min = rng.random((n, N_MIN)) < miss_p[None, :]
        m2 = m2 & (~drop_min[:, :, None])
        drop_cell = rng.random((n, N_MIN, N_BAND)) < 0.002
        m2 = m2 & (~drop_cell)
        outs_A.append(A2)
        outs_m.append(m2)
    return outs_A, outs_m


def load_realonly_site(name):
    """Like train_realonly.load_realonly but keeps site for the gate."""
    t = pq.read_table(ROOT / 'data' / 'processed' / f'synthetic_{name}.parquet',
                      columns=['vessel_type', 'mixed_dB', 'share', 'site'])
    df = t.to_pandas()
    df = df[df['vessel_type'] == REAL_VTYPE].reset_index(drop=True)
    n = len(df)
    A = np.stack(df['mixed_dB'].to_numpy()).reshape(n, N_MIN, N_BAND)
    y = df['share'].to_numpy(dtype=np.float64)
    mask = np.isfinite(A)
    return A, mask, y, df['site'].to_numpy()


def train_seed(X_tr, y_tr, X_va, y_va, seed, rounds):
    dtr = lgb.Dataset(X_tr, label=y_tr, feature_name=ALL_NAMES)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr,
                      feature_name=ALL_NAMES)
    params = dict(objective='regression', metric='rmse', learning_rate=0.05,
                  num_leaves=31, min_data_in_leaf=40, feature_fraction=0.9,
                  bagging_fraction=0.9, bagging_freq=1, seed=seed, verbose=-1)
    model = lgb.train(params, dtr, num_boost_round=rounds,
                      valid_sets=[dva], valid_names=['val'])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=600)
    ap.add_argument('--seeds', type=int, nargs='+',
                    default=[7, 11, 13, 17, 19])
    ap.add_argument('--aug_copies', type=int, default=2)
    args = ap.parse_args()
    rng = np.random.default_rng(42)

    print('loading real-only (hms_clip) data...', flush=True)
    A_tr, m_tr, y_tr, _ = load_realonly_site('train')
    A_va, m_va, y_va, site_va = load_realonly_site('val')
    print(f'train {A_tr.shape}  val {A_va.shape}', flush=True)

    print('augmenting training data (dB offsets + NaN masks)...', flush=True)
    aug_A, aug_m = augment(A_tr, m_tr, rng, n_copies=args.aug_copies)
    A_tr_all = np.concatenate([A_tr] + aug_A, axis=0)
    m_tr_all = np.concatenate([m_tr] + aug_m, axis=0)
    y_tr_all = np.concatenate([y_tr] * (1 + args.aug_copies))
    print(f'augmented train {A_tr_all.shape}', flush=True)

    print('featurizing v2...', flush=True)
    X_tr = featurize_v2(A_tr_all, m_tr_all)
    X_va = featurize_v2(A_va, m_va)
    print(f'features: {X_tr.shape[1]}', flush=True)

    oof, test_preds = [], []
    ids, A_te, m_te = load_test(ROOT / 'data' / 'Test.csv')
    X_te = featurize_v2(A_te, m_te)
    for s in args.seeds:
        model = train_seed(X_tr, y_tr_all, X_va, y_va, s, args.rounds)
        model.save_model(ROOT / 'models' / f'v2_s{s}.txt')
        oof.append(model.predict(X_va))
        test_preds.append(model.predict(X_te))
        print(f'seed {s} done', flush=True)
    oof = np.clip(np.mean(oof, axis=0), 0, 1)
    test_pred = np.clip(np.mean(test_preds, axis=0), 0, 1)

    # validation gate: Fish Hoek vs Sea Point
    err = np.abs(oof - y_va)
    fh = err[site_va == '09_Fish_Hoek'].mean()
    sp = err[site_va == '10_Sea_Point'].mean()
    rmse = float(np.sqrt(np.mean((oof - y_va) ** 2)))
    print(f'val 1-RMSE = {1 - rmse:.4f}  FishHoek abs err = {fh:.4f}  '
          f'SeaPoint abs err = {sp:.4f}', flush=True)
    gate = (fh <= 0.045) and (sp <= 0.040)
    print(f'GATE: {"PASS" if gate else "FAIL"} '
          f'(need FH<=0.045, SP<=0.040)', flush=True)

    score = detection_score(A_te, m_te)
    sub = pd.DataFrame({'ID': ids, 'share_added_vessel': test_pred,
                        'score_added_vessel': score})
    check_submission(sub, len(ids))
    sub.to_csv(ROOT / 'submissions' / 'nov6_v2.csv', index=False)
    print('wrote submissions/nov6_v2.csv', flush=True)

    # slot-1 blend: 50/50 with current best (10-seed real-only + v1 blend)
    best = pd.read_csv(ROOT / 'submissions' / 'nov4_slot2_ro_v1_avg.csv')
    m = sub.merge(best, on='ID', suffixes=('', '_best'))
    blend_share = (m['share_added_vessel'] + m['share_added_vessel_best']) / 2
    blend = pd.DataFrame({'ID': m['ID'], 'share_added_vessel': blend_share,
                          'score_added_vessel': m['score_added_vessel']})
    blend.to_csv(ROOT / 'submissions' / 'nov6_v2_blend.csv', index=False)
    print('wrote submissions/nov6_v2_blend.csv', flush=True)

    with open(ROOT / 'submissions' / 'nov6_v2_results.txt', 'w') as f:
        f.write(f'val_1-RMSE={1 - rmse:.4f}\nFishHoek_abs_err={fh:.4f}\n'
                f'SeaPoint_abs_err={sp:.4f}\nGATE={"PASS" if gate else "FAIL"}\n'
                f'seeds={args.seeds}\n')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
