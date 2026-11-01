"""Seed-ensemble LightGBM training for the vessel-noise challenge.

Trains K LightGBM models with different seeds on the synthetic set, averages
their predictions (bagging over seeds reduces variance), and optionally
applies a VALIDATION-fit calibration (isotonic or affine) to correct the
systematic bias observed on held-out sites (over-predict near 0,
under-predict at high shares). Calibration is fit on synthetic validation
only -- never on test data.

Usage:
    python -m src.train_ensemble --seeds 7 11 13 17 19 --rounds 600 \
        --out-name ensemble_v1.csv --model-prefix ens_v1 \
        [--calibrate isotonic|affine|none]

Writes models/<prefix>_s<seed>.txt and submissions/<out-name>.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

from src.common import SEED, load_test, detection_score, check_submission
from src.train_lgbm import featurize, load_synthetic, FEATURE_NAMES

ROOT = Path(__file__).resolve().parent.parent


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def bias_report(y_true, y_pred, tag):
    print(f'--- {tag} ---')
    print(f'  RMSE={rmse(y_pred, y_true):.4f}  (1-RMSE={1 - rmse(y_pred, y_true):.4f})')
    for lo, hi in [(0, 0.001), (0.001, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0)]:
        m = (y_true >= lo) & (y_true < hi)
        if m.sum():
            r = rmse(y_pred[m], y_true[m])
            print(f'  share[{lo:4.2f},{hi:4.2f}): n={m.sum():4d} rmse={r:.4f} '
                  f'bias={np.mean(y_pred[m] - y_true[m]):+.4f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[7, 11, 13, 17, 19])
    ap.add_argument('--rounds', type=int, default=600)
    ap.add_argument('--out-name', type=str, default='ensemble_v1.csv')
    ap.add_argument('--model-prefix', type=str, default='ens_v1')
    ap.add_argument('--calibrate', type=str, default='none',
                    choices=['none', 'isotonic', 'affine'])
    ap.add_argument('--data-suffix', type=str, default='',
                    help='suffix of synthetic data files, e.g. _v2')
    args = ap.parse_args()

    print('loading synthetic data...', flush=True)
    A_tr, m_tr, y_tr = load_synthetic('train', args.data_suffix)
    A_va, m_va, y_va = load_synthetic('val', args.data_suffix)
    print(f'train {A_tr.shape}  val {A_va.shape}', flush=True)

    print('featurizing...', flush=True)
    X_tr, X_va = featurize(A_tr, m_tr), featurize(A_va, m_va)

    models_dir = ROOT / 'models'
    models_dir.mkdir(exist_ok=True)

    va_preds, te_preds = [], []
    ids, A_te, m_te = load_test(ROOT / 'data' / 'Test.csv')
    print('featurizing test...', flush=True)
    X_te = featurize(A_te, m_te)

    for s in args.seeds:
        print(f'--- seed {s} ---', flush=True)
        dtr = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_NAMES)
        dva = lgb.Dataset(X_va, label=y_va, reference=dtr,
                          feature_name=FEATURE_NAMES)
        params = dict(objective='regression', metric='rmse', learning_rate=0.05,
                      num_leaves=31, min_data_in_leaf=40, feature_fraction=0.9,
                      bagging_fraction=0.9, bagging_freq=1, seed=s, verbose=-1)
        model = lgb.train(params, dtr, num_boost_round=args.rounds,
                          valid_sets=[dva], valid_names=['val'])
        model.save_model(models_dir / f'{args.model_prefix}_s{s}.txt')
        va_preds.append(np.clip(model.predict(X_va), 0, 1))
        te_preds.append(np.clip(model.predict(X_te), 0, 1))
        print(f'  seed {s} val RMSE={rmse(va_preds[-1], y_va):.4f}')

    va_ens = np.mean(va_preds, axis=0)
    te_ens = np.mean(te_preds, axis=0)
    bias_report(y_va, va_ens, 'ensemble (uncalibrated) on val')

    # ---- validation-fit calibration (synthetic val only, never test) ----
    if args.calibrate == 'isotonic':
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(va_ens, y_va)
        va_cal = np.clip(ir.predict(va_ens), 0, 1)
        te_cal = np.clip(ir.predict(te_ens), 0, 1)
        print('isotonic calibration fit on val')
    elif args.calibrate == 'affine':
        A_mat = np.column_stack([va_ens, np.ones_like(va_ens)])
        coef, *_ = np.linalg.lstsq(A_mat, y_va, rcond=None)
        print(f'affine calibration: true ~= {coef[0]:.4f}*pred + {coef[1]:+.4f}')
        va_cal = np.clip(coef[0] * va_ens + coef[1], 0, 1)
        te_cal = np.clip(coef[0] * te_ens + coef[1], 0, 1)
    else:
        va_cal, te_cal = va_ens, te_ens

    if args.calibrate != 'none':
        bias_report(y_va, va_cal, f'ensemble + {args.calibrate} calibration on val')

    share = te_cal
    score = detection_score(A_te, m_te)
    sub = pd.DataFrame({'ID': ids, 'share_added_vessel': share,
                        'score_added_vessel': score})
    check_submission(sub, len(ids))
    out = ROOT / 'submissions' / args.out_name
    sub.to_csv(out, index=False)
    print(f'wrote {out}')
    print(f'test share: mean={share.mean():.4f} p50={np.median(share):.4f} '
          f'max={share.max():.4f} frac<0.05={(share < 0.05).mean():.3f}')


if __name__ == '__main__':
    main()
