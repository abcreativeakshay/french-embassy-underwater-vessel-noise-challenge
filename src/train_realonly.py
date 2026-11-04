"""Nov 4 slot-1: REAL-ONLY LightGBM variant.

Colab setup (fresh runtime — run these cells first):
    # 1. clone the repo
    !git clone https://github.com/abcreativeakshay/french-embassy-underwater-vessel-noise-challenge
    %cd french-embassy-underwater-vessel-noise-challenge
    # 2. install deps (all pip-installable, no system packages)
    !pip install -q lightgbm==4.7.0 numpy==1.26.4 pandas==2.1.4 pyarrow==14.0.2 scikit-learn==1.9.1 scipy==1.11.4
    # 3. upload into the repo root (Colab file panel or Drive mount):
    #      data/processed/synthetic_train.parquet
    #      data/processed/synthetic_val.parquet
    #      data/Test.csv
    #    (These are the ONLY data inputs this script needs. They are built
    #    locally by: HearMyShip WAVs (322 clips, https://hearmyship.fer.hr/,
    #    into data/external/hearmyship/) -> python -m src.vessel_signatures
    #    -> python -m src.make_synthetic  [needs Zindi SA acoustics in
    #    data/raw/acoustics/]. On Colab, uploading the three processed files
    #    skips all of that.)
    # 4. train (writes models/realonly_s<seed>.txt and
    #    submissions/nov4_slot1_realonly.csv):
    !python -m src.train_realonly

Why: Nov 3's pair (slot1 v1/v2 avg 0.7923, slot2 10-seed v1-family 0.7936)
both landed under v1's 0.7939 — same-family averaging is tapped out. The best
validation ever (1-RMSE 0.9193) came from HearMyShip-derived passages, but
every submission so far trained on a MIX of real WAV-clip passages
(vessel_type == 'hms_clip': actual 20-s recordings embedded in 1-2 minutes)
and CPA spectral-template passages (vessel_type == 'hms_*': published spectra
with synthetic raised-cosine envelopes). This variant trains EXCLUSIVELY on
mixtures built from the real recordings, betting that genuine passage shapes
transfer better to the real test set than template+envelope mixtures.

NO isotonic calibration (v2 lesson: val-fit calibration overfits the
synthetic distribution). 10 seeds, v1 LightGBM params, pure average.

Usage:
    python -m src.train_realonly [--rounds 600] [--seeds 7 11 ...]
    python -m src.train_realonly --smoke      # 2 rounds, 1 seed, no outputs
Writes models/realonly_s<seed>.txt, submissions/nov4_slot1_realonly.csv and
submissions/nov4_slot1_realonly_results.txt
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb

from src.common import N_MIN, N_BAND, load_test, detection_score, check_submission
from src.train_lgbm import featurize, FEATURE_NAMES

ROOT = Path(__file__).resolve().parent.parent
REAL_VTYPE = 'hms_clip'          # real WAV-clip passages only
DEFAULT_SEEDS = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def load_realonly(name):
    """Load synthetic mixtures built ONLY from real WAV-clip passages.

    Same return convention as train_lgbm.load_synthetic: A (n,15,25) dB,
    mask (n,15,25) bool, y (n,) share.
    """
    t = pq.read_table(ROOT / 'data' / 'processed' / f'synthetic_{name}.parquet',
                      columns=['vessel_type', 'mixed_dB', 'share'])
    df = t.to_pandas()
    df = df[df['vessel_type'] == REAL_VTYPE].reset_index(drop=True)
    assert len(df) > 0, f'no {REAL_VTYPE} rows in synthetic_{name}.parquet'
    n = len(df)
    A = np.stack(df['mixed_dB'].to_numpy()).reshape(n, N_MIN, N_BAND)
    y = df['share'].to_numpy(dtype=np.float64)
    mask = np.ones_like(A, dtype=bool)   # synthetic windows are complete
    return A, mask, y


def train_seed(X_tr, y_tr, X_va, y_va, seed, rounds, model_path=None):
    dtr = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_NAMES)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr,
                      feature_name=FEATURE_NAMES)
    params = dict(objective='regression', metric='rmse', learning_rate=0.05,
                  num_leaves=31, min_data_in_leaf=40, feature_fraction=0.9,
                  bagging_fraction=0.9, bagging_freq=1, seed=seed, verbose=-1)
    model = lgb.train(params, dtr, num_boost_round=rounds,
                      valid_sets=[dva], valid_names=['val'])
    if model_path is not None:
        model.save_model(model_path)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=600)
    ap.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    ap.add_argument('--out', type=str, default='nov4_slot1_realonly.csv',
                    help='output CSV name in submissions/')
    ap.add_argument('--smoke', action='store_true',
                    help='2 rounds on 1 seed, prints diagnostics, writes nothing')
    args = ap.parse_args()

    print('loading real-only (hms_clip) data...', flush=True)
    A_tr, m_tr, y_tr = load_realonly('train')
    A_va, m_va, y_va = load_realonly('val')
    print(f'train {A_tr.shape} (real-only)  val {A_va.shape} (real-only)',
          flush=True)

    print('featurizing...', flush=True)
    X_tr, X_va = featurize(A_tr, m_tr), featurize(A_va, m_va)
    assert X_tr.shape[1] == len(FEATURE_NAMES) == 30, X_tr.shape

    rounds = 2 if args.smoke else args.rounds
    seeds = [args.seeds[0]] if args.smoke else args.seeds

    models_dir = ROOT / 'models'
    if not args.smoke:
        models_dir.mkdir(exist_ok=True)

    va_preds, te_preds = [], []
    if not args.smoke:
        ids, A_te, m_te = load_test(ROOT / 'data' / 'Test.csv')
        print('featurizing test...', flush=True)
        X_te = featurize(A_te, m_te)

    for s in seeds:
        print(f'--- seed {s} ({rounds} rounds) ---', flush=True)
        mp = None if args.smoke else models_dir / f'realonly_s{s}.txt'
        model = train_seed(X_tr, y_tr, X_va, y_va, s, rounds, mp)
        pv = np.clip(model.predict(X_va), 0, 1)
        va_preds.append(pv)
        print(f'  seed {s} real-only val 1-RMSE={1 - rmse(pv, y_va):.4f}',
              flush=True)
        if not args.smoke:
            te_preds.append(np.clip(model.predict(X_te), 0, 1))

    va_ens = np.mean(va_preds, axis=0)
    s_real = 1 - rmse(va_ens, y_va)
    print(f'ENSEMBLE real-only val 1-RMSE={s_real:.4f}', flush=True)

    if args.smoke:
        print('smoke OK: filter + featurize + train + predict all work; '
              'no files written', flush=True)
        return

    # bonus diagnostic: how does the real-only ensemble do on the FULL
    # synthetic val (incl. template passages it never trained on)?
    from src.train_lgbm import load_synthetic
    A_fv, m_fv, y_fv = load_synthetic('val')
    X_fv = featurize(A_fv, m_fv)
    # re-predict with the trained models (kept in memory via va loop is gone,
    # so reload from disk)
    full_preds = []
    for s in seeds:
        m = lgb.Booster(model_file=str(models_dir / f'realonly_s{s}.txt'))
        full_preds.append(np.clip(m.predict(X_fv), 0, 1))
    s_full = 1 - rmse(np.mean(full_preds, axis=0), y_fv)
    print(f'ENSEMBLE full-synthetic val 1-RMSE={s_full:.4f} (for reference)',
          flush=True)

    share = np.mean(te_preds, axis=0)   # pure average, NO calibration
    score = detection_score(A_te, m_te)
    sub = pd.DataFrame({'ID': ids, 'share_added_vessel': share,
                        'score_added_vessel': score})
    check_submission(sub, len(ids))
    out = ROOT / 'submissions' / args.out
    sub.to_csv(out, index=False)
    print(f'wrote {out}', flush=True)
    print(f'test share: mean={share.mean():.4f} p50={np.median(share):.4f} '
          f'max={share.max():.4f} frac<0.05={(share < 0.05).mean():.3f}')

    res = ROOT / 'submissions' / (args.out.replace('.csv', '_results.txt'))
    with open(res, 'w') as f:
        f.write(f'n_train_realonly={len(y_tr)}\n')
        f.write(f'n_val_realonly={len(y_va)}\n')
        f.write(f'seeds={seeds}\n')
        f.write(f'rounds={args.rounds}\n')
        f.write(f'ensemble_realonly_val_1rmse={s_real:.4f}\n')
        f.write(f'ensemble_fullsynth_val_1rmse={s_full:.4f}\n')
        f.write('calibration=none\n')
    print(f'wrote {res}', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
