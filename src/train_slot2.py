"""Nov 3 slot-2: v1-family variant with validation-SAFE calibration.

Variant A: 10-seed v1 ensemble, NO isotonic calibration (pure variance
reduction; the val-fit isotonic was the overfit vector in v2).
Variant B: honest calibration test -- split synthetic val in half, fit
isotonic on half 1, evaluate calibrated vs uncalibrated on half 2.
Ship whichever wins on honest validation. Never touches test data for
fitting; never reintroduces the test-guided prior.

Usage:
    python src/train_slot2.py
Writes submissions/nov3_slot2_v1safe.csv and models/nov3_s2_s<seed>.txt
"""
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

from src.common import load_test, detection_score, check_submission
from src.train_lgbm import featurize, load_synthetic, FEATURE_NAMES

ROOT = Path(__file__).resolve().parent.parent
V1_SEEDS = [7, 11, 13, 17, 19]
NEW_SEEDS = [23, 29, 31, 37, 41]
ROUNDS = 600


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def train_seed(X_tr, y_tr, X_va, y_va, seed, model_path):
    dtr = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_NAMES)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr,
                      feature_name=FEATURE_NAMES)
    params = dict(objective='regression', metric='rmse', learning_rate=0.05,
                  num_leaves=31, min_data_in_leaf=40, feature_fraction=0.9,
                  bagging_fraction=0.9, bagging_freq=1, seed=seed, verbose=-1,
                  num_threads=2)  # CPU-polite: finstress 20-seed job has priority
    model = lgb.train(params, dtr, num_boost_round=ROUNDS,
                      valid_sets=[dva], valid_names=['val'])
    model.save_model(model_path)
    return model


def main():
    rng = np.random.default_rng(0)
    print('loading v1 synthetic data...', flush=True)
    A_tr, m_tr, y_tr = load_synthetic('train', '')
    A_va, m_va, y_va = load_synthetic('val', '')
    print(f'train {A_tr.shape}  val {A_va.shape}', flush=True)
    print('featurizing...', flush=True)
    X_tr, X_va = featurize(A_tr, m_tr), featurize(A_va, m_va)

    models_dir = ROOT / 'models'
    va_preds, models = {}, {}

    # ---- train the 5 NEW seeds (v1's 5 already exist on disk) ----
    for s in NEW_SEEDS:
        mp = models_dir / f'nov3_s2_s{s}.txt'
        if mp.exists():
            print(f'seed {s}: loading cached {mp.name}', flush=True)
            models[s] = lgb.Booster(model_file=str(mp))
        else:
            print(f'seed {s}: training...', flush=True)
            models[s] = train_seed(X_tr, y_tr, X_va, y_va, s, mp)
        va_preds[s] = np.clip(models[s].predict(X_va), 0, 1)
        print(f'  seed {s} val 1-RMSE={1 - rmse(va_preds[s], y_va):.4f}',
              flush=True)

    # ---- load v1's 5 original seeds ----
    for s in V1_SEEDS:
        mp = models_dir / f'ens_v1_s{s}.txt'
        assert mp.exists(), f'missing {mp}'
        models[s] = lgb.Booster(model_file=str(mp))
        va_preds[s] = np.clip(models[s].predict(X_va), 0, 1)
    print('loaded v1 original 5 seeds', flush=True)

    va5 = np.mean([va_preds[s] for s in V1_SEEDS], axis=0)
    va10 = np.mean([va_preds[s] for s in V1_SEEDS + NEW_SEEDS], axis=0)
    s5 = 1 - rmse(va5, y_va)
    s10 = 1 - rmse(va10, y_va)
    print(f'VARIANT A val: 5-seed 1-RMSE={s5:.4f} | 10-seed 1-RMSE={s10:.4f} '
          f'(delta={s10 - s5:+.4f})', flush=True)

    # ---- Variant B: honest calibration test (split val in half) ----
    idx = rng.permutation(len(y_va))
    h1, h2 = idx[:len(idx) // 2], idx[len(idx) // 2:]
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(va10[h1], y_va[h1])
    uncal_h2 = 1 - rmse(va10[h2], y_va[h2])
    cal_h2 = 1 - rmse(np.clip(ir.predict(va10[h2]), 0, 1), y_va[h2])
    print(f'VARIANT B honest cal test on val-half2: uncalibrated 1-RMSE='
          f'{uncal_h2:.4f} | isotonic-fit-on-half1 1-RMSE={cal_h2:.4f} '
          f'(delta={cal_h2 - uncal_h2:+.4f})', flush=True)

    # ---- pick winner ----
    use_cal = cal_h2 > uncal_h2 + 1e-4  # require clear win, else stay clean
    # refit calibrator on FULL val only if the honest test passed
    # (fit on val is then honest: the decision was made on held-out half2)
    if use_cal:
        ir_full = IsotonicRegression(out_of_bounds='clip')
        ir_full.fit(va10, y_va)
        print('honest test PASSED: fitting isotonic on full val', flush=True)
    else:
        ir_full = None
        print('honest test FAILED: shipping pure 10-seed average, no '
              'calibration', flush=True)

    # ---- test predictions ----
    print('featurizing test...', flush=True)
    ids, A_te, m_te = load_test(ROOT / 'data' / 'Test.csv')
    X_te = featurize(A_te, m_te)
    te_preds = [np.clip(models[s].predict(X_te), 0, 1)
                for s in V1_SEEDS + NEW_SEEDS]
    te10 = np.mean(te_preds, axis=0)
    share = np.clip(ir_full.predict(te10), 0, 1) if ir_full else te10
    score = detection_score(A_te, m_te)
    sub = pd.DataFrame({'ID': ids, 'share_added_vessel': share,
                        'score_added_vessel': score})
    check_submission(sub, len(ids))
    out = ROOT / 'submissions' / 'nov3_slot2_v1safe.csv'
    sub.to_csv(out, index=False)
    print(f'wrote {out}', flush=True)
    print(f'calibrated={use_cal} test share: mean={share.mean():.4f} '
          f'p50={np.median(share):.4f} max={share.max():.4f} '
          f'frac<0.05={(share < 0.05).mean():.3f}')

    # ---- summary file ----
    with open(ROOT / 'submissions' / 'nov3_slot2_results.txt', 'w') as f:
        f.write(f'variantA_5seed_val_1rmse={s5:.4f}\n')
        f.write(f'variantA_10seed_val_1rmse={s10:.4f}\n')
        f.write(f'variantB_honest_half2_uncal={uncal_h2:.4f}\n')
        f.write(f'variantB_honest_half2_cal={cal_h2:.4f}\n')
        f.write(f'shipped_calibrated={use_cal}\n')
    print('done', flush=True)


if __name__ == '__main__':
    main()
