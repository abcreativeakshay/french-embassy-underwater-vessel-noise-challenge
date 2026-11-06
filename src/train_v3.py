"""Nov 7 candidate, direction #3: two-stage background + spectral unmixing.

Trains LightGBM on the 68 v3 features (src/features_v3.py) on hms_clip
mixtures only (5,797 train / 378 val rows, same protocol as the real-only
baseline). Same LightGBM params, 10 seeds, no calibration, no absolute
levels.

Gate: ensemble hms-only val 1-RMSE must beat the real-only baseline 0.9229.
Also reports the 50/50 blend of the v3 ensemble with the *current best*
public blend (real-only + v1 ensemble, evaluated on the same val) — the
blend file is only justified if that val blend also beats 0.9229.

Resume-safe: per-seed val/test predictions checkpointed to
submissions/v3_ckpt_s<seed>.npz. Models to models/v3_s<seed>.txt.

If the gate passes, writes submissions/nov7_v3.csv (standalone) and
submissions/nov7_v3_blend.csv (50/50 with nov4_slot2_ro_v1_avg.csv,
public 0.79519). Both verified (1,500 rows, ID order = Test.csv, finite).

Usage:
    python -m src.train_v3 [--rounds 600] [--seeds 7 11 ...]
    python -m src.train_v3 --smoke
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.common import load_test, detection_score, check_submission
from src.train_lgbm import featurize, FEATURE_NAMES
from src.train_realonly import (load_realonly, REAL_VTYPE, DEFAULT_SEEDS,
                                train_seed)
from src.features_v3 import featurize_v3, FEATURE_NAMES_V3

ROOT = Path(__file__).resolve().parent.parent
BASELINE_HMS_VAL = 0.9229


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=600)
    ap.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    ap.add_argument('--out', type=str, default='nov7_v3.csv')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    rounds = 2 if args.smoke else args.rounds
    seeds = [args.seeds[0]] if args.smoke else args.seeds

    print('loading hms_clip data...', flush=True)
    A_tr, m_tr, y_tr = load_realonly('train')
    A_va, m_va, y_va = load_realonly('val')
    import pyarrow.parquet as pq
    t = pq.read_table(ROOT / 'data' / 'processed' / 'synthetic_val.parquet',
                      columns=['vessel_type', 'site'])
    dfv = t.to_pandas()
    site_va = dfv[dfv['vessel_type'] == REAL_VTYPE]['site'].to_numpy()
    print(f'train {A_tr.shape}  val {A_va.shape}', flush=True)

    print('featurizing v3 (train)...', flush=True)
    X_tr = featurize_v3(A_tr, m_tr)
    print('featurizing v3 (val)...', flush=True)
    X_va = featurize_v3(A_va, m_va)
    assert X_tr.shape[1] == len(FEATURE_NAMES_V3) == 68, X_tr.shape

    # standalone unmixing-feature scores on val (no training)
    idx = {f: i for i, f in enumerate(FEATURE_NAMES_V3)}
    fh = site_va == '09_Fish_Hoek'
    print('--- standalone v3 feature scores on val ---', flush=True)
    for f in ['unmix_share', 'unmix_share_clip', 'unmix_share_cpa',
              'unmix_share_perminute', 'unmix_excess_share',
              'ts_analytic_share', 'f30_analytic_share']:
        p = np.clip(X_va[:, idx[f]], 0, 1)
        print(f'  {f:24s} 1-RMSE={1-rmse(p, y_va):.4f} '
              f'FH={1-rmse(p[fh], y_va[fh]):.4f} '
              f'SP={1-rmse(p[~fh], y_va[~fh]):.4f} '
              f'bias={np.mean(p-y_va):+.4f}', flush=True)

    ids, A_te, m_te = (None, None, None) if args.smoke else load_test(ROOT / 'data' / 'Test.csv')
    X_te = None if args.smoke else featurize_v3(A_te, m_te)
    if not args.smoke:
        print('featurized test', X_te.shape, flush=True)

    ckpt_dir = ROOT / 'submissions'
    models_dir = ROOT / 'models'
    if not args.smoke:
        models_dir.mkdir(exist_ok=True)

    va_preds, te_preds = [], []
    for s in seeds:
        ck = ckpt_dir / f'v3_ckpt_s{s}.npz'
        if not args.smoke and ck.exists():
            z = np.load(ck)
            va_preds.append(z['va_pred'])
            te_preds.append(z['te_pred'])
            print(f'--- seed {s}: loaded checkpoint ---', flush=True)
            continue
        print(f'--- seed {s} ({rounds} rounds) ---', flush=True)
        mp = None if args.smoke else models_dir / f'v3_s{s}.txt'
        dtr = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_NAMES_V3)
        dva = lgb.Dataset(X_va, label=y_va, reference=dtr,
                          feature_name=FEATURE_NAMES_V3)
        params = dict(objective='regression', metric='rmse', learning_rate=0.05,
                      num_leaves=31, min_data_in_leaf=40, feature_fraction=0.9,
                      bagging_fraction=0.9, bagging_freq=1, seed=s, verbose=-1)
        model = lgb.train(params, dtr, num_boost_round=rounds,
                          valid_sets=[dva], valid_names=['val'])
        if mp is not None:
            model.save_model(mp)
        pv = np.clip(model.predict(X_va), 0, 1)
        va_preds.append(pv)
        print(f'  seed {s} v3 val 1-RMSE={1-rmse(pv, y_va):.4f}', flush=True)
        if not args.smoke:
            pt = np.clip(model.predict(X_te), 0, 1)
            te_preds.append(pt)
            np.savez_compressed(ck, va_pred=pv, te_pred=pt)

    va_ens = np.mean(va_preds, axis=0)
    s_v3 = 1 - rmse(va_ens, y_va)
    print(f'ENSEMBLE v3 val 1-RMSE={s_v3:.4f}  (baseline {BASELINE_HMS_VAL:.4f})',
          flush=True)
    for site in ['09_Fish_Hoek', '10_Sea_Point']:
        m = site_va == site
        print(f'  {site}: n={m.sum()} 1-RMSE={1-rmse(va_ens[m], y_va[m]):.4f} '
              f'mean_abs_err={np.mean(np.abs(va_ens[m]-y_va[m])):.4f}',
              flush=True)
    for lo, hi in [(0, 0.001), (0.001, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0)]:
        m = (y_va >= lo) & (y_va < hi)
        if m.sum():
            print(f'  share[{lo:4.2f},{hi:4.2f}): n={m.sum():4d} '
                  f'rmse={rmse(va_ens[m], y_va[m]):.4f} '
                  f'bias={np.mean(va_ens[m]-y_va[m]):+.4f}', flush=True)
    gate = s_v3 >= BASELINE_HMS_VAL
    print(f'GATE (v3 val >= {BASELINE_HMS_VAL}): {"PASS" if gate else "FAIL"}',
          flush=True)

    # --- blend with the current best (real-only + v1), evaluated on val ---
    X_va30 = featurize(A_va, m_va)
    ro_preds, v1_preds = [], []
    for s in DEFAULT_SEEDS:
        m = lgb.Booster(model_file=str(models_dir / f'realonly_s{s}.txt'))
        ro_preds.append(np.clip(m.predict(X_va30), 0, 1))
    for s in [7, 11, 13, 17, 19]:   # ens_v1 was trained on 5 seeds only
        m = lgb.Booster(model_file=str(models_dir / f'ens_v1_s{s}.txt'))
        v1_preds.append(np.clip(m.predict(X_va30), 0, 1))
    ro_ens = np.mean(ro_preds, axis=0)
    v1_ens = np.mean(v1_preds, axis=0)
    best_va = 0.5 * ro_ens + 0.5 * v1_ens
    print(f'val: real-only ens={1-rmse(ro_ens, y_va):.4f}  '
          f'v1 ens={1-rmse(v1_ens, y_va):.4f}  '
          f'current-best (ro+v1)/2={1-rmse(best_va, y_va):.4f}', flush=True)
    for w in [0.5, 0.7, 0.3]:
        bl = w * va_ens + (1 - w) * best_va
        print(f'  blend w_v3={w}: val 1-RMSE={1-rmse(bl, y_va):.4f} '
              f'FH={1-rmse(bl[fh], y_va[fh]):.4f} '
              f'SP={1-rmse(bl[~fh], y_va[~fh]):.4f}', flush=True)
    blend_va = 0.5 * va_ens + 0.5 * best_va
    blend_gate = (1 - rmse(blend_va, y_va)) >= BASELINE_HMS_VAL
    print(f'BLEND GATE (50/50 val >= {BASELINE_HMS_VAL}): '
          f'{"PASS" if blend_gate else "FAIL"}', flush=True)
    # also: unmixing standalone blended with current best
    un = np.clip(X_va[:, idx['unmix_share']], 0, 1)
    bl_u = 0.5 * un + 0.5 * best_va
    print(f'  unmixing+best 50/50: val 1-RMSE={1-rmse(bl_u, y_va):.4f}',
          flush=True)

    if args.smoke:
        print('smoke OK', flush=True)
        return

    if not (gate or blend_gate):
        print('NO GATE PASSED — no submission files built (per task).',
              flush=True)
        res = ROOT / 'submissions' / 'nov7_v3_results.txt'
        with open(res, 'w') as f:
            f.write(f'ensemble_v3_val_1rmse={s_v3:.4f}\n')
            f.write(f'blend50_val_1rmse={1-rmse(blend_va, y_va):.4f}\n')
            f.write(f'baseline_hms_val_1rmse={BASELINE_HMS_VAL:.4f}\n')
            f.write(f'gate_pass={gate}\nblend_gate_pass={blend_gate}\n')
            f.write('features=68 (v3: two-stage bg + NNLS unmixing)\n')
            f.write('calibration=none\n')
            f.write('submission_files=NONE (gate fail)\n')
        print(f'wrote {res}', flush=True)
        print('done', flush=True)
        return

    share = np.mean(te_preds, axis=0)
    score = detection_score(A_te, m_te)
    sub = pd.DataFrame({'ID': ids, 'share_added_vessel': share,
                        'score_added_vessel': score})
    check_submission(sub, len(ids))
    out = ROOT / 'submissions' / args.out
    sub.to_csv(out, index=False)
    print(f'wrote {out}', flush=True)
    print(f'test share: mean={share.mean():.4f} p50={np.median(share):.4f} '
          f'max={share.max():.4f} frac<0.05={(share < 0.05).mean():.3f}')

    best = pd.read_csv(ROOT / 'submissions' / 'nov4_slot2_ro_v1_avg.csv')
    assert (best['ID'].to_numpy() == ids).all(), 'best-blend ID order mismatch'
    blend_share = 0.5 * share + 0.5 * best['share_added_vessel'].to_numpy()
    blend = pd.DataFrame({'ID': ids, 'share_added_vessel': blend_share,
                          'score_added_vessel': score})
    check_submission(blend, len(ids))
    bout = ROOT / 'submissions' / args.out.replace('.csv', '_blend.csv')
    blend.to_csv(bout, index=False)
    print(f'wrote {bout}', flush=True)
    print(f'blend test share: mean={blend_share.mean():.4f} '
          f'corr vs best={np.corrcoef(share, best["share_added_vessel"])[0, 1]:.4f} '
          f'mean|diff|={np.mean(np.abs(share - best["share_added_vessel"])):.4f}')

    res = ROOT / 'submissions' / (args.out.replace('.csv', '_results.txt'))
    with open(res, 'w') as f:
        f.write(f'ensemble_v3_val_1rmse={s_v3:.4f}\n')
        f.write(f'blend50_val_1rmse={1-rmse(blend_va, y_va):.4f}\n')
        f.write(f'baseline_hms_val_1rmse={BASELINE_HMS_VAL:.4f}\n')
        f.write(f'gate_pass={gate}\nblend_gate_pass={blend_gate}\n')
        f.write('features=68 (v3: two-stage bg + NNLS unmixing)\n')
        f.write('calibration=none\n')
    print(f'wrote {res}', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
