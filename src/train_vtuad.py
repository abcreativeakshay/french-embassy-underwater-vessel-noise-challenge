"""Nov 7 candidate: real-only LightGBM + VTUAD large-vessel passages.

Trains on hms_clip mixtures (5,797 train rows, the real-only baseline pool)
PLUS VTUAD mixtures (same pipeline, same 30 features, same LightGBM params
as src/train_realonly.py). No calibration, no absolute-level features — the
v2 86-feature set just failed publicly (blend 0.78965 / standalone 0.78088
vs best 0.79519), so this candidate differs from v2 by construction: same
30 relative features, only the vessel pool changes.

Honest validation: held-out sites (Fish Hoek, Sea Point) + disjoint vessel
clips for BOTH pools. Reported separately:
  * hms-only val  — the apples-to-apples number vs the real-only baseline
    (val 1-RMSE 0.9229). GATE: candidate must be >= 0.9229 here.
  * vtuad-only val — how the model does on large-vessel mixtures.
  * combined val, Fish Hoek / Sea Point breakdowns.

Resume-safe: per-seed test/val predictions are checkpointed to
submissions/vtuad_ckpt{suffix}_s<seed>.npz as each seed finishes; re-running
skips finished seeds. Models saved to models/vtuad{suffix}_s<seed>.txt.

Writes submissions/nov7_vtuad{suffix}.csv (standalone) and
       submissions/nov7_vtuad{suffix}_blend.csv (50/50 with the current best
       blend submissions/nov4_slot2_ro_v1_avg.csv, public 0.79519).
Both files are verified (1,500 rows, ID order = Test.csv, no NaN/inf).

Usage:
    python -m src.train_vtuad [--suffix _large] [--rounds 600]
    python -m src.train_vtuad --suffix _large --seeds 7 11
    python -m src.train_vtuad --smoke   # 2 rounds, 1 seed, no outputs
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb

from src.common import (N_MIN, N_BAND, load_test, detection_score,
                        check_submission)
from src.train_lgbm import featurize, FEATURE_NAMES
from src.train_realonly import REAL_VTYPE, DEFAULT_SEEDS, train_seed

ROOT = Path(__file__).resolve().parent.parent
BASELINE_HMS_VAL = 0.9229   # real-only ensemble, hms_clip-only val


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def load_rows(name, suffix, vtype_filter):
    """Load (A, mask, y, site) for rows matching vtype_filter in a parquet.

    vtype_filter: callable on vessel_type Series -> bool mask.
    """
    t = pq.read_table(ROOT / 'data' / 'processed' / f'synthetic_{name}{suffix}.parquet',
                      columns=['vessel_type', 'site', 'mixed_dB', 'share'])
    df = t.to_pandas()
    keep = vtype_filter(df['vessel_type'])
    df = df[keep].reset_index(drop=True)
    assert len(df) > 0, f'no matching rows in synthetic_{name}{suffix}.parquet'
    n = len(df)
    A = np.stack(df['mixed_dB'].to_numpy()).reshape(n, N_MIN, N_BAND)
    y = df['share'].to_numpy(dtype=np.float64)
    site = df['site'].to_numpy()
    mask = np.ones_like(A, dtype=bool)
    return A, mask, y, site


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suffix', type=str, default='_large',
                    help='suffix of synthetic_vtuad_{train,val}<suffix>.parquet')
    ap.add_argument('--rounds', type=int, default=600)
    ap.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    ap.add_argument('--out', type=str, default='',
                    help='output CSV name in submissions/ (default nov7_vtuad<suffix>.csv)')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--max-vtuad-train', type=int, default=0,
                    help='cap VTUAD train rows to N (0 = all); fixed seed 0 '
                         'subsample. Tests whether VTUAD rows dilute hms '
                         'performance when they dominate the train set.')
    args = ap.parse_args()

    sfx = args.suffix
    cap_tag = f'_cap{args.max_vtuad_train}' if args.max_vtuad_train else ''
    out_name = args.out or f'nov7_vtuad{sfx}{cap_tag}.csv'
    rounds = 2 if args.smoke else args.rounds
    seeds = [args.seeds[0]] if args.smoke else args.seeds

    print('loading hms_clip + vtuad data...', flush=True)
    A_h_tr, m_h_tr, y_h_tr, _ = load_rows('train', '', lambda v: v == REAL_VTYPE)
    A_v_tr, m_v_tr, y_v_tr, _ = load_rows('vtuad_train', sfx, lambda v: v.str.startswith('vtuad_'))
    if args.max_vtuad_train and len(y_v_tr) > args.max_vtuad_train:
        rng_cap = np.random.default_rng(0)
        idx = np.sort(rng_cap.choice(len(y_v_tr), args.max_vtuad_train, replace=False))
        A_v_tr, m_v_tr, y_v_tr = A_v_tr[idx], m_v_tr[idx], y_v_tr[idx]
        print(f'capped vtuad train rows to {len(y_v_tr)}', flush=True)
    A_tr = np.concatenate([A_h_tr, A_v_tr]); m_tr = np.concatenate([m_h_tr, m_v_tr])
    y_tr = np.concatenate([y_h_tr, y_v_tr])
    A_h_va, m_h_va, y_h_va, s_h_va = load_rows('val', '', lambda v: v == REAL_VTYPE)
    A_v_va, m_v_va, y_v_va, s_v_va = load_rows('vtuad_val', sfx, lambda v: v.str.startswith('vtuad_'))
    A_va = np.concatenate([A_h_va, A_v_va]); m_va = np.concatenate([m_h_va, m_v_va])
    y_va = np.concatenate([y_h_va, y_v_va]); s_va = np.concatenate([s_h_va, s_v_va])
    n_h_va = len(y_h_va)
    print(f'train hms={len(y_h_tr)} vtuad={len(y_v_tr)} total={len(y_tr)}', flush=True)
    print(f'val   hms={len(y_h_va)} vtuad={len(y_v_va)} total={len(y_va)}', flush=True)

    print('featurizing...', flush=True)
    X_tr, X_va = featurize(A_tr, m_tr), featurize(A_va, m_va)
    assert X_tr.shape[1] == len(FEATURE_NAMES) == 30, X_tr.shape

    ids, A_te, m_te = (None, None, None) if args.smoke else load_test(ROOT / 'data' / 'Test.csv')
    X_te = None if args.smoke else featurize(A_te, m_te)

    ckpt_dir = ROOT / 'submissions'
    models_dir = ROOT / 'models'
    if not args.smoke:
        models_dir.mkdir(exist_ok=True)

    va_preds, te_preds = [], []
    for s in seeds:
        ck = ckpt_dir / f'vtuad_ckpt{sfx}{cap_tag}_s{s}.npz'
        if not args.smoke and ck.exists():
            z = np.load(ck)
            va_preds.append(z['va_pred'])
            te_preds.append(z['te_pred'])
            print(f'--- seed {s}: loaded checkpoint ---', flush=True)
            continue
        print(f'--- seed {s} ({rounds} rounds) ---', flush=True)
        mp = None if args.smoke else models_dir / f'vtuad{sfx}{cap_tag}_s{s}.txt'
        model = train_seed(X_tr, y_tr, X_va, y_va, s, rounds, mp)
        pv = np.clip(model.predict(X_va), 0, 1)
        va_preds.append(pv)
        print(f'  seed {s} combined val 1-RMSE={1 - rmse(pv, y_va):.4f}  '
              f'hms-only={1 - rmse(pv[:n_h_va], y_h_va):.4f}', flush=True)
        if not args.smoke:
            pt = np.clip(model.predict(X_te), 0, 1)
            te_preds.append(pt)
            np.savez_compressed(ck, va_pred=pv, te_pred=pt)

    va_ens = np.mean(va_preds, axis=0)
    s_comb = 1 - rmse(va_ens, y_va)
    s_hms = 1 - rmse(va_ens[:n_h_va], y_h_va)
    s_vtu = 1 - rmse(va_ens[n_h_va:], y_v_va)
    print(f'ENSEMBLE combined val 1-RMSE={s_comb:.4f}', flush=True)
    print(f'ENSEMBLE hms-only val 1-RMSE={s_hms:.4f}  (baseline {BASELINE_HMS_VAL:.4f})', flush=True)
    print(f'ENSEMBLE vtuad-only val 1-RMSE={s_vtu:.4f}', flush=True)
    for site in ['09_Fish_Hoek', '10_Sea_Point']:
        m = s_va == site
        print(f'  {site}: n={m.sum()} 1-RMSE={1 - rmse(va_ens[m], y_va[m]):.4f} '
              f'mean_abs_err={np.mean(np.abs(va_ens[m] - y_va[m])):.4f}', flush=True)
    for lo, hi in [(0, 0.001), (0.001, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0)]:
        m = (y_va >= lo) & (y_va < hi)
        if m.sum():
            print(f'  share[{lo:4.2f},{hi:4.2f}): n={m.sum():4d} rmse={rmse(va_ens[m], y_va[m]):.4f} '
                  f'bias={np.mean(va_ens[m] - y_va[m]):+.4f}', flush=True)
    gate = s_hms >= BASELINE_HMS_VAL
    print(f'GATE (hms-only val >= {BASELINE_HMS_VAL}): {"PASS" if gate else "FAIL"}', flush=True)

    if args.smoke:
        print('smoke OK', flush=True)
        return

    share = np.mean(te_preds, axis=0)
    score = detection_score(A_te, m_te)
    sub = pd.DataFrame({'ID': ids, 'share_added_vessel': share,
                        'score_added_vessel': score})
    check_submission(sub, len(ids))
    out = ROOT / 'submissions' / out_name
    sub.to_csv(out, index=False)
    print(f'wrote {out}', flush=True)
    print(f'test share: mean={share.mean():.4f} p50={np.median(share):.4f} '
          f'max={share.max():.4f} frac<0.05={(share < 0.05).mean():.3f}')

    # 50/50 blend with the current best (Nov 4 blend, public 0.79519)
    best = pd.read_csv(ROOT / 'submissions' / 'nov4_slot2_ro_v1_avg.csv')
    assert (best['ID'].to_numpy() == ids).all(), 'best-blend ID order mismatch'
    blend_share = 0.5 * share + 0.5 * best['share_added_vessel'].to_numpy()
    blend = pd.DataFrame({'ID': ids, 'share_added_vessel': blend_share,
                          'score_added_vessel': score})
    check_submission(blend, len(ids))
    bout = ROOT / 'submissions' / out_name.replace('.csv', '_blend.csv')
    blend.to_csv(bout, index=False)
    print(f'wrote {bout}', flush=True)
    print(f'blend test share: mean={blend_share.mean():.4f} '
          f'corr vs best={np.corrcoef(share, best["share_added_vessel"])[0, 1]:.4f} '
          f'mean|diff|={np.mean(np.abs(share - best["share_added_vessel"])): .4f}')

    res = ROOT / 'submissions' / (out_name.replace('.csv', '_results.txt'))
    with open(res, 'w') as f:
        f.write(f'suffix={sfx}\n')
        f.write(f'n_train_hms={len(y_h_tr)} n_train_vtuad={len(y_v_tr)}\n')
        f.write(f'n_val_hms={len(y_h_va)} n_val_vtuad={len(y_v_va)}\n')
        f.write(f'seeds={seeds}\nrounds={args.rounds}\n')
        f.write(f'ensemble_combined_val_1rmse={s_comb:.4f}\n')
        f.write(f'ensemble_hms_only_val_1rmse={s_hms:.4f}\n')
        f.write(f'ensemble_vtuad_only_val_1rmse={s_vtu:.4f}\n')
        f.write(f'baseline_hms_val_1rmse={BASELINE_HMS_VAL:.4f}\n')
        f.write(f'gate_pass={gate}\n')
        f.write('features=30 (same as real-only baseline; NO absolute-level features)\n')
        f.write('calibration=none\n')
    print(f'wrote {res}', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
