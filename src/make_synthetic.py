"""Stage C synthetic training data (PLAN.md).

The organisers release no labelled/synthetic training set, so we build our own:
  1. Backgrounds: contiguous 15-minute windows sampled from the real SA acoustic
     parquets (one site/recorder, ~1-min resolution, qc_25band_finite).
  2. Vessel passages: PARAMETRIC spectral-temporal templates (placeholder for
     real HearMyShip/ONC recordings - see README "next steps"). A vessel is a
     transient broadband energy bump, strongest at low-mid frequencies, with a
     rise/peak/decay envelope over a few minutes.
  3. Mixing is done in the ENERGY domain at a controlled ground-truth share:
         mixed = B + alpha * V
         alpha = s * sum(B) / ((1 - s) * sum(V))   -> share exactly s

Held-out SITES are reserved for validation (no site leakage), so the val score
honestly reflects performance on unseen backgrounds.

Usage:
    python -m src.make_synthetic [--n-per-site 750]
Writes data/processed/synthetic_train.parquet and data/processed/synthetic_val.parquet
Columns: window_id, site, share (target), mixed_dB (15x25 float32 array),
         bg_dB (15x25), vessel_type, passage_minutes.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.common import BAND_COLS, N_MIN, N_BAND, SEED, db_to_energy

ROOT = Path(__file__).resolve().parent.parent
ACOUSTICS = ROOT / 'data' / 'raw' / 'acoustics'
VAL_SITES = {'09_Fish_Hoek', '10_Sea_Point'}   # held-out backgrounds
BAND_LOGF = np.log10(np.array(
    [31.6, 39.8, 50.1, 63.1, 79.4, 100, 125.9, 158.5, 199.5, 251.2, 316.2,
     398.1, 501.2, 631, 794.3, 1000, 1258.9, 1584.9, 1995.3, 2511.9, 3162.3,
     3981.1, 5011.9, 6309.6, 7943.3]))


# ---------------------------------------------------------------- backgrounds
def sample_background_windows(n_per_site, rng):
    """Contiguous 15-min windows from each site parquet. Returns list of dicts
    with bg_dB (15,25) float32 and site name."""
    windows = []
    for pq_path in sorted(ACOUSTICS.glob('*.parquet')):
        site = pq_path.stem
        cols = ['t', 'dur', 'recorder', 'qc_25band_finite'] + BAND_COLS
        df = pq.read_table(pq_path, columns=cols).to_pandas()
        df = df[df['qc_25band_finite'] & (df['dur'] >= 55)].copy()
        df['t'] = pd.to_datetime(df['t'])
        site_windows = []
        for rec, g in df.groupby('recorder', sort=False):
            g = g.sort_values('t').reset_index(drop=True)
            times = g['t'].to_numpy()
            vals = g[BAND_COLS].to_numpy(dtype=np.float32)
            i = 0
            while i + N_MIN <= len(g):
                seg_t = times[i:i + N_MIN]
                gaps = np.diff(seg_t).astype('timedelta64[s]').astype(float)
                if (gaps <= 180).all() and np.isfinite(vals[i:i + N_MIN]).all():
                    site_windows.append(vals[i:i + N_MIN])
                    i += N_MIN          # non-overlapping
                else:
                    i += 1
        rng.shuffle(site_windows)
        for w in site_windows[:n_per_site]:
            windows.append({'bg_dB': w.astype(np.float32), 'site': site})
        print(f'{site}: {len(site_windows)} candidate windows, kept '
              f'{min(len(site_windows), n_per_site)}', flush=True)
    return windows


# ------------------------------------------------------- vessel templates
def vessel_template(rng):
    """Parametric vessel passage -> (T_dB[15,25] bump, vessel_type, k_minutes).

    T_dB is the dB *increase* over background the vessel would cause at full
    strength (alpha=1); the actual mix level is set by the target share.
    """
    vtype = rng.choice(['small_craft', 'large_ship'], p=[0.6, 0.4])
    if vtype == 'small_craft':
        f0 = 10 ** rng.uniform(np.log10(80), np.log10(900))
        width = rng.uniform(0.35, 0.8)          # log10-Hz std
        amp = rng.uniform(5, 16)                # dB peak above background
        floor = rng.uniform(1.0, 3.5)           # broadband cavitation floor
        k = int(rng.integers(2, 9))             # passage minutes
    else:
        f0 = 10 ** rng.uniform(np.log10(40), np.log10(220))
        width = rng.uniform(0.5, 1.0)
        amp = rng.uniform(6, 20)
        floor = rng.uniform(1.5, 4.5)
        k = int(rng.integers(6, 16))
    spec = amp * np.exp(-0.5 * ((BAND_LOGF - np.log10(f0)) / width) ** 2) + floor
    # temporal envelope: raised cosine passage centred at random position
    start = int(rng.integers(0, N_MIN - k + 1))
    tt = np.arange(k)
    env = 0.5 - 0.5 * np.cos(2 * np.pi * (tt + 0.5) / k)
    env = env / env.max()
    T = np.zeros((N_MIN, N_BAND))
    T[start:start + k, :] = env[:, None] * spec[None, :]
    return T.astype(np.float32), vtype, k


def mix_window(bg_dB, T_dB, share):
    """Add vessel energy to background at exact ground-truth `share`."""
    B = db_to_energy(bg_dB)
    V = B * (db_to_energy(T_dB) - 1.0)          # added component at alpha=1
    sumB, sumV = B.sum(), V.sum()
    if share <= 0 or sumV <= 0:
        return bg_dB.copy(), 0.0
    alpha = share * sumB / ((1.0 - share) * sumV)
    mixed_E = B + alpha * V
    with np.errstate(divide='ignore'):
        mixed_dB = 10.0 * np.log10(mixed_E)
    return mixed_dB.astype(np.float32), float(share)


def sample_share(rng):
    if rng.random() < 0.25:
        return 0.0
    return float(rng.uniform(0.03, 0.85))


# ------------------------------------------------------------------ main
def build_split(windows, rng, split_name):
    rows = []
    for i, w in enumerate(windows):
        T, vtype, k = vessel_template(rng)
        s = sample_share(rng)
        mixed, s = mix_window(w['bg_dB'], T, s)
        rows.append({'window_id': f'{split_name}_{i:05d}',
                     'site': w['site'], 'share': s,
                     'mixed_dB': mixed.reshape(-1),
                     'bg_dB': w['bg_dB'].reshape(-1),
                     'vessel_type': vtype, 'passage_minutes': k})
    df = pd.DataFrame(rows)
    # store the 375-vectors as fixed-size list columns
    table = pa.table({
        'window_id': df['window_id'].to_numpy(),
        'site': df['site'].to_numpy(),
        'share': df['share'].to_numpy(dtype=np.float64),
        'vessel_type': df['vessel_type'].to_numpy(),
        'passage_minutes': df['passage_minutes'].to_numpy(dtype=np.int64),
        'mixed_dB': pa.array([x.astype(np.float32) for x in df['mixed_dB']],
                             type=pa.list_(pa.float32(), N_MIN * N_BAND)),
        'bg_dB': pa.array([x.astype(np.float32) for x in df['bg_dB']],
                          type=pa.list_(pa.float32(), N_MIN * N_BAND)),
    })
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-per-site', type=int, default=750)
    ap.add_argument('--seed', type=int, default=SEED)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    windows = sample_background_windows(args.n_per_site, rng)
    train_w = [w for w in windows if w['site'] not in VAL_SITES]
    val_w = [w for w in windows if w['site'] in VAL_SITES]
    print(f'train windows: {len(train_w)}  val windows: {len(val_w)}', flush=True)

    out = ROOT / 'data' / 'processed'
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(build_split(train_w, rng, 'tr'), out / 'synthetic_train.parquet')
    pq.write_table(build_split(val_w, rng, 'va'), out / 'synthetic_val.parquet')
    print('wrote', out / 'synthetic_train.parquet', 'and', out / 'synthetic_val.parquet')


if __name__ == '__main__':
    main()
