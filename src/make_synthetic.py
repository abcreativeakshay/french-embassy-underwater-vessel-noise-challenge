"""Stage C synthetic training data with REAL vessel passages (HearMyShip).

The organisers release no labelled/synthetic training set, so we build our own:
  1. Backgrounds: contiguous 15-minute windows sampled from the real SA acoustic
     parquets (one site/recorder, ~1-min resolution, qc_25band_finite).
  2. Vessel passages: REAL recordings from the HearMyShip database
     (https://hearmyship.fer.hr/, Shipton et al., 1149 small-vessel passages,
     Sibenik canal, Croatia, open access). Each passage is a 20-s 48 kHz mono
     WAV at closest point of approach, converted (src/vessel_signatures.py)
     to per-second third-octave band energies for the 25 challenge bands,
     with the clip's own ambient floor subtracted. A second pool of 1149 CPA
     spectral templates comes from the site's published thirdOctaveData
     (30 bands 31.5 Hz-20 kHz, mapped onto the 25 challenge bands).
  3. Mixing is done in the ENERGY domain at a controlled ground-truth share:
         V_norm = V / V.sum() * B.sum()   (vessel shape, background scale)
         mixed  = B + alpha * V_norm,  alpha = s / (1 - s)  -> share exactly s
     The 20-s clip's band totals are embedded in a single minute bin (or split
     across two adjacent minutes); CPA templates get a 1-3 min raised-cosine
     passage envelope. Only the *relative* spectral-temporal shape of the
     recordings is used, so hydrophone calibration differences cancel out.

Held-out SITES are reserved for validation (no site leakage) AND vessel clips
are split disjointly between train/val (no passage leakage).

Usage:
    python -m src.vessel_signatures   # first: build signatures from WAVs
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
EXT = ROOT / 'data' / 'external'
VAL_SITES = {'09_Fish_Hoek', '10_Sea_Point'}   # held-out backgrounds


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


# ------------------------------------------------------- real vessel passages
class VesselPool:
    """Real HearMyShip vessel signatures, split disjointly into train/val."""

    def __init__(self, split='train', val_frac_every=5, seed=SEED):
        z = np.load(EXT / 'vessel_signatures.npz')
        ids = z['ids'].astype(str)
        sig = z['sig']                                   # (n, 20, 25) energy
        order = np.argsort(ids)
        ids, sig = ids[order], sig[order]
        is_val = (np.arange(len(ids)) % val_frac_every) == 0
        keep = is_val if split == 'val' else ~is_val
        self.ids = ids[keep]
        self.sig = sig[keep]
        print(f'clip pool {split}: {len(self.ids)} clips')

        t = np.load(EXT / 'vessel_cpa_templates.npz')
        tids = t['ids'].astype(str)
        tcats = t['cats'].astype(str)
        tspec = t['spectra']                              # (m, 25) normalised
        order = np.argsort(tids)
        tids, tcats, tspec = tids[order], tcats[order], tspec[order]
        is_val = (np.arange(len(tids)) % val_frac_every) == 0
        keep = is_val if split == 'val' else ~is_val
        self.tids, self.tcats, self.tspec = tids[keep], tcats[keep], tspec[keep]
        print(f'CPA template pool {split}: {len(self.tids)} templates')
        self.rng = np.random.default_rng(seed + (1 if split == 'val' else 0))

    def sample(self):
        """Return (V_minute (15,25) energy shape, vessel_type str, k minutes)."""
        rng = self.rng
        if rng.random() < 0.65:
            # WAV clip: 20 s of real passage, band totals into 1-2 minutes
            i = rng.integers(len(self.ids))
            band_tot = self.sig[i].sum(axis=0)            # (25,)
            vtype = 'hms_clip'
            k = 1 if rng.random() < 0.8 else 2
        else:
            # CPA spectral template: 1-3 min raised-cosine passage
            i = rng.integers(len(self.tids))
            band_tot = self.tspec[i]                      # (25,) normalised
            vtype = f"hms_{self.tcats[i]}"
            k = int(rng.integers(1, 4))
        V = np.zeros((N_MIN, N_BAND))
        start = int(rng.integers(0, N_MIN - k + 1))
        tt = np.arange(k)
        env = 0.5 - 0.5 * np.cos(2 * np.pi * (tt + 0.5) / k)
        env = env / env.max()
        V[start:start + k, :] = env[:, None] * band_tot[None, :]
        return V.astype(np.float32), vtype, k


def mix_window(bg_dB, V_shape, share):
    """Add real vessel energy shape to background at exact ground-truth share.

    V_shape: (15,25) non-negative energy with arbitrary scale (relative shape).
    """
    B = db_to_energy(bg_dB)
    sumB, sumV = B.sum(), V_shape.sum()
    if share <= 0 or sumV <= 0 or sumB <= 0:
        return bg_dB.copy(), 0.0
    Vn = V_shape / sumV * sumB
    alpha = share / (1.0 - share)
    mixed_E = B + alpha * Vn
    with np.errstate(divide='ignore'):
        mixed_dB = 10.0 * np.log10(mixed_E)
    return mixed_dB.astype(np.float32), float(share)


def sample_share(rng):
    if rng.random() < 0.25:
        return 0.0
    return float(rng.uniform(0.03, 0.85))


# ------------------------------------------------------------------ main
def build_split(windows, pool, rng, split_name):
    rows = []
    for i, w in enumerate(windows):
        V, vtype, k = pool.sample()
        s = sample_share(rng)
        mixed, s = mix_window(w['bg_dB'], V, s)
        rows.append({'window_id': f'{split_name}_{i:05d}',
                     'site': w['site'], 'share': s,
                     'mixed_dB': mixed.reshape(-1),
                     'bg_dB': w['bg_dB'].reshape(-1),
                     'vessel_type': vtype, 'passage_minutes': k})
    df = pd.DataFrame(rows)
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

    pool_tr = VesselPool('train', seed=args.seed)
    pool_va = VesselPool('val', seed=args.seed)

    out = ROOT / 'data' / 'processed'
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(build_split(train_w, pool_tr, rng, 'tr'),
                   out / 'synthetic_train.parquet')
    pq.write_table(build_split(val_w, pool_va, rng, 'va'),
                   out / 'synthetic_val.parquet')
    print('wrote', out / 'synthetic_train.parquet', 'and',
          out / 'synthetic_val.parquet')


if __name__ == '__main__':
    main()
