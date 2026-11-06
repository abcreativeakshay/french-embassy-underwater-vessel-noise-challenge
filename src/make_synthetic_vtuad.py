"""Build VTUAD synthetic mixtures (Direction #2, Nov 7).

Same pipeline as src/make_synthetic.py, with the vessel pool swapped:
  * Backgrounds: identical sampling (same seed, same n_per_site) as the
    original synthetic set, so VTUAD rows pair with the hms_clip rows
    background-for-background and share-for-share.
  * Vessel passages: VTUAD large-vessel signatures
    (data/external/vtuad_signatures.npz) — the loudest-60s band totals of
    each clip, embedded in 1-2 minutes exactly like hms_clip (flat envelope,
    80% 1-min / 20% 2-min). Vessel type recorded as 'vtuad_<label>'.
  * Same share prior (25% zeros, rest U(0.03, 0.85)) and same energy-domain
    mixing at exact ground-truth share.
  * Same held-out sites for validation (Fish Hoek, Sea Point) and disjoint
    vessel clips between train/val (every 5th sorted clip id -> val, same
    convention as VesselPool).

Writes data/processed/synthetic_vtuad_train.parquet and
     data/processed/synthetic_vtuad_val.parquet (same schema as
     synthetic_train/val.parquet).

Usage: python -m src.make_synthetic_vtuad [--n-per-site 1500] [--large-only]
  --large-only restricts the pool to cargo/tanker/passengership/dredger/other
  (default: all non-background VTUAD clips, incl. tug/fishing/pleasurecraft).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.common import N_MIN, N_BAND, SEED
from src.make_synthetic import (sample_background_windows, mix_window,
                                sample_share, build_split, VAL_SITES)

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / 'data' / 'external'

LARGE_LABELS = {'cargo', 'tanker', 'passengership', 'dredger', 'other'}


class VtuadPool:
    """VTUAD vessel signatures, split disjointly into train/val."""

    def __init__(self, split='train', val_frac_every=5, seed=SEED,
                 large_only=False):
        z = np.load(EXT / 'vtuad_signatures.npz')
        ids = z['ids'].astype(str)
        labels = z['labels'].astype(str)
        tot = z['band_tot']                              # (n, 25) energy
        order = np.argsort(ids)
        ids, labels, tot = ids[order], labels[order], tot[order]
        if large_only:
            keep_large = np.array([l in LARGE_LABELS for l in labels])
            ids, labels, tot = ids[keep_large], labels[keep_large], tot[keep_large]
            print(f'large-only pool: {len(ids)} clips')
        is_val = (np.arange(len(ids)) % val_frac_every) == 0
        keep = is_val if split == 'val' else ~is_val
        self.ids = ids[keep]
        self.labels = labels[keep]
        self.tot = tot[keep]
        print(f'VTUAD pool {split}: {len(self.ids)} clips '
              f'({pd.Series(self.labels).value_counts().to_dict()})')
        self.rng = np.random.default_rng(seed + (1 if split == 'val' else 0))

    def sample(self):
        """Return (V_minute (15,25) energy shape, vessel_type, k minutes).

        Mirrors VesselPool.sample() for hms_clip: band totals into 1-2
        minutes with a flat envelope.
        """
        rng = self.rng
        i = rng.integers(len(self.ids))
        band_tot = self.tot[i].astype(np.float64)
        vtype = f"vtuad_{self.labels[i]}"
        k = 1 if rng.random() < 0.8 else 2
        V = np.zeros((N_MIN, N_BAND))
        start = int(rng.integers(0, N_MIN - k + 1))
        V[start:start + k, :] = band_tot[None, :]
        return V.astype(np.float32), vtype, k


def build_vtuad_split(windows, pool, rng, split_name,
                      zero_frac=0.25, share_lo=0.03, share_hi=0.85):
    rows = []
    for i, w in enumerate(windows):
        V, vtype, k = pool.sample()
        s = sample_share(rng, zero_frac, share_lo, share_hi)
        mixed, s = mix_window(w['bg_dB'], V, s)
        rows.append({'window_id': f'{split_name}_{i:05d}',
                     'site': w['site'], 'share': s,
                     'mixed_dB': mixed.reshape(-1),
                     'bg_dB': w['bg_dB'].reshape(-1),
                     'vessel_type': vtype, 'passage_minutes': k})
    df = pd.DataFrame(rows)
    import pyarrow as pa
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
    ap.add_argument('--n-per-site', type=int, default=1500)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--zero-frac', type=float, default=0.25)
    ap.add_argument('--share-lo', type=float, default=0.03)
    ap.add_argument('--share-hi', type=float, default=0.85)
    ap.add_argument('--large-only', action='store_true',
                    help='pool = cargo/tanker/passengership/dredger/other only')
    ap.add_argument('--out-suffix', type=str, default='',
                    help='suffix for output filenames, e.g. _large')
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    windows = sample_background_windows(args.n_per_site, rng)
    train_w = [w for w in windows if w['site'] not in VAL_SITES]
    val_w = [w for w in windows if w['site'] in VAL_SITES]
    print(f'train windows: {len(train_w)}  val windows: {len(val_w)}', flush=True)

    pool_tr = VtuadPool('train', seed=args.seed, large_only=args.large_only)
    pool_va = VtuadPool('val', seed=args.seed, large_only=args.large_only)

    out = ROOT / 'data' / 'processed'
    out.mkdir(parents=True, exist_ok=True)
    sfx = args.out_suffix
    pq.write_table(build_vtuad_split(train_w, pool_tr, rng, 'tr',
                                     args.zero_frac, args.share_lo, args.share_hi),
                   out / f'synthetic_vtuad_train{sfx}.parquet')
    pq.write_table(build_vtuad_split(val_w, pool_va, rng, 'va',
                                     args.zero_frac, args.share_lo, args.share_hi),
                   out / f'synthetic_vtuad_val{sfx}.parquet')
    print('wrote', out / f'synthetic_vtuad_train{sfx}.parquet', 'and',
          out / f'synthetic_vtuad_val{sfx}.parquet')
    t = pq.read_table(out / f'synthetic_vtuad_train{sfx}.parquet',
                      columns=['share']).to_pandas()['share'].to_numpy()
    print(f'prior: mean={t.mean():.4f} median={np.median(t):.4f} '
          f'frac_zero={(t == 0).mean():.3f} frac<0.1={(t < 0.1).mean():.3f}')


if __name__ == '__main__':
    main()
