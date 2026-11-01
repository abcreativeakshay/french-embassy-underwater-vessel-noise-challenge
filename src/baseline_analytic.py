"""Stage B analytic baseline (PLAN.md).

Per test window, estimate the per-band background as the 10th percentile over
the 15 minute positions in the ENERGY domain, then attribute any energy above
that background to the added vessel component:

    share = sum(max(E_obs - bg_b, 0)) / sum(E_obs)   (valid cells only)

No training data is used; this is a fixed per-window transformation, which the
rules explicitly allow on test inputs.

Usage:
    python -m src.baseline_analytic [--pct 10]
Writes submissions/baseline_analytic.csv
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.common import load_test, analytic_share, detection_score, check_submission

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pct', type=float, default=10.0,
                    help='background percentile over minutes (default 10)')
    args = ap.parse_args()

    ids, A, mask = load_test(ROOT / 'data' / 'Test.csv')
    share, _ = analytic_share(A, mask, pct=args.pct)
    score = detection_score(A, mask, pct=args.pct)

    sub = pd.DataFrame({'ID': ids,
                        'share_added_vessel': share,
                        'score_added_vessel': score})
    check_submission(sub, len(ids))
    out = ROOT / 'submissions' / 'baseline_analytic.csv'
    sub.to_csv(out, index=False)
    print(f'wrote {out}  rows={len(sub)}')
    print(f'share: mean={share.mean():.4f} p50={np.median(share):.4f} '
          f'max={share.max():.4f} zeros={(share == 0).sum()}')


if __name__ == '__main__':
    main()
