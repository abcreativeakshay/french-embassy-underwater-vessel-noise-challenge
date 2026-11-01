"""Shared helpers for the Underwater Vessel Noise Challenge.

Test windows: 15 minute-positions x 25 third-octave bands (31.6 Hz - 7943.3 Hz),
values in dB of mean-square pressure per minute per band (see starter notebook
`minute_band_power`). Blanks in the CSV are NaN and mean "no observation",
NOT silence - never fill them with 0 dB.

Target: share_added_vessel = E_added / E_total over the whole window, computed
in the ENERGY domain E = 10^(dB/10). score_added_vessel is diagnostic only.
"""
from pathlib import Path
import numpy as np
import pandas as pd

N_MIN, N_BAND = 15, 25
BANDS = ['31.6', '39.8', '50.1', '63.1', '79.4', '100', '125.9', '158.5',
         '199.5', '251.2', '316.2', '398.1', '501.2', '631', '794.3', '1000',
         '1258.9', '1584.9', '1995.3', '2511.9', '3162.3', '3981.1', '5011.9',
         '6309.6', '7943.3']
BAND_COLS = [f'TOL_{b}' for b in BANDS]
SEED = 42


def db_to_energy(db):
    """dB -> linear energy. NaN stays NaN."""
    return np.power(10.0, np.asarray(db, dtype=np.float64) / 10.0)


def feature_columns(df):
    return [c for c in df.columns if c != 'ID']


def load_test(path):
    """Return (ids, A[n,15,25] float64 dB, mask[n,15,25] bool valid)."""
    df = pd.read_csv(path, dtype={'ID': str})
    cols = feature_columns(df)
    assert len(cols) == N_MIN * N_BAND, len(cols)
    A = df[cols].to_numpy(dtype=np.float64).reshape(-1, N_MIN, N_BAND)
    mask = np.isfinite(A)
    return df['ID'].to_numpy(), A, mask


def window_background_energy(A, mask, pct=10):
    """Per-window, per-band background in ENERGY domain.

    Background is persistent, vessel passages are transient, so the background
    of band b is a low percentile over the 15 minute positions. NaN-aware.
    Returns bg[n,25] energy; NaN where a band has no valid minutes.
    """
    E = db_to_energy(A)
    E = np.where(mask, E, np.nan)
    bg = np.nanpercentile(E, pct, axis=1)  # (n, 25)
    return bg


def analytic_share(A, mask, pct=10):
    """Stage B heuristic: share of window energy above per-band background.

    share = sum(max(E_obs - bg_b, 0)) / sum(E_obs) over valid cells only.
    Returns share[n] clipped to [0,1], plus the per-band background energy.
    """
    bg = window_background_energy(A, mask, pct=pct)          # (n,25)
    E = np.where(mask, db_to_energy(A), 0.0)
    bg_full = np.where(np.isfinite(bg)[:, None, :], bg[:, None, :], 0.0)
    valid_E = np.where(mask, E, 0.0)
    excess = np.clip(valid_E - np.where(mask, bg_full, 0.0), 0.0, None)
    tot = valid_E.sum(axis=(1, 2))
    exc = excess.sum(axis=(1, 2))
    share = np.divide(exc, tot, out=np.zeros_like(exc), where=tot > 0)
    return np.clip(share, 0.0, 1.0), bg


def detection_score(A, mask, pct=10):
    """Diagnostic evidence score in [0,1]: broadband transient strength.

    Max over minutes of (broadband energy - median minute broadband energy),
    expressed as a fraction of the window's total energy, squashed to [0,1].
    Not used for ranking.
    """
    E = np.where(mask, db_to_energy(A), 0.0)
    minute_E = E.sum(axis=2)                                  # (n,15)
    minute_valid = mask.any(axis=2)
    med = np.array([np.median(m[v]) if v.any() else 0.0
                    for m, v in zip(minute_E, minute_valid)])
    peak = np.array([m[v].max() if v.any() else 0.0
                     for m, v in zip(minute_E, minute_valid)])
    tot = E.sum(axis=(1, 2))
    frac = np.divide(np.clip(peak - med, 0, None), tot,
                     out=np.zeros_like(tot), where=tot > 0)
    # squash: 0 -> 0, typical transient fractions (~1e-3..1e-1) -> mid range
    score = 1.0 - np.exp(-30.0 * frac)
    return np.clip(score, 0.0, 1.0)


def check_submission(df, n_expected):
    assert df.columns.tolist() == ['ID', 'share_added_vessel', 'score_added_vessel']
    assert len(df) == n_expected and df['ID'].is_unique
    vals = df[['share_added_vessel', 'score_added_vessel']].to_numpy(float)
    assert np.isfinite(vals).all() and (vals >= 0).all() and (vals <= 1).all()
