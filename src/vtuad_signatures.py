"""Extract VTUAD large-vessel signatures (Direction #2, Nov 7).

VTUAD (rebuilt) v1.3.3, Zenodo 23773305, CC BY 4.0 — 509 strict-interval
single-vessel scenario clips, 32 kHz mono WAV, ONC hydrophone, Strait of
Georgia, May 2016-May 2017. Declared by competitors already; we must declare
it too before submitting anything built from it (see README.md).

Method mirrors src/vessel_signatures.py (HearMyShip) so the two pools are
directly comparable:
  * per-second third-octave band energies for the 25 challenge bands
    (31.6 Hz - 7943.3 Hz), rfft on 1-s Hann windows;
  * per-band noise floor (5th percentile over the clip) subtracted, so the
    signature is the *vessel-added* component, not the clip's own ambient;
  * the clip is a multi-minute scenario (2-23 min), not a 20-s CPA clip, so
    the signature is the LOUDEST 60-s window (closest to CPA) — the direct
    analogue of HearMyShip's 20-s CPA clip. Its band totals are embedded in
    1-2 minutes by src/make_synthetic_vtuad.py, exactly like hms_clip.

Only the *relative* spectral-temporal shape is used downstream: signatures
are normalised and rescaled to an exact target share in the energy domain,
so hydrophone / water differences cancel out.

Memory-safe: WAVs are read with scipy.io.wavfile and processed one second
at a time (longest clip is 23 min = 44M samples; the VM has ~8 GB RAM).

Input:  data/external/vtuad/wav/vessel/*.wav  (unzipped strict archive)
         data/external/vtuad/metadata.csv     (clip_uid -> label mapping)
Output: data/external/vtuad_signatures.npz
         ids (clip stem), labels, lengths_m, durations_s,
         sig60 (n,60,25) float32 per-second energies of loudest window,
         band_tot (n,25) float32 = sig60 summed over 60 s.
Usage: python -m src.vtuad_signatures [--limit N]
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import wavfile

from src.common import BANDS

ROOT = Path(__file__).resolve().parent.parent
VT = ROOT / 'data' / 'external' / 'vtuad'
WAVDIR = VT / 'wav' / 'strict_interval_wav' / 'inclusion_4000_exclusion_6000' / 'vessel'

FC = np.array([float(b) for b in BANDS])
LO = FC * 2 ** (-1 / 6)
HI = FC * 2 ** (+1 / 6)
WIN_SEC = 60


def band_energy_per_sec(path):
    """Return (n_sec, 25) mean-square band energies, 60 s at a time.

    Same math as src.vessel_signatures.band_energy_per_sec (rfft, Hann,
    one-sided power, summed over third-octave bands), but vectorised over
    60-s chunks so a 23-minute clip processes in seconds, not minutes.
    """
    fs, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    # normalise integer PCM to [-1, 1]; float WAVs pass through
    if np.issubdtype(data.dtype, np.integer):
        x = data.astype(np.float64) / max(1.0, float(np.iinfo(data.dtype).max))
    else:
        x = data.astype(np.float64)
    n_sec = len(x) // fs
    x = x[:n_sec * fs]
    freqs = np.fft.rfftfreq(fs, 1 / fs)
    # indicator matrix (n_freq, 25): 1 if the freq bin falls in band b
    M = np.zeros((len(freqs), len(FC)), dtype=np.float64)
    for b in range(len(FC)):
        M[(freqs >= LO[b]) & (freqs < HI[b]), b] = 1.0
    win = np.hanning(fs)
    out = np.empty((n_sec, len(FC)), dtype=np.float32)
    for s0 in range(0, n_sec, 60):
        s1 = min(s0 + 60, n_sec)
        chunk = x[s0 * fs:s1 * fs].reshape(-1, fs) * win[None, :]
        X = np.fft.rfft(chunk, axis=1)
        pxx = (np.abs(X) ** 2) / (fs ** 2)
        pxx[:, 1:-1] *= 2.0
        out[s0:s1] = pxx @ M
    return out, fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0,
                    help='process at most N clips (0 = all)')
    args = ap.parse_args()

    meta = pd.read_csv(VT / 'metadata.csv')
    meta = meta[meta['path'].str.startswith('./vessel/') &
                (meta['label'] != 'background')].copy()
    meta['stem'] = meta['path'].str.extract(r'vessel/(\d+)\.wav')[0]
    meta = meta.sort_values('stem')
    if args.limit:
        meta = meta.head(args.limit)
    print(f'{len(meta)} vessel clips (label != background)')

    ids, labels, lengths, durs, sig60s, tots = [], [], [], [], [], []
    skipped = []
    for _, row in meta.iterrows():
        p = WAVDIR / f"{row['stem']}.wav"
        if not p.exists():
            skipped.append((row['stem'], 'missing file'))
            continue
        try:
            E, fs = band_energy_per_sec(p)
        except Exception as e:
            skipped.append((row['stem'], str(e)))
            continue
        if E.shape[0] < WIN_SEC:
            skipped.append((row['stem'], f'too short: {E.shape[0]}s'))
            continue
        floor = np.percentile(E, 5, axis=0, keepdims=True)
        sig = np.clip(E - floor, 0, None)
        # loudest 60-s window (broadband energy of floor-subtracted sig)
        bb = sig.sum(axis=1)
        csum = np.concatenate([[0.0], np.cumsum(bb)])
        wsum = csum[WIN_SEC:] - csum[:-WIN_SEC]
        start = int(np.argmax(wsum))
        s60 = sig[start:start + WIN_SEC]
        ids.append(row['stem'])
        labels.append(row['label'])
        lengths.append(float(row['length']))
        durs.append(float(row['duration_sec']))
        sig60s.append(s60.astype(np.float32))
        tots.append(s60.sum(axis=0).astype(np.float32))
        if len(ids) % 50 == 0:
            print(f'  {len(ids)} clips done', flush=True)

    sig60s = np.stack(sig60s)
    tots = np.stack(tots)
    out = ROOT / 'data' / 'external' / 'vtuad_signatures.npz'
    np.savez_compressed(out, ids=np.array(ids), labels=np.array(labels),
                        lengths_m=np.array(lengths, dtype=np.float32),
                        durations_s=np.array(durs, dtype=np.float32),
                        sig60=sig60s, band_tot=tots)
    print(f'wrote {out} {sig60s.shape} band_tot {tots.shape}')
    print('labels:', pd.Series(labels).value_counts().to_dict())
    large_lengths = [l for l, lab in zip(lengths, labels)
                     if lab in ("cargo", "tanker", "passengership", "dredger", "other")]
    if large_lengths:
        print(f'lengths: large-vessel mean={np.mean(large_lengths):.1f} m '
              f'min={np.min(large_lengths):.0f} max={np.max(large_lengths):.0f}')
    if skipped:
        print(f'skipped {len(skipped)}: {skipped[:10]}')


if __name__ == '__main__':
    main()
