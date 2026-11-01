"""Extract real vessel signatures from HearMyShip 20-sec CPA recordings.

For each WAV (mono, 48 kHz): per-second third-octave band energies for the
25 challenge bands (31.6 Hz - 7943.3 Hz), computed via rfft on 1-s Hann
windows (non-overlapping). A per-band noise floor (5th percentile over the
20 s) is subtracted so the signature represents the *vessel-added* component,
not the clip's own ambient.

Also builds CPA spectral templates from the HearMyShip site metadata
(thirdOctaveData, 30 bands 31.5 Hz-20 kHz -> mapped onto the 25 challenge
bands by nearest log-centre).

Only the *relative* spectral-temporal shape is used downstream: signatures
are normalised and rescaled to an exact target share in the energy domain,
so hydrophone calibration differences cancel out.

Writes:
  data/external/vessel_signatures.npz   (per-clip 20x25 energy signatures)
  data/external/vessel_cpa_templates.npz (per-vessel 25-band CPA spectra)

Usage: python -m src.vessel_signatures
"""
from pathlib import Path
import json
import numpy as np
import wave

from src.common import BANDS

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / 'data' / 'external'
WAVDIR = EXT / 'hearmyship'

# challenge band edges (third-octave, exact centres from common.BANDS)
FC = np.array([float(b) for b in BANDS])
LO = FC * 2 ** (-1 / 6)
HI = FC * 2 ** (+1 / 6)


def band_energy_per_sec(path):
    """Return (n_sec, 25) mean-square band energies for a mono WAV."""
    with wave.open(str(path), 'rb') as w:
        nch, sw, fs, nfr = (w.getnchannels(), w.getsampwidth(),
                            w.getframerate(), w.getnframes())
        raw = w.readframes(nfr)
    if sw == 3:
        # 24-bit little-endian: pad each 3-byte sample to 4 bytes with sign
        u8 = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        x = u8[:, 0] | (u8[:, 1] << 8) | (u8[:, 2] << 16)
        x = np.where(x >= 2 ** 23, x - 2 ** 24, x).astype(np.float64)
        x = x / 2 ** 23
    else:
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
        x = np.frombuffer(raw, dtype=dtype).astype(np.float64)
        x = x / (2 ** (8 * sw - 1))
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
    n_sec = len(x) // fs
    x = x[:n_sec * fs].reshape(n_sec, fs)
    win = np.hanning(fs)
    X = np.fft.rfft(x * win[None, :], axis=1)          # (n_sec, fs/2+1)
    freqs = np.fft.rfftfreq(fs, 1 / fs)
    pxx = (np.abs(X) ** 2) / (fs ** 2)                 # proportional to MS
    pxx[:, 1:-1] *= 2.0                               # one-sided
    out = np.empty((n_sec, len(FC)))
    for b in range(len(FC)):
        m = (freqs >= LO[b]) & (freqs < HI[b])
        out[:, b] = pxx[:, m].sum(axis=1)
    return out.astype(np.float32)


def main():
    wavs = sorted(WAVDIR.glob('*.wav'))
    print(f'{len(wavs)} wav files')
    ids, cats, sigs, durs = [], [], [], []
    for p in wavs:
        try:
            E = band_energy_per_sec(p)
        except Exception as e:
            print('skip', p.name, e)
            continue
        floor = np.percentile(E, 5, axis=0, keepdims=True)
        sig = np.clip(E - floor, 0, None)
        if sig.sum() <= 0:
            print('skip (silent)', p.name)
            continue
        n_sec = sig.shape[0]
        pad = np.zeros((max(0, 20 - n_sec), sig.shape[1]), dtype=np.float32)
        sig20 = np.vstack([sig, pad])[:20]          # uniform (20,25)
        ids.append(p.stem)
        sigs.append(sig20)
        durs.append(n_sec)
    sigs = np.stack(sigs)                              # (n, 20, 25)
    np.savez_compressed(EXT / 'vessel_signatures.npz',
                        ids=np.array(ids), sig=sigs,
                        durations=np.array(durs, dtype=np.int32))
    print('wrote vessel_signatures.npz', sigs.shape)

    # CPA spectral templates from site metadata (all 1149 vessels)
    meta = json.load(open('/tmp/hms_vessels.json'))
    hms_fc = np.array([31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315,
                       400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500,
                       3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000,
                       20000], dtype=float)
    # nearest challenge band for each HMS band
    mapping = np.argmin(np.abs(np.log10(hms_fc)[:, None]
                               - np.log10(FC)[None, :]), axis=1)
    tids, tcats, tmpls = [], [], []
    for v in meta:
        tod = v.get('thirdOctaveData') or {}
        if not tod:
            continue
        E = np.zeros(len(FC))
        for fstr, db in tod.items():
            if db is None:
                continue
            try:
                f = float(fstr)
            except ValueError:
                continue
            i = int(np.argmin(np.abs(hms_fc - f)))
            E[mapping[i]] += 10 ** (db / 10.0)
        if E.sum() <= 0:
            continue
        tids.append(str(v['id']))
        tcats.append(v.get('category', ''))
        tmpls.append(E / E.sum())
    tmpls = np.stack(tmpls).astype(np.float32)
    np.savez_compressed(EXT / 'vessel_cpa_templates.npz',
                        ids=np.array(tids), cats=np.array(tcats),
                        spectra=tmpls)
    print('wrote vessel_cpa_templates.npz', tmpls.shape)


if __name__ == '__main__':
    main()
