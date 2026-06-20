"""check_calibration.py — diagnose calibration data quality before training.

Reports:
  - trial counts per class
  - per-channel RMS (spots railed/dead channels)
  - per-class log-band-power on key motor channels (mu 8-13, beta 13-30)
  - class-vs-rest separation z-score (the real signal we care about)
  - overall verdict
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy import signal as sig

FS = 125
HW_CHANS = ['FC5', 'FC1', 'FCz', 'FC2', 'FC6',
            'C5',  'C3',  'C1',  'Cz',  'C2',  'C4', 'C6',
            'CP5', 'CP1', 'CP2', 'CP6']
KEY_CHANS = ['C3', 'Cz', 'C4', 'FC1', 'FC2', 'FCz']  # motor + frontal (jaw)
MU   = (8, 13)
BETA = (13, 30)

def bandpower(x, band, fs=FS):
    sos = sig.butter(5, band, btype='band', fs=fs, output='sos')
    y = sig.sosfiltfilt(sos, x, axis=-1)
    return np.mean(y ** 2, axis=-1)   # per channel

def main(path):
    d = np.load(path, allow_pickle=True)
    X = d['X']                # (n, 16, T)
    y = d['y']
    cls = list(d['class_names']) if 'class_names' in d.files else \
          [str(i) for i in sorted(set(y))]
    n, n_ch, T = X.shape
    print(f'File: {path}')
    print(f'Shape: {X.shape}   classes: {cls}')
    print()
    for i, c in enumerate(cls):
        print(f'  class {i} ({c:7s}): {int((y == i).sum())} trials')
    print()

    # ── per-channel RMS (raw microvolts) ───────────────────────────────
    rms = np.sqrt(np.mean(X ** 2, axis=(0, 2)))   # average across trials and time
    print('PER-CHANNEL RMS (raw uV)')
    print('  channel    rms     status')
    for i, name in enumerate(HW_CHANS):
        r = rms[i]
        if   r > 80_000: tag = 'RAILED'
        elif r > 200:    tag = 'NOISY'
        elif r < 0.3:    tag = 'DEAD'
        elif r > 50:     tag = 'high'
        else:            tag = 'OK'
        print(f'   {name:4s}   {r:9.2f}   {tag}')
    print()

    # ── per-class band power on KEY channels ───────────────────────────
    print('PER-CLASS LOG BAND POWER (lower = stronger ERD = better imagery)')
    print('Key channels:', KEY_CHANS)
    print()
    key_idx = [HW_CHANS.index(c) for c in KEY_CHANS]
    for band_name, band in [('mu (8-13)', MU), ('beta (13-30)', BETA)]:
        print(f'  {band_name}:')
        header = '    class         ' + '  '.join(f'{c:>7s}' for c in KEY_CHANS)
        print(header)
        for i, c in enumerate(cls):
            trials = X[y == i][:, key_idx, :]
            bp = bandpower(trials, band)        # (n_trials, n_key)
            logbp = np.log(bp + 1e-12).mean(axis=0)
            print(f'    {c:7s}       ' + '  '.join(f'{v:7.2f}' for v in logbp))
        print()

    # Always compute log-band-power features (used by both blocks below)
    all_feats = []
    for i in range(n):
        mu_p = np.log(bandpower(X[i, key_idx, :], MU) + 1e-12)
        be_p = np.log(bandpower(X[i, key_idx, :], BETA) + 1e-12)
        all_feats.append(np.concatenate([mu_p, be_p]))
    all_feats = np.stack(all_feats)

    # ── class separation: distance between each class and rest ────────
    if 'rest' in cls:
        print('CLASS-VS-REST SEPARATION (z-score in mu+beta on KEY channels)')
        print('  >= 0.5 = real signal,  >= 1.0 = strong,  ~0 = noise')
        print()
        rest_id = cls.index('rest')
        rest_mean = all_feats[y == rest_id].mean(axis=0)
        rest_std  = all_feats[y == rest_id].std(axis=0) + 1e-9
        for i, c in enumerate(cls):
            if i == rest_id: continue
            cls_mean = all_feats[y == i].mean(axis=0)
            zs = (cls_mean - rest_mean) / rest_std
            print(f'  {c:7s} vs rest:  max|z|={np.max(np.abs(zs)):.2f}   '
                  f'mean|z|={np.mean(np.abs(zs)):.2f}')

    # ── pairwise class separation (all pairs) ─────────────────────────
    print()
    print('PAIRWISE FISHER RATIO (higher = more separable)')
    print('  >= 0.3 = decent,  >= 0.5 = strong,  < 0.1 = barely distinguishable')
    print()
    for i in range(len(cls)):
        for j in range(i + 1, len(cls)):
            a = all_feats[y == i]
            b = all_feats[y == j]
            # univariate Fisher per feature, then take the max — captures best separator
            fisher = (a.mean(0) - b.mean(0)) ** 2 / (a.var(0) + b.var(0) + 1e-9)
            print(f'  {cls[i]:7s} vs {cls[j]:7s}:  '
                  f'max F={fisher.max():.3f}   mean F={fisher.mean():.3f}')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else 'cal_am.npz'))
