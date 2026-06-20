"""train_classical.py — FBCSP + LDA trained on YOUR calibration data.

Why this works better than fine-tuning EEGNet on small data
-----------------------------------------------------------
CSP has the right inductive bias for motor imagery baked in: it learns
spatial filters that maximise the variance ratio between classes, which
is exactly what mu/beta ERD looks like. With 50-100 trials per class,
FBCSP+LDA reliably beats a CNN trained from scratch (which would just
overfit).

USAGE
-----
    # train on one session
    python train_classical.py --data my_calibration.npz

    # concatenate multiple sessions (recommended)
    python train_classical.py --data am.npz pm.npz --out my_classical.joblib

    # override bad-channel masking (default: C4 C6 CP5 CP1)
    python train_classical.py --data my_calibration.npz \
        --bad-channels C4 C6 CP5 CP1

OUTPUT
------
    my_classical.joblib            FBCSP extractor + LDA classifier + metadata
    my_classical.classes.npy       class-name list (live-test reads this)
    my_classical.bad_channels.npy  bad-channel list (live-test reads this)

It prints 5-fold cross-validated accuracy + confusion matrix.
Rule of thumb for a live demo:
    > 75%  : great
    65-75% : usable; tune the confidence gate
    < 60%  : need more/cleaner trials
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from mne.decoding import CSP
from scipy import signal as sig
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

FS = 125
HW_CHANS = ['FC5', 'FC1', 'FCz', 'FC2', 'FC6',
            'C5',  'C3',  'C1',  'Cz',  'C2',  'C4', 'C6',
            'CP5', 'CP1', 'CP2', 'CP6']

# Filter bank — four sub-bands inside the mu+beta MI range
BANDS = [(8, 12), (12, 16), (16, 22), (22, 30)]
N_CSP = 4   # components per band  ->  4*4 = 16 features


# ── preprocessing ───────────────────────────────────────────────────────────
def bandpass(x, low, high, fs=FS, order=5):
    sos = sig.butter(order, [low, high], btype='band', fs=fs, output='sos')
    return sig.sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def apply_car_masked_batch(X, bad_idx):
    """X: (n_trials, n_channels, T). CAR over good channels only; zero the bad ones."""
    out = X.copy()
    good = np.ones(X.shape[1], dtype=bool)
    good[bad_idx] = False
    mean = out[:, good, :].mean(axis=1, keepdims=True)
    out = out - mean
    out[:, bad_idx, :] = 0.0
    return out


def drop_bad(X, bad_idx):
    """Drop bad channel rows. CSP needs full-rank covariance; zero rows break it."""
    keep = [i for i in range(X.shape[1]) if i not in bad_idx]
    return X[:, keep, :], keep


# ── FBCSP feature extractor ─────────────────────────────────────────────────
class FBCSPExtractor:
    """Filter-bank CSP. One CSP per sub-band, log-variance features concatenated."""

    def __init__(self, bands=BANDS, n_components=N_CSP):
        self.bands = bands
        self.n_components = n_components
        self.csps = []

    def fit(self, X, y):
        self.csps = []
        for low, high in self.bands:
            Xf = np.stack([bandpass(t, low, high) for t in X]).astype(np.float64)
            csp = CSP(n_components=self.n_components, reg='ledoit_wolf',
                      log=True, transform_into='average_power')
            csp.fit(Xf, y)
            self.csps.append(csp)
        return self

    def transform(self, X):
        feats = []
        for (low, high), csp in zip(self.bands, self.csps):
            Xf = np.stack([bandpass(t, low, high) for t in X]).astype(np.float64)
            feats.append(csp.transform(Xf))
        return np.concatenate(feats, axis=1).astype(np.float32)


# ── main ────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data', nargs='+', required=True,
                   help='one or more calibration .npz files; concatenated')
    p.add_argument('--out', default='my_classical.joblib')
    p.add_argument('--bad-channels', nargs='*',
                   default=['C4', 'C6', 'CP5', 'CP1'],
                   help='Channels to mask before CSP (default: C4 C6 CP5 CP1)')
    p.add_argument('--exclude-classes', nargs='*', default=[],
                   help='Drop these classes from the data before training '
                        '(e.g. --exclude-classes rest)')
    p.add_argument('--cv', type=int, default=5)
    args = p.parse_args()

    unknown = [c for c in args.bad_channels if c not in HW_CHANS]
    if unknown:
        sys.exit(f'unknown channel(s): {unknown}\n  valid: {HW_CHANS}')
    bad_idx = [HW_CHANS.index(c) for c in args.bad_channels]
    print(f'Masking bad channels: {args.bad_channels}  (indices {bad_idx})')

    # ── load + concatenate sessions ─────────────────────────────────────────
    Xs, ys, names_ref = [], [], None
    for path in args.data:
        d = np.load(path, allow_pickle=True)
        Xs.append(d['X'])
        ys.append(d['y'])
        names = list(d['class_names']) if 'class_names' in d.files else None
        if names_ref is None:
            names_ref = names
        elif names != names_ref:
            sys.exit(f'class_names mismatch between sessions:\n'
                     f'  {path}: {names}\n  earlier: {names_ref}')
    X_raw = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys)
    class_names = names_ref

    if args.exclude_classes:
        unknown = [c for c in args.exclude_classes if c not in class_names]
        if unknown:
            sys.exit(f'unknown class(es) to exclude: {unknown}\n'
                     f'  available: {class_names}')
        keep_old = [i for i, n in enumerate(class_names) if n not in args.exclude_classes]
        mask = np.isin(y, keep_old)
        X_raw, y = X_raw[mask], y[mask]
        remap = {old: new for new, old in enumerate(keep_old)}
        y = np.array([remap[v] for v in y], dtype=np.int64)
        class_names = [class_names[i] for i in keep_old]
        print(f'\nExcluded classes: {args.exclude_classes}')

    n_classes = len(class_names)
    print(f'\nLoaded {len(X_raw)} trials, {n_classes} classes: {class_names}')
    for i, name in enumerate(class_names):
        print(f'  class {i} = {name:7s}  ({int((y == i).sum())} trials)')

    if any(int((y == i).sum()) < 10 for i in range(n_classes)):
        print('\n  WARNING: some classes have <10 trials. Accuracy will be noisy.')

    # ── preprocess: CAR-masked → drop bad ───────────────────────────────────
    X = apply_car_masked_batch(X_raw, bad_idx)
    X, kept = drop_bad(X, bad_idx)
    kept_names = [HW_CHANS[i] for i in kept]
    print(f'\nKept {len(kept)}/{len(HW_CHANS)} channels: {kept_names}')

    # ── cross-validation ────────────────────────────────────────────────────
    print(f'\n{args.cv}-fold cross-validation:')
    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=0)
    accs, all_pred, all_true = [], [], []
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        fbcsp = FBCSPExtractor()
        fbcsp.fit(X[tr], y[tr])
        Ftr = fbcsp.transform(X[tr])
        Fte = fbcsp.transform(X[te])
        clf = LinearDiscriminantAnalysis(shrinkage='auto', solver='lsqr')
        clf.fit(Ftr, y[tr])
        pred = clf.predict(Fte)
        acc = float((pred == y[te]).mean())
        accs.append(acc)
        all_pred.append(pred)
        all_true.append(y[te])
        print(f'  fold {fold}: {acc:.1%}')
    cv_mean, cv_std = float(np.mean(accs)), float(np.std(accs))
    print(f'\n  CV accuracy: {cv_mean:.1%} ± {cv_std:.1%}')

    # ── confusion matrix (row-normalised) ──────────────────────────────────
    cm = confusion_matrix(np.concatenate(all_true), np.concatenate(all_pred),
                          labels=list(range(n_classes)))
    print('\n  Confusion matrix (rows=true, cols=pred):')
    print('             ' + '  '.join(f'{n:>7s}' for n in class_names))
    for i, name in enumerate(class_names):
        s = cm[i].sum() or 1
        row = cm[i] / s
        print(f'    {name:7s}  ' + '  '.join(f'{v:6.0%}' for v in row))

    # ── train final model on ALL data ──────────────────────────────────────
    print('\nTraining final model on all data...')
    fbcsp = FBCSPExtractor()
    fbcsp.fit(X, y)
    F = fbcsp.transform(X)
    clf = LinearDiscriminantAnalysis(shrinkage='auto', solver='lsqr')
    clf.fit(F, y)

    pkg = {
        'fbcsp':         fbcsp,
        'clf':           clf,
        'class_names':   class_names,
        'bad_channels':  args.bad_channels,
        'kept_channels': kept_names,
        'cv_accuracy':   cv_mean,
        'cv_std':        cv_std,
        'fs':            FS,
        'bands':         BANDS,
        'n_csp':         N_CSP,
    }
    joblib.dump(pkg, args.out)
    np.save(Path(args.out).with_suffix('.classes.npy'), np.array(class_names))
    np.save(Path(args.out).with_suffix('.bad_channels.npy'),
            np.array(args.bad_channels))

    print('\n' + '=' * 60)
    print(f'  saved -> {args.out}')
    print(f'  saved -> {Path(args.out).with_suffix(".classes.npy")}')
    print(f'  saved -> {Path(args.out).with_suffix(".bad_channels.npy")}')
    print('=' * 60)
    print(f'  Next:  python live-test-classical.py --model {args.out} --port COM4')
    return 0


if __name__ == '__main__':
    sys.exit(main())
