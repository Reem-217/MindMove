"""live-test-classical.py — live test the FBCSP+LDA classical pipeline.

Loads a model trained by train_classical.py and runs it against a live
BrainFlow stream from your Cyton+Daisy. Uses the SAME preprocessing as
training (CAR over good channels, drop bad channels, FBCSP) — anything
else would silently destroy accuracy.

USAGE
-----
    # classify only (print predictions)
    python live-test-classical.py --model my_classical.joblib --port COM4

    # classify + send commands to ESP32 over WiFi
    python live-test-classical.py --model my_classical.joblib --port /dev/ttyUSB0 \
        --esp-ip 192.168.1.50

The model package itself stores:
    - class names
    - bad channels (so live and training mask the same way)
    - kept channels
    - CV accuracy
We just trust what's in the .joblib — no flags to pass.
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

import joblib
import numpy as np


def send_esp(ip: str, cmd: str, timeout: float = 0.5) -> None:
    """Send a motor command to the ESP32 HTTP server (non-blocking on failure)."""
    try:
        url = f"http://{ip}/cmd?v={cmd}"
        urllib.request.urlopen(url, timeout=timeout)
    except Exception as e:
        print(f"  [ESP] send failed ({cmd}): {e}")

# joblib needs to find FBCSPExtractor — same trick live-test-svm.py uses.
sys.path.insert(0, str(Path(__file__).parent))
from train_classical import (
    HW_CHANS,
    FBCSPExtractor,        # noqa: F401  (imported so unpickling works)
    apply_car_masked_batch,
    drop_bad,
)


FS         = 125
WINDOW_SEC = 4.0
N_SAMPLES  = int(FS * WINDOW_SEC)   # 500

WHEELCHAIR_BY_NAME = {
    'left':   'LEFT',
    'right':  'RIGHT',
    'feet':   'FORWARD',
    'tongue': 'BACKWARD',
    'jaw':    'LEFT',
    'rest':   'STOP',
}

# Audio cue — same fallback pattern as calibrate.py
try:
    import winsound
    def beep(freq=880, ms=150):
        winsound.Beep(freq, ms)
except Exception:
    def beep(freq=880, ms=150):
        print('\a', end='', flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--model', default='my_classical.joblib')
    p.add_argument('--port', default='COM4')
    p.add_argument('-n', '--n-predictions', type=int, default=20)
    p.add_argument('--transition', type=float, default=3.0,
                   help='REST seconds between imagery windows so the brain '
                        'state can switch cleanly (default 3.0)')
    p.add_argument('--gate', type=float, default=0.55,
                   help='confidence threshold for issuing a command (default 0.55)')
    p.add_argument('--no-cue', action='store_true',
                   help='disable the audio beep at the start/end of each window')
    p.add_argument('--esp-ip', default=None,
                   help='ESP32 IP address (e.g. 192.168.1.50). '
                        'If set, sends motor commands via HTTP.')
    args = p.parse_args()

    print(f'Loading {args.model}...')
    pkg = joblib.load(args.model)
    fbcsp       = pkg['fbcsp']
    clf         = pkg['clf']
    class_names = list(pkg['class_names'])
    bad_chans   = list(pkg['bad_channels'])
    n_classes   = len(class_names)
    bad_idx     = [HW_CHANS.index(c) for c in bad_chans]
    wheelchair  = {i: WHEELCHAIR_BY_NAME.get(c, f'CLASS_{i}')
                   for i, c in enumerate(class_names)}

    print(f'  classes      : {class_names}')
    print(f'  bad channels : {bad_chans}')
    print(f'  CV accuracy  : {pkg.get("cv_accuracy", 0):.1%}')
    print(f'  mapping      : ' + '  '.join(f'{c}->{wheelchair[i]}'
                                            for i, c in enumerate(class_names)))
    if args.esp_ip:
        print(f'  ESP32        : http://{args.esp_ip}/cmd')
        print(f'  (testing connection...)')
        try:
            urllib.request.urlopen(f'http://{args.esp_ip}/status', timeout=3)
            print(f'  ESP32 reachable ✓')
        except Exception as e:
            print(f'  WARNING: ESP32 not reachable — {e}')
            print(f'  Commands will be sent but may fail silently.')

    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    params = BrainFlowInputParams()
    params.serial_port = args.port
    board = BoardShim(BoardIds.CYTON_DAISY_BOARD, params)
    board.prepare_session()
    board.start_stream()
    eeg_idx = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD)

    print('\nProtocol per prediction:')
    print(f'  1. REST          {args.transition:.1f}s   (clear your mind)')
    print(f'  2. beep          ->  IMAGINE for {WINDOW_SEC:.1f}s')
    print(f'  3. beep + predict')
    print('\nstreaming... Ctrl+C to stop\n')

    # Let the BrainFlow ring buffer fill for one full window so the first
    # prediction isn't using stale data from before this loop started.
    time.sleep(WINDOW_SEC + 0.5)

    try:
        for i in range(1, args.n_predictions + 1):
            # 1. REST gap — old imagery flushes from the analysis window
            print(f'[{i:2d}/{args.n_predictions}] REST '
                  f'{args.transition:.1f}s ...', end='', flush=True)
            time.sleep(args.transition)

            # 2. Cue + imagery — discard the rest-gap data by waiting one full
            #    WINDOW_SEC after the cue, then grabbing only the most recent
            #    N_SAMPLES (which is exactly this fresh window).
            if not args.no_cue:
                beep(880, 150)
            t_imagery_start = time.time()
            print('  IMAGINE NOW         ', end='\r', flush=True)
            time.sleep(WINDOW_SEC)
            if not args.no_cue:
                beep(440, 120)

            data = board.get_current_board_data(N_SAMPLES)
            if data.shape[1] < N_SAMPLES:
                print(' (not enough samples, skipping)')
                continue
            hw = data[eeg_idx][:16, -N_SAMPLES:].astype(np.float32)

            # Same preprocessing as training: CAR-masked → drop bad → FBCSP
            X = apply_car_masked_batch(hw[np.newaxis, ...], bad_idx)
            X, _ = drop_bad(X, bad_idx)
            feats = fbcsp.transform(X)
            probs = clf.predict_proba(feats)[0]
            pred  = int(np.argmax(probs))
            conf  = float(probs[pred])
            cmd   = wheelchair[pred] if conf >= args.gate else '(below gate)'

            in_energy = float(np.mean(np.abs(hw)))
            print(f'[{i:2d}/{args.n_predictions}] {time.strftime("%H:%M:%S")}'
                  f'  -> {class_names[pred]:7s}  conf={conf:5.1%}  cmd={cmd}')
            print('     ' + '  '.join(f'{class_names[c]}:{probs[c]:4.0%}'
                                       for c in range(n_classes))
                  + f'   |hw|={in_energy:7.2f}')

            if args.esp_ip and cmd != '(below gate)':
                send_esp(args.esp_ip, cmd)

    except KeyboardInterrupt:
        print('\nstopped by user.')
    finally:
        try:
            board.stop_stream()
            board.release_session()
        except Exception:
            pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
