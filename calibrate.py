"""calibrate.py — Record YOUR OWN motor-imagery data to calibrate the model.

WHY THIS EXISTS
---------------
The shipped model (eegnet_hw_subject_3.keras) was trained on BCI-IV-2a
subject 3's brain. It does NOT work on your brain — that's why live tests
were near-random. To get real live control, we record labelled trials of
YOU imagining each movement, then fine-tune the model on this data
(see finetune.py).

PROTOCOL (per trial)
--------------------
    1. "GET READY: <CLASS>"   — 2 s to prepare
    2. beep + "IMAGINE NOW"   — 4 s recording window (imagine the WHOLE time)
    3. "REST"                 — 3 s break

Classes are randomised so the model can't cheat on order.

USAGE
-----
    # 2-class (recommended to start — easiest to get working):
    python calibrate.py --port COM4 --classes left right --trials 20

    # 2-class with a rest/idle state (great for "go vs stop"):
    python calibrate.py --port COM4 --classes left rest --trials 20

    # full 4-class (only if signal is excellent):
    python calibrate.py --port COM4 --classes left right feet tongue --trials 15

OUTPUT
------
    my_calibration.npz  with:
        X           float32 (n_trials, 16, 500)   raw microvolts
        y           int64   (n_trials,)            class index
        class_names list of str
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

# Windows beep is optional — degrade gracefully on other platforms
try:
    import winsound
    def beep(freq=880, ms=200):
        winsound.Beep(freq, ms)
except Exception:
    def beep(freq=880, ms=200):
        print('\a', end='', flush=True)


FS          = 125
WINDOW_SEC  = 4.0
N_SAMPLES   = int(FS * WINDOW_SEC)   # 500
N_CHANNELS  = 16

HW_CHANS = ['FC5', 'FC1', 'FCz', 'FC2', 'FC6',
            'C5',  'C3',  'C1',  'Cz',  'C2',  'C4', 'C6',
            'CP5', 'CP1', 'CP2', 'CP6']

# Map friendly names -> (model class index, on-screen imagery instruction)
# Indices match the model: 0=Left hand, 1=Right hand, 2=Feet, 3=Tongue.
CLASS_SPEC = {
    'left':   (0, 'LEFT HAND  — squeeze an imaginary ball with your LEFT hand'),
    'right':  (1, 'RIGHT HAND — squeeze an imaginary ball with your RIGHT hand'),
    'feet':   (2, 'FEET       — wiggle your toes / press both feet down'),
    'tongue': (3, 'TONGUE     — press your tongue to the roof of your mouth'),
    # Hybrid-BCI deliberate gesture; produces a strong frontal EMG signal.
    'jaw':    (4, 'JAW CLENCH — bite your back teeth firmly for the whole 4 sec'),
    # Synthetic rest label for explicit STOP class.
    'rest':   (9, 'REST       — relax, clear your mind, stare at the dot'),
}


def open_board(port: str):
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    params = BrainFlowInputParams()
    params.serial_port = port
    board = BoardShim(BoardIds.CYTON_DAISY_BOARD, params)
    board.prepare_session()
    board.start_stream()
    eeg_idx = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD)
    return board, eeg_idx


def grab_window(board, eeg_idx) -> np.ndarray | None:
    data = board.get_current_board_data(N_SAMPLES)
    if data.shape[1] < N_SAMPLES:
        return None
    return data[eeg_idx][:N_CHANNELS, -N_SAMPLES:].astype(np.float32)


def countdown(prefix: str, secs: int):
    for s in range(secs, 0, -1):
        print(f'\r  {prefix} {s}...  ', end='', flush=True)
        time.sleep(1.0)
    print('\r' + ' ' * 40 + '\r', end='', flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--port', default='COM4', help='OpenBCI dongle serial port')
    p.add_argument('--classes', nargs='+', default=['left', 'right'],
                   choices=list(CLASS_SPEC.keys()),
                   help='Which classes to record (default: left right)')
    p.add_argument('--trials', type=int, default=20,
                   help='Trials PER CLASS (default 20)')
    p.add_argument('--out', default='my_calibration.npz')
    p.add_argument('--rest-secs', type=float, default=3.0,
                   help='Break between trials (default 3 s)')
    args = p.parse_args()

    # Build a randomised trial list
    trial_order = []
    for cls in args.classes:
        trial_order += [cls] * args.trials
    random.shuffle(trial_order)
    total = len(trial_order)

    print('=' * 60)
    print(' CALIBRATION RECORDING')
    print('=' * 60)
    print(f'  classes : {", ".join(args.classes)}')
    print(f'  trials  : {args.trials} each  ->  {total} total')
    print(f'  time    : ~{total * (2 + WINDOW_SEC + args.rest_secs) / 60:.0f} min')
    print()
    print('  TIPS:  sit still · hands in lap · DON\'T blink during "IMAGINE"')
    print('         imagine the FEELING of moving, not a picture of it')
    print('=' * 60)
    input('\n  Put the headset on, get comfortable, then press ENTER to start...')

    print('\n  Opening board...')
    try:
        board, eeg_idx = open_board(args.port)
    except Exception as e:
        sys.exit(f'  Could not open board on {args.port}: {e}')
    time.sleep(2.0)  # let the stream fill

    X, y = [], []
    # Re-map the synthetic 'rest' index (9) down to a contiguous label set
    used = sorted({CLASS_SPEC[c][0] for c in args.classes})
    relabel = {orig: i for i, orig in enumerate(used)}
    class_names = [None] * len(used)
    for c in args.classes:
        orig = CLASS_SPEC[c][0]
        class_names[relabel[orig]] = c

    try:
        for n, cls in enumerate(trial_order, 1):
            orig_idx, instruction = CLASS_SPEC[cls]
            label = relabel[orig_idx]

            print(f'\n[{n}/{total}]  >>> {instruction}')
            countdown('get ready', 2)

            beep(880, 180)
            print('  *** IMAGINE NOW *** (keep going the whole time)')
            time.sleep(WINDOW_SEC)        # accumulate exactly one window
            win = grab_window(board, eeg_idx)
            beep(440, 120)

            if win is None:
                print('  ! not enough samples, skipping this trial')
                continue
            X.append(win)
            y.append(label)

            print('  rest...')
            time.sleep(args.rest_secs)

    except KeyboardInterrupt:
        print('\n\n  Stopped early — saving what we have.')
    finally:
        try:
            board.stop_stream()
            board.release_session()
        except Exception:
            pass

    if not X:
        sys.exit('\n  No trials recorded. Nothing saved.')

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    np.savez(args.out, X=X, y=y, class_names=np.array(class_names))

    print('\n' + '=' * 60)
    print(f'  SAVED {len(X)} trials -> {args.out}')
    for i, name in enumerate(class_names):
        print(f'    class {i} = {name:7s}  ({int((y == i).sum())} trials)')
    print('=' * 60)
    print('  Next:  python finetune.py --data my_calibration.npz')
    return 0


if __name__ == '__main__':
    sys.exit(main())
