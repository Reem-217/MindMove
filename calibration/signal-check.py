"""signal-check.py — Live signal viewer for OpenBCI Cyton+Daisy.

Replaces the OpenBCI GUI for the signal-quality check step.
Streams 16 channels via BrainFlow, plots them live, and reports per-channel
RMS so you can spot bad electrodes immediately.

Usage:
    python signal-check.py --port COM3
    python signal-check.py --port COM3 --duration 60     # run for 60 s then quit
    python signal-check.py --simulate                    # no hardware needed

What to look for:
  *  Good channel  : wavy line ±50 µV, RMS roughly 5-30 µV
  *  Railed        : flat line at extreme values, RMS > 500 — bad contact
  *  Flat / dead   : RMS < 1 — electrode disconnected
  *  Blink test    : blink hard → ALL 16 channels show a big spike
  *  Alpha test    : close eyes 10 s → posterior channels (CP*) show 8-12 Hz bump
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal as scipy_sig


HW_CHANS = ['FC5', 'FC1', 'FCz', 'FC2', 'FC6',
            'C5',  'C3',  'C1',  'Cz',  'C2',  'C4', 'C6',
            'CP5', 'CP1', 'CP2', 'CP6']

FS_TARGET     = 125
WINDOW_SEC    = 4.0
N_CHANNELS    = 16


def open_board(port: str, simulate: bool, cyton_only: bool = False):
    """Returns (board, fs, eeg_idx). board is None in simulate mode."""
    if simulate:
        return None, FS_TARGET, list(range(N_CHANNELS))

    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    board_id = BoardIds.CYTON_BOARD if cyton_only else BoardIds.CYTON_DAISY_BOARD
    params = BrainFlowInputParams()
    params.serial_port = port
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()
    fs = BoardShim.get_sampling_rate(board_id)
    eeg_idx = BoardShim.get_eeg_channels(board_id)
    return board, fs, eeg_idx


def grab_window(board, eeg_idx, n_samples: int, simulate: bool) -> np.ndarray:
    """Return latest (16, n_samples) microvolt array."""
    if simulate:
        t = np.arange(n_samples) / FS_TARGET
        sig = np.zeros((N_CHANNELS, n_samples), dtype=np.float32)
        for c in range(N_CHANNELS):
            sig[c] = 20 * np.sin(2 * np.pi * 10 * t + c)
            sig[c] += 5 * np.random.randn(n_samples)
        return sig

    data = board.get_current_board_data(n_samples)
    if data.shape[1] < n_samples:
        return None
    return data[eeg_idx][:N_CHANNELS, -n_samples:].astype(np.float32)


def channel_health(rms: float) -> str:
    if rms > 500:    return 'RAILED'
    if rms > 100:    return 'noisy'
    if rms < 1:      return 'DEAD  '
    if rms < 3:      return 'low   '
    return 'OK    '


def bandpass_rms(eeg: np.ndarray, fs: int, low: float = 8.0, high: float = 30.0) -> np.ndarray:
    """RMS of each channel after the SAME 8-30 Hz band the model uses.

    Raw RMS includes 50 Hz mains hum + slow drift that the model's bandpass
    throws away. This shows the in-band power that actually reaches the model.
    """
    sos = scipy_sig.butter(4, [low, high], btype='band', fs=fs, output='sos')
    filt = scipy_sig.sosfiltfilt(sos, eeg, axis=1)
    return np.sqrt((filt ** 2).mean(axis=1))


def band_health(rms: float) -> str:
    """Health from in-band (8-30 Hz) RMS — the signal quality the model sees."""
    if rms > 80:    return 'BAD   '
    if rms > 35:    return 'high  '
    if rms < 0.3:   return 'DEAD  '
    return 'OK    '


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--port', default='COM3',
                   help='Serial port for OpenBCI dongle (Windows: COM3, Linux: /dev/ttyUSB0)')
    p.add_argument('--simulate', action='store_true',
                   help='No hardware — generate fake EEG to test the viewer')
    p.add_argument('--duration', type=float, default=None,
                   help='Auto-stop after N seconds (default: run until Ctrl+C)')
    p.add_argument('--cyton-only', action='store_true',
                   help='Diagnostic: connect as plain Cyton (8 ch), ignore Daisy. '
                        'If this works but full mode does not, the Daisy is the problem.')
    args = p.parse_args()

    print('Opening board...')
    board, fs, eeg_idx = open_board(args.port, args.simulate, args.cyton_only)
    print(f'  fs={fs} Hz   eeg_channels={len(eeg_idx)}   '
          f'mode={"sim" if args.simulate else ("cyton-only" if args.cyton_only else "live 16ch")}')
    n_samples = int(WINDOW_SEC * fs)

    fig, axes = plt.subplots(8, 2, figsize=(12, 9), sharex=True)
    axes = axes.flatten()
    lines = []
    for i, ax in enumerate(axes):
        line, = ax.plot(np.zeros(n_samples), lw=0.7)
        ax.set_ylim(-100, 100)
        ax.set_ylabel(HW_CHANS[i], rotation=0, ha='right', va='center', fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.3)
        lines.append(line)
    axes[-1].set_xlabel('samples')
    axes[-2].set_xlabel('samples')
    fig.suptitle('OpenBCI Cyton+Daisy — Live Signal (close window to quit)', fontsize=11)
    fig.tight_layout()

    t_start = time.time()
    last_report = 0.0

    def update(_frame):
        nonlocal last_report
        eeg = grab_window(board, eeg_idx, n_samples, args.simulate)
        if eeg is None:
            return lines

        # detrend + clip for display
        eeg_disp = eeg - eeg.mean(axis=1, keepdims=True)
        n_ch = eeg_disp.shape[0]
        for i, line in enumerate(lines):
            if i < n_ch:
                line.set_ydata(np.clip(eeg_disp[i], -200, 200))

        # adaptive y-lim once data starts flowing
        peak = np.percentile(np.abs(eeg_disp), 95)
        if peak > 5:
            for ax in axes:
                ax.set_ylim(-peak * 1.5, peak * 1.5)

        # console RMS report every 2 s
        now = time.time()
        if now - last_report > 2.0:
            last_report = now
            rms_raw  = np.sqrt((eeg_disp ** 2).mean(axis=1))
            rms_band = bandpass_rms(eeg_disp, fs)
            print(f'\n[t={now-t_start:5.1f}s] per-channel RMS (uV):')
            print('   ch name    raw_rms (status)  |  8-30Hz (model sees)')
            for i in range(n_ch):
                ch = HW_CHANS[i] if i < len(HW_CHANS) else f'ch{i+1}'
                print(f'  {i+1:2d} {ch:4s} {rms_raw[i]:8.1f}  {channel_health(rms_raw[i])}  | '
                      f'{rms_band[i]:7.1f}  {band_health(rms_band[i])}')

        if args.duration and (now - t_start) >= args.duration:
            plt.close(fig)
        return lines

    ani = FuncAnimation(fig, update, interval=200, blit=False, cache_frame_data=False)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        if board is not None:
            board.stop_stream()
            board.release_session()
            print('\nBoard released.')

    return 0


if __name__ == '__main__':
    sys.exit(main())
