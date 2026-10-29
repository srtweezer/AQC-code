"""Steering the tweezer array with CH3: constant tones, and every trap tracked over a CH3 sweep.

Functions take the open AWG (from aqc.hardware.awg.aods) and camera, so importing this
module does not load libspcm.
"""

import numpy as np

from aqc import config
from aqc.tones import multitone
from aqc.traps import fit_all

F_CENTER = 70e6          # Hz, CH1 always; CH3 for the reference frames
NB_SAMPLES = 200_000


def play(awg, f3, scale=1.0):
    """CH1 at 70 MHz, CH3 at f3 (snapped to the 3.125 kHz buffer grid) times scale; returns the snapped f3."""
    ch1, _ = multitone([F_CENTER], NB_SAMPLES, config.AWG_SAMPLING_RATE)
    ch3, (f3,) = multitone([f3], NB_SAMPLES, config.AWG_SAMPLING_RATE)
    awg.stop()
    awg.load_waveform(np.stack([ch1, scale * ch3]), -1)
    awg.force_trigger()
    return f3


def grab(cam, n):
    return np.mean([cam.grab().astype(float) for _ in range(n)], axis=0)


def sweep_ch3(awg, cam, p0, offs, average):
    """Displacement (len(offs), N, 2) in (y, x) px of traps p0 (N, 2), fitted at 70 MHz, at CH3 = 70 + offs MHz.

    Steps outward from 70 MHz on each side, predicting each trap from the last two
    points, so no trap is ever matched to its neighbour. Before every point a 70 MHz
    reference frame is taken and the displacement is measured against it, so slow drift
    of the whole array cancels. NaN where a trap was lost.
    """
    i0 = int(np.argmin(np.abs(offs)))
    track = np.full((len(offs), len(p0), 2), np.nan)
    refs = np.full_like(track, np.nan)
    track[i0] = refs[i0] = p0
    for side in (range(i0 + 1, len(offs)), range(i0 - 1, -1, -1)):
        prev, prev2 = i0, None
        for i in side:
            play(awg, F_CENTER)
            refs[i] = fit_all(grab(cam, average), p0)
            play(awg, F_CENTER + offs[i] * 1e6)
            guess = track[prev] if prev2 is None else (
                track[prev] + (track[prev] - track[prev2]) * (offs[i] - offs[prev]) / (offs[prev] - offs[prev2]))
            track[i] = fit_all(grab(cam, average), guess)
            prev2, prev = prev, i
    return track - refs


def slopes(offs, d):
    """Per-trap steering vector (N, 2), (y, x) px/MHz, from a linear fit; NaN for traps lost in the sweep."""
    ok = np.all(np.isfinite(d), axis=(0, 2))
    S = np.full((d.shape[1], 2), np.nan)
    S[ok] = np.stack([np.polyfit(offs, d[:, ok, j], 1)[0] for j in range(2)], axis=1)
    return S
