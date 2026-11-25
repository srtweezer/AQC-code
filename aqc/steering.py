"""Steering the tweezer array with the AODs: constant tones, and every trap tracked over a sweep of CH3 or CH1.

Functions take the open AWG (from aqc.hardware.awg.aods) and camera, so importing this
module does not load libspcm.
"""

import numpy as np

from aqc import config
from aqc.aod_amplitude import ch1_scale, ch3_scale
from aqc.tones import multitone
from aqc.traps import SPOT_FRAC, find_traps, fit_all, fit_spots

F_CENTER = 70e6          # Hz, CH1 always; CH3 for the reference frames
NB_SAMPLES = 200_000


def play(awg, f3, scale=1.0, f1=F_CENTER, scale1=1.0):
    """CH1 at f1 times scale1, CH3 at f3 times scale (both snapped to the 3.125 kHz buffer grid); returns the snapped f3."""
    ch1, _ = multitone([f1], NB_SAMPLES, config.AWG_SAMPLING_RATE)
    ch3, (f3,) = multitone([f3], NB_SAMPLES, config.AWG_SAMPLING_RATE)
    awg.stop()
    awg.load_waveform(np.stack([scale1 * ch1, scale * ch3]), -1)
    awg.force_trigger()
    return f3


def grab(cam, n):
    return np.mean([cam.grab().astype(float) for _ in range(n)], axis=0)


def play_transport(awg, f, vertical, df3=0.0, df1=0.0):
    """Transport channel at f (Hz), flattened by aqc.aod_amplitude, the other at 70 MHz unscaled; returns the snapped f.

    CH1 carries the vertical 90x10 array, CH3 the horizontal one. df3, df1 (Hz, multiples
    of the 3.125 kHz grid) shift CH3 and CH1 on top, as the blade scan does; the
    flattening follows the shifted tone.
    """
    if vertical:
        play(awg, F_CENTER + df3, f1=f + df1, scale1=float(ch1_scale(f + df1)))
        return multitone([f], NB_SAMPLES, config.AWG_SAMPLING_RATE)[1][0]
    return play(awg, f + df3, float(ch3_scale(f + df3)), f1=F_CENTER + df1) - df3


def fit_array(cam, nv, nh, average, spot_frac=SPOT_FRAC):
    """Averaged frame and the Gaussian fits (nv * nh, 5) (aqc.traps.fit_spots) of its traps, in camera order.

    spot_frac: see aqc.traps.find_traps (an unhomogenized vertical array needs ~0.1).
    """
    frame = grab(cam, average)
    pts, r, c = find_traps(frame, nv, nh, spot_frac)
    guess = np.empty((nv, nh, 2))
    guess[r, c] = pts
    return frame, fit_spots(frame, guess.reshape(-1, 2))


def sweep(awg, cam, p0, offs, average, channel):
    """Displacement (len(offs), N, 2) in (y, x) px of traps p0 (N, 2), fitted at 70 MHz, at CH<channel> = 70 + offs MHz.

    channel is 3 or 1; the other channel stays at 70 MHz.

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
            f = F_CENTER + offs[i] * 1e6
            if channel == 3:
                play(awg, f)
            else:
                play(awg, F_CENTER, f1=f)
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
