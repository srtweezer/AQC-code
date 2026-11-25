"""Blade centres of the tweezer array at the two tones of a handoff pair, and the pair mismatch between them.

A pair is the transport channel at tone A and at tone B one site lower: the two tones the
Shepard crossfade plays together (aqc.shepard). Every tone is shifted by CH3 offsets of
-span..span MHz (CH1 at its tone), then by the same CH1 offsets (CH3 at 70 MHz), in
shuffled order, with the A and B frames of each offset taken back to back, so slow drift
is common to both tones and cancels in the mismatch. The spot's shift is J df, J measured
at 70 MHz: the offsets are symmetric, so J's change with the tone enters the centres only
at second order. Offsets with even and odd grid index are fitted separately
(aqc.blade.blade); their difference is the noise. Gaussian fits to an unshifted A and B
frame seed the windows.

Functions take the open AWG (aqc.hardware.awg.aods) and the free-running camera, so
importing this module does not load libspcm.
"""

import numpy as np

from aqc.blade import blade, kept_pixels
from aqc.handoff import mismatch, step_angle_basis
from aqc.intensity import FULL_SCALE
from aqc.steering import fit_array, grab, play_transport
from aqc.traps import SPOT_FRAC


def shepard_pairs(f_center, spacing, positions):
    """Tones (A, B) in Hz, (len(positions), 2), of the continuous Shepard pair at each position through the crossfade.

    In the continuous Shepard centred on f_center every tone chirps up from f_center -
    spacing to f_center + spacing; while tone A runs from f_center to f_center + spacing it
    fades out and B = A - spacing fades in. Position 0 is mid-crossfade (A = f_center +
    spacing / 2), -0.5 and +0.5 are its ends.
    """
    a = f_center + spacing * (0.5 + np.asarray(positions, float))
    return np.stack([a, a - spacing], axis=1)


def scan(span_mhz, step_mhz, J, rng=None):
    """The blade offsets, as a dict: offsets (n, 2) (df3, df1) Hz in shuffled order, half (n,) 0 or 1, shifts (n, 2) (y, x) px.

    J: camera (x, y) px per MHz of (CH3, CH1) (config.AOD_STEERING_CAL or _VERTICAL_CAL).
    step_mhz must be a multiple of the 3.125 kHz tone grid.
    """
    k = np.arange(-round(span_mhz / step_mhz), round(span_mhz / step_mhz) + 1)
    df = k * step_mhz * 1e6
    offsets = np.array([(d, 0.0) for d in df] + [(0.0, d) for d in df[k != 0]])
    half = np.concatenate([k % 2, k[k != 0] % 2])
    order = (rng or np.random.default_rng()).permutation(len(offsets))
    offsets, half = offsets[order], half[order]
    return dict(offsets=offsets, half=half, shifts=(offsets / 1e6 @ J.T)[:, ::-1])


def take(awg, cam, tones, vertical, sc, average, nv, nh, spot_frac=SPOT_FRAC):
    """Frames of one pair for fit: a dict of seeds (2, N, 2) (Gaussian centres, camera order, px), the snapped tones (2,) Hz, index and frames.

    tones (A, B) in Hz on the transport channel (CH1 with vertical, else CH3); sc from
    scan; average frames per tone and offset. frames (2, offsets, pixels) hold only the
    pixels a blade window can reach, index maps the sensor onto them (aqc.blade.kept_pixels).
    """
    seeds, snapped = [], []
    for tone in tones:
        snapped.append(play_transport(awg, tone, vertical))
        frame, fit = fit_array(cam, nv, nh, average, spot_frac)
        _check(frame, tone)
        seeds.append(fit[:, :2])
    seeds = np.stack(seeds)
    reach = np.linalg.norm(sc["shifts"], axis=1).max()
    pixels, index = kept_pixels(frame.shape, seeds.reshape(-1, 2), reach)
    frames = np.empty((2, len(sc["offsets"]), len(pixels)), np.float32)
    for i, (df3, df1) in enumerate(sc["offsets"]):
        for t, tone in enumerate(tones):
            play_transport(awg, tone, vertical, df3, df1)
            frame = grab(cam, average)
            _check(frame, tone)
            frames[t, i] = frame.ravel()[pixels]
    return dict(seeds=seeds, snapped=np.array(snapped), index=index, frames=frames)


def _check(frame, tone):
    if frame.max() >= FULL_SCALE:
        raise RuntimeError(f"saturated at the {tone / 1e6:.4f} MHz tone; shorten the exposure")


def fit(m, sc):
    """Blade fits (2 halves, 2 tones, N, 5) of a take: y, x, sigma y, sigma x (px) and light (counts); NaN where its seed is.

    See aqc.blade.blade for the columns.
    """
    out = np.full((2,) + m["seeds"].shape[:2] + (5,), np.nan)
    for h in range(2):
        sel = sc["half"] == h
        for t in range(2):
            frames = m["frames"][t, sel]
            for n, seed in enumerate(m["seeds"][t]):
                if np.isfinite(seed).all():
                    out[h, t, n] = blade(frames, m["index"], seed, sc["shifts"][sel])
    return out


def analyse(c, nv, nh):
    """Pair mismatch of blade fits c (2 halves, 2 tones, N, >= 2) from fit (only the centres are used), as a dict (px).

    centres (2, N, 2): A and B, the mean of the halves; e (nv, nh, 2): their pair mismatch
    (aqc.handoff.mismatch) and dc; s, r: its step and angle basis
    (aqc.handoff.step_angle_basis); floor: the fit noise of the rms of e - <e>,
    rms|D - <D>| / 2 with D the difference of the two halves' mismatch.
    """
    c = c[..., :2]
    (e0, _), (e1, _) = (mismatch(c[h, 0], c[h, 1], nv, nh) for h in range(2))
    d = (e1 - e0) - np.nanmean(e1 - e0, axis=(0, 1))
    centres = c.mean(axis=0)
    e, dc = mismatch(centres[0], centres[1], nv, nh)
    s, r = step_angle_basis(centres[0], centres[1], nv, nh, dc)
    return dict(centres=centres, e=e, dc=dc, s=s, r=r, floor=np.sqrt(np.nanmean(np.sum(d ** 2, axis=-1))) / 2)
