"""Column (or row) spacing of the tweezer array, in tweezer waists, for SLM patterns at several CH3 frequencies.

Setup: SLM pattern with a blaze (the usual 8 px along y unless --blaze-y/--blaze-x),
both AODs on CW tones (CH1 at 70 MHz, CH3 at each --f3), Basler free-running (no hardware trigger). Every spot is
fitted on the raw camera pixels with a Gaussian whose axes are fixed along and across the
rows (aqc.traps.row_fit gives the angle). Resampling onto the row grid first, as
spot_shape.py does, would broaden these ~2 px sigma spots by the linear interpolation
(about 8% here). The spacing of each pair of
neighbouring columns is their centre distance along the row divided by the mean waist
w = 2 sigma_along (1/e^2 intensity radius) of the two spots, so it measures how well
neighbouring tweezers are separated in units of their own size. With --px the plain centre
distance in camera px is plotted instead, which shows the pitch chirp the handoff calibration
copies from the CH3 steering map without the waist variation along the row. With --rows the same is
done across the rows: the distance between rows r and r+1 across the rows, over the mean
w = 2 sigma_across of the two spots.

With --blade SPAN STEP the centres and widths come from the blade method instead
(docs/blade_method.md, aqc.blade), free of the pixel-phase bias and pixel blur of a
Gaussian fit to one frame: at each CH3 frequency and pattern, CH3 and then CH1 is shifted
over -SPAN..SPAN MHz in STEP steps (shuffled), which moves the array rigidly by J df on the
camera (config.AOD_STEERING_CAL), with a reference frame at the unshifted tones before
every shift to measure drift. The camera-axis sigmas are converted to along and across the
rows assuming the spot axes lie along the rows.

Plots one panel per CH3 frequency: the spacing of each gap, averaged over the rows (with
--rows, over the columns; error bar: standard error of that mean), one curve per pattern.

Usage:
    python scripts/characterization/column_spacing.py \\
        aqc/calibration/phase-patterns/20261101_184357_wgs_cwgs_pos_homogenized_zflat.npy \\
        aqc/calibration/phase-patterns/20261106_175439_handoff.npy
    python scripts/characterization/column_spacing.py <pattern>.npy --f3 65e6 70e6 --exposure-us 50
    python scripts/characterization/column_spacing.py <zflat>.npy <handoff>.npy --rows
    python scripts/characterization/column_spacing.py <zflat>.npy <handoff>.npy --px
    python scripts/characterization/column_spacing.py <zflat>.npy <handoff>.npy --px --f3 70e6 --blade 0.9 0.025
"""

import argparse
from pathlib import Path

import e2style as e2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from aqc import config
from aqc.blade import HALF, blade
from aqc.blaze import blaze
from aqc.data import save_plot
from aqc.hardware.awg import aods
from aqc.hardware.camera import Camera
from aqc.hardware.slm import send_phase
from aqc.steering import F_CENTER, grab, play
from aqc.traps import find_traps, fit_all, fit_spots, rotate, row_fit

NV, NH = 10, 90
AMPLITUDE = 1.6          # V peak, the CW drive of aod_tone.py
BOX = 9                  # fit window half-width, px


def gauss_uv(uv, amp, u0, v0, su, sv, off):
    u, v = uv
    return amp * np.exp(-(u - u0) ** 2 / (2 * su ** 2) - (v - v0) ** 2 / (2 * sv ** 2)) + off


def fit_uv(img, guess, phi):
    """Centre (u, v) and sigma (u, v) (along, across the rows, px) of each spot of guess (fit_spots output)."""
    out = np.empty((len(guess), 4))
    for k, (y, x, a, sy, sx) in enumerate(guess):
        yi, xi = int(round(y)), int(round(x))
        yy, xx = np.mgrid[yi - BOX:yi + BOX + 1, xi - BOX:xi + BOX + 1]
        p = img[yy, xx].ravel()
        uv = rotate(np.stack([yy.ravel(), xx.ravel()], -1).astype(float), phi)
        u0, v0 = rotate(np.array([y, x]), phi)
        popt, _ = curve_fit(gauss_uv, uv, p, p0=[a, u0, v0, sx, sy, np.percentile(p, 10)], maxfev=4000)
        out[k] = popt[1], popt[2], abs(popt[3]), abs(popt[4])
    return out


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("patterns", type=Path, nargs="+", help="phase maps (.npy, radians, no blaze)")
parser.add_argument("--f3", type=float, nargs="+", default=[60e6, 65e6, 70e6], help="CH3 frequencies, Hz")
parser.add_argument("--exposure-us", type=float, default=100, help="camera exposure, us")
parser.add_argument("--average", type=int, default=4, help="frames averaged per image")
parser.add_argument("--rows", action="store_true", help="spacing between neighbouring rows, across the rows")
parser.add_argument("--px", action="store_true", help="plot the centre distance in camera px, not in waists")
parser.add_argument("--blade", type=float, nargs=2, metavar=("SPAN", "STEP"),
                    help="blade-method centres: CH3, then CH1, shifted over -SPAN..SPAN MHz in STEP steps "
                         "(STEP a multiple of the 3.125 kHz tone grid)")
parser.add_argument("--blaze-y", type=int, default=8, help="y blaze period, SLM px")
parser.add_argument("--blaze-x", type=int, default=0, help="x blaze period, SLM px (0: none)")
args = parser.parse_args()

if args.blade:
    span, step = args.blade
    grid = np.arange(-round(span / step), round(span / step) + 1) * step
    grid = grid[grid != 0]
    shifts_mhz = np.vstack([(0.0, 0.0), np.random.default_rng().permutation(
        [(f, 0.0) for f in grid] + [(0.0, f) for f in grid])])                 # (df3, df1), unshifted first
    shifts_px = (shifts_mhz @ np.load(config.AOD_STEERING_CAL)["J"].T)[:, ::-1]  # (y, x)
    reach = int(np.ceil(np.abs(shifts_px).max())) + HALF + 2


def blade_scan(awg, cam, f3, frame):
    """Pixel stack (runs, P), its (H, W) index map, the traps' Gaussian centres p0 (N, 2) and the drift (runs, 2)."""
    p0 = fit_all(frame, find_traps(frame, NV, NH)[0])
    keep = np.zeros(frame.shape, bool)
    for y, x in np.round(p0).astype(int):
        keep[y - reach:y + reach + 1, x - reach:x + reach + 1] = True
    pixels = np.flatnonzero(keep)
    index = np.full(frame.size, -1)
    index[pixels] = np.arange(len(pixels))
    stack = np.empty((len(shifts_mhz), len(pixels)), np.float32)
    drift = np.empty((len(shifts_mhz), 2))
    for s, (df3, df1) in enumerate(shifts_mhz):
        ref = img = frame
        if s > 0:
            play(awg, f3, f1=F_CENTER)
            ref = grab(cam, args.average)
            play(awg, f3 + df3 * 1e6, f1=F_CENTER + df1 * 1e6)
            img = grab(cam, args.average)
        drift[s] = np.nanmean(fit_all(ref, p0) - p0, axis=0)
        stack[s] = img.ravel()[pixels]
    return stack, index.reshape(frame.shape), p0, drift


frames, scans = {}, {}
with aods(amplitudes=(AMPLITUDE, AMPLITUDE)) as awg, Camera(args.exposure_us, hardware_trigger=False) as cam:
    for f3 in args.f3:
        f3 = play(awg, f3, f1=F_CENTER)
        for path in args.patterns:
            send_phase(np.load(path) + blaze(args.blaze_y, args.blaze_x))
            frames[f3, path.stem] = grab(cam, args.average)
            if args.blade:
                scans[f3, path.stem] = blade_scan(awg, cam, f3, frames[f3, path.stem])
                play(awg, f3, f1=F_CENTER)

WHAT = {"along": "along the rows", "across": "across the rows"}
spacing = {}
for (f3, name), frame in frames.items():
    pts, rows, cols = find_traps(frame, NV, NH)
    fits = fit_spots(frame, pts)
    if not np.all(np.isfinite(fits)):
        raise RuntimeError(f"{name} at {f3 / 1e6:g} MHz: Gaussian fit failed for {np.isnan(fits[:, 0]).sum()} "
                           f"of {NV * NH} traps")
    if args.blade:
        stack, index, p0, drift = scans[f3, name]
        fits = np.array([blade(stack, index, p, shifts_px + drift)[:4] for p in p0])   # (y0, x0, sigma_y, sigma_x)
        phi = np.radians(row_fit(fits[:, :2], rows, NV))
        c2, s2 = np.cos(phi) ** 2, np.sin(phi) ** 2
        sy2, sx2 = fits[:, 2] ** 2, fits[:, 3] ** 2
        uv = np.column_stack([*rotate(fits[:, :2], phi), np.sqrt((c2 * sx2 - s2 * sy2) / (c2 - s2)),
                              np.sqrt((c2 * sy2 - s2 * sx2) / (c2 - s2))])
        print(f"CH3 {f3 / 1e6:g} MHz, {name}: blade, {len(shifts_mhz)} runs, drift up to {np.abs(drift).max():.3f} px")
    else:
        phi = np.radians(row_fit(fits[:, :2], rows, NV))
        uv = fit_uv(frame, fits, phi)
    k = 1 if args.rows else 0                   # v (across) or u (along the rows)
    centre, waist = np.empty((NV, NH)), np.empty((NV, NH))
    centre[rows, cols] = uv[:, k]
    waist[rows, cols] = 2 * uv[:, 2 + k]
    # Gaps run along axis 0 (rows) or 1 (columns); s[..., gap] after the transpose.
    if args.rows:
        centre, waist = centre.T, waist.T
    gap = np.abs(np.diff(centre, axis=1))
    s = gap / (0.5 * (waist[:, 1:] + waist[:, :-1]))
    spacing[f3, name] = gap if args.px else s
    print(f"CH3 {f3 / 1e6:g} MHz, {name}: peak {frame.max():.0f}, pitch {np.median(gap):.3f} px, waist "
          f"{WHAT['across'] if args.rows else WHAT['along']} {waist.mean():.3f} px, "
          f"spacing {s.mean():.3f} +- {s.std():.3f} waists")

e2.apply()
colors = [e2.OKABE_ITO[k] for k in ("blue", "vermilion", "green", "purple", "orange")]
fig, axes = plt.subplots(len(args.f3), 1, figsize=(180 * e2.MM, 50 * e2.MM * len(args.f3)),
                         sharex=True, sharey=True, layout="constrained", squeeze=False)
for ax, f3 in zip(axes[:, 0], dict.fromkeys(f for f, _ in frames)):
    for path, color in zip(args.patterns, colors):
        s = spacing[f3, path.stem]
        ax.errorbar(np.arange(s.shape[1]), s.mean(axis=0), s.std(axis=0) / np.sqrt(len(s)), fmt="o-", ms=2, lw=0.8,
                    color=color, label=f"{path.stem}, mean {s.mean():.3f}")
    ax.set_title(f"CH3 {f3 / 1e6:g} MHz, CH1 {F_CENTER / 1e6:g} MHz")
    ax.set_ylabel(f"spacing, {'column' if args.rows else 'row'} mean ({'px' if args.px else 'waists'})")
    ax.legend()
axes[-1, 0].set_xlabel("gap between rows r and r+1" if args.rows else "gap between columns c and c+1")
what = WHAT["across"] if args.rows else WHAT["along"]
unit = "camera px" if args.px else f"waists (w = 2σ {what})"
method = f"blade ±{args.blade[0]:g} MHz / {args.blade[1] * 1e3:g} kHz" if args.blade else "Gaussian fits"
fig.suptitle(f"{'Row' if args.rows else 'Column'} spacing {what} in {unit}, {method}, "
             f"{args.exposure_us:g} µs x {args.average}")
print(save_plot(fig, ("row_spacing" if args.rows else "column_spacing") + ("_px" if args.px else "")
                + ("_blade" if args.blade else "")))
