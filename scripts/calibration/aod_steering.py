"""Calibrate how AOD frequency steps move the tweezer array on the camera: the 2x2 steering matrix J.

    (dx, dy) = J (df_CH3, df_CH1),   camera px per MHz

Step 0 of the blade method (docs/blade_method.md). The pattern is displayed with an 8 px
y blaze and each AOD channel in turn is swept around 70 MHz, the other held at 70 MHz,
while every trap is tracked (aqc.steering.sweep). Each point is measured against a
70 MHz frame taken just before it, so slow drift of the whole array cancels. A single
trap's fitted centre is biased by where its spot falls within a pixel, but over 900
traps at all pixel phases that bias averages out of the array-mean displacement. J's
columns are straight-line fits of the mean displacement against frequency. A quadratic
fit of the same data shows any nonlinearity over the sweep, and per-trap slopes show how
the steering varies across the array.

With --vertical the pattern is the 90 rows x 10 columns array (align_rows_to_aod.py
--vertical), found on the camera as 10 lines of 90 traps (see homogenize.py for how they
map to design rows and columns). Its calibration goes to config.AOD_STEERING_VERTICAL_CAL
(handoff_calibration.py --vertical), and config.AOD_STEERING_CAL is left pointing at the
horizontal array's, which the horizontal blade and handoff scripts use.

--blaze-y / --blaze-x set the blaze the pattern is displayed with (default 8 px along y).

With --cols N only the central N columns are tracked (aqc.traps.find_central_traps), for
an array whose ends are cut off somewhere in the beam path. Their column indices then count
from the first kept column, not from the edge of the full array.

Setup: the SLM beam imaged onto the Basler (free-run), AODs on CH1/CH3 at 1.6 V peak.
Nothing else may hold the AWG (stop aod_tone.py first).

Saves aqc/calibration/<timestamp>_aod_steering.npz (J, the array-mean displacements and
their linear and quadratic fits, per-trap steering, the trap positions at 70/70 MHz),
points config.AOD_STEERING_CAL at it (with --vertical, <timestamp>_aod_steering_vertical.npz
and config.AOD_STEERING_VERTICAL_CAL), and saves data/<MM_DD>/<timestamp>_aod_steering.png
and <timestamp>_aod_steering_heatmap.png (per-trap steering direction). With --plot-only
only the plots are saved: no calibration file, and config is left as it is (for comparing
the steering of different patterns).
Leaves the pattern on the SLM and the AWG stopped.

Usage:
    python scripts/calibration/aod_steering.py \\
        aqc/calibration/phase-patterns/20261101_184357_wgs_cwgs_pos_homogenized_zflat.npy --exposure-us 100
    python scripts/calibration/aod_steering.py <pattern>.npy --exposure-us 200 --cols 80
    python scripts/calibration/aod_steering.py \
        aqc/calibration/phase-patterns/vertical/<timestamp>_wgs_90x10_homogenized.npy \
        --exposure-us 150 --vertical --blaze-x -8 --blaze-y 0
    python scripts/calibration/aod_steering.py <pattern>.npy --exposure-us 100 --vertical \
        --blaze-x -8 --blaze-y 0 --plot-only
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

import e2style as e2
import matplotlib.pyplot as plt
import numpy as np

from aqc import config
from aqc.blaze import blaze
from aqc.data import save_plot
from aqc.hardware.awg import aods
from aqc.hardware.camera import Camera
from aqc.hardware.slm import send_phase
from aqc.steering import F_CENTER, NB_SAMPLES, grab, play, slopes, sweep
from aqc.traps import find_central_traps, find_traps, fit_all

# On the camera the array is NV lines of NH traps along its long axis: rows by default,
# columns with --vertical.
NV, NH = 10, 90
AMPLITUDE = 1.6          # V peak, the CW drive of aod_tone.py
CHANNELS = (3, 1)        # J's columns
CAL_DIR = Path(__file__).resolve().parents[2] / "aqc" / "calibration"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("pattern", type=Path, help="phase map (.npy, radians, no blaze)")
parser.add_argument("--exposure-us", type=float, required=True, help="camera exposure, us")
parser.add_argument("--f-span", type=float, default=2.0, help="sweep half-span around 70 MHz, MHz")
parser.add_argument("--f-step", type=float, default=0.25, help="sweep step, MHz")
parser.add_argument("--average", type=int, default=8, help="frames averaged per grab")
parser.add_argument("--cols", type=int, help="track only the central COLS columns (array ends cut off)")
parser.add_argument("--vertical", action="store_true", help="90 rows x 10 columns array")
parser.add_argument("--blaze-y", type=int, default=8, help="blaze period along y, px (0 = none)")
parser.add_argument("--blaze-x", type=int, default=0, help="blaze period along x, px (0 = none)")
parser.add_argument("--plot-only", action="store_true", help="save the plots only: no calibration file, config untouched")
args = parser.parse_args()
# Names of the camera lines and of the axis along them, and which channel moves along them
LINE, ALONG, STEER = ("design column", "89 - design row", 1) if args.vertical else ("row", "column", 3)

offs = np.round(np.arange(-args.f_span, args.f_span + args.f_step / 2, args.f_step), 6)
grid_mhz = config.AWG_SAMPLING_RATE / NB_SAMPLES / 1e6
if np.any(np.abs(offs / grid_mhz - np.round(offs / grid_mhz)) > 1e-6):
    raise ValueError(f"--f-span and --f-step must be multiples of the {grid_mhz * 1e3:g} kHz tone grid")

send_phase(np.load(args.pattern) + blaze(args.blaze_y, args.blaze_x))
d = {}
with aods(amplitudes=(AMPLITUDE, AMPLITUDE)) as awg, Camera(args.exposure_us, hardware_trigger=False) as cam:
    play(awg, F_CENTER)
    ref = grab(cam, args.average)
    if args.cols:
        peaks, rows, cols = find_central_traps(ref, NV, args.cols)
    else:
        # The ends of the vertical array come out dim on a uniform-weight solve
        peaks, rows, cols = find_traps(ref, NV, NH, 0.1 if args.vertical else 0.3)
    p0 = fit_all(ref, peaks)
    if not np.all(np.isfinite(p0)):
        raise RuntimeError(f"Gaussian fit failed for {np.isnan(p0[:, 0]).sum()} of {len(p0)} traps on the 70 MHz frame")
    for ch in CHANNELS:
        print(f"sweeping CH{ch}")
        d[ch] = sweep(awg, cam, p0, offs, args.average, ch)

J = np.empty((2, 2))
res = {}
for j, ch in enumerate(CHANNELS):
    ok = np.all(np.isfinite(d[ch]), axis=(0, 2))
    mean = d[ch][:, ok].mean(axis=1)[:, ::-1]          # (offs, 2) as (x, y) px
    lin, quad = np.polyfit(offs, mean, 1), np.polyfit(offs, mean, 2)
    J[:, j] = lin[0]
    S = slopes(offs, d[ch])[:, ::-1]                   # per trap (x, y) px/MHz
    res[ch] = dict(ok=ok, mean=mean, lin=lin, quad=quad, S=S)
    mag = np.hypot(*J[:, j])
    trap_mag = np.hypot(S[:, 0], S[:, 1])
    rms = np.sqrt(np.mean((mean - np.polyval(lin, offs[:, None])) ** 2, axis=0)) * 1e3
    print(f"CH{ch}: J column (x, y) = ({J[0, j]:+.5f}, {J[1, j]:+.5f}) px/MHz, |J| {mag:.5f} px/MHz at "
          f"{np.degrees(np.arctan2(J[1, j], J[0, j])):+.3f} deg to camera x; quadratic (x, y) "
          f"({quad[0, 0] * 1e3:+.2f}, {quad[0, 1] * 1e3:+.2f}) mpx/MHz^2; linear residual rms "
          f"({rms[0]:.1f}, {rms[1]:.1f}) mpx; per-trap |S| spread "
          f"{np.nanstd(trap_mag) / np.nanmean(trap_mag) * 100:.2f}% ({ok.sum()} traps)")
angle = np.degrees(np.arccos(J[:, 0] @ J[:, 1] / np.hypot(*J[:, 0]) / np.hypot(*J[:, 1])))
print(f"J = [[{J[0, 0]:+.5f}, {J[0, 1]:+.5f}], [{J[1, 0]:+.5f}, {J[1, 1]:+.5f}] px/MHz (rows x, y; columns CH3, CH1); "
      f"angle between the channels {angle:.3f} deg")

stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
path = CAL_DIR / f"{stamp}_aod_steering{'_vertical' if args.vertical else ''}.npz"
if not args.plot_only:
    np.savez(path, J=J, channels=np.array(CHANNELS), offs_mhz=offs, rows=rows, cols=cols, p0_yx=p0,
             **{f"mean_xy_ch{ch}": r["mean"] for ch, r in res.items()},
             **{f"linear_ch{ch}": r["lin"] for ch, r in res.items()},
             **{f"quadratic_ch{ch}": r["quad"] for ch, r in res.items()},
             **{f"trap_steering_xy_ch{ch}": r["S"] for ch, r in res.items()},
             pattern=args.pattern.name, exposure_us=args.exposure_us, average=args.average,
             vertical=args.vertical, blaze_y_px=args.blaze_y, blaze_x_px=args.blaze_x)
    print(f"wrote {path}")
    config_path = Path(config.__file__)
    name = "AOD_STEERING_VERTICAL_CAL" if args.vertical else "AOD_STEERING_CAL"
    line = f'{name} = Path(__file__).parent / "calibration" / "{path.name}"  # {datetime.now():%Y-%m-%d}'
    text, n = re.subn(rf"^{name} = .*$", line, config_path.read_text(), flags=re.M)
    if n != 1:
        raise RuntimeError(f"expected one {name} line in {config_path}, found {n}")
    config_path.write_text(text)
    print(f"pointed config.{name} at {path.name}")

e2.apply()
c = e2.OKABE_ITO
fig = plt.figure(figsize=(180 * e2.MM, 140 * e2.MM), layout="constrained")
gs = fig.add_gridspec(2, 2)
ax_d, ax_r = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
f_fine = np.linspace(offs[0], offs[-1], 200)
for ch, color in zip(CHANNELS, (c["blue"], c["vermilion"])):
    r = res[ch]
    for k, (comp, marker) in enumerate((("x", "o"), ("y", "s"))):
        ax_d.plot(offs, r["mean"][:, k], marker, ms=3, color=color, mfc="none" if comp == "y" else color,
                  label=f"CH{ch}, δ{comp}")
        ax_d.plot(f_fine, np.polyval(r["lin"][:, k], f_fine), lw=0.6, color=color)
        resid = r["mean"][:, k] - np.polyval(r["lin"][:, k], offs)
        ax_r.plot(offs, resid * 1e3, marker, ms=3, color=color, mfc="none" if comp == "y" else color,
                  label=f"CH{ch}, δ{comp}")
        ax_r.plot(f_fine, (np.polyval(r["quad"][:, k], f_fine) - np.polyval(r["lin"][:, k], f_fine)) * 1e3,
                  lw=0.6, color=color, ls="-" if comp == "x" else "--")
ax_d.set_xlabel("δf (MHz)")
ax_d.set_ylabel("array-mean shift (camera px)")
ax_d.set_title("mean shift of all traps, linear fit")
ax_d.legend(frameon=True, facecolor="white")
ax_r.axhline(0, color=e2.GREY.edge, lw=0.5)
ax_r.set_xlabel("δf (MHz)")
ax_r.set_ylabel("residual from linear fit (mpx)")
ax_r.set_title("nonlinearity; lines: quadratic fit")
for j, ch in enumerate(CHANNELS):
    ax = fig.add_subplot(gs[1, j])
    S = res[ch]["S"]
    along = np.full((NV, cols.max() + 1), np.nan)
    along[rows, cols] = (S @ J[:, j] / np.hypot(*J[:, j]) ** 2 - 1) * 100   # % of |J| along J
    lim = np.nanmax(np.abs(along))
    im = ax.imshow(along, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="per-trap steering − J (%)")
    ax.set_xlabel(ALONG)
    ax.set_ylabel(LINE)
    ax.set_title(f"CH{ch}: per-trap steering along J")
fig.suptitle(f"AOD steering on the camera: J = [[{J[0, 0]:+.4f}, {J[0, 1]:+.4f}], [{J[1, 0]:+.4f}, {J[1, 1]:+.4f}]] "
             f"px/MHz (x, y; CH3, CH1)\n{args.pattern.name}, ±{args.f_span:g} MHz in {args.f_step:g} MHz steps, "
             f"{args.average} frames per point, exposure {args.exposure_us:g} µs")
save_plot(fig, "aod_steering", stamp)

# Per-trap steering direction: the angle of S from the array-mean direction J of its channel,
# on the array and as its mean over the lines at each position along them. A trend along the
# lines is a fan-out: the shift direction rotates across the field, which a magnitude map
# can't show (0.8 deg changes |S| by 0.002%).
fig = plt.figure(figsize=(180 * e2.MM, 170 * e2.MM), layout="constrained")
gs = fig.add_gridspec(3, 1)
n_along = cols.max() + 1
for j, ch in enumerate(CHANNELS):
    ax = fig.add_subplot(gs[j])
    S, Jc = res[ch]["S"], J[:, j]
    ang = np.full((NV, n_along), np.nan)
    ang[rows, cols] = np.degrees(np.arctan2(Jc[0] * S[:, 1] - Jc[1] * S[:, 0], S @ Jc))
    lim = np.nanmax(np.abs(ang))
    im = ax.imshow(ang, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="angle from J (deg)")
    ax.set_xlabel(ALONG)
    ax.set_ylabel(LINE)
    ax.set_title(f"CH{ch} (moves {'along' if ch == STEER else 'across'} the {LINE.split()[-1]}s): J at "
                 f"{np.degrees(np.arctan2(Jc[1], Jc[0])):+.3f} deg to camera x; per-trap angle from J "
                 f"{np.nanmin(ang):+.2f} to {np.nanmax(ang):+.2f} deg")
    res[ch]["along_mean"] = np.nanmean(ang, axis=0)
    res[ch]["along_sem"] = np.nanstd(ang, axis=0) / np.sqrt(np.isfinite(ang).sum(axis=0))
ax = fig.add_subplot(gs[2])
for ch, color in zip(CHANNELS, (c["blue"], c["vermilion"])):
    ax.errorbar(np.arange(n_along), res[ch]["along_mean"], res[ch]["along_sem"], fmt="o-", ms=3,
                color=color, label=f"CH{ch}")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel(ALONG)
ax.set_ylabel("angle from J (deg)")
ax.set_title(f"steering direction at each position along the {LINE.split()[-1]}s, mean over the {NV} "
             f"{LINE.split()[-1]}s")
ax.legend(frameon=True, facecolor="white")
fig.suptitle(f"AOD steering direction per trap (angle of d position / d frequency from J), {stamp + ' (plot only)' if args.plot_only else path.name}\n"
             f"pattern {args.pattern.name}, blaze y {args.blaze_y:+d} x {args.blaze_x:+d} px,\nsweep "
             f"{-args.f_span:+g} to {args.f_span:+g} MHz around 70 MHz, other channel at 70 MHz")
save_plot(fig, "aod_steering_heatmap", stamp)
