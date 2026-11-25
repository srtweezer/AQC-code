"""Map the shape of every trap across the tweezer array, for one or two phase patterns.

Each pattern is displayed with a blaze (8 px along y unless --blaze-y/--blaze-x) and the
10x90 traps are found on the camera. The frame is resampled onto the array rows
(aqc.traps.row_grid, 1 camera px per pixel), so a 2-D Gaussian with axes along that grid gives each trap's sigma along and
across the rows (the rows sit at ~17 deg to the camera axes, where an axis-aligned fit
would read a row-aligned ellipse as rounder than it is). Per trap this maps sigma along,
sigma across, their ratio, and the peak (the smoothed maximum that homogenize.py
equalizes) relative to the array mean, as in Fig. 4 of Machu et al., arXiv:2512.15967.
Comparing an uncorrected with an aberration-corrected pattern shows whether the
correction sharpens the off-axis traps. An ellipse tilted to the rows, and coma's
asymmetry, are not captured.

Setup: the SLM beam imaged onto the Basler (free-run), AODs on CH1/CH3 at 70 MHz and 1.6 V
peak. Nothing else may hold the AWG (stop aod_tone.py first). Pick an exposure that keeps
the brightest trap of every pattern below saturation.

Saves data/<MM_DD>/<timestamp>_spot_shape.png. Leaves the last pattern on the SLM and the AWG stopped.

Usage:
    python scripts/characterization/spot_shape.py <pattern.npy> --exposure-us 100
    python scripts/characterization/spot_shape.py <uncorrected.npy> <corrected.npy> --exposure-us 100
"""

import argparse
from pathlib import Path

import e2style as e2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates

from aqc.blaze import blaze
from aqc.data import save_plot
from aqc.hardware.awg import aods
from aqc.hardware.camera import Camera
from aqc.hardware.slm import send_phase
from aqc.intensity import peaks
from aqc.steering import F_CENTER, grab, play
from aqc.traps import find_traps, fit_all, fit_spots, rotate, row_grid

NV, NH = 10, 90
AMPLITUDE = 1.6          # V peak, the CW drive of aod_tone.py

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("patterns", type=Path, nargs="+", help="one or two phase maps (.npy, radians, no blaze)")
parser.add_argument("--exposure-us", type=float, required=True, help="camera exposure, us")
parser.add_argument("--average", type=int, default=16, help="frames averaged per image")
parser.add_argument("--blaze-y", type=int, default=8, help="y blaze period, SLM px")
parser.add_argument("--blaze-x", type=int, default=0, help="x blaze period, SLM px (0: none)")
args = parser.parse_args()
if len(args.patterns) > 2:
    parser.error("give one or two patterns")

maps = []
with aods(amplitudes=(AMPLITUDE, AMPLITUDE)) as awg, Camera(args.exposure_us, hardware_trigger=False) as cam:
    play(awg, F_CENTER)
    for path in args.patterns:
        send_phase(np.load(path) + blaze(args.blaze_y, args.blaze_x))
        frame = grab(cam, args.average)
        pts, rows, cols = find_traps(frame, NV, NH)
        pos = fit_all(frame, pts)
        grid, phi, pitch = row_grid(pos, rows, cols, NV)
        u, v = rotate(pos, phi)
        # Grid (y, x) of each trap: row_grid starts one pitch before the first trap.
        fits = fit_spots(map_coordinates(frame, grid, order=1),
                         np.column_stack([v - np.nanmin(v) + pitch, u - np.nanmin(u) + pitch]))
        if not np.all(np.isfinite(fits)):
            raise RuntimeError(f"{path.name}: Gaussian fit failed for {np.isnan(fits[:, 0]).sum()} of {NV * NH} traps")
        m = {k: np.empty((NV, NH)) for k in ("along", "across", "ratio", "peak")}
        m["along"][rows, cols] = fits[:, 4]
        m["across"][rows, cols] = fits[:, 3]
        m["ratio"] = m["along"] / m["across"]
        k = peaks(frame, pos)
        m["peak"][rows, cols] = k / k.mean()
        maps.append(m)
        print(f"{path.name}: sigma along the rows median {np.median(m['along']):.3f} px, across "
              f"{np.median(m['across']):.3f} px, along/across median {np.median(m['ratio']):.3f} "
              f"(quartiles {np.percentile(m['ratio'], 25):.3f}-{np.percentile(m['ratio'], 75):.3f}); "
              f"peak std/mean {k.std() / k.mean():.3f}")

e2.apply()
colors = [e2.OKABE_ITO["blue"], e2.OKABE_ITO["vermilion"]]
fig = plt.figure(figsize=(180 * e2.MM, (45 + 50 * len(maps)) * e2.MM), layout="constrained")
gs = fig.add_gridspec(len(maps) + 1, 4)
labels = {"along": "sigma along rows, px", "across": "sigma across rows, px", "ratio": "sigma along / across",
          "peak": "peak / array mean"}
limits = {key: (min(np.min(m[key]) for m in maps), max(np.max(m[key]) for m in maps)) for key in labels}
for i, (m, path) in enumerate(zip(maps, args.patterns)):
    for j, key in enumerate(labels):
        ax = fig.add_subplot(gs[i, j])
        im = ax.imshow(m[key], cmap="viridis", vmin=limits[key][0], vmax=limits[key][1],
                       aspect="auto", interpolation="nearest")
        fig.colorbar(im, ax=ax, label=labels[key])
        ax.set_title(f"{path.stem}" if j == 0 else "", loc="left")
        ax.set_xlabel("column")
        ax.set_ylabel("row")
for j, key in enumerate(labels):
    ax = fig.add_subplot(gs[len(maps), j])
    for m, path, color in zip(maps, args.patterns, colors):
        ax.plot(m[key].mean(axis=0), "o-", ms=2, color=color, label=path.stem)
    ax.set_xlabel("column")
    ax.set_ylabel(f"{labels[key]} (row mean)")
if len(maps) > 1:
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncols=len(maps))
fig.suptitle(f"Trap shape across the array; exposure {args.exposure_us:g} us, {args.average} frames")
save_plot(fig, "spot_shape")
