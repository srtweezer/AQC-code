"""Tweezer waist along and across the rows against the Shepard cycle period, from saved strobe runs.

Reads one handoff_strobe.py --save run per period (e.g. the continuous Shepard at several
--cycle-us) and, in each, the latest Gaussian trajectory of every row
(<run>/analysis/*_handoff_trajectory_row<r>.npz of handoff_trajectory.py). A row's waist
is w = 2 sigma along (x) and across (y) the row, the median over its trajectory: one
tweezer followed through every phase of the cycle, along the whole row. The points are the
mean over the rows, the bars their standard deviation. The dashed lines are the static
array (the transport channel held at 70 MHz, the run's reference frame minus its leak
frame), fitted the same way (aqc.traps.fit_spot on the frame resampled onto the rows),
median over the traps of columns 2 .. NH - 3 and over the runs.

Gaussian sigmas include the pixel blur (~10% wider than the blade method's), and the
strobe adds the array's motion during it along the rows, (pitch / T * pulse)^2 / 12 to
sigma^2. The right axis is the same waist on the sensor (config.CAMERA_PIXEL_UM).

Saves data/<MM_DD>/<stamp>_waist_vs_period.png. Needs no hardware.

Usage:
    python scripts/characterization/plot_waist_vs_period.py data/11_17/*_shepard_continuous_*us_1site_gauss_period_sweep
"""

import argparse
import json
from pathlib import Path

import e2style as e2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter, ScalarFormatter

from aqc import config
from aqc.data import save_plot
from aqc.movie import on_rows
from aqc.traps import fit_spot

NV, NH = 10, 90

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("runs", type=Path, nargs="+", help="handoff_strobe.py --save run folders, one per period")
args = parser.parse_args()

periods, w_mean, w_std, static = [], [], [], []
for run in args.runs:
    meta = json.loads((run / "metadata.json").read_text())
    latest = {}
    for f in sorted(run.glob("analysis/*_handoff_trajectory_row*.npz")):
        latest[int(f.stem.rsplit("_row", 1)[1])] = f
    ref = np.load(run / "raw/ref.npy").astype(float) - np.load(run / "raw/leak.npy").astype(float)
    grid_ref, u, v, rows, cols, _ = on_rows(ref, ref[None], NV, NH)
    fits = np.array([fit_spot(grid_ref[0], v[k], u[k]) for k in np.flatnonzero((cols >= 2) & (cols < NH - 2))])
    w_static = [2 * np.nanmedian(fits[:, 4]), 2 * np.nanmedian(fits[:, 3])]
    if not latest:
        # e.g. periods so short that the spots smear into lines along the rows, which the tracker can't follow
        print(f"{run.name}: no trajectories, left out; static {w_static[0]:.2f}, {w_static[1]:.2f} px")
        continue
    w = np.array([[np.nanmedian(np.load(f)[k]) / config.CAMERA_PIXEL_UM for k in ("waist_x_cam_um", "waist_y_cam_um")]
                  for f in latest.values()])                          # rows x (along, across), px
    periods.append(meta["parameters"]["cycle_us"])
    w_mean.append(w.mean(0))
    w_std.append(w.std(0))
    static.append(w_static)
    print(f"{run.name}: {len(latest)} rows; w along {w_mean[-1][0]:.2f} ± {w_std[-1][0]:.2f} px, "
          f"across {w_mean[-1][1]:.2f} ± {w_std[-1][1]:.2f} px; static {static[-1][0]:.2f}, {static[-1][1]:.2f} px")
order = np.argsort(periods)
T, w_mean, w_std, static = (np.array(a)[order] for a in (periods, w_mean, w_std, static))

e2.apply()
plt.rcParams.update({"axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13})
c = e2.OKABE_ITO
fig, ax = plt.subplots(figsize=(150 * e2.MM, 100 * e2.MM), layout="constrained")
for i, color, label in ((0, c["blue"], "$w_x$, along the row"), (1, c["vermilion"], "$w_y$, across the row")):
    ax.errorbar(T, w_mean[:, i], w_std[:, i], fmt="o-", ms=5, lw=1.2, capsize=3, color=color, label=label)
    ax.axhline(np.median(static[:, i]), color=color, ls="--", lw=1)
ax.set_xscale("log")
ax.set_xticks(T)
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("Shepard cycle period τ (µs)")
ax.set_ylabel("Waist w = 2σ (camera px)")
right = ax.secondary_yaxis("right", functions=(lambda p: p * config.CAMERA_PIXEL_UM,
                                               lambda um: um / config.CAMERA_PIXEL_UM))
right.set_ylabel("Waist on the sensor (µm)")
ax.legend(title="dashed: static array", frameon=False)
save_plot(fig, "waist_vs_period")
