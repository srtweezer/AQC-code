"""Plot handoff heating simulations from the .npz files that handoff_heating.py saved.

Two panels against the sites traversed (|site - site at t = 0| from the trajectory, so time
runs left to right whichever way the row moves), with the distance travelled (sites x
pitch, on the mean-waist-1-um scale of the simulation) on the top axis:
- <E>/3k_B (uK) of the atoms still held, the mean energy per axis;
- the fraction of atoms lost (%), zero for a --harmonic simulation.
Within a handoff cycle <E> breathes reversibly with the trap (~0.75 uK peak to peak).
--average removes this: each row's curves are averaged over every full cycle (n_sites
sites, from the run's metadata.json next to the trajectories), one point per cycle, and
the last, incomplete cycle is dropped.
With several rows the curves are the mean over rows and the bands ± 1 standard deviation
across rows. <E> of a row is left out once it holds fewer than 5% of its atoms, so a curve
ends when every row has lost 95%. Several .npz are overlaid in the given order, one colour each, labelled by
--labels or else by depth (and harmonic, frozen channels).

--column (repeatable) puts each group of .npz in its own column, sharing the y axes, with
--titles above the columns; --labels then names the i-th curve of every column.

--summary instead plots one point per .npz: <E>/3k_B and the loss over the last full
handoff cycle (mean ± s.d. across rows), with --labels on the x axis. With several --column,
each column is one series (e.g. one frozen-channel variant of the same runs, in the same
order), drawn side by side at each x and named in the legend by --titles.

One .npz: writes <npz stem>.png next to it. Several: data/<MM_DD>/<timestamp>_<--name>.png
(default handoff_heating_<run stamp>_overlay). --out-dir writes there instead. Needs no hardware.

Usage:
    python scripts/characterization/plot_handoff_heating.py data/<MM_DD>/<stamp>_handoff_heating_<run stamp>_rows0-9.npz
    python scripts/characterization/plot_handoff_heating.py data/<MM_DD>/*_handoff_heating_<run stamp>_rows0-9.npz --average
    python scripts/characterization/plot_handoff_heating.py --column <a50>.npz <b50>.npz --column <a20>.npz <b20>.npz \\
        --titles "50 µs cycle" "20 µs cycle" --labels continuous sudden --average --name continuous_vs_sudden
    python scripts/characterization/plot_handoff_heating.py <npz> <npz> ... --summary --labels <label> <label> ...
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import e2style as e2
import matplotlib.pyplot as plt
import numpy as np

from aqc.data import data_path

COLORS = ("blue", "vermilion", "green", "orange", "purple", "sky", "yellow", "black")
MIN_HELD = 0.05          # <E> of a row is dropped once fewer than this fraction of its atoms are held

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("npz", type=Path, nargs="*", help="<stamp>_handoff_heating_<run stamp>_row<r>[...].npz")
parser.add_argument("--column", type=Path, nargs="+", action="append", help="the .npz of one column")
parser.add_argument("--titles", nargs="+", help="one title per --column")
parser.add_argument("--labels", nargs="+", help="legend label of each curve (or x label with --summary)")
parser.add_argument("--summary", action="store_true", help="last-cycle <E> and loss, one point per .npz")
parser.add_argument("--average", action="store_true", help="one point per handoff cycle (the cycle average)")
parser.add_argument("--name", help="output name for several .npz")
parser.add_argument("--out-dir", type=Path, help="write the PNG here instead")
args = parser.parse_args()

columns = args.column or [args.npz]
paths = [p for col in columns for p in col]


def cycle_average(d, y):
    """y (rows x records) averaged over each full handoff cycle, and the mean sites traversed of each cycle."""
    run_dir = Path(str(d["trajectories"][0])).parents[1]
    n_sites = json.loads((run_dir / "metadata.json").read_text())["parameters"]["n_sites"]
    cycle = np.floor(d["sites"].mean(0) / n_sites).astype(int)
    full = np.unique(cycle)[:-1]           # the last cycle is cut short
    return (np.array([d["sites"].mean(0)[cycle == c].mean() for c in full]),
            np.stack([y[:, cycle == c].mean(1) for c in full], axis=1))


def energy(d):
    """<E>/3k_B (rows x records), NaN where a row holds fewer than MIN_HELD of its atoms."""
    return np.where(1 - d["lost_frac"] < MIN_HELD, np.nan, d["energy_uk"])


def band(a, s, y, color, **kw):
    """Mean over rows (those with a value) and ± 1 s.d. band."""
    m, sd = np.nanmean(y, 0), np.nanstd(y, 0)
    a.plot(s, m, color=color, lw=1.6, **kw)
    if y.shape[0] > 1:
        a.fill_between(s, m - sd, m + sd, color=color, alpha=0.25, lw=0)


def default_label(d):
    freeze = list(d["freeze"])
    harmonic = "harmonic" in d.files and bool(d["harmonic"])
    return (f"$U_0$ = {float(d['depth_mk']):g} mK" + (", harmonic" if harmonic else "")
            + (f", frozen: {', '.join(freeze)}" if freeze else ""))


e2.apply()
plt.rcParams.update({"axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
                     "axes.titlesize": 15})

if args.summary:
    series = [[np.load(p) for p in col] for col in columns]
    n = len(series[0])
    labels = args.labels or [default_label(d) for d in series[0]]
    fig, (ax, ax_lost) = plt.subplots(2, 1, sharex=True, figsize=(180 * e2.MM, 130 * e2.MM), layout="constrained",
                                      gridspec_kw=dict(height_ratios=(2, 1)))
    x = np.arange(n)
    start = np.mean([d["energy_uk"][:, 0].mean() for runs in series for d in runs])
    width = 0.7 / len(series)
    for k, (runs, c) in enumerate(zip(series, COLORS)):
        xk = x + (k - (len(series) - 1) / 2) * width
        for a, get in ((ax, energy), (ax_lost, lambda d: d["lost_frac"] * 100)):
            last = np.array([cycle_average(d, get(d))[1][:, -1] for d in runs])     # runs x rows
            a.errorbar(xk, np.nanmean(last, 1), np.nanstd(last, 1), fmt="o", color=e2.OKABE_ITO[c], ms=6,
                       capsize=3, lw=1.6, label=args.titles[k] if args.titles and a is ax else None)
            if a is ax:
                for xi in xk[np.isnan(last).all(1)]:
                    a.text(xi, start, f"≥{100 - MIN_HELD * 100:.0f}% lost", rotation=90, ha="center", va="bottom",
                           fontsize=12)
    if args.titles:
        ax.legend()
    ax.axhline(start, color=e2.OKABE_ITO["black"], lw=1, ls="--")
    ax.text(-0.4, start, f"start, {start:.1f} µK", va="bottom", ha="left", fontsize=12)
    ax.set_ylabel("$\\langle E\\rangle / 3k_B$ [µK]")
    ax_lost.set_ylabel("$P_\\mathrm{atom\\ loss}$ (%)")
    ax_lost.set_xticks(x, labels)
    ax_lost.set_xlim(-0.5, n - 0.5)
else:
    ncol = len(columns)
    fig, axes = plt.subplots(2, ncol, sharex=True, sharey="row", squeeze=False,
                             figsize=(180 * e2.MM * (1 if ncol == 1 else 0.75 * ncol), 120 * e2.MM),
                             layout="constrained", gridspec_kw=dict(height_ratios=(2, 1)))
    for j, col in enumerate(columns):
        ax, ax_lost = axes[:, j]
        runs = [np.load(p) for p in col]
        for i, (d, c) in enumerate(zip(runs, COLORS)):
            color = e2.OKABE_ITO[c]
            label = args.labels[i] if args.labels else default_label(d)
            for a, y in ((ax, energy(d)), (ax_lost, d["lost_frac"] * 100)):
                s, y = cycle_average(d, y) if args.average else (d["sites"].mean(0), y)
                band(a, s, y, color, label=label if a is ax else None)
        pitch_um = float(runs[0]["pitch_um"])
        top = ax.secondary_xaxis("top", functions=(lambda s, p=pitch_um: s * p, lambda x, p=pitch_um: x / p))
        top.set_xlabel("Distance travelled [µm]")
        if args.titles:
            ax.set_title(args.titles[j])
        ax_lost.set_xlabel("Sites traversed")
    axes[0, 0].set_ylabel("$\\langle E\\rangle / 3k_B$ [µK]")
    axes[1, 0].set_ylabel("$P_\\mathrm{atom\\ loss}$ (%)")
    if len(columns[0]) > 1:
        axes[0, 0].legend()

if len(paths) == 1:
    out = (args.out_dir or paths[0].parent) / paths[0].with_suffix(".png").name
else:
    name = (args.name or f"handoff_heating_{np.load(paths[0])['run_stamp']}_overlay") + ".png"
    out = args.out_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{name}" if args.out_dir else data_path(name)
e2.save(fig, out, dpi=300, transparent=False)
