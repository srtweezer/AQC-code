"""Plot a Shepard-handoff trajectory from the .npz that handoff_trajectory.py saved.

Three panels against the number of sites traveled (time since the atom left its first
site, in cycles, times the sites moved per cycle). Lengths are in units of the mean waist
w_bar = <(w_x + w_y) / 2> over the trajectory (the length unit aqc.trajectory sets to 1 um for
the heating simulation), with camera px, which need no focal length, on the right axis; the
--overlay figure stays in px, since its runs have different w_bar. x is along the row in
the direction the array moves, y across it (x turned by +90 deg in the (u, v) row frame of
handoff_trajectory.py). A strip above the
panels sketches the array with both axes, in the colours of their curves, and the mean
motion:
- Position residual, labelled in the legend: x - (x_0 + v t), with x_0 + v t the
  straight-line fit of x over the trajectory (v the mean speed), and y - y_row(x), with
  y_row the straight line through the row's traps in the reference frame. With a
  --quadratic trajectory both are relative to their best quadratic in time instead.
- Waist: w = 2 sigma along x and y.
- Peak intensity change (%): 100 (I - <I>)/<I>. For a Gaussian trajectory I is the fitted
  peak amplitude; for a blade trajectory it is the window light divided by w_x w_y
  (proportional to the peak of a Gaussian of that power and those widths).
The cycle and the sites per cycle come from the run's metadata.json (the .npz is in
<run>/analysis/), --quadratic from the analysis .json next to the .npz.

Writes <npz stem>.png next to each .npz, or into --out-dir. With --overlay, all the .npz
(e.g. the same row from different runs) go into one figure, coloured by run and labelled
from their metadata, with x and y in separate panels; it is saved with aqc.data.save_plot
as data/<MM_DD>/<stamp>_handoff_trajectory_overlay_row<row>.png. Needs no hardware.

Usage:
    python scripts/characterization/plot_handoff_trajectory.py <run>/analysis/<stamp>_handoff_trajectory_blade_row4.npz
    python scripts/characterization/plot_handoff_trajectory.py <run>/analysis/*_handoff_trajectory_blade_row*.npz
    python scripts/characterization/plot_handoff_trajectory.py <npz> --out-dir /tmp/preview
    python scripts/characterization/plot_handoff_trajectory.py <runA>/analysis/<a>_row4.npz <runB>/analysis/<b>_row4.npz --overlay
"""

import argparse
import json
from pathlib import Path

import e2style as e2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from aqc import config
from aqc.data import save_plot


def array_sketch(ax, cx, cy, colors):
    """Schematic of the array (a few rows of traps) with the x and y axes and the mean motion along x."""
    nx, ny = 14, 4
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny))
    ax.plot(gx.ravel(), gy.ravel(), "o", ms=2.5, color=colors["black"])
    ax.annotate("", xy=(nx - 3, ny + 0.4), xytext=(2, ny + 0.4),
                arrowprops=dict(arrowstyle="-|>", color=e2.GREY.edge, lw=2))
    ax.text((nx - 1) / 2, ny + 0.9, "mean motion", ha="center", va="bottom", fontsize=10, color=e2.GREY.edge)
    o = (-2.2, -0.8)                                   # origin of the axes, outside the lower-left corner
    for (dx, dy), color, name, off in (((2.2, 0), cx, "x", (2.6, 0)), ((0, 2.2), cy, "y", (0, 2.75))):
        ax.annotate("", xy=(o[0] + dx, o[1] + dy), xytext=o, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5))
        ax.text(o[0] + off[0], o[1] + off[1], name, color=color, ha="center", va="center", fontsize=12)
    ax.set_xlim(-3, nx)
    ax.set_ylim(-1.3, ny + 2)
    ax.set_aspect("equal")
    ax.axis("off")
    e2.skip_frame(ax)                                  # e2.save would otherwise draw its rounded frame


def traces(npz):
    """Sites traveled and the plotted quantities of one .npz, plus its run's parameters and --quadratic."""
    d = np.load(npz)
    params = json.loads((npz.parents[1] / "metadata.json").read_text())["parameters"]
    quadratic = json.loads(npz.with_suffix(".json").read_text())["arguments"]["quadratic"]
    sites = d["t_us"] / (params["t_cycle_s"] * 1e6) * params["n_sites"]
    px = config.CAMERA_PIXEL_UM
    # x along the motion: flip the row frame (x, y) by 180 deg if the array moves towards -x.
    ok = np.isfinite(d["x_cam_um"])
    sign = np.sign(np.polyfit(d["t_us"][ok], d["x_cam_um"][ok], 1)[0])
    w_x, w_y = d["waist_x_cam_um"] / px, d["waist_y_cam_um"] / px
    peak = d["amp"] / (w_x * w_y) if str(d["method"]) == "blade" else d["amp"]
    return dict(sites=sites, x_px=sign * d["x_residual_cam_um"] / px, y_px=sign * d["y_residual_cam_um"] / px,
                w_x=w_x, w_y=w_y, w_mean=np.nanmean((w_x + w_y) / 2), rel=peak / np.nanmean(peak) - 1,
                row=int(d["row"]), params=params, quadratic=quadratic)


def plot(npz, out_dir):
    npz = Path(npz)
    t = traces(npz)
    sites, x_px, y_px, w_x, w_y, w_mean, rel, quadratic = (t[k] for k in ("sites", "x_px", "y_px", "w_x", "w_y",
                                                                            "w_mean", "rel", "quadratic"))

    e2.apply()
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11})
    colors = e2.OKABE_ITO
    cx, cy = colors["blue"], colors["vermilion"]
    fig = plt.figure(figsize=(180 * e2.MM, 185 * e2.MM), layout="constrained")
    gs = fig.add_gridspec(4, 1, height_ratios=(0.45, 1, 1, 1))
    axes = [fig.add_subplot(gs[1])]
    axes += [fig.add_subplot(gs[i], sharex=axes[0]) for i in (2, 3)]
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    array_sketch(fig.add_subplot(gs[0]), cx, cy, colors)
    if quadratic:
        names = ("$x - $ quadratic in $t$", "$y - $ quadratic in $t$")
    else:
        names = ("$x - (x_0 + v\\,t)$", "$y - y_{\\mathrm{row}}(x)$")
    w = w_mean
    panels = (("Position residual ($\\bar w$)" if not quadratic else "Residual from\nquadratic ($\\bar w$)",
               ((x_px / w, cx, names[0]), (y_px / w, cy, names[1])), "px"),
              ("Waist ($\\bar w$)", ((w_x / w, cx, None), (w_y / w, cy, None)), "px"),
              ("Peak intensity\nchange (%)", ((100 * rel, colors["green"], None),), None))
    for ax, (ylabel, curves, px_label) in zip(axes, panels):
        for values, color, name in curves:
            ax.plot(sites, values, lw=0.5, color=color, label=name)
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_locator(MaxNLocator(6))
        if px_label:
            ax.secondary_yaxis("right", functions=(lambda v: v * w, lambda p: p / w)).set_ylabel(px_label)
    fig.text(0.02, 0.98, f"$\\bar w = \\langle (w_x + w_y)/2 \\rangle$\n$\\bar w$ = {w:.2f} px", ha="left", va="top",
             fontsize=12)
    leg = axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=12,
                         handlelength=1.5, borderaxespad=0.2)
    for line in leg.get_lines():
        line.set_linewidth(2)
    axes[-1].set_xlabel("Number of sites traveled")

    out = Path(out_dir or npz.parent) / f"{npz.stem}.png"
    e2.save(fig, out, dpi=300, transparent=False)          # prints "wrote <out>"
    plt.close(fig)


def plot_overlay(npzs):
    runs = [traces(Path(npz)) for npz in npzs]
    rows = {t["row"] for t in runs}
    assert len(rows) == 1, f"--overlay expects one row, got rows {sorted(rows)}"
    quadratic = runs[0]["quadratic"]
    assert all(t["quadratic"] == quadratic for t in runs), "--overlay mixes --quadratic and straight-line residuals"

    e2.apply()
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11})
    colors = e2.OKABE_ITO
    run_colors = [colors[c] for c in ("blue", "vermilion", "green", "orange", "purple")]
    fig = plt.figure(figsize=(180 * e2.MM, 250 * e2.MM), layout="constrained")
    gs = fig.add_gridspec(6, 1, height_ratios=(0.45, 1, 1, 1, 1, 1))
    axes = [fig.add_subplot(gs[1])]
    axes += [fig.add_subplot(gs[i], sharex=axes[0]) for i in range(2, 6)]
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    array_sketch(fig.add_subplot(gs[0]), colors["black"], colors["black"], colors)
    if quadratic:
        labels = ("$x - $ quadratic\nin $t$ (px)", "$y - $ quadratic\nin $t$ (px)")
    else:
        labels = ("$x - (x_0 + v\\,t)$ (px)", "$y - y_{\\mathrm{row}}(x)$ (px)")
    panels = ((labels[0], "x_px", 1), (labels[1], "y_px", 1), ("Waist $w_x$ (px)", "w_x", 1),
              ("Waist $w_y$ (px)", "w_y", 1), ("Peak intensity\nchange (%)", "rel", 100))
    for ax, (ylabel, key, scale) in zip(axes, panels):
        for t, color in zip(runs, run_colors):
            p = t["params"]
            mode = "continuous" if p["continuous"] else "sudden" if p["sudden"] else "chirped"
            label = f"{mode}, {p['cycle_us']:g} µs, {p['n_sites']} site{'s' if p['n_sites'] > 1 else ''}"
            ax.plot(t["sites"], scale * t[key], lw=0.5, color=color, alpha=0.8, label=label)
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_locator(MaxNLocator(6))
    leg = axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(runs), frameon=False, fontsize=12,
                         handlelength=1.5, borderaxespad=0.2)
    for line in leg.get_lines():
        line.set_linewidth(2)
        line.set_alpha(1)
    axes[-1].set_xlabel("Number of sites traveled")
    axes[0].set_title(f"Row {rows.pop()}", loc="left", fontsize=13)

    save_plot(fig, f"handoff_trajectory_overlay_row{runs[0]['row']}")   # e2.save prints "wrote <path>"
    plt.close(fig)


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("npz", type=Path, nargs="+", help=".npz files of handoff_trajectory.py")
parser.add_argument("--out-dir", type=Path, help="write the PNGs here instead of next to the .npz")
parser.add_argument("--overlay", action="store_true", help="draw all the .npz (one row, several runs) in one figure")
args = parser.parse_args()
if args.overlay:
    plot_overlay(args.npz)
else:
    for npz in args.npz:
        plot(npz, args.out_dir)
