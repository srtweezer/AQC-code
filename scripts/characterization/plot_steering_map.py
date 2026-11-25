"""Plot the AOD steering map of a steering_map.py run: v3 = dX/df3 and v1 = dX/df1 at every spot over the camera.

Each spot's v is the slope of a straight-line fit of its displacement against frequency
offset (aqc.steering.slopes). A polynomial of degree --degree in camera (x, y) is fitted to
each component over all spots; its residual is the spot-to-spot scatter (measurement noise
plus anything finer than the polynomial).

Panels (rows CH3, CH1): |v| (px/MHz) and the direction of v on the camera (deg,
atan2(dy, dx), y down), one dot per spot. The fit residual and the angle between v1 and v3
are printed.

A second figure draws the field dependence of v3 and v1 as vector fields: the spots are
averaged in square blocks along the grid's own two axes, --arrows of them across its short
side, and each block's deviation from the field mean, v - <v>, is an arrow at the block's
mean position. The mean <v> itself is drawn once per panel (grey, direction only).

Writes <run>/analysis/<stamp>_steering_map.png, <stamp>_steering_map_vectors.png, .npz (positions, v3, v1, polynomial fits)
and .json (arguments). Needs no hardware.

Usage:
    python scripts/characterization/plot_steering_map.py data/11_16/aod_steering/<stamp>_steering_map
    python scripts/characterization/plot_steering_map.py <run> --degree 3
    python scripts/characterization/plot_steering_map.py <run> --arrows 8
"""

import argparse
from pathlib import Path

import e2style as e2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import offset_copy

from aqc.run import analysis, load
from aqc.steering import slopes
from aqc.traps import principal_angle, rotate

HALF_PT = 16             # half-length of the mean-v arrow, points

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("run", type=Path, help="steering_map run folder")
parser.add_argument("--degree", type=int, default=2, help="degree of the smooth polynomial fit in (x, y)")
parser.add_argument("--arrows", type=int, default=8, help="blocks across the grid's short side in the vector-field figure")
args = parser.parse_args()

meta, a = load(args.run)
offs, p0 = np.asarray(a["offs_mhz"]), np.asarray(a["p0"])
y, x = p0[:, 0], p0[:, 1]
# Monomials in coordinates scaled to about [-1, 1], so the fit is well conditioned.
c, s = p0.mean(axis=0), np.ptp(p0, axis=0) / 2
yn, xn = (y - c[0]) / s[0], (x - c[1]) / s[1]
powers = [(i, j) for i in range(args.degree + 1) for j in range(args.degree + 1 - i)]
A = np.column_stack([xn ** i * yn ** j for i, j in powers])

v, fit, coef = {}, {}, {}
for ch in (3, 1):
    v[ch] = slopes(offs, np.asarray(a[f"displacement_ch{ch}"]))           # (N, 2) y, x px/MHz
    ok = np.all(np.isfinite(v[ch]), axis=1)
    coef[ch] = np.linalg.lstsq(A[ok], v[ch][ok], rcond=None)[0]
    fit[ch] = A @ coef[ch]
    mag, mag_fit = np.hypot(*v[ch].T), np.hypot(*fit[ch].T)
    ang = np.degrees(np.arctan2(v[ch][:, 0], v[ch][:, 1]))
    res = 100 * np.hypot(*(v[ch] - fit[ch]).T) / mag_fit
    lo, hi = np.nanpercentile(mag, [2, 98])
    alo, ahi = np.nanpercentile(ang, [2, 98])
    print(f"CH{ch}: {ok.sum()} spots, |v| mean {np.nanmean(mag):.4f} px/MHz, 2-98% {lo:.4f}..{hi:.4f} "
          f"({100 * (hi - lo) / np.nanmean(mag):.2f}% of mean); direction {alo:+.2f}..{ahi:+.2f} deg; "
          f"residual from degree-{args.degree} fit {np.sqrt(np.nanmean(res ** 2)):.2f}% rms")
between = np.degrees(np.arccos(np.sum(v[3] * v[1], axis=1) / np.hypot(*v[3].T) / np.hypot(*v[1].T)))
print(f"angle between v1 and v3: mean {np.nanmean(between):.3f} deg, 2-98% "
      f"{np.nanpercentile(between, 2):.3f}..{np.nanpercentile(between, 98):.3f} deg")

stem = analysis(args.run, "steering_map", args)
np.savez(stem.with_suffix(".npz"), p0_yx=p0, v3_yx=v[3], v1_yx=v[1], fit3_yx=fit[3], fit1_yx=fit[1],
         coef3=coef[3], coef1=coef[1], powers_xy=np.array(powers), center_yx=c, scale_yx=s, degree=args.degree)

e2.apply()
fig, axes = plt.subplots(2, 2, figsize=(125 * e2.MM, 150 * e2.MM), layout="constrained", sharex=True, sharey=True)
for row, ch in enumerate((3, 1)):
    for col, (val, name, unit) in enumerate((
            (np.hypot(*v[ch].T), f"|v$_{ch}$|", "px/MHz"),
            (np.degrees(np.arctan2(v[ch][:, 0], v[ch][:, 1])), f"direction of v$_{ch}$", "deg"))):
        ax = axes[row, col]
        lo, hi = np.nanpercentile(val, [2, 98])
        im = ax.scatter(x, y, c=val, s=2.5, cmap="viridis", vmin=lo, vmax=hi, linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        ax.set_xlim(x.min() - 60, x.max() + 60)
        ax.set_ylim(y.max() + 60, y.min() - 60)
        ax.set_title(f"CH{ch}: {name}")
        fig.colorbar(im, ax=ax, label=unit, shrink=0.9, pad=0.02)
for ax in axes[1]:
    ax.set_xlabel("camera x (px)")
for ax in axes[:, 0]:
    ax.set_ylabel("camera y (px)")
fig.suptitle(f"AOD steering over the field: {len(p0)} spots, each channel swept "
             f"{offs[0]:+g} to {offs[-1]:+g} MHz around 70 MHz")
e2.save(fig, stem.with_suffix(".png"), dpi=300, transparent=False)
print(f"wrote {stem.with_suffix('.png')}")

# Vector fields: means over square blocks along the grid's own axes, --arrows across its short side.
along, across = rotate(p0, principal_angle(p0))
pitch = min(np.ptp(along), np.ptp(across)) / args.arrows                # block side, px
n_al, n_ac = max(1, round(np.ptp(along) / pitch)), max(1, round(np.ptp(across) / pitch))
block = (np.clip(((along - along.min()) / np.ptp(along) * n_al).astype(int), 0, n_al - 1) * n_ac
         + np.clip(((across - across.min()) / np.ptp(across) * n_ac).astype(int), 0, n_ac - 1))
used = np.unique(block)
pos_b = np.array([p0[block == b].mean(axis=0) for b in used])

fig, axes = plt.subplots(1, 2, figsize=(125 * e2.MM, 85 * e2.MM), layout="constrained")
dev = {ch: np.array([np.nanmean(v[ch][block == b], axis=0) for b in used]) - np.nanmean(v[ch], axis=0)
       for ch in (3, 1)}
key = float(np.round(max(np.abs(d).max() for d in dev.values()), 2)) or 0.01
for ax, (ch, color) in zip(axes, ((3, e2.OKABE_ITO["blue"]), (1, e2.OKABE_ITO["vermilion"]))):
    ax.scatter(x, y, s=0.4, color=e2.GREY.face, linewidths=0, rasterized=True)
    # angles="xy" draws (dx, dy) in data coordinates, so the arrows follow the inverted camera y.
    q = ax.quiver(pos_b[:, 1], pos_b[:, 0], dev[ch][:, 1], dev[ch][:, 0], color=color, angles="xy",
                  scale_units="xy", scale=key / (0.75 * pitch),
                  width=0.005, headwidth=3.5, headlength=4, headaxislength=3.5)
    ax.quiverkey(q, 0.80, 0.06, key, f"{key:g} px/MHz", labelpos="N", coordinates="axes",
                 fontproperties={"size": 6})
    # The mean v itself, in the top-left corner of the axes (empty for both grid layouts so
    # far), direction only: offsets in points, with camera y (down) flipped to screen y (up).
    m = np.nanmean(v[ch], axis=0)
    u = m / np.hypot(*m)
    at = lambda k: offset_copy(ax.transAxes, fig=fig, x=k * HALF_PT * u[1], y=-k * HALF_PT * u[0], units="points")
    ax.annotate("", xy=(0.2, 0.8), xycoords=at(1), xytext=(0.2, 0.8), textcoords=at(-1),
                arrowprops=dict(arrowstyle="-|>", color=e2.GREY.edge, lw=1.5))
    ax.annotate(f"<v$_{ch}$> = {np.hypot(*m):.3f} px/MHz\nat {np.degrees(np.arctan2(m[0], m[1])):+.1f}°",
                xy=(0.2, 0.8), xycoords="axes fraction", xytext=(0, HALF_PT * abs(u[0]) + 4),
                textcoords="offset points", color=e2.GREY.edge, ha="center", va="bottom", fontsize=6)
    # The limits, not the box, give way to the equal aspect, so a narrow grid gets side margin.
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlim(x.min() - 60, x.max() + 60)
    ax.set_ylim(y.max() + 60, y.min() - 60)
    ax.set_title(f"CH{ch}: v$_{ch}$ $-$ <v$_{ch}$>")
    ax.set_xlabel("camera x (px)")
axes[0].set_ylabel("camera y (px)")
axes[1].tick_params(labelleft=False)
fig.suptitle(f"Deviation of the AOD steering from its mean, over {len(used)} blocks of ~{len(p0) // len(used)} spots")
path = stem.with_name(stem.name + "_vectors.png")
e2.save(fig, path, dpi=300, transparent=False)
print(f"wrote {path}")
