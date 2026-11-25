"""Looping GIF of the Shepard handoff on some rows of the array, from a saved strobe run.

Reads a run folder of handoff_strobe.py --save or --blade (DATA_TAKING.md, aqc.run): the
leak-subtracted strobe frames of one cycle. The frames are resampled onto the array rows
(aqc.movie.on_rows), with the traps found on the static reference minus the leak frame
(the leak frame also carries the room light, which the strobe frames already have
subtracted), and cropped to rows --rows R0 R1 (camera lines; for the vertical array the
design columns) over the full length of the rows. One strobe per video frame; a cycle
moves the array by exactly one site, so the loop is seamless away from the row ends.

With --follow ROW it instead zooms on the one spot that handoff_trajectory.py tracked along
ROW (its latest trajectory .npz in <run>/analysis/), through one cycle: the loop of the
trajectory nearest the middle of the row, or --cycle. The top panel is the frame resampled
bilinearly onto a 0.2 px grid along (horizontal, in the direction of travel) and across
(vertical) the row, over the site the spot crosses, with the trajectory's 1/e^2 ellipse
(waists w along and across, blade or Gaussian as in the .npz) drawn at its centre. The
bottom panel is those waists through the cycle against the strobe time (AWG time from the
cycle start; the light lags it by the AOD's acoustic transit), with the current strobe
marked. Lengths are um on the sensor (px * config.CAMERA_PIXEL_UM); the bottom panel also has a px axis.
--fps sets the speed.

--handoffs N instead shows N cycles from the middle of cycle C (--cycle) on, so the N handoffs
(the cycle wraps, where the trajectory moves on from strobe frame n-1 to frame 0 of the
next loop) fall inside, each marked by a dashed line. The zoom then moves with the uniform
transport (a line fitted to the spot's position along the row over the frames shown), so
the spot stays near the centre and the ellipse sits at its residual. --trace position plots
the centre residuals along and across the row (each minus its mean over the frames shown)
instead of the waists, --trace quadrature the two waists in quadrature, sqrt(w_along^2 +
w_across^2); several kinds (--trace position width) give one panel each.

--raw shows the camera pixels themselves instead of the 0.2 px resampling: integer-pixel crops
of the frame in camera axes (the axis nearest the travel horizontal, flipped to match the
resampled zoom), the crop following the zoom centre in 1 px steps with its extent shifted by the
sub-pixel rest, so the ellipse and axes mean the same as without it. The leftover row tilt
shows as a rotated ellipse.

No hardware is used.

Saves <run>/analysis/<stamp>_handoff_rows<R0>-<R1>.gif, or with --follow
<stamp>_handoff_follow_row<ROW>_cycle<C>[_handoffs<N>][_<trace>...][_raw].gif (no provenance .json).

Usage:
    python scripts/characterization/handoff_gif.py data/11_13/20261113_020430_shepard_sudden_50us_1site_blade --rows 0 4
    python scripts/characterization/handoff_gif.py <run folder> --rows 0 9 --scale 2
    python scripts/characterization/handoff_gif.py data/11_13/20261113_020430_shepard_sudden_50us_1site_blade --follow 0
    python scripts/characterization/handoff_gif.py <run folder> --follow 4 --cycle 30 --fps 2
    python scripts/characterization/handoff_gif.py <run folder> --follow 0 --handoffs 2 --trace position
    python scripts/characterization/handoff_gif.py <run folder> --follow 0 --cycle 43 --handoffs 2 --trace position width --raw
"""

import argparse
from datetime import datetime
from pathlib import Path

import e2style as e2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter
from matplotlib.patches import Ellipse
from scipy.ndimage import map_coordinates

from aqc import config
from aqc.movie import on_rows, save_gif
from aqc.run import analysis_dir, load
from aqc.traps import find_traps, fit_all, row_fit

NV, NH = 10, 90
STEP_PX = 0.2            # resampling step of the --follow zoom
HALF_ACROSS_PX = 5       # half-height of the --follow zoom across the row

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("run", type=Path, help="run folder of handoff_strobe.py --save or --blade")
parser.add_argument("--rows", type=int, nargs=2, default=[0, NV - 1], metavar=("R0", "R1"),
                    help="first and last row shown (0 = top)")
parser.add_argument("--scale", type=int, default=1, help="video pixels per camera px")
parser.add_argument("--follow", type=int, metavar="ROW", help="zoom on the tracked spot of this row for one cycle")
parser.add_argument("--cycle", type=int, help="with --follow: the trajectory loop shown (default: mid-row)")
parser.add_argument("--fps", type=float, default=3, help="with --follow: video frames (strobes) per second")
parser.add_argument("--handoffs", type=int, help="with --follow: span this many handoffs, from the middle of the cycle")
parser.add_argument("--trace", choices=("width", "quadrature", "position"), nargs="+", default=["width"],
                    help="with --follow: time-trace panels under the zoom, top to bottom")
parser.add_argument("--raw", action="store_true", help="with --follow: show camera pixels instead of the 0.2 px resampling")
args = parser.parse_args()

meta, d = load(args.run)


def gif_stem(name):
    return analysis_dir(args.run) / f"{datetime.now():%Y%m%d_%H%M%S}_{name}"


p = meta["parameters"]
ref = d["ref"].astype(float) - d["leak"].astype(float)
variant = "continuous" if p["continuous"] else "sudden switch" if p["sudden"] else "standard"
rel_us = np.asarray(d["strobe_times_us"])

if args.follow is None:
    r0, r1 = args.rows
    movie, u, v, rows, cols, pitch = on_rows(ref, np.asarray(d["frames"], dtype=float), NV, NH)
    y0 = int(np.nanmean(v[rows == r0]) - pitch / 2)
    y1 = int(np.nanmean(v[rows == r1]) + pitch / 2) + 1
    title = (f"Shepard handoff ({variant}{', random tone phases' if p['random_phases'] else ''}) on CH{p['channel']}: "
             f"{p['n_sites']} site per {p['cycle_us']:g} µs, rows {r0}-{r1}; run {meta['run']['stamp']}")
    stem = gif_stem(f"handoff_rows{r0}-{r1}")
    save_gif(stem.with_suffix(".gif"), movie[:, y0:y1], args.scale, title, rel_us)
else:
    row = args.follow
    npzs = sorted(list(args.run.glob(f"analysis/*_handoff_trajectory_row{row}.npz"))
                  + list(args.run.glob(f"analysis/*_handoff_trajectory_blade_row{row}.npz")), key=lambda f: f.name)
    if not npzs:
        raise FileNotFoundError(f"no handoff_trajectory .npz of row {row} in {args.run / 'analysis'}")
    tr = np.load(npzs[-1])
    pts, rows, _ = find_traps(ref, NV, NH)
    phi = np.radians(row_fit(fit_all(ref, pts), rows, NV))
    um = config.CAMERA_PIXEL_UM
    pitch = np.polyfit(tr["site"], tr["x_cam_um"], 1)[0] / um       # px

    loop = args.cycle if args.cycle is not None else int(np.median(tr["loop"]))
    n_strobe = len(rel_us)
    if args.handoffs:
        # The trajectory is stored loop by loop, n_strobe frames each
        first = np.flatnonzero((tr["loop"] == loop) & (tr["frame"] == n_strobe // 2))[0]
        sel = np.arange(first, first + args.handoffs * n_strobe)
    else:
        sel = np.flatnonzero(tr["loop"] == loop)
    k, cv, cu = tr["frame"][sel], tr["v_px"][sel], tr["u_px"][sel]
    w_x, w_y = tr["waist_x_cam_um"][sel], tr["waist_y_cam_um"][sel]
    site, loops = tr["site"][sel], tr["loop"][sel]
    t = tr["t_us"][sel] - tr["t_us"][sel[0]]
    xr, yr = tr["x_residual_cam_um"][sel], tr["y_residual_cam_um"][sel]
    xr, yr = xr - xr.mean(), yr - yr.mean()
    w_q = np.hypot(w_x, w_y)
    # kind: (traces, legend labels, y label, title line of frame i)
    panels = {"position": ((xr, yr), ("along, x − (x₀ + v t)", "across, y − y_row(x)"), "residual − mean (µm)",
                           lambda i: f"centre residual: along {xr[i]:+.3f} µm, across {yr[i]:+.3f} µm"),
              "width": ((w_x, w_y), ("along the row", "across the row"), "waist w (µm)",
                        lambda i: f"waist w: along {w_x[i]:.2f} µm, across {w_y[i]:.2f} µm"),
              "quadrature": ((w_q,), ("√(w_along² + w_across²)",), "waists in quadrature (µm)",
                             lambda i: f"√(w_along² + w_across²): {w_q[i]:.2f} µm")}

    # Row-aligned grid around the cycle's mean position, x pointing along the travel.
    e_x = np.array([np.sin(phi), np.cos(phi)])                 # (v, u) unit step along the row
    e_y = np.array([np.cos(phi), -np.sin(phi)])                # across
    c = np.array([np.nanmean(cv), np.nanmean(cu)])
    rel = np.stack([cv, cu], axis=1) - c
    if np.polyfit(np.arange(len(sel)), rel @ e_x, 1)[0] < 0:
        e_x = -e_x
    dx, dy = rel @ e_x, rel @ e_y
    shift = np.polyval(np.polyfit(t, dx, 1), t) if args.handoffs else np.zeros(len(sel))
    dx = dx - shift
    half_x = 0.75 * pitch
    if args.raw:
        # Whole camera pixels around the moving centre m, the crop stepping by 1 px and its extent carrying
        # the sub-pixel rest. Horizontal is camera axis a, the one nearest the travel, vertical the other
        # (b), each flipped to match the resampled zoom, so the two differ only by the leftover row tilt.
        m = c + shift[:, None] * e_x
        q = np.stack([cv, cu], axis=1)
        a = int(np.argmax(np.abs(e_x)))
        b = 1 - a
        sa, sb = int(np.sign(e_x[a])), int(np.sign(e_y[b]))
        h = np.zeros(2, int)
        h[a], h[b] = int(np.ceil(half_x)), HALF_ACROSS_PX
        patches, exts = [], []
        for f, mi in zip(k, m):
            n0 = np.round(mi).astype(int)
            crop = np.asarray(d["frames"][f, n0[0] - h[0]:n0[0] + h[0] + 1, n0[1] - h[1]:n0[1] + h[1] + 1],
                              dtype=float)
            crop = (crop.T if a == 0 else crop)[::sb, ::sa]
            xc, yc = sa * (n0[a] - mi[a]), sb * (n0[b] - mi[b])
            exts.append([xc - h[a] - 0.5, xc + h[a] + 0.5, yc + h[b] + 0.5, yc - h[b] - 0.5])
            patches.append(crop)
        exts = np.array(exts) * um
        ell_xy = np.stack([sa * (q[:, a] - m[:, a]), sb * (q[:, b] - m[:, b])], axis=1) * um
        ell_deg = np.degrees(np.arctan2(sb * e_x[b], sa * e_x[a]))
    else:
        gx = np.arange(-half_x, half_x + STEP_PX / 2, STEP_PX)
        gy = np.arange(-HALF_ACROSS_PX, HALF_ACROSS_PX + STEP_PX / 2, STEP_PX)
        patches = []
        for f, s in zip(k, shift):
            yx = ((c + s * e_x)[:, None, None] + e_x[:, None, None] * gx[None, None, :]
                  + e_y[:, None, None] * gy[None, :, None])
            lo = np.floor(yx.reshape(2, -1).min(1)).astype(int) - 1
            hi = np.ceil(yx.reshape(2, -1).max(1)).astype(int) + 2
            patches.append(map_coordinates(np.asarray(d["frames"][f, lo[0]:hi[0], lo[1]:hi[1]], dtype=float),
                                           yx - lo[:, None, None], order=1))
        exts = np.tile(np.array([gx[0], gx[-1], gy[-1], gy[0]]) * um, (len(sel), 1))
        ell_xy = np.stack([dx, dy], axis=1) * um
        ell_deg = 0.0
    patches = np.array(patches)

    e2.apply()
    plt.rcParams["savefig.transparent"] = False         # transparent GIF frames are drawn over the previous ones
    colors = e2.OKABE_ITO
    n_tr = len(args.trace)
    fig, (ax, *axs) = plt.subplots(1 + n_tr, 1, figsize=(170 * e2.MM, (60 + 60 * n_tr) * e2.MM),
                                   layout="constrained", gridspec_kw=dict(height_ratios=(1,) * (1 + n_tr)))
    im = ax.imshow(patches[0], cmap="inferno", vmin=0, vmax=np.percentile(patches, 99.9), extent=exts[0],
                   interpolation="nearest", aspect="equal")
    ax.set_xlim(-half_x * um, half_x * um)
    ax.set_ylim(HALF_ACROSS_PX * um, -HALF_ACROSS_PX * um)
    ax.set_autoscale_on(False)                         # set_extent per frame must not move the axes
    ell = Ellipse((0, 0), 0, 0, angle=ell_deg, fill=False, color=colors["sky"], lw=1.2)
    ax.add_patch(ell)
    ax.set_xlabel((f"camera {'vu'[a]}, nearest the travel (µm), raw pixels" if args.raw
                   else "along the row, direction of travel (µm)")
                  + (", moving with the uniform transport" if args.handoffs else ""))
    ax.set_ylabel(f"camera {'vu'[b]} (µm)" if args.raw else "across (µm)")
    title = ax.set_title("", fontsize=10, loc="left")

    marks, nows = [], []
    for ax_s, kind in zip(axs, args.trace):
        traces, labels, ylabel, _ = panels[kind]
        for tr_i, col, lab in zip(traces, (colors["blue"], colors["vermilion"]), labels):
            ax_s.plot(t, tr_i, color=col, lw=1.2, label=lab)
        for tw in t[1:][k[1:] == 0]:
            ax_s.axvline(tw, color=e2.GREY.edge, lw=0.8, ls="--")
        marks += [(ax_s.plot([], [], "o", color=col, ms=6)[0], tr_i)
                  for col, tr_i in zip((colors["blue"], colors["vermilion"]), traces)]
        nows.append(ax_s.axvline(t[0], color=e2.GREY.edge, lw=0.8))
        ax_s.set_ylabel(ylabel)
        ax_s.secondary_yaxis("right", functions=(lambda v: v / um, lambda v: v * um)).set_ylabel("camera px")
        ax_s.legend(loc="upper right", frameon=False)
    axs[-1].set_xlabel("time from the first strobe shown (µs, AWG time)"
                       + (", dashed: handoff" if args.handoffs else ""))
    fig.suptitle(f"Shepard handoff ({variant}), {p['n_sites']} site per {p['cycle_us']:g} µs; row {row}, "
                 + (f"cycles {loop}-{loops[-1]}" if args.handoffs else f"cycle {loop}")
                 + f"; run {meta['run']['stamp']}", fontsize=10)

    name = (f"handoff_follow_row{row}_cycle{loop}" + (f"_handoffs{args.handoffs}" if args.handoffs else "")
            + "".join(f"_{kind}" for kind in args.trace if args.trace != ["width"]) + ("_raw" if args.raw else ""))
    stem = gif_stem(name)
    writer = PillowWriter(fps=args.fps)
    with writer.saving(fig, stem.with_suffix(".gif"), 150):
        for i in range(len(sel)):
            im.set_data(patches[i])
            im.set_extent(exts[i])
            ell.set_center(ell_xy[i])
            ell.set_width(2 * w_x[i])                          # full width 2 w
            ell.set_height(2 * w_y[i])
            for mk, s in marks:
                mk.set_data([t[i]], [s[i]])
            for now in nows:
                now.set_xdata([t[i]])
            lines = [f"cycle {loops[i]}, strobe {k[i]:2d}/{n_strobe}, t = {t[i]:6.2f} µs, site {site[i]:.2f}"]
            lines += [panels[kind][3](i) for kind in args.trace]
            title.set_text("\n".join(lines))
            writer.grab_frame()
    plt.close(fig)
    print(f"wrote {stem.with_suffix('.gif')}")
