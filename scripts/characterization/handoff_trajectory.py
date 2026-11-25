"""Trajectory of one atom carried along a row by the Shepard handoff, from a saved strobe run.

Reads a run folder of handoff_strobe.py --save (DATA_TAKING.md, aqc.run): strobed frames
at --n-frames points of one cycle, during which CH3 moves the array by one site. Looping the frames gives the motion
an atom riding one tweezer would see: within a cycle it follows one spot, and at the wrap
it carries on into the spot one site further, since frame 0 of the next cycle is the same
picture one site on. Every pass through the loop therefore crosses a different pair of
neighbouring SLM traps, and one atom is followed along the whole row --row, from one site
inside one end to one site inside the other.

Each spot is a 2-D Gaussian fit (aqc.traps.fit_spot), seeded by the previous point moved
on by the mean velocity (n_sites pitches per cycle). Through the crossfade a
spot is two overlapping spots from neighbouring SLM traps, and the fit gives their
weighted centre. The reference frame (CH3 static at 70 MHz) gives the row's sites, its
direction (aqc.traps.row_fit) and the pitch.

Time: the beam lags the RF by the AOD's acoustic transit. A tone is at 70 MHz at the start
of its cycle, so the light puts the atom on a reference site at AWG time transit (mod
the cycle); transit is fitted from where the trajectory crosses the sites, and the time
axis is AWG time - transit, so the atom sits on a site at every multiple of the cycle.

Coordinates, in the code and the saved keys: u, v are the camera axes (sensor column and
row, px; aqc.traps and aqc.blade take and return them in (v, u) order); x, y are along
and across the row (the camera position rotated by the row angle), in um on the sensor
(px * config.CAMERA_PIXEL_UM, suffix _cam_um), which is the tweezer plane on this rig. The
residuals are x minus a straight line in time (uniform transport) and y minus the row's
line y_row(x). The spot's waist w = 2 sigma along and across the row comes from a second
fit of each point on the frame resampled onto the row grid (aqc.traps.row_grid), since the
first fit's axes are the camera's; amp is the Gaussian peak amplitude (counts).

This script only fits and saves; plot_handoff_trajectory.py makes the figure from the .npz.

With --blade (a run of handoff_strobe.py --blade) the Gaussian fits only follow the atom:
every tracked point's centre and width are then refitted by the blade method (aqc.blade,
docs/blade_method.md) on that strobe frame of every scan run, with the spot moved by
J df (the steering matrix recorded in the run's metadata.json) plus the run's drift, the
array-mean displacement of its 70/70 MHz reference from the first run's. The widths along
and across the row follow from the camera-axis ones, assuming the spot's axes lie along
the rows; amp is then the light in the blade window (counts), averaged over the runs.
The outputs are named handoff_trajectory_blade_row<row>.

With --quadratic, the along- and across-row offsets are taken relative to their best quadratic in time
instead, which removes the bow of the row (a pitch that changes linearly along it) and
leaves the variation about it. The outputs get _quad before _row<row>.

Saves <run>/analysis/<stamp>_handoff_trajectory_row<row>.json (script, git commit,
arguments; aqc.run.analysis) and .npz: t_us, u_px, v_px, waist_u_px, waist_v_px (camera
axes), x_cam_um, y_cam_um, x_residual_cam_um, y_residual_cam_um, waist_x_cam_um,
waist_y_cam_um, waist_median_cam_um (half the sum of the medians of waist_x and waist_y),
site, amp, loop, frame, transit_us (DATA_TAKING.md lists them).

Usage:
    python scripts/characterization/handoff_trajectory.py data/<MM_DD>/<stamp>_shepard_continuous_50us_1site_gauss
    python scripts/characterization/handoff_trajectory.py <run folder> --row 2
    python scripts/characterization/handoff_trajectory.py <blade run folder> --blade --row 2
    python scripts/characterization/handoff_trajectory.py <blade run folder> --blade --row 0 --quadratic
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from aqc import config
from aqc.blade import blade
from aqc.run import analysis, load
from aqc.traps import find_traps, fit_all, fit_spot, rotate, row_fit, row_grid

NV, NH = 10, 90
PX_UM = config.CAMERA_PIXEL_UM

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("run", type=Path, help="run folder of handoff_strobe.py --save or --blade")
parser.add_argument("--row", type=int, default=4, help="array row (0 = top)")
parser.add_argument("--blade", action="store_true", help="centres and widths by the blade method (handoff_strobe.py --blade run)")
parser.add_argument("--quadratic", action="store_true", help="x and y relative to their best quadratic in time (removes the bow)")
args = parser.parse_args()

meta, d = load(args.run)
# The leak frame (0 for --raw runs) also carries any room light, which the strobed frames
# already have subtracted; left in the reference it buries the array.
ref = d["ref"].astype(float) - d["leak"].astype(float)
frames = d["frames"].astype(float)
rel_us = np.asarray(d["strobe_times_us"])   # strobe centres from the reset start, AWG time
T = meta["parameters"]["t_cycle_s"] * 1e6   # us
n_frames = len(rel_us)
if rel_us[-1] - rel_us[0] > T:
    raise ValueError(f"the frames span {rel_us[-1] - rel_us[0]:.2f} us, more than one {T:g} us cycle")
# Time to the next frame; the last one wraps to frame 0 of the next cycle. handoff_strobe
# spaces the strobes evenly from the start to the end of the cycle, so the wrap step is
# shorter (one strobe width) than the rest.
dt_next = np.append(np.diff(rel_us), rel_us[0] + T - rel_us[-1])

# Sites of the row in the reference frame
pts, rows, cols = find_traps(ref, NV, NH)
pos = fit_all(ref, pts)                                # (N, 2) (v, u) px
phi = np.radians(row_fit(pos, rows, NV))
unit = np.array([np.sin(phi), np.cos(phi)])            # (v, u) step along the row
in_row = np.flatnonzero(rows == args.row)
site_pos = pos[in_row[np.argsort(cols[in_row])]]       # (NH, 2), column order
site_x, site_y = rotate(site_pos, phi)
pitch = np.nanmedian(np.diff(site_x))
y_row = np.polyfit(site_x, site_y, 1)                  # the row as a line y(x)


def fit(img, guess):
    """(v, u, amp, sigma_v, sigma_u) of the spot near guess (v, u)."""
    return np.array(fit_spot(img, *guess))


# Shift of frame 0 against the reference (mod pitch): the shift that puts the most light
# on the inner sites.
smooth = gaussian_filter(frames[0], 1.5)
shifts = np.arange(0, pitch, 0.25)
light = [map_coordinates(smooth, (site_pos[1:-1] + s * unit).T, order=1).sum() for s in shifts]
shift0 = shifts[int(np.argmax(light))]

# Direction of travel from the mean motion of the inner spots over a quarter pitch (a quarter
# cycle per site moved): over half a pitch the two directions look the same.
n_sites = meta["parameters"]["n_sites"]                # sites moved per cycle
q = n_frames // (4 * n_sites)
f0 = np.array([fit(frames[0], p) for p in site_pos[2:-2] + shift0 * unit])
fq = np.array([fit(frames[q], p) for p in f0[:, :2]])
dx = np.nanmean(rotate(fq[:, :2], phi)[0] - rotate(f0[:, :2], phi)[0])
direction = np.sign(dx)
print(f"row {args.row}: pitch {pitch:.2f} px, frame 0 shifted {shift0:.2f} px from the reference sites, "
      f"motion {'+' if direction > 0 else '-'}x ({dx:+.2f} px in {q} frames)")

# Start on the frame-0 spot nearest one site inside the end the array moves away from,
# then follow it until it passes one site inside the other end.
start_site, stop_site = (1, NH - 2) if direction > 0 else (NH - 2, 1)
f0_site = np.interp(rotate(f0[:, :2], phi)[0], site_x, np.arange(NH))
ahead = (f0_site - start_site) * direction
k0 = np.flatnonzero(ahead >= 0)[np.argmin(ahead[ahead >= 0])]
p = f0[k0, :2]
max_loops = 2 * NH // n_sites                          # twice the cycles the crossing takes
track = []
loop = 0
while True:
    if loop > max_loops:
        raise RuntimeError(f"row {args.row}: the tracked spot did not cross the row in {max_loops} cycles "
                           f"(stuck at site {site:.1f})")
    for k in range(n_frames):
        if loop or k:
            guess = p + direction * n_sites * pitch * dt_next[k - 1] / T * unit
            out = fit(frames[k], guess)
            p = out[:2] if np.isfinite(out[0]) else guess
        else:
            out = f0[k0]
        site = np.interp(rotate(p, phi)[0], site_x, np.arange(NH))
        if (site - stop_site) * direction > 0:
            break
        track.append((loop, k, *out))
    else:
        loop += 1
        continue
    break
track = np.array(track)
loops, ks = track[:, 0].astype(int), track[:, 1].astype(int)
v, u, amp, sigma_v, sigma_u = track[:, 2:].T
failed = np.isnan(v)
print(f"followed {loops.max() + 1} cycles, {len(track)} points, {failed.sum()} failed fits")
name = f"handoff_trajectory{'_blade' if args.blade else ''}{'_quad' if args.quadratic else ''}_row{args.row}"
if args.blade:
    stack = d["blade_frames"]                          # (frames, runs, pixels)
    pixels = np.asarray(d["blade_pixels"])
    index = np.full(ref.size, -1)
    index[pixels] = np.arange(len(pixels))
    index = index.reshape(ref.shape)

    def unmask(a):
        img = np.zeros(ref.shape)
        img.flat[pixels] = a
        return img

    p_ref = fit_all(unmask(d["blade_refs"][0]), pos)
    drift = np.array([np.nanmean(fit_all(unmask(r), p_ref) - p_ref, axis=0) for r in d["blade_refs"]])
    steering = [c for name, c in meta["calibrations"].items() if name.startswith("AOD_STEERING")]
    J = np.array(steering[0]["values"]["J"])           # rows u, v (camera); columns CH3, CH1; px/MHz
    uv = d["blade_shifts_hz"] / 1e6 @ J.T
    shifts = uv[:, ::-1] + drift                       # (runs, 2) (v, u) px
    print(f"blade: {len(shifts)} runs, drift up to {np.abs(drift).max():.3f} px")
    refit = np.full((len(track), 5), np.nan)
    for k in range(n_frames):
        sel = np.flatnonzero((ks == k) & ~failed)
        if len(sel):
            frames_k = np.asarray(stack[k])
            for i in sel:
                refit[i] = blade(frames_k, index, np.array([v[i], u[i]]), shifts)
    print(f"blade shift from the Gaussian centres: median {np.nanmedian(refit[:, 0] - v):+.3f} px in v, "
          f"{np.nanmedian(refit[:, 1] - u):+.3f} px in u; rms {np.nanstd(refit[:, 0] - v):.3f}, "
          f"{np.nanstd(refit[:, 1] - u):.3f} px")
    v, u, sigma_v, sigma_u, amp = refit.T
    failed = np.isnan(v)

t_awg = loops * T + rel_us[ks]
x, y = rotate(np.stack([v, u], axis=-1), phi)
site = np.interp(x, site_x, np.arange(NH))
ok = ~failed
# The atom is on a reference site whenever site is an integer: transit = where the
# straight-line fit of site(t) is an integer, mod the cycle.
b, a = np.polyfit(t_awg[ok], site[ok], 1)
transit = (-(a - np.round(a)) / b) % T
t = t_awg - transit
line = np.polyfit(t[ok], x[ok], 1)
x_residual = (x - np.polyval(line, t)) * PX_UM
y_residual = (y - np.polyval(y_row, x)) * PX_UM
speed = line[0] * PX_UM
if args.quadratic:
    x_residual -= np.polyval(np.polyfit(t[ok], x_residual[ok], 2), t)
    y_residual -= np.polyval(np.polyfit(t[ok], y_residual[ok], 2), t)
print(f"acoustic transit {transit:.2f} us (mod {T:g}); speed {speed:+.4f} um/us on the sensor "
      f"({b * T:+.4f} site per cycle); along-row residual rms {np.nanstd(x_residual[ok]):.3f} um, "
      f"across-row rms {np.nanstd(y_residual[ok]):.3f} um")

# Spot shape along and across the row: refit every tracked point on the frames resampled
# onto the row grid, where the Gaussian's axes lie along and across the rows (the fits
# above are along the camera axes, ~17 deg off).
if args.blade:
    # sigma_u^2 = c^2 sigma_x^2 + s^2 sigma_y^2, sigma_v^2 = s^2 sigma_x^2 + c^2 sigma_y^2
    c2, s2 = np.cos(phi) ** 2, np.sin(phi) ** 2
    sigma_x = np.sqrt((c2 * sigma_u ** 2 - s2 * sigma_v ** 2) / (c2 - s2))
    sigma_y = np.sqrt((c2 * sigma_v ** 2 - s2 * sigma_u ** 2) / (c2 - s2))
else:
    grid, _, grid_pitch = row_grid(pos, rows, cols, NV)
    x_all, y_all = rotate(pos, phi)
    gx, gy = x - np.nanmin(x_all) + grid_pitch, y - np.nanmin(y_all) + grid_pitch   # grid column, row of each point
    on_grid = [map_coordinates(f, grid, order=1) for f in frames]
    shape = np.array([fit_spot(on_grid[k], gy[i], gx[i]) if ok[i] else (np.nan,) * 5 for i, k in enumerate(ks)])
    sigma_x, sigma_y = shape[:, 4], shape[:, 3]
# Tweezer waist w = 2 sigma (1/e^2 intensity radius)
waist_x, waist_y = 2 * sigma_x * PX_UM, 2 * sigma_y * PX_UM
waist = (np.nanmedian(waist_x) + np.nanmedian(waist_y)) / 2
print(f"waist median {np.nanmedian(waist_x):.3f} um along the row, {np.nanmedian(waist_y):.3f} um across (sensor)")

stem = analysis(args.run, name, args)
np.savez(stem.with_suffix(".npz"), method="blade" if args.blade else "gaussian", t_us=t, u_px=u, v_px=v,
         waist_u_px=2 * sigma_u, waist_v_px=2 * sigma_v, x_cam_um=x * PX_UM, y_cam_um=y * PX_UM,
         x_residual_cam_um=x_residual, y_residual_cam_um=y_residual, waist_x_cam_um=waist_x, waist_y_cam_um=waist_y,
         waist_median_cam_um=waist, site=site, amp=amp, loop=loops, frame=ks, transit_us=transit, row=args.row,
         run=str(args.run))
print(f"wrote {stem.with_suffix('.npz')}")
