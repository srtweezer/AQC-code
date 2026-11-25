"""Align the tweezer rows with the one-site step of the transport AOD, closing a loop on the WGS design angle.

The 10x90 array (aqc.wgs.PITCH_UM, 50.6 um pitch) is solved with aqc.wgs at design angle theta and displayed
with an 8 px y blaze. The misalignment is measured where the handoff happens: at the pairs
of tones the continuous Shepard plays together (aqc.pair.shepard_pairs, tone A =
--center-mhz + (1/2 + p) step, tone B = A - step, config.CH3_SITE_STEP_MHZ, for each
crossfade position p in --positions). Every trap is located at both tones by the blade
method (aqc.pair), and the pair mismatch E (aqc.handoff.mismatch) is the B centre of the
trap that lands on A trap (r, c) minus A trap (r, c). A lattice turned by alpha against
the steering puts every B trap beside its partner; alpha and the step error are fitted
to the E of all pairs together (aqc.handoff.fit_step_angle), so an error of the site step
doesn't bias alpha. theta is stepped by alpha / config.ROW_ANGLE_PER_DESIGN_DEG until
|alpha| < --tol twice in a row at the same theta.

The fit measures the angle against the direction of the one-site step itself, which is
what the handoff needs. Steering slopes from a wide sweep (the method before 2026-11-23)
average the AOD's curvature over +-8 MHz and missed ~3.5 mrad on the vertical array.

With --vertical the array is 90 rows x 10 columns instead, its long axis along design y.
CH3 steers along design x and CH1 along design y, so the loop then aligns the 10 lines of
90 traps (the columns) with CH1, at the pairs of config.CH1_SITE_STEP_MHZ. Rotating theta
turns the whole lattice, so it can align only one axis; the AODs are not quite
orthogonal. After the loop the other channel is swept (aqc.steering.sweep, --f-span /
--f-step around 70 MHz, a 70 MHz reference before every point) and its misalignment with
the short lines (across the long ones, from the affine fit of the lattice) is reported,
not corrected.

--blaze-y / --blaze-x set the blaze the pattern is displayed with (default 8 px along y).

Both angles are measured on the camera. The trap plane is imaged onto it by a near-linear
map (rotation, unequal x/y magnification), which keeps parallel lines parallel, so
alpha = 0 on the camera means alpha = 0 in the trap plane, even though the camera row
angle itself (~20.7 deg) has nothing to do with theta (~1.7 deg).

Setup: the SLM beam imaged onto the Basler (free-run), AODs on CH1/CH3 at 1.6 V peak.
Nothing else may hold the AWG (stop aod_tone.py first). Pick an exposure that keeps the
brightest trap below saturation at every tone. The site step must be within a fraction of
a site (aod_site_step.py), so that each trap's partner is its nearest neighbour.

Saves data/<MM_DD>/<timestamp>_align_rows.png and, on convergence,
aqc/calibration/phase-patterns/<timestamp>_wgs_aligned.npy (phase in radians, no blaze;
display it with slm_upload.py and the same blaze), or
phase-patterns/vertical/<timestamp>_wgs_90x10_aligned.npy with --vertical. Leaves the last
pattern on the SLM and the AWG stopped. About 30 s per pair and measurement with the
defaults, plus the WGS solve.

Usage:
    python scripts/calibration/align_rows_to_aod.py --exposure-us 200
    python scripts/calibration/align_rows_to_aod.py --exposure-us 200 --theta0 1.9
    python scripts/calibration/align_rows_to_aod.py --exposure-us 150 --vertical --theta0 -1.188 \\
        --blaze-x -8 --blaze-y 0
"""

import argparse
from datetime import datetime
from pathlib import Path

import e2style as e2
import matplotlib.pyplot as plt
import numpy as np

from aqc import config, pair
from aqc.blaze import blaze
from aqc.data import save_plot
from aqc.handoff import fit_step_angle
from aqc.hardware.awg import aods
from aqc.hardware.camera import Camera
from aqc.hardware.slm import send_phase
from aqc.steering import F_CENTER, grab, play, slopes, sweep
from aqc.traps import affine_residual, find_traps, fit_all
from aqc.wgs import PITCH_UM, run_wgs

PHASE_DIR = Path(__file__).resolve().parents[2] / "aqc" / "calibration" / "phase-patterns"
FULL_SCALE = 4095        # Mono12
# On the camera the array is NV lines of NH traps along its long axis: rows by default,
# columns with --vertical.
NV, NH = 10, 90
AMPLITUDE = 1.6          # V peak, the CW drive of aod_tone.py
PX_UM = config.CAMERA_PIXEL_UM

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--exposure-us", type=float, required=True, help="camera exposure, us")
parser.add_argument("--theta0", type=float, default=1.6899, help="starting WGS design angle, deg")
# Successive blade measurements of one pattern agree to ~0.1 mrad (0.006 deg).
parser.add_argument("--tol", type=float, default=0.01, help="|alpha| tolerance, deg")
parser.add_argument("--max-iter", type=int, default=6, help="max measurements")
parser.add_argument("--center-mhz", type=float, default=70.0, help="centre frequency of the Shepard, MHz")
parser.add_argument("--positions", type=float, nargs="+", default=(-0.25, 0.0, 0.25),
                    help="crossfade positions of the pairs (0 = middle, +-0.5 = ends)")
parser.add_argument("--blade", type=float, nargs=2, default=(0.5, 0.05), metavar=("SPAN", "STEP"),
                    help="blade offsets -SPAN..SPAN MHz in STEP steps on each channel")
parser.add_argument("--blade-average", type=int, default=2, help="frames averaged per tone and blade offset")
parser.add_argument("--f-span", type=float, default=8.0, help="cross-channel sweep half-span around 70 MHz, MHz")
parser.add_argument("--f-step", type=float, default=0.5, help="cross-channel sweep step, MHz")
parser.add_argument("--average", type=int, default=8, help="frames averaged per grab in the cross-channel sweep")
parser.add_argument("--vertical", action="store_true", help="90 rows x 10 columns, aligned with CH1")
parser.add_argument("--blaze-y", type=int, default=8, help="blaze period along y, px (0 = none)")
parser.add_argument("--blaze-x", type=int, default=0, help="blaze period along x, px (0 = none)")
args = parser.parse_args()

# The channel steering along the NV lines of NH traps, and the one steering across them
STEER, CROSS = (1, 3) if args.vertical else (3, 1)
STEP_MHZ = config.CH1_SITE_STEP_MHZ if args.vertical else config.CH3_SITE_STEP_MHZ
# The ends of the long axis of the vertical array reach toward the edge of the SLM's
# diffraction field and come out at ~0.1 of the brightest traps on a uniform-weight solve.
SPOT_FRAC = 0.1 if args.vertical else 0.3
sc = pair.scan(*args.blade, np.load(config.AOD_STEERING_VERTICAL_CAL if args.vertical else config.AOD_STEERING_CAL)["J"])
tones = pair.shepard_pairs(args.center_mhz * 1e6, STEP_MHZ * 1e6, args.positions)


def measure(awg, cam):
    """Misalignment alpha (deg, camera) of the pattern on the SLM from the pair mismatch at every Shepard pair."""
    pairs = []
    for fa, fb in tones:
        m = pair.take(awg, cam, (fa, fb), args.vertical, sc, args.blade_average, NV, NH, SPOT_FRAC)
        q = pair.analyse(pair.fit(m, sc), NV, NH)
        q["a"], q["theta"], _ = fit_step_angle(q["e"], q["s"], q["r"])
        pairs.append(q)
    a, theta, _ = fit_step_angle(*(np.array([q[k] for q in pairs]) for k in "esr"))
    per_pair = np.degrees([q["theta"] for q in pairs])
    return dict(alpha=float(np.degrees(theta)), per_pair=per_pair, step_err_khz=a * STEP_MHZ * 1e3,
                alpha_sem=float(per_pair.std() / np.sqrt(len(pairs))), pairs=pairs)


log = []
theta, hits = args.theta0, 0
with aods(amplitudes=(AMPLITUDE, AMPLITUDE)) as awg, Camera(args.exposure_us, hardware_trigger=False) as cam:
    for _ in range(args.max_iter):
        n_horiz, n_vert = (NV, NH) if args.vertical else (NH, NV)
        phase, _ = run_wgs(n_horiz, n_vert, PITCH_UM, iterations=120, scal=4, momentum=0.6, fix_phase_at=0.5,
                           angle_deg=theta, seed=0, verbose=False)
        send_phase(phase + blaze(args.blaze_y, args.blaze_x))
        print(f"design angle {theta:+.4f} deg:")
        m = measure(awg, cam)
        m["theta"] = theta
        log.append(m)
        print(f"    alpha {m['alpha']:+.4f} +- {m['alpha_sem']:.4f} deg (pairs "
              f"{', '.join(f'{x:+.4f}' for x in m['per_pair'])}); site step off by {m['step_err_khz']:+.1f} kHz",
              flush=True)
        if abs(m["alpha"]) < args.tol:
            hits += 1
            if hits == 2:
                break
        else:
            hits = 0
            theta += m["alpha"] / config.ROW_ANGLE_PER_DESIGN_DEG

    # The other axis, on the last pattern: CROSS steering against the lattice vector from
    # one line to the next (the affine fit absorbs any shear between the two axes).
    play(awg, F_CENTER)
    ref = grab(cam, args.average)
    peaks, rows, cols = find_traps(ref, NV, NH, SPOT_FRAC)
    pos = fit_all(ref, peaks)
    _, M = affine_residual(pos, rows, cols)
    across_deg = np.degrees(np.arctan2(M[0, 1], M[1, 1]))
    offs = np.round(np.arange(-args.f_span, args.f_span + args.f_step / 2, args.f_step), 6)
    S = slopes(offs, sweep(awg, cam, pos, offs, args.average, channel=CROSS))
    alpha_cross = (np.degrees(np.arctan2(S[:, 0], S[:, 1])) - across_deg + 90) % 180 - 90
    n_cross = int(np.isfinite(alpha_cross).sum())
    cross = dict(alpha=float(np.nanmedian(alpha_cross)),
                 sem=float(1.2533 * np.nanstd(alpha_cross) / np.sqrt(n_cross)), n=n_cross)
    print(f"CH{CROSS} vs the lattice across the lines ({across_deg:.4f} deg on camera): "
          f"alpha {cross['alpha']:+.4f} +- {cross['sem']:.4f} deg over {n_cross} traps")

converged = hits == 2
stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
if converged:
    out_dir = PHASE_DIR / "vertical" if args.vertical else PHASE_DIR
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{stamp}_wgs_{'90x10_' if args.vertical else ''}aligned.npy"
    np.save(path, phase.astype(np.float32))
    print(f"Aligned at design angle {theta:.4f} deg. Saved {path}")
else:
    print(f"Not converged within {args.max_iter} measurements (last alpha {log[-1]['alpha']:+.4f} deg); "
          f"phase not saved")

last = log[-1]
e2.apply()
colors = [e2.OKABE_ITO[c] for c in ("blue", "vermilion", "green", "orange", "purple")]
fig = plt.figure(figsize=(180 * e2.MM, 130 * e2.MM), layout="constrained")
gs = fig.add_gridspec(2, 2)
ax = fig.add_subplot(gs[0, 0])
ax.errorbar(range(len(log)), [m["alpha"] for m in log], [m["alpha_sem"] for m in log], fmt="o-", ms=3,
            color=e2.OKABE_ITO["blue"])
for i, m in enumerate(log):
    ax.annotate(f"{m['theta']:.4f}", (i, m["alpha"]), fontsize=6, textcoords="offset points", xytext=(3, 3))
ax.axhspan(-args.tol, args.tol, color=e2.GREY.face, alpha=0.5, lw=0)
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("measurement (labels: design angle, deg)")
ax.set_ylabel("alpha = step direction − rows (deg)")
ax.set_title("misalignment per measurement")
# The across-transport part of E per trap, last measurement, mid pair: a misalignment
# alpha makes it alpha x pitch everywhere; what varies is the SLM traps' own scatter.
q = last["pairs"][int(np.argmin(np.abs(args.positions)))]
u = np.nanmean(q["s"], axis=(0, 1))
u /= np.linalg.norm(u)
across = q["e"] @ np.array([u[1], -u[0]]) * PX_UM
ax = fig.add_subplot(gs[0, 1])
lim = np.nanpercentile(np.abs(across), 99)
im = ax.imshow(across, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
fig.colorbar(im, ax=ax, label="µm on the sensor")
ax.set_xlabel("tone A trap along the line")
ax.set_ylabel("line")
ax.set_title(f"last measurement, mid pair: mismatch across the transport, mean {np.nanmean(across):+.3f} µm")
ax = fig.add_subplot(gs[1, :])
for (fa, _), q, col in zip(tones, last["pairs"], colors):
    u = np.nanmean(q["s"], axis=(0, 1))
    u /= np.linalg.norm(u)
    ax.plot(np.nanmean(q["e"] @ np.array([u[1], -u[0]]), axis=0) * PX_UM, lw=0.8, color=col,
            label=f"tone A {fa / 1e6:.4f} MHz: {np.degrees(q['theta']):+.4f} deg")
ax.axhline(0, color=e2.GREY.edge, lw=0.5)
ax.set_xlabel("tone A trap along the line")
ax.set_ylabel("mismatch across, mean over lines (µm)")
ax.set_title(f"last measurement (design {last['theta']:.4f} deg), every pair")
ax.legend(fontsize=6)
fig.suptitle(f"{'Columns' if args.vertical else 'Rows'} vs the CH{STEER} one-site step "
             f"({STEP_MHZ:.4f} MHz, Shepard pairs around {args.center_mhz:g} MHz): "
             f"{'converged' if converged else 'NOT converged'} at design angle {last['theta']:.4f} deg; "
             f"CH{CROSS} vs the other axis {cross['alpha']:+.3f} deg\n"
             f"blaze y {args.blaze_y:+d} px, x {args.blaze_x:+d} px; exposure {args.exposure_us:g} µs; "
             f"blade ±{args.blade[0]:g} MHz / {args.blade[1] * 1e3:g} kHz, {args.blade_average} frames per offset")
save_plot(fig, "align_rows", stamp)
