"""Straighten the tweezer lattice by camera feedback on the WGS target positions.

The 10x90 array (aqc.wgs.PITCH_UM, 50.6 um pitch, design angle --theta) is solved with aqc.wgs and displayed
with an 8 px y blaze, with fixed weights (--weights, default uniform). Each trap's
Gaussian centre is fitted on the camera, and its offset from the best-fit affine lattice
(aqc.traps.affine_residual) is the error: the array is not moved, turned or rescaled,
only its nonlinear distortion is removed. Every round nudges each trap's WGS target by
-gain x that error and solves again. The target offsets are in padded-grid pixels and
are rounded to the grid in the solve, so they accumulate below a grid pixel (0.22 um in
y, 0.14 um in x) but the position can't be finer than that.

A camera offset becomes a target offset through two Jacobians, d(grid) = J_grid
J_cam^-1 d(camera): J_cam (camera px per column and row) is the linear part of the
fitted affine map, and J_grid (grid px per column and row) follows from the design
lattice. No sign or scale is set by hand. Before any feedback, a fiducial moves design
rows 3-5, cols 20-29 by FIDUCIAL_GRID_PX in grid y and checks that they land where the
Jacobians predict. A wrong sign or a row/column mix-up fails it, and the script stops.

Ported from slm_local/posfeedback_08_27/position_feedback.py, which took the 2026-08-27
array from 6.7 px to 0.27 px mean offset (0.34 um/px camera) in 5 rounds.

With --correct, every solve is the corrected WGS, as in homogenize.py --correct.

With --vertical the array is the 90 rows x 10 columns one (align_rows_to_aod.py
--vertical). The camera finds it as 10 lines of 90 traps, and the feedback works in that
camera order (the fiducial's rows and columns are camera lines and traps along them); the
offsets are put in design order (aqc.traps.vertical_design_order) for the solve and the
saved file. --weights is then in design order (90 x 10), as homogenize.py saves it.

--blaze-y / --blaze-x set the blaze the pattern is displayed with (default 8 px along y).

Setup: the SLM beam imaged onto the Basler (free-run), AODs on CH1/CH3 at 70 MHz and
1.6 V peak. Nothing else may hold the AWG (stop aod_tone.py first). Pick an exposure that
keeps the brightest trap below saturation.

Saves the best round's offsets (design order) to aqc/calibration/<timestamp>_position_offsets.npz
(use them with homogenize.py --offsets) and data/<MM_DD>/<timestamp>_position_feedback.png. Leaves the
best round's pattern on the SLM and the AWG stopped.

Usage:
    python scripts/calibration/position_feedback.py --exposure-us 170
    python scripts/calibration/position_feedback.py --exposure-us 170 --correct \\
        --weights aqc/calibration/phase-patterns/<timestamp>_wgs_cwgs_homogenized_weights.npy
    python scripts/calibration/position_feedback.py --exposure-us 150 --correct --vertical \\
        --theta -1.1877 --blaze-x -8 --blaze-y 0 \\
        --weights aqc/calibration/phase-patterns/vertical/<timestamp>_wgs_90x10_cwgs_homogenized_weights.npy
"""

import argparse
from datetime import datetime
from pathlib import Path

import e2style as e2
import matplotlib.pyplot as plt
import numpy as np

from aqc import aberration, config
from aqc.blaze import blaze
from aqc.data import save_plot
from aqc.hardware.awg import aods
from aqc.hardware.camera import Camera
from aqc.hardware.slm import send_phase
from aqc.steering import F_CENTER, grab, play
from aqc.traps import affine_residual, find_traps, fit_all, vertical_design_order
from aqc.wgs import PITCH_UM, run_wgs, trap_indices

CAL_DIR = Path(__file__).resolve().parents[2] / "aqc" / "calibration"
# On the camera the array is NV lines of NH traps along its long axis: rows by default,
# columns with --vertical.
NV, NH = 10, 90
SCAL = 4
AMPLITUDE = 1.6          # V peak, the CW drive of aod_tone.py
FIDUCIAL_GRID_PX = 5     # ~3 camera px: within aqc.traps.fit_spot's reach of the unshifted centre
FIDUCIAL_TOL = 0.3       # max |measured - predicted| / |predicted|
ROWS, COLS = (a.ravel() for a in np.mgrid[0:NV, 0:NH])

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--exposure-us", type=float, required=True, help="camera exposure, us")
parser.add_argument("--theta", type=float, default=3.1910,
                    help="WGS design angle, deg (default: that of 20261028_182104_wgs_aligned.npy)")
parser.add_argument("--rounds", type=int, default=10, help="feedback rounds after round 0")
parser.add_argument("--gain", type=float, default=0.6, help="fraction of the measured offset corrected per round")
parser.add_argument("--average", type=int, default=16, help="frames averaged per measurement")
parser.add_argument("--weights", type=Path, help="WGS weights (.npy, design order), held fixed; default uniform")
parser.add_argument("--correct", action="store_true", help="correct the aberrations of config.ABERRATION_CAL")
parser.add_argument("--vertical", action="store_true", help="90 rows x 10 columns")
parser.add_argument("--blaze-y", type=int, default=8, help="blaze period along y, px (0 = none)")
parser.add_argument("--blaze-x", type=int, default=0, help="blaze period along x, px (0 = none)")
args = parser.parse_args()

N_HORIZ, N_VERT = (NV, NH) if args.vertical else (NH, NV)
# Camera index of every design trap; the feedback works in camera order
ORDER = vertical_design_order(NV, NH) if args.vertical else np.arange(NV * NH)
# The ends of the vertical array come out at ~0.1 of the brightest traps on a uniform-weight solve
SPOT_FRAC = 0.1 if args.vertical else 0.3
rr, cc = trap_indices(N_HORIZ, N_VERT, PITCH_UM, SCAL, angle_deg=args.theta)
correction = None
if args.correct:
    if config.ABERRATION_CAL is None:
        raise RuntimeError("config.ABERRATION_CAL is not set; run scripts/calibration/aberration_scan.py")
    correction = aberration.correction(aberration.load(config.ABERRATION_CAL), rr, cc, SCAL, args.blaze_y, args.blaze_x)
weights = None if args.weights is None else np.load(args.weights)

# Grid px (y, x) per camera (column, row) step, from the design lattice in camera order
cam_index = np.argsort(ORDER)
J_GRID = np.linalg.lstsq(np.column_stack([COLS, ROWS, np.ones(NV * NH)]),
                         np.column_stack([rr.ravel()[cam_index], cc.ravel()[cam_index]]), rcond=None)[0][:2].T


def to_design(off):
    """Camera-ordered offsets (NV, NH, 2) in design order (N_VERT, N_HORIZ, 2)."""
    return off.reshape(-1, 2)[ORDER].reshape(N_VERT, N_HORIZ, 2)


def solve(off):
    phase, _ = run_wgs(N_HORIZ, N_VERT, PITCH_UM, weights=weights, iterations=120, scal=SCAL, momentum=0.6,
                       fix_phase_at=0.5, angle_deg=args.theta, seed=0, verbose=False,
                       correction=correction, offset_px=to_design(off))
    return phase


def measure(cam, phase, guess):
    """Gaussian centres (NV * NH, 2) in camera order, fitted from guess (None: find the traps)."""
    send_phase(phase + blaze(args.blaze_y, args.blaze_x))
    frame = grab(cam, args.average)
    if guess is None:
        pts, rows, cols = find_traps(frame, NV, NH, SPOT_FRAC)
        guess = np.empty((NV, NH, 2))
        guess[rows, cols] = pts
        guess = guess.reshape(-1, 2)
    pos = fit_all(frame, guess)
    if not np.all(np.isfinite(pos)):
        raise RuntimeError(f"Gaussian fit failed for {np.isnan(pos[:, 0]).sum()} of {NV * NH} traps")
    return pos


def cam_per_grid(M):
    """2x2 map from a target offset (grid px) to the camera shift (px) it causes."""
    return M[:, :2] @ np.linalg.inv(J_GRID)


def offsets_um(d, M):
    """Magnitude (NV, NH) of camera offsets d (N, 2), in focal-plane um (camera scale from the fitted pitch)."""
    um_per_px = PITCH_UM / np.linalg.norm(M[:, 0])
    return np.linalg.norm(d, axis=1).reshape(NV, NH) * um_per_px


off = np.zeros((NV, NH, 2))
log = []
with aods(amplitudes=(AMPLITUDE, AMPLITUDE)) as awg, Camera(args.exposure_us, hardware_trigger=False) as cam:
    play(awg, F_CENTER)
    phase = solve(off)
    pos = measure(cam, phase, None)
    d, M = affine_residual(pos, ROWS, COLS)

    fid = np.zeros((NV, NH, 2))
    fid[3:6, 20:30, 0] = FIDUCIAL_GRID_PX
    block = (fid[..., 0] != 0).ravel()
    moved = measure(cam, solve(fid), pos) - pos
    pred = cam_per_grid(M) @ [FIDUCIAL_GRID_PX, 0]
    got = moved[block].mean(axis=0)
    rel = np.linalg.norm(got - pred) / np.linalg.norm(pred)
    print(f"Fiducial: rows 3-5, cols 20-29 moved {FIDUCIAL_GRID_PX} grid px in y. Camera (dy, dx) "
          f"predicted ({pred[0]:+.2f}, {pred[1]:+.2f}) px, measured ({got[0]:+.2f}, {got[1]:+.2f}) px "
          f"({rel:.1%} off); other traps moved {np.linalg.norm(moved[~block], axis=1).mean():.2f} px")
    if rel > FIDUCIAL_TOL:
        raise RuntimeError("fiducial landed away from the prediction; not feeding back")

    print(f"{'round':>5} {'mean px':>8} {'mean um':>8} {'rms px':>7} {'max px':>7} {'max |offset| grid px':>21}")
    for rnd in range(args.rounds + 1):
        if rnd > 0:
            step = -args.gain * (np.linalg.inv(cam_per_grid(M)) @ d.T).T.reshape(NV, NH, 2)
            off = off + step
            phase = solve(off)
            # Seed the fits where the step should have put each trap, so big first steps aren't lost.
            pos = measure(cam, phase, pos + (cam_per_grid(M) @ step.reshape(-1, 2).T).T)
            d, M = affine_residual(pos, ROWS, COLS)
        r = np.linalg.norm(d, axis=1)
        log.append(dict(off=off.copy(), phase=phase, d=d, M=M))
        print(f"{rnd:5d} {r.mean():8.3f} {offsets_um(d, M).mean():8.4f} {np.sqrt((r ** 2).mean()):7.3f} "
              f"{r.max():7.3f} {np.abs(off).max():21.2f}", flush=True)
    best = min(range(len(log)), key=lambda i: np.linalg.norm(log[i]["d"], axis=1).mean())
    send_phase(log[best]["phase"] + blaze(args.blaze_y, args.blaze_x))

hist = np.array([offsets_um(m["d"], m["M"]).mean() for m in log])
print(f"Best: round {best} at {hist[best]:.4f} um mean offset (round 0 {hist[0]:.4f} um)")

stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
path = CAL_DIR / f"{stamp}_position_offsets.npz"
np.savez(path, offset_px=to_design(log[best]["off"]), theta=args.theta, correct=args.correct,
         vertical=args.vertical, blaze_y_px=args.blaze_y, blaze_x_px=args.blaze_x)
print(f"Saved {path}")

e2.apply()
u0, ub = offsets_um(log[0]["d"], log[0]["M"]), offsets_um(log[best]["d"], log[best]["M"])
fig = plt.figure(figsize=(180 * e2.MM, 190 * e2.MM), layout="constrained")
gs = fig.add_gridspec(4, 1, height_ratios=[1.2, 1, 1, 1])
ax = fig.add_subplot(gs[0])
ax.plot(hist, "o-", ms=3, color=e2.OKABE_ITO["blue"])
ax.set_ylim(0, None)
ax.set_xlabel("round")
ax.set_ylabel("mean offset from affine lattice (µm)")
ax.set_title(f"round 0 {hist[0]:.3f} µm -> round {best} {hist[best]:.3f} µm")
for row, (u, label) in enumerate([(u0, "round 0"), (ub, f"round {best}")], start=1):
    ax = fig.add_subplot(gs[row])
    im = ax.imshow(u, cmap="viridis", vmin=0, vmax=u0.max(), aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="offset (µm)")
    ax.set_ylabel("row")
    ax.set_title(f"{label}: mean {u.mean():.3f} µm, max {u.max():.3f} µm")
ax = fig.add_subplot(gs[3])
im = ax.imshow(np.linalg.norm(log[best]["off"], axis=2), cmap="magma", aspect="auto", interpolation="nearest")
fig.colorbar(im, ax=ax, label="|target offset| (grid px)")
ax.set_xlabel("column")
ax.set_ylabel("row")
ax.set_title(f"round {best} target offsets")
fig.suptitle(f"{'Corrected ' if args.correct else ''}WGS position feedback, {'90x10' if args.vertical else '10x90'} "
             f"(camera lines), design angle {args.theta:.4f} deg, blaze y {args.blaze_y:+d} x {args.blaze_x:+d} px, "
             f"gain {args.gain:g}\n{args.weights.name if args.weights else 'uniform weights'}; "
             f"exposure {args.exposure_us:g} us, {args.average} frames per measurement")
save_plot(fig, "position_feedback", stamp)
