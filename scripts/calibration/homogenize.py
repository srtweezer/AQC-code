"""Equalize the trap peaks of the tweezer array by camera feedback on the WGS weights.

The 10x90 array (aqc.wgs.PITCH_UM, 50.6 um pitch, design angle --theta) is solved with aqc.wgs and displayed
with an 8 px y blaze. The traps are found on the round-0 frame; after that, every round
re-fits each trap's centre starting from its previous one (the traps shift slightly
between solves), and measures each trap's peak K there. A fresh search every round would
also pick up stray spots, which pass find_traps' cuts once the array is more uniform. The WGS target weights are
then updated as w *= (mean K / K) ** gain and the array is solved again. Round 0 with
uniform weights is the plain solve, i.e. the pattern align_rows_to_aod.py saved for that
angle.

Every round is solved from scratch with the same seed. With the trap phases frozen
halfway through the solve (fix_phase_at), this keeps the solution close between rounds,
so the measured change follows the weight update. A re-solve still moves the other traps
a little, and the camera is noisy, so the gain is dropped after the first rounds to stop
the loop chasing both.

With --correct, every solve is the corrected WGS (aqc.wgs.run_wgs with correction): each
trap gets the field-dependent aberration correction of config.ABERRATION_CAL
(scripts/calibration/aberration_scan.py, whose reference spots must cover the field the
array sits in with this blaze). Start it from uniform weights: weights homogenized without
the correction don't carry over.

With --offsets, every solve moves each trap's target by the offsets of
position_feedback.py (which must have used the same --theta), so the array stays on the
straightened lattice while its weights are equalized.

Camera trap (row, col) is design trap (row, col): checked 2026-10-29 by dimming design
rows 2-4, cols 6-13, which dimmed the same rows and cols on the camera.

With --vertical the array is 90 rows x 10 columns (long axis along design y), as aligned by
align_rows_to_aod.py --vertical; pass that design angle as --theta. On the camera it is
found as 10 lines of 90 traps: line i is design column i, and trap j along a line is design
row 89 - j. Checked 2026-11-12 by matching the lattice steps of the vertical array (blaze
x -8, y +8 px) to those of the horizontal one in its steering calibration, whose camera
indices are the design ones (aqc.traps.vertical_design_order). --offsets must then come
from position_feedback.py --vertical (design order, 90 x 10).

--blaze-y / --blaze-x set the blaze the pattern is displayed with (default 8 px along y).

With --state, the pattern is not solved by WGS but rebuilt from a hologram state
(focus_flatten.py or handoff_calibration.py _state.npz): the trap positions, lenses,
aberration correction and trap phases stay as they are, and only the amplitudes are
refined (aqc.wgs.refine_amplitudes) towards the state's target intensities times the
weights. A WGS re-solve would lose the lenses and the handoff moves. The blaze is the
state's; --theta, --correct, --offsets and --weights0 don't apply. Needed on 2026-11-12:
the handoff moves took the vertical array from 11.3% to 16.0% std/mean of integrated trap
power, although they held every trap's simulated intensity.

Setup: the SLM beam imaged onto the Basler (free-run), AODs on CH1/CH3 at 70 MHz and
1.6 V peak. Nothing else may hold the AWG (stop aod_tone.py first). Pick an exposure that
keeps the brightest trap below saturation at round 0.

Saves the best round's phase (radians, no blaze; display it with slm_upload.py and the
same blaze) and its weights to aqc/calibration/phase-patterns/<timestamp>_wgs_homogenized.npy
and <timestamp>_wgs_homogenized_weights.npy (_wgs_cwgs_homogenized with --correct,
_wgs_pos_homogenized with --offsets, vertical/<timestamp>_wgs_90x10_homogenized with
--vertical; with --state, <timestamp>_<state name>_homogenized.npy and its _state.npz next
to the state), and data/<MM_DD>/<timestamp>_homogenize.png.
Leaves the best pattern on the SLM and the AWG stopped.

Usage:
    python scripts/calibration/homogenize.py --exposure-us 100
    python scripts/calibration/homogenize.py --exposure-us 100 --rounds 10 \\
        --weights0 aqc/calibration/phase-patterns/<timestamp>_wgs_homogenized_weights.npy
    python scripts/calibration/homogenize.py --exposure-us 100 --correct
    python scripts/calibration/homogenize.py --exposure-us 170 --correct \
        --offsets aqc/calibration/<timestamp>_position_offsets.npz
    python scripts/calibration/homogenize.py --exposure-us 150 --vertical --theta -1.1877 \
        --blaze-x -8 --blaze-y 0
    python scripts/calibration/homogenize.py --exposure-us 150 --vertical --correct --theta -1.1877 \\
        --blaze-x -8 --blaze-y 0
    python scripts/calibration/homogenize.py --exposure-us 150 --vertical \\
        --state aqc/calibration/phase-patterns/vertical/<timestamp>_handoff_state.npz
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
from aqc.intensity import peaks
from aqc.steering import F_CENTER, grab, play
from aqc.traps import find_traps, fit_all, vertical_design_order
from aqc.wgs import PITCH_UM, corrected_waves, refine_amplitudes, run_wgs, trap_indices

PHASE_DIR = Path(__file__).resolve().parents[2] / "aqc" / "calibration" / "phase-patterns"
# On the camera the array is NV lines of NH traps along its long axis: rows by default,
# columns with --vertical.
NV, NH = 10, 90
AMPLITUDE = 1.6          # V peak, the CW drive of aod_tone.py
# Gain schedule and per-round step limit of andrii-test/calibration/uniformize90.py, which
# took the same array to its wgs_aligned_uniform.npy on 2026-10-22.
GAIN, GAIN_LATE, LATE_FROM = 0.5, 0.15, 6
MAX_STEP = 1.7           # max factor a weight changes by in one round
REFINE = 8               # amplitude refinements per round with --state, as focus_flatten.py

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--exposure-us", type=float, required=True, help="camera exposure, us")
parser.add_argument("--theta", type=float, default=3.1910,
                    help="WGS design angle, deg (default: that of 20261028_182104_wgs_aligned.npy)")
parser.add_argument("--rounds", type=int, default=20, help="feedback rounds (measurements)")
parser.add_argument("--average", type=int, default=16, help="frames averaged per measurement")
parser.add_argument("--weights0", type=Path, help="starting weights (.npy, NV x NH); default uniform")
parser.add_argument("--correct", action="store_true", help="correct the aberrations of config.ABERRATION_CAL")
parser.add_argument("--offsets", type=Path, help="per-trap target offsets (.npz from position_feedback.py)")
parser.add_argument("--vertical", action="store_true", help="90 rows x 10 columns")
parser.add_argument("--blaze-y", type=int, default=8, help="blaze period along y, px (0 = none)")
parser.add_argument("--blaze-x", type=int, default=0, help="blaze period along x, px (0 = none)")
parser.add_argument("--state", type=Path, help="hologram state (.npz) to homogenize without a WGS re-solve")
args = parser.parse_args()
if args.state and (args.correct or args.offsets or args.weights0):
    raise ValueError("--state carries its own correction, positions and target; not with --correct, --offsets, --weights0")

N_HORIZ, N_VERT = (NV, NH) if args.vertical else (NH, NV)
# The ends of the long axis of the vertical array reach toward the edge of the SLM's
# diffraction field and come out at ~0.1 of the brightest traps on a uniform-weight solve.
SPOT_FRAC = 0.1 if args.vertical else 0.3


def to_design(k):
    """Camera-ordered (NV, NH) per-trap values in design order (N_VERT, N_HORIZ); see the docstring."""
    return k.ravel()[vertical_design_order(NV, NH)].reshape(N_VERT, N_HORIZ) if args.vertical else k

offsets = None
if args.offsets is not None:
    cal = np.load(args.offsets)
    if not np.isclose(cal["theta"], args.theta):
        raise ValueError(f"{args.offsets.name} was measured at theta {float(cal['theta'])}, not {args.theta}")
    offsets = cal["offset_px"]

blaze_y, blaze_x = args.blaze_y, args.blaze_x
if args.state:
    st = dict(np.load(args.state))
    blaze_y, blaze_x = int(st["blaze_y_px"]), int(st["blaze_x_px"])
    M = corrected_waves(st["rows"], st["cols"], int(st["scal"]), st["correction"], st["lens_rad"])

correction = None
if args.correct:
    if config.ABERRATION_CAL is None:
        raise RuntimeError("config.ABERRATION_CAL is not set; run scripts/calibration/aberration_scan.py")
    rr, cc = trap_indices(N_HORIZ, N_VERT, PITCH_UM, 4, angle_deg=args.theta)
    correction = aberration.correction(aberration.load(config.ABERRATION_CAL), rr, cc, 4, args.blaze_y, args.blaze_x)


def solve(weights, spots):
    """Phase for these weights, and with --state the refined spot field (warm-started from spots)."""
    if args.state:
        phase, spots, _ = refine_amplitudes(spots, M, st["target"] * weights.ravel(), REFINE)
        return phase, spots
    phase, _ = run_wgs(N_HORIZ, N_VERT, PITCH_UM, weights=weights, iterations=120, scal=4, momentum=0.6,
                       fix_phase_at=0.5, angle_deg=args.theta, seed=0, verbose=False,
                       correction=correction, offset_px=offsets)
    return phase, None


def measure(cam, phase, pos):
    """Peak (NV, NH) of every trap, and its centre (NV * NH, 2) re-fitted from pos (None: find the traps)."""
    send_phase(phase + blaze(blaze_y, blaze_x))
    frame = grab(cam, args.average)
    if pos is None:
        pts, rows, cols = find_traps(frame, NV, NH, SPOT_FRAC)
        pos = np.empty((NV, NH, 2))
        pos[rows, cols] = pts
        pos = pos.reshape(-1, 2)
    pos = fit_all(frame, pos)
    if not np.all(np.isfinite(pos)):
        raise RuntimeError(f"Gaussian fit failed for {np.isnan(pos[:, 0]).sum()} of {NV * NH} traps")
    return peaks(frame, pos).reshape(NV, NH), pos


def nonuniformity(k):
    return float(k.std() / k.mean())


w = np.ones((N_VERT, N_HORIZ)) if args.weights0 is None else np.load(args.weights0)
w = w / w.mean()
log, pos, spots = [], None, st["spots"] if args.state else None
with aods(amplitudes=(AMPLITUDE, AMPLITUDE)) as awg, Camera(args.exposure_us, hardware_trigger=False) as cam:
    play(awg, F_CENTER)
    print(f"{'round':>5} {'std/mean':>9} {'mean':>8} {'min':>7} {'max':>7} {'weights':>14} {'gain':>5}")
    for rnd in range(args.rounds):
        phase, spots = solve(w, spots)
        k, pos = measure(cam, phase, pos)
        gain = GAIN if rnd < LATE_FROM else GAIN_LATE
        log.append(dict(phase=phase, w=w, k=k, spots=spots))
        print(f"{rnd:5d} {nonuniformity(k):9.4f} {k.mean():8.1f} {k.min():7.1f} {k.max():7.1f} "
              f"[{w.min():.2f}, {w.max():5.2f}] {gain:5.2f}", flush=True)
        step = np.clip((k.mean() / to_design(k)) ** gain, 1 / MAX_STEP, MAX_STEP)
        w = w * step
        w = w / w.mean()
    best = min(range(len(log)), key=lambda i: nonuniformity(log[i]["k"]))
    # Re-measure the best round on a fresh display, so it wasn't picked for a lucky frame.
    k_check, _ = measure(cam, log[best]["phase"], pos)

hist = np.array([nonuniformity(m["k"]) for m in log])
print(f"Best: round {best} at std/mean {hist[best]:.4f} (round 0 {hist[0]:.4f}); "
      f"re-measured {nonuniformity(k_check):.4f}")

stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
out_dir = PHASE_DIR / "vertical" if args.vertical else PHASE_DIR
out_dir.mkdir(exist_ok=True)
if args.state:
    path = args.state.parent / f"{stamp}_{args.state.stem[16:].removesuffix('_state')}_homogenized.npy"
    np.save(path, log[best]["phase"].astype(np.float32))
    np.savez(path.with_name(f"{path.stem}_state.npz"),
             **{**st, "target": st["target"] * log[best]["w"].ravel(), "spots": log[best]["spots"],
                "source": f"homogenize.py round {best} of {args.state.name}"})
    print(f"Saved {path} and its _state.npz")
else:
    name = (f"{stamp}_wgs_{'90x10_' if args.vertical else ''}{'cwgs_' if args.correct else ''}"
            f"{'pos_' if args.offsets else ''}homogenized")
    path = out_dir / f"{name}.npy"
    np.save(path, log[best]["phase"].astype(np.float32))
    np.save(out_dir / f"{name}_weights.npy", log[best]["w"])
    print(f"Saved {path} and its weights")

# Plots keep the camera order: along = trap along the long axis, across = line
along, across = ("89 - design row", "design column") if args.vertical else ("column", "row")
e2.apply()
k0, kb = log[0]["k"] / log[0]["k"].mean(), log[best]["k"] / log[best]["k"].mean()
fig = plt.figure(figsize=(180 * e2.MM, 170 * e2.MM), layout="constrained")
gs = fig.add_gridspec(3, 2, width_ratios=[1, 2])
ax = fig.add_subplot(gs[0, 0])
ax.plot(hist, "o-", ms=3, color=e2.OKABE_ITO["blue"])
ax.plot(best, nonuniformity(k_check), "s", ms=4, color=e2.OKABE_ITO["vermilion"],
        label=f"round {best} re-measured")
ax.set_ylim(0, None)
ax.set_xlabel("round")
ax.set_ylabel("std / mean of trap peaks")
ax.legend()
ax.set_title(f"round 0 {hist[0]:.3f} -> best {hist[best]:.3f}")
ax = fig.add_subplot(gs[0, 1])
ax.plot(k0.mean(axis=0), "o-", ms=2, color=e2.OKABE_ITO["orange"], label="round 0")
ax.plot(kb.mean(axis=0), "o-", ms=2, color=e2.OKABE_ITO["blue"], label=f"round {best}")
ax.axhline(1, color="k", lw=0.5)
ax.set_xlabel(along)
ax.set_ylabel(f"peak / array mean (mean over {across}s)")
ax.legend()
lim = max(np.abs(k0 - 1).max(), np.abs(kb - 1).max())
for row, (k, label) in enumerate([(k0, "round 0"), (kb, f"round {best}")], start=1):
    ax = fig.add_subplot(gs[row, :])
    im = ax.imshow(k, cmap="coolwarm", vmin=1 - lim, vmax=1 + lim, aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="peak / array mean")
    ax.set_xlabel(along)
    ax.set_ylabel(across)
    ax.set_title(f"{label}: std/mean {k.std():.3f}, min {k.min():.2f}, max {k.max():.2f}")
fig.suptitle(f"{'Corrected ' if args.correct else ''}WGS camera feedback, {'90x10' if args.vertical else '10x90'}, "
             f"{'state ' + args.state.name if args.state else f'design angle {args.theta:.4f} deg'}, "
             f"blaze y {blaze_y:+d} x {blaze_x:+d} px"
             f"{', offsets ' + args.offsets.name if args.offsets else ''}\nexposure {args.exposure_us:g} us, "
             f"{args.average} frames per measurement")
save_plot(fig, "homogenize", stamp)
