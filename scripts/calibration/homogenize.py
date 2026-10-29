"""Equalize the trap peaks of the tweezer array by camera feedback on the WGS weights.

The 10x90 array (10 um pitch, design angle --theta) is solved with aqc.wgs and displayed
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

Camera trap (row, col) is design trap (row, col): checked 2026-10-29 by dimming design
rows 2-4, cols 6-13, which dimmed the same rows and cols on the camera.

Setup: the SLM beam imaged onto the Basler (free-run), AODs on CH1/CH3 at 70 MHz and
1.6 V peak. Nothing else may hold the AWG (stop aod_tone.py first). Pick an exposure that
keeps the brightest trap below saturation at round 0.

Saves the best round's phase (radians, no blaze; display it with slm_upload.py
--blaze-y 8) and its weights to aqc/calibration/phase-patterns/<timestamp>_wgs_homogenized.npy
and <timestamp>_wgs_homogenized_weights.npy, and data/<timestamp>_homogenize.png.
Leaves the best pattern on the SLM and the AWG stopped.

Usage:
    python scripts/calibration/homogenize.py --exposure-us 100
    python scripts/calibration/homogenize.py --exposure-us 100 --rounds 10 \\
        --weights0 aqc/calibration/phase-patterns/<timestamp>_wgs_homogenized_weights.npy
"""

import argparse
from datetime import datetime
from pathlib import Path

import e2style as e2
import matplotlib.pyplot as plt
import numpy as np

from aqc.blaze import blaze
from aqc.hardware.awg import aods
from aqc.hardware.camera import Camera
from aqc.hardware.slm import send_phase
from aqc.intensity import peaks
from aqc.steering import F_CENTER, grab, play
from aqc.traps import find_traps, fit_all
from aqc.wgs import run_wgs

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PHASE_DIR = Path(__file__).resolve().parents[2] / "aqc" / "calibration" / "phase-patterns"
NV, NH, PITCH_UM = 10, 90, 10.0
BLAZE_Y_PX = 8
AMPLITUDE = 1.6          # V peak, the CW drive of aod_tone.py
# Gain schedule and per-round step limit of andrii-test/calibration/uniformize90.py, which
# took the same array to its wgs_aligned_uniform.npy on 2026-10-22.
GAIN, GAIN_LATE, LATE_FROM = 0.5, 0.15, 6
MAX_STEP = 1.7           # max factor a weight changes by in one round

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--exposure-us", type=float, required=True, help="camera exposure, us")
parser.add_argument("--theta", type=float, default=3.1910,
                    help="WGS design angle, deg (default: that of 20261028_182104_wgs_aligned.npy)")
parser.add_argument("--rounds", type=int, default=20, help="feedback rounds (measurements)")
parser.add_argument("--average", type=int, default=16, help="frames averaged per measurement")
parser.add_argument("--weights0", type=Path, help="starting weights (.npy, NV x NH); default uniform")
args = parser.parse_args()


def solve(weights):
    phase, _ = run_wgs(NH, NV, PITCH_UM, weights=weights, iterations=120, scal=4, momentum=0.6,
                       fix_phase_at=0.5, angle_deg=args.theta, seed=0, verbose=False)
    return phase


def measure(cam, phase, pos):
    """Peak (NV, NH) of every trap, and its centre (NV * NH, 2) re-fitted from pos (None: find the traps)."""
    send_phase(phase + blaze(BLAZE_Y_PX, 0))
    frame = grab(cam, args.average)
    if pos is None:
        pts, rows, cols = find_traps(frame, NV, NH)
        pos = np.empty((NV, NH, 2))
        pos[rows, cols] = pts
        pos = pos.reshape(-1, 2)
    pos = fit_all(frame, pos)
    if not np.all(np.isfinite(pos)):
        raise RuntimeError(f"Gaussian fit failed for {np.isnan(pos[:, 0]).sum()} of {NV * NH} traps")
    return peaks(frame, pos).reshape(NV, NH), pos


def nonuniformity(k):
    return float(k.std() / k.mean())


w = np.ones((NV, NH)) if args.weights0 is None else np.load(args.weights0)
w = w / w.mean()
log, pos = [], None
with aods(amplitudes=(AMPLITUDE, AMPLITUDE)) as awg, Camera(args.exposure_us, hardware_trigger=False) as cam:
    play(awg, F_CENTER)
    print(f"{'round':>5} {'std/mean':>9} {'mean':>8} {'min':>7} {'max':>7} {'weights':>14} {'gain':>5}")
    for rnd in range(args.rounds):
        phase = solve(w)
        k, pos = measure(cam, phase, pos)
        gain = GAIN if rnd < LATE_FROM else GAIN_LATE
        log.append(dict(phase=phase, w=w, k=k))
        print(f"{rnd:5d} {nonuniformity(k):9.4f} {k.mean():8.1f} {k.min():7.1f} {k.max():7.1f} "
              f"[{w.min():.2f}, {w.max():5.2f}] {gain:5.2f}", flush=True)
        step = np.clip((k.mean() / k) ** gain, 1 / MAX_STEP, MAX_STEP)
        w = w * step
        w = w / w.mean()
    best = min(range(len(log)), key=lambda i: nonuniformity(log[i]["k"]))
    # Re-measure the best round on a fresh display, so it wasn't picked for a lucky frame.
    k_check, _ = measure(cam, log[best]["phase"], pos)

hist = np.array([nonuniformity(m["k"]) for m in log])
print(f"Best: round {best} at std/mean {hist[best]:.4f} (round 0 {hist[0]:.4f}); "
      f"re-measured {nonuniformity(k_check):.4f}")

stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
PHASE_DIR.mkdir(exist_ok=True)
path = PHASE_DIR / f"{stamp}_wgs_homogenized.npy"
np.save(path, log[best]["phase"].astype(np.float32))
np.save(PHASE_DIR / f"{stamp}_wgs_homogenized_weights.npy", log[best]["w"])
print(f"Saved {path} and its weights")

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
ax.set_xlabel("column")
ax.set_ylabel("peak / array mean (mean over rows)")
ax.legend()
lim = max(np.abs(k0 - 1).max(), np.abs(kb - 1).max())
for row, (k, label) in enumerate([(k0, "round 0"), (kb, f"round {best}")], start=1):
    ax = fig.add_subplot(gs[row, :])
    im = ax.imshow(k, cmap="coolwarm", vmin=1 - lim, vmax=1 + lim, aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="peak / array mean")
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.set_title(f"{label}: std/mean {k.std():.3f}, min {k.min():.2f}, max {k.max():.2f}")
fig.suptitle(f"WGS camera feedback, design angle {args.theta:.4f} deg; exposure {args.exposure_us:g} us, "
             f"{args.average} frames per measurement")
DATA_DIR.mkdir(exist_ok=True)
e2.save(fig, DATA_DIR / f"{stamp}_homogenize.png", dpi=300, transparent=False)
