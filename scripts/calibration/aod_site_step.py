"""Measure the transport-channel frequency step that moves the tweezer array by exactly one site, at the Shepard pairs.

The step is set where the handoff uses it: at the pairs of tones the continuous Shepard
plays together (aqc.pair.shepard_pairs), tone A = --center-mhz + (1/2 + p) step and tone
B = A - step for each crossfade position p in --positions (default mid-crossfade and a
quarter step either side, which averages out the few-kHz ripple of the AOD steering).
Every trap is located at both tones of each pair by the blade method (aqc.pair), and the
pair mismatch E (aqc.handoff.mismatch) is the B centre of the trap that lands on A trap
(r, c) minus A trap (r, c).

A step too short by a fraction a leaves every B trap short of its partner along its own
A -> B displacement; a lattice turned by theta against the steering puts it beside the
partner. One (a, theta) is fitted to the E of all pairs (aqc.handoff.fit_step_angle), and
the step becomes (played step) x (1 + a). The tones are snapped to the 3.125 kHz buffer
grid, so the played step is what the AWG actually played. The measurement is repeated
with the new step until two successive estimates agree within --tol-khz.

theta is printed as a camera rotation and as the WGS design angle change that would
remove it (config.ROW_ANGLE_PER_DESIGN_DEG); this script doesn't correct it
(align_rows_to_aod.py re-solves at a new design angle; handoff_calibration.py --rotate
moves the traps of an existing pattern).

With --vertical the pattern is the 90 rows x 10 columns array, found on the camera as 10
lines of 90 traps; CH1 is the transport channel (CH3 held at 70 MHz) and the step is
written to CH1_SITE_STEP_MHZ, otherwise CH3 and CH3_SITE_STEP_MHZ. The transport channel
is flattened (aqc.aod_amplitude).

Setup: the SLM beam imaged onto the Basler (free-run), AODs on CH1/CH3 at 1.6 V peak.
Nothing else may hold the AWG (stop aod_tone.py first). Pick an exposure that keeps the
brightest trap below saturation at every tone. The current step must be within a fraction
of a site, so that each trap's partner is its nearest neighbour.

Writes the step into aqc/config.py (CH3_SITE_STEP_MHZ, or CH1_ with --vertical) and saves
data/<MM_DD>/<timestamp>_aod_site_step.png. Leaves the pattern on the SLM and the AWG stopped.
About 30 s per pair and round with the defaults.

Usage:
    python scripts/calibration/aod_site_step.py aqc/calibration/phase-patterns/20261028_182104_wgs_aligned.npy --exposure-us 100
    python scripts/calibration/aod_site_step.py \\
        aqc/calibration/phase-patterns/vertical/20261112_234338_handoff_homogenized.npy \\
        --exposure-us 200 --vertical --blaze-x -8 --blaze-y 0
"""

import argparse
import re
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

NV, NH = 10, 90          # camera lines, traps along each line
AMPLITUDE = 1.6          # V peak, the CW drive of aod_tone.py
PX_UM = config.CAMERA_PIXEL_UM

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("pattern", type=Path, help="phase map (.npy, radians, no blaze)")
parser.add_argument("--exposure-us", type=float, required=True, help="camera exposure, us")
parser.add_argument("--vertical", action="store_true", help="90 rows x 10 columns array, transported with CH1")
parser.add_argument("--blaze-y", type=int, default=8, help="y blaze period, SLM px")
parser.add_argument("--blaze-x", type=int, default=0, help="x blaze period, SLM px (0: none)")
parser.add_argument("--center-mhz", type=float, default=70.0, help="centre frequency of the Shepard, MHz")
parser.add_argument("--positions", type=float, nargs="+", default=(-0.25, 0.0, 0.25),
                    help="crossfade positions of the pairs (0 = middle, +-0.5 = ends)")
parser.add_argument("--blade", type=float, nargs=2, default=(0.5, 0.05), metavar=("SPAN", "STEP"),
                    help="blade offsets -SPAN..SPAN MHz in STEP steps on each channel")
parser.add_argument("--average", type=int, default=2, help="frames averaged per tone and offset")
parser.add_argument("--tol-khz", type=float, default=1.0, help="stop when two successive estimates agree this well")
parser.add_argument("--max-rounds", type=int, default=3, help="measurements at most")
args = parser.parse_args()
CH = 1 if args.vertical else 3
NAME = f"CH{CH}_SITE_STEP_MHZ"

sc = pair.scan(*args.blade, np.load(config.AOD_STEERING_VERTICAL_CAL if args.vertical else config.AOD_STEERING_CAL)["J"])
step0 = step = getattr(config, NAME)
send_phase(np.load(args.pattern) + blaze(args.blaze_y, args.blaze_x))
log = []
with aods(amplitudes=(AMPLITUDE, AMPLITUDE)) as awg, Camera(args.exposure_us, hardware_trigger=False) as cam:
    for rnd in range(args.max_rounds):
        pairs = []
        for (fa, fb), p in zip(pair.shepard_pairs(args.center_mhz * 1e6, step * 1e6, args.positions), args.positions):
            m = pair.take(awg, cam, (fa, fb), args.vertical, sc, args.average, NV, NH)
            a = pair.analyse(pair.fit(m, sc), NV, NH)
            a.update(position=p, snapped=m["snapped"], played=(m["snapped"][0] - m["snapped"][1]) / 1e6)
            a["a"], a["theta"], _ = fit_step_angle(a["e"], a["s"], a["r"])
            pairs.append(a)
            print(f"round {rnd}, CH{CH} {fa / 1e6:.4f} -> {fb / 1e6:.4f} MHz: best step {a['played'] * (1 + a['a']):.4f} MHz "
                  f"(played {a['played']:.6f}), lattice {a['theta'] * 1e3:+.2f} mrad; noise floor "
                  f"{a['floor'] * PX_UM:.3f} um", flush=True)
        played = np.mean([q["played"] for q in pairs])
        a, theta, _ = fit_step_angle(*(np.array([q[k] for q in pairs]) for k in "esr"))
        log.append(dict(pairs=pairs, step=step, played=played, best=played * (1 + a), theta=theta))
        print(f"round {rnd}: step {played:.6f} -> {log[-1]['best']:.6f} MHz; lattice {theta * 1e3:+.3f} mrad on the "
              f"camera", flush=True)
        step = log[-1]["best"]
        if len(log) > 1 and abs(log[-1]["best"] - log[-2]["best"]) * 1e3 < args.tol_khz:
            break

converged = len(log) > 1 and abs(log[-1]["best"] - log[-2]["best"]) * 1e3 < args.tol_khz
theta = log[-1]["theta"]
design_deg = np.degrees(theta) / config.ROW_ANGLE_PER_DESIGN_DEG
print(f"One site = {step:.4f} MHz ({(step - step0) * 1e3:+.1f} kHz from {step0:.4f}); "
      f"{'converged' if converged else f'NOT converged within {args.max_rounds} rounds'}")
print(f"Lattice angle error: turn the lattice {theta * 1e3:+.3f} mrad on the camera, i.e. WGS design angle "
      f"{design_deg:+.4f} deg")

config_path = Path(config.__file__)
line = (f"{NAME} = {step:.4f}        # {datetime.now():%Y-%m-%d}, {args.pattern.name}, "
        f"Shepard pairs around {args.center_mhz:g} MHz")
text, n = re.subn(rf"^{NAME} = .*$", line, config_path.read_text(), flags=re.M)
if n != 1:
    raise RuntimeError(f"expected one {NAME} line in {config_path}, found {n}")
config_path.write_text(text)
print(f"Wrote {NAME} = {step:.4f} to {config_path}")

e2.apply()
colors = [e2.OKABE_ITO[c] for c in ("blue", "vermilion", "green", "orange", "purple")]
fig = plt.figure(figsize=(180 * e2.MM, 130 * e2.MM), layout="constrained")
gs = fig.add_gridspec(2, 2)
ax = fig.add_subplot(gs[0, 0])
for i, (rec, col) in enumerate(zip(log, colors)):
    fa = [q["snapped"][0] / 1e6 for q in rec["pairs"]]
    ax.plot(fa, [(q["played"] * (1 + q["a"]) - step0) * 1e3 for q in rec["pairs"]], "o", ms=3, color=col,
            label=f"round {i}: played {(rec['played'] - step0) * 1e3:+.1f}, best {(rec['best'] - step0) * 1e3:+.1f} kHz")
    ax.axhline((rec["best"] - step0) * 1e3, color=col, ls="--", lw=0.8)
ax.axhline(0, color=e2.GREY.edge, lw=0.5)
ax.set_xlabel(f"tone A, CH{CH} (MHz)")
ax.set_ylabel(f"best step − {step0:.4f} MHz (kHz)")
ax.set_title("site step per pair (dashed: all pairs)")
ax.legend(fontsize=6)
ax = fig.add_subplot(gs[0, 1])
for i, (rec, col) in enumerate(zip(log, colors)):
    ax.plot([q["snapped"][0] / 1e6 for q in rec["pairs"]], [q["theta"] * 1e3 for q in rec["pairs"]], "o", ms=3,
            color=col, label=f"round {i}")
    ax.axhline(rec["theta"] * 1e3, color=col, ls="--", lw=0.8)
ax.axhline(0, color=e2.GREY.edge, lw=0.5)
ax.set_xlabel(f"tone A, CH{CH} (MHz)")
ax.set_ylabel("lattice rotation (mrad, camera)")
ax.set_title(f"lattice angle: design {design_deg:+.4f} deg (not corrected)")
ax.legend(fontsize=6)
# Where along the array the mismatch sits, last round: the along-transport part of E per
# column, mean over the lines. A step error is flat; a slope is the one-site step changing
# across the field, which no single step removes.
ax = fig.add_subplot(gs[1, :])
for q, col in zip(log[-1]["pairs"], colors):
    u = np.nanmean(q["s"], axis=(0, 1))
    u /= np.linalg.norm(u)
    along = np.nanmean(q["e"] @ u, axis=0) * PX_UM
    across = np.nanmean(q["e"] @ np.array([u[1], -u[0]]), axis=0) * PX_UM
    ax.plot(along, "-", lw=0.8, color=col, label=f"tone A {q['snapped'][0] / 1e6:.4f} MHz, along")
    ax.plot(across, ":", lw=0.8, color=col, label="across")
ax.axhline(0, color=e2.GREY.edge, lw=0.5)
ax.set_xlabel("tone A trap along the line")
ax.set_ylabel("pair mismatch, mean over lines (µm)")
ax.set_title(f"last round (step {log[-1]['played']:.6f} MHz played): mismatch along and across the transport")
ax.legend(ncols=len(log[-1]["pairs"]), fontsize=6)
fig.suptitle(f"CH{CH} one-site step at the Shepard pairs around {args.center_mhz:g} MHz: {step:.4f} MHz "
             f"({(step - step0) * 1e3:+.1f} kHz){'' if converged else ', NOT converged'}\n"
             f"{args.pattern.name}; blade ±{args.blade[0]:g} MHz / {args.blade[1] * 1e3:g} kHz, "
             f"{args.average} frames per offset, exposure {args.exposure_us:g} µs; µm on the sensor")
save_plot(fig, "aod_site_step")
