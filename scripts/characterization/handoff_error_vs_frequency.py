"""Total handoff error of SLM patterns as a function of the transport tone f, for a step from f to f + --shift sites,
and how much of it a better site step and design angle would remove.

At every f, tone A is the transport channel at f and tone B at f + --shift * site step.
The pair mismatch E (aqc.handoff.mismatch) is the centre at tone B of the trap that lands
on A trap (r, c), minus that trap. Its single-number summary (aqc.handoff.error) is
eps = sqrt(<|E|^2>), split exactly into the trap-to-trap scatter eps_slm, which only the
SLM fixes, and the common offset |<E>|.

Step and angle. A site step too short by a fraction a leaves every B trap short of its
partner along its own A -> B displacement; a lattice turned by theta against the AOD
steering puts it beside the partner. Both are fitted to E by least squares
(aqc.handoff.fit_step_angle, with each trap's own displacement and partner vector, so a
steering that varies over the field is included):
  - per f: what a step and angle tuned at that f would leave;
  - one (a, theta) for all f: what retuning config.CH<n>_SITE_STEP_MHZ and the WGS design
    angle would leave. Its step change is a * step, its design angle change theta /
    config.ROW_ANGLE_PER_DESIGN_DEG.
What remains after the per-f fit is SLM scatter (and fit noise); what the per-f fit
removes beyond the global one is the AOD's steering changing with f, which one step and
one angle cannot follow.

Centres. Without --blade: 2-D Gaussian fits (aqc.traps.fit_spots) to frames at f, at
f + shift, and at f again, the reference being the mean of the two f frames, so drift
linear in time cancels; with per-component noise s per frame the floor of eps_slm is
sqrt(0.75) * rms|D - <D>|, D the difference of the two f frames. With --blade SPAN STEP:
the blade method on CH3 and CH1 offsets of -SPAN..SPAN MHz in STEP steps (aqc.pair, with
its noise floor from two halves of the offsets). All frames are taken first and fitted
after the AWG and camera are released.

Several patterns are measured interleaved: at every f each is displayed in turn.

The transport channel is flattened (aqc.aod_amplitude) at every tone. Lengths are um on
the sensor (px * config.CAMERA_PIXEL_UM). With --vertical the pattern is the 90 rows x 10
columns array, CH1 is the transport channel (config.CH1_SITE_STEP_MHZ), CH3 is held at
70 MHz, and the pattern gets an 8 px blaze along -x and none along y; otherwise CH3
transports the horizontal array with an 8 px y blaze. The camera finds the array as 10
lines of 90 traps either way.

Setup: the SLM beam imaged onto the Basler (free-run), AODs on CH1/CH3 at 1.6 V peak.
Nothing else may hold the AWG (stop aod_tone.py first). Pick an exposure that keeps the
brightest trap below saturation at every tone.

Saves data/<MM_DD>[/<--subdir>]/<timestamp>_handoff_error_vs_frequency.png and, with the
same stamp, _centres.npz: the snapped tones and every trap's A and B centres (camera
order, px) per pattern, f and (with --blade) half. Leaves the last pattern on the SLM and
the AWG stopped.

Usage:
    python scripts/characterization/handoff_error_vs_frequency.py \\
        aqc/calibration/phase-patterns/vertical/20261112_234338_handoff_homogenized.npy \\
        --exposure-us 250 --vertical --subdir handoff_error
    python scripts/characterization/handoff_error_vs_frequency.py \\
        aqc/calibration/phase-patterns/vertical/20261113_103928_wgs_90x10_cwgs_pos_homogenized.npy \\
        aqc/calibration/phase-patterns/vertical/20261112_234338_handoff_homogenized.npy \\
        --exposure-us 250 --vertical --blade 0.5 0.05 --average 2 --subdir handoff_error
"""

import argparse
from datetime import datetime
from pathlib import Path

import e2style as e2
import matplotlib.pyplot as plt
import numpy as np

from aqc import config, pair
from aqc.blaze import blaze
from aqc.data import data_path, save_plot
from aqc.handoff import error, fit_step_angle, mismatch, step_angle_basis
from aqc.hardware.awg import aods
from aqc.hardware.camera import Camera
from aqc.hardware.slm import send_phase
from aqc.intensity import FULL_SCALE
from aqc.steering import fit_array, play_transport

NV, NH = 10, 90           # camera lines, traps along each line
PX_UM = config.CAMERA_PIXEL_UM
AMPLITUDE = 1.6           # V peak, the CW drive of aod_tone.py
ROW_GAIN = config.ROW_ANGLE_PER_DESIGN_DEG

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("patterns", type=Path, nargs="+", help="phase maps (.npy, radians, no blaze)")
parser.add_argument("--exposure-us", type=float, required=True, help="camera exposure, us")
parser.add_argument("--f-mhz", type=float, nargs=2, default=(65.0, 75.0), metavar=("START", "STOP"),
                    help="range of the reference tone f, MHz")
parser.add_argument("--step-mhz", type=float, default=0.5, help="step of f, MHz")
parser.add_argument("--shift", type=int, default=-1, help="second tone: f + shift * site step")
parser.add_argument("--average", type=int, default=32, help="frames averaged per tone (per offset with --blade)")
parser.add_argument("--blade", type=float, nargs=2, metavar=("SPAN", "STEP"),
                    help="blade centres from CH3 and CH1 offsets of -SPAN..SPAN MHz in STEP steps")
parser.add_argument("--vertical", action="store_true", help="90 rows x 10 columns array, transported with CH1")
parser.add_argument("--subdir", help="save in data/<MM_DD>/<subdir>/")
args = parser.parse_args()

CH = 1 if args.vertical else 3
STEP_MHZ = config.CH1_SITE_STEP_MHZ if args.vertical else config.CH3_SITE_STEP_MHZ
f_ref = np.arange(args.f_mhz[0], args.f_mhz[1] + args.step_mhz / 2, args.step_mhz) * 1e6
phases = [np.load(p) + (blaze(0, -8) if args.vertical else blaze(8, 0)) for p in args.patterns]
names = [p.stem for p in args.patterns]

sc = pair.scan(*args.blade, np.load(config.AOD_STEERING_VERTICAL_CAL if args.vertical else config.AOD_STEERING_CAL)["J"]) \
    if args.blade else None


def measure_gauss(awg, cam, f):
    """Gaussian centres (2, N, 2) px of tones A (mean of two frames) and B, the snapped tones, and the floor of eps_slm (px)."""
    fits, snapped = [], []
    for tone in (f, f + args.shift * STEP_MHZ * 1e6, f):
        snapped.append(play_transport(awg, tone, args.vertical))
        frame, fit = fit_array(cam, NV, NH, args.average)
        if frame.max() >= FULL_SCALE:
            raise RuntimeError(f"saturated at CH{CH} {snapped[-1] / 1e6:.4f} MHz; shorten --exposure-us")
        fits.append(fit[:, :2])
    a1, b, a2 = fits
    d = (a2 - a1) - np.nanmean(a2 - a1, axis=0)
    return np.stack([(a1 + a2) / 2, b]), np.array(snapped[:2]), np.sqrt(0.75 * np.nanmean(np.sum(d ** 2, axis=1)))


# Measure: every f, every pattern in turn
raw = [[None] * len(f_ref) for _ in phases]
with aods(amplitudes=(AMPLITUDE, AMPLITUDE)) as awg, Camera(args.exposure_us, hardware_trigger=False) as cam:
    for i, f in enumerate(f_ref):
        for p, phase in enumerate(phases):
            send_phase(phase)
            raw[p][i] = (pair.take(awg, cam, (f, f + args.shift * STEP_MHZ * 1e6), args.vertical, sc, args.average,
                                   NV, NH) if args.blade else measure_gauss(awg, cam, f))
            print(f"{names[p]}: CH{CH} {f / 1e6:7.3f} MHz measured", flush=True)

# Fit and analyse
step_hz = []
res = []                  # per pattern: dict of arrays over f
for p in range(len(phases)):
    out = dict(eps=[], eps_slm=[], mean=[], floor=[], e=[], s=[], r=[], centres=[], snapped=[])
    for i, f in enumerate(f_ref):
        if args.blade:
            c = pair.fit(raw[p][i], sc)[..., :2]
            snapped = raw[p][i]["snapped"]
            raw[p][i] = None
            m = pair.analyse(c, NV, NH)
        else:
            c, snapped, floor = raw[p][i]
            e, dc = mismatch(c[0], c[1], NV, NH)
            m = dict(e=e, floor=floor, **dict(zip("sr", step_angle_basis(c[0], c[1], NV, NH, dc))))
        eps, eps_slm, mean = error(m["e"] * PX_UM)
        floor = m["floor"] * PX_UM
        for key, v in (("eps", eps), ("eps_slm", eps_slm), ("mean", mean), ("floor", floor), ("e", m["e"] * PX_UM),
                       ("s", m["s"] * PX_UM), ("r", m["r"] * PX_UM), ("centres", c), ("snapped", snapped)):
            out[key].append(v)
        print(f"{names[p]}: CH{CH} {snapped[0] / 1e6:7.3f} -> {snapped[1] / 1e6:7.4f} MHz: eps {eps:.3f} um = "
              f"eps_slm {eps_slm:.3f} (+) |<E>| {np.linalg.norm(mean):.3f} um; floor {floor:.3f} um", flush=True)
    out = {key: np.array(v) for key, v in out.items()}
    # Step and angle: per f, and one pair for all f
    per_f = [fit_step_angle(e, s, r) for e, s, r in zip(out["e"], out["s"], out["r"])]
    out["a_f"] = np.array([q[0] for q in per_f])
    out["theta_f"] = np.array([q[1] for q in per_f])
    out["eps_f"] = np.array([error(q[2])[0] for q in per_f])
    out["a"], out["theta"], resid = fit_step_angle(out["e"], out["s"], out["r"])
    out["eps_g"] = np.array([error(q)[0] for q in resid])
    res.append(out)
    step_hz.append(np.abs(np.diff(out["snapped"], axis=1)[:, 0]))

stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
np.savez(data_path("handoff_error_vs_frequency_centres.npz", stamp, args.subdir),
         patterns=[str(p) for p in args.patterns], f_hz=f_ref, snapped_hz=np.array([o["snapped"] for o in res]),
         centres_px=np.array([o["centres"] for o in res]), blade=args.blade or [], vertical=args.vertical,
         axes="pattern, f, [blade half,] tone (A, B), trap (camera order), (y, x)")

for name, o, st in zip(names, res, step_hz):
    fine = np.median(o["eps"] ** 2)
    print(f"\n{name}:\n  median eps {np.median(o['eps']):.3f} um, eps_slm {np.median(o['eps_slm']):.3f} um, "
          f"floor {np.median(o['floor']):.3f} um\n"
          f"  one step and angle for all f: step {o['a'] * st.mean() / 1e3:+.1f} kHz "
          f"(-> {(1 + o['a']) * st.mean() / 1e6:.4f} MHz), lattice {o['theta'] * 1e3:+.3f} mrad on the camera "
          f"(design angle {np.degrees(o['theta']) / ROW_GAIN:+.4f} deg); median eps -> {np.median(o['eps_g']):.3f} um "
          f"({1 - np.median(o['eps_g'] ** 2) / fine:.0%} of median eps^2)\n"
          f"  step and angle per f: median eps -> {np.median(o['eps_f']):.3f} um "
          f"({1 - np.median(o['eps_f'] ** 2) / fine:.0%} of median eps^2)")

fm = f_ref / 1e6
e2.apply()
colors = [e2.OKABE_ITO[c] for c in ("blue", "vermilion", "green", "orange")]
fig = plt.figure(figsize=(180 * e2.MM, (110 + 45 * len(res)) * e2.MM), layout="constrained")
gs = fig.add_gridspec(2 + len(res), 2, height_ratios=[1.3, 1] + [0.8] * len(res))
ax = fig.add_subplot(gs[0, :])
for name, o, col in zip(names, res, colors):
    ax.plot(fm, o["eps"], "o-", ms=3, color=col, label=f"{name}: measured")
    ax.plot(fm, o["eps_g"], "s--", ms=2.5, color=col, label="one step + angle for all f")
    ax.plot(fm, o["eps_f"], "^:", ms=2.5, color=col, label="step + angle per f")
    ax.plot(fm, o["floor"], "-", lw=0.6, color=col, alpha=0.5, label=r"fit noise floor of $\varepsilon_\mathrm{SLM}$")
ax.set_ylim(0, None)
ax.set_xlabel(f"reference tone f, CH{CH} (MHz)")
ax.set_ylabel(r"$\varepsilon$ (µm on sensor)")
ax.set_title(r"handoff error $\varepsilon = \sqrt{\langle |E|^2 \rangle}$, as measured and after the best site step "
             "and lattice angle")
ax.legend(ncols=len(res), fontsize=6)

ax = fig.add_subplot(gs[1, 0])
for name, o, col, st in zip(names, res, colors, step_hz):
    ax.plot(fm, o["a_f"] * st / 1e3, "o-", ms=3, color=col, label=name)
    ax.axhline(o["a"] * st.mean() / 1e3, color=col, ls="--", lw=0.8)
ax.axhline(0, color=e2.GREY.edge, lw=0.5)
ax.set_xlabel(f"reference tone f, CH{CH} (MHz)")
ax.set_ylabel("best step − used step (kHz)")
ax.set_title(f"site step (used {STEP_MHZ:.4f} MHz); dashed: one for all f")

ax = fig.add_subplot(gs[1, 1])
for name, o, col in zip(names, res, colors):
    ax.plot(fm, o["theta_f"] * 1e3, "o-", ms=3, color=col, label=name)
    ax.axhline(o["theta"] * 1e3, color=col, ls="--", lw=0.8)
ax.axhline(0, color=e2.GREY.edge, lw=0.5)
ax.set_xlabel(f"reference tone f, CH{CH} (MHz)")
ax.set_ylabel("lattice rotation (mrad, camera)")
ax.set_title(f"lattice angle (design deg = camera / {ROW_GAIN}); dashed: one for all f")

for j, (name, o) in enumerate(zip(names, res)):
    ax = fig.add_subplot(gs[2 + j, :])
    # rms over the lines of E - <E> per reference column: where along the array the scatter sits
    col_rms = np.sqrt(np.nanmean(np.sum((o["e"] - o["mean"][:, None, None]) ** 2, axis=-1), axis=1))
    im = ax.imshow(col_rms, aspect="auto", origin="lower", cmap="viridis", interpolation="nearest",
                   extent=(-0.5, NH - 0.5, fm[0] - args.step_mhz / 2, fm[-1] + args.step_mhz / 2))
    fig.colorbar(im, ax=ax, label=r"rms $|E - \langle E\rangle|$ over lines (µm)")
    ax.set_xlabel("reference column (trap along the line)")
    ax.set_ylabel("f (MHz)")
    ax.set_title(f"{name}: trap-to-trap scatter along the array")

method = (f"blade, CH3 and CH1 offsets ±{args.blade[0]:g} MHz in {args.blade[1] * 1e3:g} kHz steps, "
          f"{args.average} frames per offset" if args.blade else
          f"Gaussian fits, reference = mean of two f frames, {args.average}-frame average")
fig.suptitle(f"Handoff error vs tone, {datetime.now():%Y-%m-%d}\n"
             rf"CH{CH} $f \to f {args.shift:+d}$ site ({args.shift * STEP_MHZ:+.4f} MHz), flattened; {method}"
             f"\nexposure {args.exposure_us:g} µs; camera x/y, µm on the sensor ({PX_UM} µm/px)")
print(save_plot(fig, "handoff_error_vs_frequency", stamp, args.subdir))
