"""Make a grid of spots from superposed blazed gratings (no WGS), display it, and optionally even out the spots with camera feedback.

Spot (n, m) is a plane wave of slope (u_m, v_n) waves per SLM pixel along x and y with a
random phase theta[n, m] and an amplitude weight w[n, m] (aqc.blaze.grating_array). The
SLM shows the argument of the summed field; about 85% of the light lands in the grid and
the rest in intermodulation spots.

A slope (u, v) puts the first order at X_0 + u a_x + v a_y on the camera, with
|a| = (f1 f3 / f2) lambda / p = 40 mm * 1.064 um / 8 um, ~2876 px for the SLM -> f = 200 mm
-> blade -> f = 200 mm -> f = 40 mm objective chain (config.SLM_ZERO_ORDER_YX and
config.SLM_PX_PER_WAVE hold the measured map). Keep |u|, |v| < 0.5 (the SLM's Nyquist
limit), and u below about -0.04, the D mirror's straight edge (it also blocks u > -0.04
above camera y ~2250).

With --rounds R the grid is evened out in R rounds of camera feedback, with the phases
theta fixed: every spot is first located (Gaussian fit at its predicted position), then each
round measures its peak (aqc.intensity.peaks, averaged over --average free-run frames) and
updates w *= (mean / peak) ** --gain, at most a factor MAX_STEP per round. A weight change
also moves the intermodulation light that interferes with every other spot, so a full step
(gain 0.5) overshoots: in simulation and on the camera, gain 0.5 takes std/mean from 0.7 to
0.3 in one round and then back up to 0.6; gain 0.25 converges (~0.07 after 8 rounds in
simulation). This needs the beam on (aod_tone.py running) and an exposure at which no spot
saturates.

Saves aqc/calibration/phase-patterns/<timestamp>_grating_array_<n_v>x<n_u>.npy (no blaze)
and leaves it on the SLM. With --rounds it also saves <...>_weights.npy (w, n_v x n_u) and
data/<MM_DD>/<timestamp>_grating_array.png (std/mean of the peaks per round, and the
final peaks on the grid).

Usage:
    python scripts/calibration/grating_array.py --n 50 50 --u -0.45 -0.06 --v -0.42 0.45
    python scripts/aod_tone.py                         # in another terminal
    python scripts/calibration/grating_array.py --n 50 50 --u -0.45 -0.06 --v -0.42 0.45 \\
        --rounds 8 --exposure-us 150
"""

import argparse
from datetime import datetime
from pathlib import Path

import e2style as e2
import matplotlib.pyplot as plt
import numpy as np

from aqc import config
from aqc.blaze import grating_array
from aqc.data import save_plot
from aqc.hardware.camera import Camera
from aqc.hardware.slm import send_phase
from aqc.intensity import peaks
from aqc.steering import grab
from aqc.traps import fit_all

PATTERN_DIR = Path(__file__).resolve().parents[2] / "aqc" / "calibration" / "phase-patterns"
MAX_STEP = 2.0           # largest weight change per round
FLOOR = 0.05             # peaks below this times the mean count as this (a spot lost in the noise)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--n", type=int, nargs=2, required=True, metavar=("N_V", "N_U"), help="spots along y and x")
parser.add_argument("--u", type=float, nargs=2, required=True, help="first and last slope along x, waves per px")
parser.add_argument("--v", type=float, nargs=2, required=True, help="first and last slope along y, waves per px")
parser.add_argument("--seed", type=int, default=0, help="seed of the random phases")
parser.add_argument("--rounds", type=int, default=0, help="camera feedback rounds (0 = none)")
parser.add_argument("--exposure-us", type=float, help="camera exposure for the feedback, us")
parser.add_argument("--average", type=int, default=8, help="frames averaged per measurement")
parser.add_argument("--gain", type=float, default=0.25, help="feedback exponent on mean / peak")
args = parser.parse_args()
if args.rounds and args.exposure_us is None:
    parser.error("--rounds needs --exposure-us")

n_v, n_u = args.n
u, v = np.linspace(*args.u, n_u), np.linspace(*args.v, n_v)
theta = np.random.default_rng(args.seed).uniform(0, 2 * np.pi, (n_v, n_u))
w = np.ones((n_v, n_u))
phase = grating_array(u, v, w * np.exp(1j * theta))

stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
path = PATTERN_DIR / f"{stamp}_grating_array_{n_v}x{n_u}.npy"
send_phase(phase)

if args.rounds:
    U, V = np.meshgrid(u, v)
    pred = (np.array(config.SLM_ZERO_ORDER_YX) + U[..., None] * np.array(config.SLM_PX_PER_WAVE[0])
            + V[..., None] * np.array(config.SLM_PX_PER_WAVE[1])).reshape(-1, 2)
    hist = []
    with Camera(args.exposure_us, hardware_trigger=False) as cam:
        img = grab(cam, args.average)
        pos = fit_all(img, pred)
        lost = ~np.all(np.isfinite(pos), axis=1)
        pos[lost] = pred[lost]
        print(f"{n_v * n_u - lost.sum()}/{n_v * n_u} spots fitted, "
              f"{np.sqrt(np.mean(np.sum((pos - pred)[~lost] ** 2, axis=1))):.2f} px rms from the prediction")
        print(f"{'round':>5} {'std/mean':>9} {'mean':>7} {'min':>6} {'max':>6} {'weights':>13}")
        for rnd in range(args.rounds + 1):
            k = peaks(img, pos).reshape(n_v, n_u)
            hist.append(k.std() / k.mean())
            print(f"{rnd:5d} {hist[-1]:9.3f} {k.mean():7.1f} {k.min():6.1f} {k.max():6.1f} "
                  f"[{w.min():.2f}, {w.max():.2f}]", flush=True)
            if rnd == args.rounds:
                break
            w *= np.clip((k.mean() / np.maximum(k, FLOOR * k.mean())) ** args.gain, 1 / MAX_STEP, MAX_STEP)
            w /= w.mean()
            phase = grating_array(u, v, w * np.exp(1j * theta))
            send_phase(phase)
            img = grab(cam, args.average)
    np.save(path.with_name(path.stem + "_weights.npy"), w)

    e2.apply()
    fig, (ax0, ax1) = plt.subplots(1, 2, layout="constrained", figsize=(150 * e2.MM, 60 * e2.MM))
    ax0.plot(hist, "o-", ms=3, color=e2.OKABE_ITO["blue"])
    ax0.set_xlabel("round")
    ax0.set_ylabel("peak std / mean")
    im = ax1.imshow(k / k.mean(), cmap="viridis", origin="upper")
    ax1.set_xlabel("u index (SLM x)")
    ax1.set_ylabel("v index (SLM y)")
    ax1.set_title(f"final peaks / mean, {args.exposure_us:g} us")
    fig.colorbar(im, ax=ax1)
    save_plot(fig, "grating_array", stamp)

np.save(path, phase)
print(f"wrote {path}")
print(f"{n_v}x{n_u} grid on the SLM: u {args.u[0]:g}..{args.u[1]:g}, v {args.v[0]:g}..{args.v[1]:g} waves/px, "
      f"seed {args.seed}" + (f", {args.rounds} feedback rounds" if args.rounds else ""))
