"""Calibrate a per-frequency CH3 amplitude scale that makes the power the AODs pass flat.

The SLM shows a defocused blaze (one large spot that never saturates; the tweezer array
gives the same efficiency curve). CH1 is held at 70 MHz and CH3 steps over
--f-min..--f-max. At each frequency the spot's integrated intensity (aqc.intensity:
counts/us summed over a fixed box) measures the power passed.

Pass 0 runs at full drive (scale 1). The weakest frequency is the anchor: it keeps scale
1, and every other frequency is turned down towards it, so the drive never exceeds the
current 1.6 V. Each later pass measures with the current scale and updates it as
scale *= sqrt(I_anchor / I). The square root is only a step size: the AOD response is
A^2 only well below saturation, and iterating converges either way. Each pass is
normalised to its own anchor reading, so slow laser drift between passes cancels. It
stops when every frequency is within --tol of the anchor.

Setup: the SLM beam imaged onto the Basler (free-run), AODs on CH1/CH3 at 1.6 V peak.
Nothing else may hold the AWG (stop aod_tone.py first). Pick an exposure that keeps the
brightest frequency (~67.7 MHz) below saturation at full drive.

On convergence saves aqc/calibration/<timestamp>_ch3_amplitude.npz and points
config.CH3_AMPLITUDE_CAL at it; always saves data/<timestamp>_aod_amplitude.png. Leaves
the defocused blaze on the SLM and the AWG stopped.

Usage:
    python scripts/calibration/aod_amplitude.py --exposure-us 105
    python scripts/calibration/aod_amplitude.py --exposure-us 105 --f-min 64 --f-max 76
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

import e2style as e2
import matplotlib.pyplot as plt
import numpy as np

from aqc import config
from aqc.blaze import blaze
from aqc.hardware.awg import aods
from aqc.hardware.camera import Camera
from aqc.hardware.slm import send_phase
from aqc.intensity import integrated, spot_box
from aqc.lens import defocus
from aqc.steering import F_CENTER, grab, play

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
AMPLITUDE = 1.6          # V peak, the CW drive of aod_tone.py
DEFOCUS_WAVES = 10
BLAZE_Y_PX = 6
PAD_PX = 100             # box margin; 10 MHz moves the spot ~49 px

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--exposure-us", type=float, required=True, help="camera exposure, us")
parser.add_argument("--f-min", type=float, default=60.0, help="CH3 band start, MHz")
parser.add_argument("--f-max", type=float, default=80.0, help="CH3 band end, MHz")
parser.add_argument("--f-step", type=float, default=0.5, help="CH3 grid step, MHz")
parser.add_argument("--average", type=int, default=8, help="frames averaged per point")
parser.add_argument("--iterations", type=int, default=4, help="correction passes after pass 0")
parser.add_argument("--tol", type=float, default=0.02, help="max |I / I_anchor - 1| to stop")
args = parser.parse_args()

freqs = np.round(np.arange(args.f_min, args.f_max + args.f_step / 2, args.f_step), 6)   # MHz


def sweep(awg, cam, box, scale):
    """Integrated intensity (counts/us) at every grid frequency, CH3 scaled by scale."""
    out = np.empty(len(freqs))
    for i, (f, s) in enumerate(zip(freqs, scale)):
        play(awg, f * 1e6, s)
        out[i] = integrated(grab(cam, args.average), box, args.exposure_us)
    return out


send_phase(defocus(DEFOCUS_WAVES) + blaze(BLAZE_Y_PX, 0))
scale = np.ones(len(freqs))
passes = []
with aods(amplitudes=(AMPLITUDE, AMPLITUDE)) as awg, Camera(args.exposure_us, hardware_trigger=False) as cam:
    play(awg, F_CENTER)
    box = spot_box(grab(cam, args.average), PAD_PX)
    for k in range(args.iterations + 1):
        if k:
            scale = np.minimum(1.0, scale * np.sqrt(passes[-1][anchor] / passes[-1]))
        passes.append(sweep(awg, cam, box, scale))
        if k == 0:
            anchor = int(np.argmin(passes[0]))
        err = np.max(np.abs(passes[-1] / passes[-1][anchor] - 1))
        print(f"pass {k}: anchor {freqs[anchor]:g} MHz, max |I / I_anchor - 1| = {err:.3f}, "
              f"scale {scale.min():.3f}-{scale.max():.3f}")
        if err < args.tol:
            break

converged = err < args.tol
i70 = int(np.argmin(np.abs(freqs - F_CENTER / 1e6)))
level = passes[0][anchor] / passes[0][i70]      # the anchor stays at full drive
stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
if converged:
    print(f"Flat to {err:.1%} over {args.f_min:g}-{args.f_max:g} MHz at {level:.3f} x the full-drive "
          f"70 MHz power")
    cal_path = Path(config.__file__).parent / "calibration" / f"{stamp}_ch3_amplitude.npz"
    np.savez(cal_path, f_hz=freqs * 1e6, scale=scale, intensity_pass0=passes[0], intensity_final=passes[-1],
             anchor_hz=freqs[anchor] * 1e6, exposure_us=args.exposure_us, ch3_amplitude_v=AMPLITUDE,
             ch1_hz=F_CENTER, date=stamp)
    config_path = Path(config.__file__)
    line = (f'CH3_AMPLITUDE_CAL = Path(__file__).parent / "calibration" / "{cal_path.name}"'
            f"  # {datetime.now():%Y-%m-%d}")
    text, n = re.subn(r"^CH3_AMPLITUDE_CAL = .*$", line, config_path.read_text(), flags=re.M)
    if n != 1:
        raise RuntimeError(f"expected one CH3_AMPLITUDE_CAL line in {config_path}, found {n}")
    config_path.write_text(text)
    print(f"Saved {cal_path}; config.CH3_AMPLITUDE_CAL points at it")
else:
    print(f"Not flat to {args.tol:.0%} after {args.iterations} correction passes (max deviation {err:.3f}); "
          f"nothing saved")

e2.apply()
fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(120 * e2.MM, 110 * e2.MM), layout="constrained",
                              gridspec_kw={"height_ratios": [1.4, 1]})
for k, (p, color) in enumerate(zip(passes, [e2.OKABE_ITO[c] for c in
                                            ("blue", "orange", "green", "purple", "vermilion")])):
    if k in (0, len(passes) - 1):
        ax.plot(freqs, p / passes[0][i70], "o-", ms=2, lw=0.8, color=color,
                label="pass 0, full drive" if k == 0 else f"pass {k}, calibrated scale")
ax.axvline(freqs[anchor], color=e2.GREY.face, lw=0.5)
ax.set_ylabel("power / full-drive 70 MHz")
ax.set_title(f"{'converged' if converged else 'NOT converged'}: max deviation {err:.1%}, "
             f"anchor {freqs[anchor]:g} MHz, exposure {args.exposure_us:g} µs")
ax.legend(frameon=True, facecolor="white", framealpha=0.9)
ax2.plot(freqs, scale, "o-", ms=2, lw=0.8, color=e2.OKABE_ITO["blue"])
ax2.set_xlabel("CH3 frequency (MHz)")
ax2.set_ylabel("CH3 amplitude scale")
fig.suptitle(f"CH3 amplitude calibration, CH1 70 MHz, {AMPLITUDE:g} V peak")
DATA_DIR.mkdir(exist_ok=True)
e2.save(fig, DATA_DIR / f"{stamp}_aod_amplitude.png", dpi=300, transparent=False)
