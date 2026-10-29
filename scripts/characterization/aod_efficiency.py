"""Integrated intensity of the SLM beam on the camera as CH3 steps across +-sites around 70 MHz.

Whatever is on the SLM stays there; a defocused blaze (one large spot) works best, since
it never saturates at a usable exposure. CH1 is held at 70 MHz and CH3 steps from
70 MHz - sites * config.CH3_SITE_STEP_MHZ to + sites in --steps equal steps. At each
point the frame is summed over a fixed box around the spot, after subtracting the
frame's dark level (median of the whole frame, which is mostly dark), and divided by
the exposure: counts/us, proportional to the power the AODs put through to the camera.

The box is the region above 5% of the peak at 70 MHz, padded so the spot stays inside it
when CH3 moves it by --sites sites. A saturated pixel in the box stops the run, since
the sum would then be too low.

With --flat, CH3 is scaled by the calibrated aqc.aod_amplitude.ch3_scale(f), so the
curve checks that calibration (flat when it holds).

Setup: the SLM beam imaged onto the Basler (free-run), AODs on CH1/CH3 at 1.6 V peak.
Nothing else may hold the AWG (stop aod_tone.py first).

Saves data/<timestamp>_aod_efficiency.png. Leaves the AWG stopped.

Usage:
    python scripts/characterization/aod_efficiency.py --exposure-us 105
    python scripts/characterization/aod_efficiency.py --exposure-us 105 --sites 2 --steps 20
    python scripts/characterization/aod_efficiency.py --exposure-us 105 --flat
"""

import argparse
from datetime import datetime
from pathlib import Path

import e2style as e2
import matplotlib.pyplot as plt
import numpy as np

from aqc import config
from aqc.hardware.awg import aods
from aqc.hardware.camera import Camera
from aqc.aod_amplitude import ch3_scale
from aqc.intensity import integrated, spot_box
from aqc.steering import F_CENTER, grab, play

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
AMPLITUDE = 1.6          # V peak, the CW drive of aod_tone.py
PAD_PX = 100             # box margin beyond the spot; one site moves it ~28 px

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--exposure-us", type=float, required=True, help="camera exposure, us")
parser.add_argument("--sites", type=float, default=1.0, help="CH3 range, +-sites around 70 MHz")
parser.add_argument("--steps", type=int, default=10, help="steps across the range (steps + 1 points)")
parser.add_argument("--average", type=int, default=8, help="frames averaged per point")
parser.add_argument("--flat", action="store_true", help="scale CH3 by aqc.aod_amplitude.ch3_scale (flat power)")
args = parser.parse_args()

offsets = np.linspace(-args.sites, args.sites, args.steps + 1) * config.CH3_SITE_STEP_MHZ
scales = ch3_scale(F_CENTER + offsets * 1e6) if args.flat else np.ones(len(offsets))
f3s, power = [], []
with aods(amplitudes=(AMPLITUDE, AMPLITUDE)) as awg, Camera(args.exposure_us, hardware_trigger=False) as cam:
    play(awg, F_CENTER)
    box = spot_box(grab(cam, args.average), PAD_PX)
    print(f"box rows {box[0]}-{box[1]}, cols {box[2]}-{box[3]}")
    for df, scale in zip(offsets, scales):
        f3 = play(awg, F_CENTER + df * 1e6, scale)
        p = integrated(grab(cam, args.average), box, args.exposure_us)
        print(f"CH3 {f3 / 1e6:8.4f} MHz ({df / config.CH3_SITE_STEP_MHZ:+.2f} site), scale {scale:.3f}: "
              f"integrated {p:.4g} counts/us")
        f3s.append(f3 / 1e6)
        power.append(p)

f3s, power = np.array(f3s), np.array(power)
p70 = np.interp(F_CENTER / 1e6, f3s, power)
e2.apply()
fig, ax = plt.subplots(figsize=(120 * e2.MM, 80 * e2.MM), layout="constrained")
ax.plot(f3s, power / 1e6, "o-", ms=3, lw=0.8, color=e2.OKABE_ITO["blue"])
ax.axvline(F_CENTER / 1e6, color=e2.GREY.face, lw=0.5)
ax.set_xlabel("CH3 frequency (MHz)")
ax.set_ylabel("integrated intensity (10⁶ counts/µs)")
sec = ax.secondary_xaxis("top", functions=(lambda f: (f - F_CENTER / 1e6) / config.CH3_SITE_STEP_MHZ,
                                           lambda s: F_CENTER / 1e6 + s * config.CH3_SITE_STEP_MHZ))
sec.set_xlabel("sites from 70 MHz")
ax.set_title(f"{'CH3 flattened (aqc.aod_amplitude), ' if args.flat else ''}CH1 70 MHz, exposure {args.exposure_us:g} µs, {args.average}-frame average; "
             f"range {power.min() / p70:.3f}-{power.max() / p70:.3f} × 70 MHz")
DATA_DIR.mkdir(exist_ok=True)
e2.save(fig, DATA_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_aod_efficiency.png", dpi=300, transparent=False)
