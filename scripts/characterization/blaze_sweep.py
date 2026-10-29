"""Step the SLM through y blaze periods and image the first diffraction order for each.

Setup: SLM beam on the Basler with the zero order blocked, so the first order is
the brightest spot in each frame (found after smoothing, so a hot pixel can't win).
Each frame is cropped around it. Free-run exposure: the AWG and AOM are left as
they are, so the beam has to be on, e.g. with aod_tone.py running. Each blaze is
imaged at every exposure in --exposure-us. The SLM is left on the last blaze.

Saves data/<timestamp>_blaze_sweep.png: one row per exposure, one column per
period, crops on a shared Mono12 colour scale.

Usage:
    python scripts/characterization/blaze_sweep.py --exposure-us 12 100
    python scripts/characterization/blaze_sweep.py --exposure-us 12 --periods 4 8 16 32 --crop 50
"""

import argparse
from datetime import datetime
from pathlib import Path

import e2style as e2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from aqc.blaze import blaze
from aqc.hardware.camera import Camera
from aqc.hardware.slm import send_phase

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FULL_SCALE = 4095   # Mono12
SMOOTH_PX = 5       # so a hot pixel can't win the peak search

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--exposure-us", type=float, nargs="+", required=True, help="camera exposures, us")
parser.add_argument("--periods", type=int, nargs="+", default=[4, 8, 16, 32], help="y blaze periods, px (signed)")
parser.add_argument("--crop", type=int, default=32, help="crop half-width, camera px")
args = parser.parse_args()


def peak(image):
    return np.unravel_index(np.argmax(gaussian_filter(image, SMOOTH_PX)), image.shape)


# frames[exposure index][period index]
frames = [[None] * len(args.periods) for _ in args.exposure_us]
with Camera(args.exposure_us[0], hardware_trigger=False) as cam:
    for j, period in enumerate(args.periods):
        send_phase(blaze(period_y_px=period))
        for i, exposure in enumerate(args.exposure_us):
            cam.set_exposure(exposure)
            frames[i][j] = cam.grab()

e2.apply()
fig, axes = plt.subplots(len(args.exposure_us), len(args.periods), squeeze=False, layout="constrained",
                         figsize=(45 * e2.MM * len(args.periods), 42 * e2.MM * len(args.exposure_us)))
for i, exposure in enumerate(args.exposure_us):
    for j, period in enumerate(args.periods):
        frame = frames[i][j]
        h, w = frame.shape
        y, x = peak(frame.astype(float))
        y0 = min(max(y - args.crop, 0), h - 2 * args.crop)
        x0 = min(max(x - args.crop, 0), w - 2 * args.crop)
        crop = frame[y0:y0 + 2 * args.crop, x0:x0 + 2 * args.crop]
        n_sat = int((crop >= FULL_SCALE).sum())
        print(f"{exposure:g} us, {period:+d} px: first order at x {x}, y {y}, "
              f"peak {crop.max()}, {n_sat} saturated px")
        ax = axes[i, j]
        im = ax.imshow(crop, cmap="inferno", vmin=0, vmax=FULL_SCALE,
                       extent=(x0, x0 + crop.shape[1], y0 + crop.shape[0], y0))
        ax.set_title(f"{exposure:g} us, blaze y {period:+d} px\npeak {crop.max()}, {n_sat} sat. px")
fig.colorbar(im, ax=axes, label="counts", shrink=0.8)
DATA_DIR.mkdir(exist_ok=True)
path = DATA_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_blaze_sweep.png"
e2.save(fig, path, dpi=300, transparent=False)
