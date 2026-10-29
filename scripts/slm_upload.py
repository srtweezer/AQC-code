"""Display a phase pattern on the SLM, with a blazed grating added.

The pattern is a 1200x1920 phase map in radians (.npy) with no blaze. A blaze along y
and/or x is added, and send_phase maps the sum to grey levels through the measured
phase response in config.SLM_PHASE_RESPONSE. Without a pattern the blaze is shown on a
flat phase; with no pattern and no blaze the SLM shows uniform phase.

Setup: the Meadowlark 1920x1200 HDMI SLM behind slm-server at config.SLM_HOST.

Blaze periods are signed integers in pixels; the sign sets the direction and 0 means
no blaze along that axis.

Usage:
    python scripts/slm_upload.py                          # uniform phase
    python scripts/slm_upload.py --blaze-y 8
    python scripts/slm_upload.py pattern.npy --blaze-x -6
"""

import argparse
from pathlib import Path

import numpy as np

from aqc import config
from aqc.blaze import blaze
from aqc.hardware.slm import phase_to_gray, send_phase

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('pattern', type=Path, nargs='?', help='phase map (.npy, radians, no blaze); omit for flat')
parser.add_argument('--blaze-y', type=int, default=0, help='blaze period along y, px (0 = none)')
parser.add_argument('--blaze-x', type=int, default=0, help='blaze period along x, px (0 = none)')
args = parser.parse_args()

if args.pattern is None:
    phase, name = np.zeros(config.SLM_SHAPE), 'flat phase'
else:
    phase, name = np.load(args.pattern), args.pattern.name
if not np.all(np.isfinite(phase)):
    raise ValueError(f"{name} contains non-finite values")

phase = phase + blaze(args.blaze_y, args.blaze_x)

gray = phase_to_gray(phase)
print(f"{name} + blaze {args.blaze_y:+d} px along y, {args.blaze_x:+d} px along x: "
      f"gray range [{gray.min()}, {gray.max()}], mean {gray.mean():.1f}")
send_phase(phase)
print("Sent.")
