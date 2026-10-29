"""Grab one frame from the Basler and save it as a PNG.

Two modes:

- Free-run (default): the camera exposes for --exposure-us on its own and images
  whatever is there now (the current SLM pattern and whatever the AWG is playing).
  The camera can't go below 12 us, which saturates on the CW beam.
- AOM-gated (--aom-us): the script drives the AWG itself, so stop aod_tone.py first.
  It loops a 1.6 ms buffer of CW tones on CH1/CH3 (--f1/--f3). In each loop X2
  triggers the camera at t = 0, and X0 opens the AOM (active-low) for --aom-us at
  t = 100 us, inside the 500 us camera exposure. So the frame sees one light pulse
  of --aom-us, which can be far shorter than the camera's minimum exposure. The
  AODs are driven the whole time, so they stay at their CW thermal steady state.

The PNG goes to data/<timestamp>_snapshot.png, with the colour scale over the full
Mono12 range so saturation shows; the peak count is printed too.

Usage:
    python scripts/camera_snapshot.py                          # free-run, 1000 us
    python scripts/camera_snapshot.py --exposure-us 200
    python scripts/camera_snapshot.py --aom-us 1               # 70 MHz on both AODs
    python scripts/camera_snapshot.py --aom-us 1 --f3 65e6 70e6 75e6
"""

import argparse
from datetime import datetime
from pathlib import Path

import e2style as e2
import matplotlib.pyplot as plt
import numpy as np

from aqc import config
from aqc.hardware.camera import Camera
from aqc.tones import multitone

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FULL_SCALE = 4095           # Mono12

# AOM-gated mode
AMPLITUDE = 1.6             # V peak into 50 ohm, both channels (as aod_tone.py)
NB_SAMPLES = 1_000_000      # 1.6 ms loop (a multiple of 32 samples); tones snap to a 625 Hz grid
TRIGGER_US = 10             # X2 pulse width
AOM_OPEN_US = 100           # AOM pulse start after the trigger, well past the camera's trigger latency
GATED_EXPOSURE_US = 500     # ends well before the next loop's pulse at 1700 us

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--exposure-us", type=float, default=1000, help="free-run camera exposure, us")
parser.add_argument("--aom-us", type=float, help="AOM-gated mode: light pulse length, us")
parser.add_argument("--f1", type=float, nargs="+", default=[70e6], help="AOM-gated mode: CH1 frequencies, Hz")
parser.add_argument("--f3", type=float, nargs="+", default=[70e6], help="AOM-gated mode: CH3 frequencies, Hz")
args = parser.parse_args()

if args.aom_us is None:
    with Camera(args.exposure_us, hardware_trigger=False) as cam:
        image = cam.grab()
    label = f"free-run, {args.exposure_us:g} us exposure"
else:
    # Imported only here: loading libspcm in a process that doesn't drive the card
    # unlinks the driver's shared memory under aod_tone.py when this process exits.
    from aqc.hardware.awg import aods

    rate = config.AWG_SAMPLING_RATE
    ch1, f1 = multitone(args.f1, NB_SAMPLES, rate)
    ch3, f3 = multitone(args.f3, NB_SAMPLES, rate)
    t_us = np.arange(NB_SAMPLES) / rate * 1e6
    aom_blocked = ~((t_us >= AOM_OPEN_US) & (t_us < AOM_OPEN_US + args.aom_us))
    camera_trigger = t_us < TRIGGER_US
    if aom_blocked.all():
        raise ValueError(f"--aom-us {args.aom_us} is shorter than one sample ({1e6 / rate:g} us)")

    with aods(amplitudes=(AMPLITUDE, AMPLITUDE), digital_outputs=True) as awg, \
            Camera(GATED_EXPOSURE_US) as cam:
        awg.load_waveform(np.stack([ch1, ch3]), -1, digital_data=np.stack([aom_blocked, camera_trigger]))
        awg.force_trigger()
        image = cam.grab()
    label = (f"AOM {args.aom_us:g} us, CH1 {', '.join(f'{f / 1e6:g}' for f in f1)} MHz, "
             f"CH3 {', '.join(f'{f / 1e6:g}' for f in f3)} MHz")

n_saturated = int((image >= FULL_SCALE).sum())
print(f"{label}: peak {image.max()} / {FULL_SCALE}, mean {image.mean():.1f}, {n_saturated} saturated px")

e2.apply()
fig, ax = plt.subplots(figsize=(180 * e2.MM, 140 * e2.MM), layout="constrained")
im = ax.imshow(image, cmap="inferno", vmin=0, vmax=FULL_SCALE)
fig.colorbar(im, ax=ax, label="counts")
ax.set_title(f"Basler, {label}\npeak {image.max()}, {n_saturated} saturated px")
DATA_DIR.mkdir(exist_ok=True)
path = DATA_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_snapshot.png"
e2.save(fig, path, dpi=300, transparent=False)
