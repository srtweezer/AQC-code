"""Grab one frame from the Basler and save it as a PNG.

Two modes:

- Free-run (default): the camera exposes for --exposure-us on its own and images
  whatever is there now (the current SLM pattern and whatever the AWG is playing).
  The camera can't go below 12 us, which saturates on the CW beam.
- AOM-gated (--aom-us): the script drives the AWG itself, so stop aod_tone.py first.
  It loops a 1.6 ms buffer of CW tones on CH1/CH3 (--f1/--f3). In each loop the AOM
  is open for the first 5 us while the intensity servo's hold engages (aqc.intensity_lock),
  then X2 triggers the camera, and X0 opens the AOM (active-low) again for --aom-us at
  t = 100 us, inside the 500 us camera exposure. So the frame sees one light pulse
  of --aom-us, which can be far shorter than the camera's minimum exposure. The
  AODs are driven the whole time, so they stay at their CW thermal steady state.

With --crop-traps, the 10x90 tweezer array is found in the frame (aqc.traps), and the
frame is resampled onto the array's row axis (angle from aqc.traps.row_fit) and cropped
to the traps plus one pitch on every side.

The PNG goes to data/<MM_DD>/<timestamp>_snapshot.png, with the colour scale over the full
Mono12 range so saturation shows; the peak count is printed too.

Usage:
    python scripts/camera_snapshot.py                          # free-run, 1000 us
    python scripts/camera_snapshot.py --exposure-us 200
    python scripts/camera_snapshot.py --aom-us 1               # 70 MHz on both AODs
    python scripts/camera_snapshot.py --aom-us 1 --f3 65e6 70e6 75e6
    python scripts/camera_snapshot.py --aom-us 170 --crop-traps
"""

import argparse

import e2style as e2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates

from aqc import config
from aqc.data import save_plot
from aqc.hardware.camera import Camera
from aqc.intensity_lock import strobe_rows
from aqc.tones import multitone
from aqc.traps import find_traps, fit_all, row_grid

FULL_SCALE = 4095           # Mono12

# AOM-gated mode
AMPLITUDE = 1.6             # V peak into 50 ohm, both channels (as aod_tone.py)
NB_SAMPLES = 1_000_000      # 1.6 ms loop (a multiple of 32 samples); tones snap to a 625 Hz grid
TRIGGER_US = 10             # X2 pulse width
AOM_OPEN_US = 100           # AOM pulse start after the trigger, well past the camera's trigger latency
GATED_EXPOSURE_US = 500     # ends well before the next loop's pulse at 1700 us

NV, NH = 10, 90             # --crop-traps

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--exposure-us", type=float, default=1000, help="free-run camera exposure, us")
parser.add_argument("--aom-us", type=float, help="AOM-gated mode: light pulse length, us")
parser.add_argument("--f1", type=float, nargs="+", default=[70e6], help="AOM-gated mode: CH1 frequencies, Hz")
parser.add_argument("--f3", type=float, nargs="+", default=[70e6], help="AOM-gated mode: CH3 frequencies, Hz")
parser.add_argument("--crop-traps", action="store_true", help="rotate the array rows horizontal and crop to the traps")
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
    # strobe_rows opens the AOM for the first INT_HOLD_LEAD_US with the tones on, so the
    # camera is triggered only after that.
    camera_trigger = (t_us >= config.INT_HOLD_LEAD_US) & (t_us < config.INT_HOLD_LEAD_US + TRIGGER_US)
    if aom_blocked.all():
        raise ValueError(f"--aom-us {args.aom_us} is shorter than one sample ({1e6 / rate:g} us)")

    with aods(amplitudes=(AMPLITUDE, AMPLITUDE), digital_outputs=True) as awg, \
            Camera(GATED_EXPOSURE_US) as cam:
        x0, hold = strobe_rows(aom_blocked)
        awg.load_waveform([ch1, ch3, hold], -1, digital_data=[x0, camera_trigger])
        awg.force_trigger()
        image = cam.grab()
    label = (f"AOM {args.aom_us:g} us, CH1 {', '.join(f'{f / 1e6:g}' for f in f1)} MHz, "
             f"CH3 {', '.join(f'{f / 1e6:g}' for f in f3)} MHz")

peak, n_saturated = int(image.max()), int((image >= FULL_SCALE).sum())
print(f"{label}: peak {peak} / {FULL_SCALE}, mean {image.mean():.1f}, {n_saturated} saturated px")

if args.crop_traps:
    image = image.astype(float)   # find_traps smooths in the input dtype; uint16 makes tied maxima
    peaks, rows, cols = find_traps(image, NV, NH)
    grid, phi, _ = row_grid(fit_all(image, peaks), rows, cols, NV)
    image = map_coordinates(image, grid, order=1)
    label += f", rotated by {-np.degrees(phi):.3f} deg and cropped to the traps"

e2.apply()
h_mm = 140 if not args.crop_traps else 180 * image.shape[0] / image.shape[1] + 30
fig, ax = plt.subplots(figsize=(180 * e2.MM, h_mm * e2.MM), layout="constrained")
im = ax.imshow(image, cmap="inferno", vmin=0, vmax=FULL_SCALE)
fig.colorbar(im, ax=ax, label="counts")
ax.set_title(f"Basler, {label}\npeak {peak}, {n_saturated} saturated px")
save_plot(fig, "snapshot")
