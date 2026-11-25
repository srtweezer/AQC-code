"""Take the data for a map of AOD steering over the camera field: every spot of a grid tracked while CH3, then CH1, is swept around 70 MHz.

The local steering vectors v3 = dX/df3 and v1 = dX/df1 (camera px/MHz) at each spot are
the slopes of these tracks; plot_steering_map.py fits and plots them. The grid is a phase
pattern with its deflection built in, e.g. the homogenized grating array of
scripts/calibration/grating_array.py, or a WGS pattern displayed with its usual blaze
(--blaze-y / --blaze-x, default none; the vertical 90x10 array takes --blaze-x -8).

Setup: SLM -> f = 200 mm -> blade (0th order blocked) -> f = 200 mm -> f = 40 mm objective
-> Basler, free-run. The script drives the AODs itself (CW, AMPLITUDE on both channels), so
stop aod_tone.py first. With CH1 and CH3 at 70 MHz the --n-spots brightest spots are found
(aqc.traps.find_spots) and fitted; then each channel in turn steps through 70 MHz +
offsets with the other at 70 MHz (aqc.steering.sweep: a 70 MHz reference frame before
every point, so drift of the whole grid cancels, and each spot's next position predicted
from its last two).

Writes a run folder (aqc.run, see DATA_TAKING.md), data/<MM_DD>[/<subdir>]/<stamp>_steering_map/:
raw/p0.npy (spots at 70/70 MHz), raw/offs_mhz.npy, raw/displacement_ch3.npy and
raw/displacement_ch1.npy. Leaves the pattern on the SLM and the AWG stopped.

Usage:
    python scripts/characterization/steering_map.py \\
        aqc/calibration/phase-patterns/20261116_204655_grating_array_50x50.npy \\
        --n-spots 2500 --exposure-us 400 --subdir aod_steering
    python scripts/characterization/steering_map.py \\
        aqc/calibration/phase-patterns/vertical/20261113_103928_wgs_90x10_cwgs_pos_homogenized.npy \\
        --n-spots 900 --exposure-us 150 --blaze-x -8 --subdir aod_steering
"""

import argparse
from pathlib import Path

import numpy as np

from aqc import config
from aqc.blaze import blaze
from aqc.hardware.awg import aods
from aqc.hardware.camera import Camera
from aqc.hardware.slm import send_phase
from aqc.run import Run
from aqc.steering import F_CENTER, NB_SAMPLES, grab, play, sweep
from aqc.traps import find_spots, fit_all

AMPLITUDE = 1.6          # V peak, the CW drive of aod_tone.py
CHANNELS = (3, 1)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("pattern", type=Path, help="phase map (.npy, radians)")
parser.add_argument("--n-spots", type=int, required=True, help="spots in the pattern")
parser.add_argument("--exposure-us", type=float, required=True, help="camera exposure, us")
parser.add_argument("--blaze-y", type=int, default=0, help="blaze period along y, px (0 = none)")
parser.add_argument("--blaze-x", type=int, default=0, help="blaze period along x, px (0 = none)")
parser.add_argument("--f-span", type=float, default=1.0, help="sweep half-span around 70 MHz, MHz")
parser.add_argument("--f-step", type=float, default=0.125, help="sweep step, MHz")
parser.add_argument("--average", type=int, default=4, help="frames averaged per grab")
parser.add_argument("--subdir", help="group the run under data/<MM_DD>/<subdir>/")
parser.add_argument("--note", help="why the run was taken")
args = parser.parse_args()

offs = np.round(np.arange(-args.f_span, args.f_span + args.f_step / 2, args.f_step), 6)
grid_mhz = config.AWG_SAMPLING_RATE / NB_SAMPLES / 1e6
if np.any(np.abs(offs / grid_mhz - np.round(offs / grid_mhz)) > 1e-6):
    raise ValueError(f"--f-span and --f-step must be multiples of the {grid_mhz * 1e3:g} kHz tone grid")

with Run("steering_map", args, note=args.note, subdir=args.subdir) as run:
    run.setup(pattern=args.pattern, blaze_y_px=args.blaze_y, blaze_x_px=args.blaze_x, aod_amplitude_v=AMPLITUDE,
              f_center_mhz=F_CENTER / 1e6, exposure_us=args.exposure_us)
    run.calibration("SLM_PHASE_RESPONSE")
    send_phase(np.load(args.pattern) + blaze(args.blaze_y, args.blaze_x))
    d = {}
    with aods(amplitudes=(AMPLITUDE, AMPLITUDE)) as awg, Camera(args.exposure_us, hardware_trigger=False) as cam:
        play(awg, F_CENTER)
        ref = grab(cam, args.average)
        p0 = fit_all(ref, find_spots(ref, args.n_spots))
        if not np.all(np.isfinite(p0)):
            raise RuntimeError(f"Gaussian fit failed for {np.isnan(p0[:, 0]).sum()} of {len(p0)} spots on the 70 MHz frame")
        print(f"70/70 MHz frame peak {ref.max():.0f} counts")
        for ch in CHANNELS:
            print(f"sweeping CH{ch}", flush=True)
            d[ch] = sweep(awg, cam, p0, offs, args.average, ch)

    run.save("p0", p0, ["spot", "yx"], "camera px", "Gaussian centres with CH1 = CH3 = 70 MHz")
    run.save("offs_mhz", offs, ["point"], "MHz", "offset of the swept channel from 70 MHz")
    for ch in CHANNELS:
        lost = int((~np.all(np.isfinite(d[ch]), axis=(0, 2))).sum())
        print(f"CH{ch}: {lost} of {len(p0)} spots lost somewhere in the sweep")
        run.save(f"displacement_ch{ch}", d[ch], ["point", "spot", "yx"], "camera px",
                 f"spot centre with CH{ch} at 70 MHz + offs_mhz, minus its centre on the 70/70 MHz frame "
                 "taken just before; NaN where the spot was lost")
