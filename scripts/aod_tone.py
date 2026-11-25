"""Play constant tones on both AODs (AWG CH1 and CH3) until Enter is pressed.

Each channel plays one tone, or the sum of several (a multitone, one deflected beam
per tone). The amplitude is the 1.6 V peak the old e2-tools-aqc/awg_tone.py used for
CW drive, below the AOD_AMPLITUDES used for the 200 us pulses in calibration; --amplitude
lowers it, e.g. for a single SLM spot, which saturates the Basler at 1.6 V even at its
12 us minimum exposure. A multitone is rescaled so its peak is that amplitude, so each
tone gets less power than a single tone would.

With --flat, each CH3 tone is weighted by aqc.aod_amplitude.ch3_scale(f), so every
CH3 beam carries the same power (the calibration holds for 1.6 V on CH3 with CH1 at
70 MHz).

With --block-aom, X0 is held high so the AOM (active-low) blocks the beam while the AODs
keep their CW drive. The AOM extinction is finite: a single SLM spot still shows its leak
(about 1000 counts/us peak on the Basler, measured 2026-10-29).

Tones are snapped to the buffer's 3.125 kHz grid (625 MS/s / 200 000 samples), so
each fits an integer number of cycles and the loop wraps without a phase jump.

Usage:
    python scripts/aod_tone.py                         # 70 MHz on both
    python scripts/aod_tone.py --f3 60e6 70e6 80e6     # 3 tones on CH3, 70 MHz on CH1
    python scripts/aod_tone.py --f3 70e6 64.3292e6 --flat
    python scripts/aod_tone.py --block-aom
    python scripts/aod_tone.py --amplitude 0.25
"""

import argparse

import numpy as np

from aqc import config
from aqc.aod_amplitude import ch3_scale
from aqc.hardware.awg import aods
from aqc.intensity_lock import int_hold
from aqc.tones import multitone

AMPLITUDE = 1.6         # V peak into 50 ohm, both channels
NB_SAMPLES = 200_000

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--f1", type=float, nargs="+", default=[70e6], help="CH1 frequencies, Hz")
parser.add_argument("--f3", type=float, nargs="+", default=[70e6], help="CH3 frequencies, Hz")
parser.add_argument("--flat", action="store_true", help="weight CH3 tones by aqc.aod_amplitude.ch3_scale (flat power)")
parser.add_argument("--amplitude", type=float, default=AMPLITUDE, help="drive of both channels, V peak")
parser.add_argument("--block-aom", action="store_true", help="hold X0 high so the AOM blocks the beam")
args = parser.parse_args()

ch1, f1 = multitone(args.f1, NB_SAMPLES, config.AWG_SAMPLING_RATE)
ch3, f3 = multitone(args.f3, NB_SAMPLES, config.AWG_SAMPLING_RATE,
                    weights=ch3_scale(args.f3) if args.flat else None)

# Digital rows are X0 (AOM, high blocks) and X2 (camera trigger, left low).
rows, digital = [ch1, ch3], None
if args.block_aom:
    aom_blocked = np.ones(NB_SAMPLES, bool)
    rows, digital = [ch1, ch3, int_hold(aom_blocked)], [aom_blocked, None]
with aods(amplitudes=(args.amplitude, args.amplitude), digital_outputs=args.block_aom) as awg:
    awg.load_waveform(rows, -1, digital_data=digital)
    awg.force_trigger()
    input(f"CH1 {', '.join(f'{f / 1e6:g}' for f in f1)} MHz, CH3 {', '.join(f'{f / 1e6:g}' for f in f3)} MHz, "
          f"{args.amplitude:g} V peak per channel{', CH3 flattened' if args.flat else ''}"
          f"{', AOM blocked' if args.block_aom else ''}. Press Enter to stop...")
print("Stopped.")
