"""Play constant tones on both AODs (AWG CH1 and CH3) until Enter is pressed.

Each channel plays one tone, or the sum of several (a multitone, one deflected beam
per tone). The amplitude is the 1.6 V peak the old e2-tools-aqc/awg_tone.py used for
CW drive, below the AOD_AMPLITUDES used for the 200 us pulses in calibration. A
multitone is rescaled so its peak is that amplitude, so each tone gets less power
than a single tone would.

With --flat, each CH3 tone is weighted by aqc.aod_amplitude.ch3_scale(f), so every
CH3 beam carries the same power (the calibration holds for 1.6 V on CH3 with CH1 at
70 MHz).

Tones are snapped to the buffer's 3.125 kHz grid (625 MS/s / 200 000 samples), so
each fits an integer number of cycles and the loop wraps without a phase jump.

Usage:
    python scripts/aod_tone.py                         # 70 MHz on both
    python scripts/aod_tone.py --f3 60e6 70e6 80e6     # 3 tones on CH3, 70 MHz on CH1
    python scripts/aod_tone.py --f3 70e6 64.3292e6 --flat
"""

import argparse

import numpy as np

from aqc import config
from aqc.aod_amplitude import ch3_scale
from aqc.hardware.awg import aods
from aqc.tones import multitone

AMPLITUDE = 1.6         # V peak into 50 ohm, both channels
NB_SAMPLES = 200_000

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--f1", type=float, nargs="+", default=[70e6], help="CH1 frequencies, Hz")
parser.add_argument("--f3", type=float, nargs="+", default=[70e6], help="CH3 frequencies, Hz")
parser.add_argument("--flat", action="store_true", help="weight CH3 tones by aqc.aod_amplitude.ch3_scale (flat power)")
args = parser.parse_args()

ch1, f1 = multitone(args.f1, NB_SAMPLES, config.AWG_SAMPLING_RATE)
ch3, f3 = multitone(args.f3, NB_SAMPLES, config.AWG_SAMPLING_RATE,
                    weights=ch3_scale(args.f3) if args.flat else None)

with aods(amplitudes=(AMPLITUDE, AMPLITUDE)) as awg:
    awg.load_waveform(np.stack([ch1, ch3]), -1)
    awg.force_trigger()
    input(f"CH1 {', '.join(f'{f / 1e6:g}' for f in f1)} MHz, CH3 {', '.join(f'{f / 1e6:g}' for f in f3)} MHz, "
          f"{AMPLITUDE} V peak per channel{', CH3 flattened' if args.flat else ''}. Press Enter to stop...")
print("Stopped.")
