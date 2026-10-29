"""Sweep CH3 linearly over 2 s (60 to 80 MHz by default), repeating, while CH1 holds 70 MHz.

The whole sweep is one buffer played in a loop until Enter is pressed. At each wrap
the frequency jumps from the stop back to the start frequency. The phase stays
continuous there only if the sweep spans a whole number of cycles, i.e. if the mean
frequency times 2 s is an integer (true for any start + stop that is a multiple of
1 Hz). Both channels use the 1.6 V peak CW amplitude from aod_tone.py.

The card runs at 500 MS/s here, not config.AWG_SAMPLING_RATE. The M4i.6622 has 4 GiB,
which is 2^30 samples per channel with two channels active: 1.72 s at 625 MS/s, so a
2 s buffer only fits at a lower rate. Nyquist is still 250 MHz.

Usage:
    python scripts/aod_sweep.py                    # CH3 60 -> 80 MHz
    python scripts/aod_sweep.py --f3 65e6 75e6
"""

import argparse

import numpy as np

from aqc.hardware.awg import aods

AMPLITUDE = 1.6             # V peak into 50 ohm, both channels
SAMPLING_RATE = 500_000_000
DURATION = 2.0              # s
F1 = 70e6                   # Hz, CH1

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--f3", type=float, nargs=2, default=[60e6, 80e6], metavar=("START", "STOP"),
                    help="CH3 sweep start and stop frequencies, Hz")
args = parser.parse_args()
F3_START, F3_STOP = args.f3

nb_samples = round(DURATION * SAMPLING_RATE)
times = np.arange(nb_samples) / SAMPLING_RATE
ch1 = np.cos(2 * np.pi * F1 * times)
ch3 = np.cos(2 * np.pi * (F3_START * times + (F3_STOP - F3_START) / (2 * DURATION) * times**2))
del times

with aods(amplitudes=(AMPLITUDE, AMPLITUDE), sampling_rate=SAMPLING_RATE) as awg:
    awg.load_waveform([ch1, ch3], -1)
    awg.force_trigger()
    input(f"CH1 {F1 / 1e6:g} MHz, CH3 {F3_START / 1e6:g} -> {F3_STOP / 1e6:g} MHz every {DURATION:g} s, "
          f"{AMPLITUDE} V peak per channel. Press Enter to stop...")
print("Stopped.")
