"""Integrator hold for the LB1005-S intensity servo while the fiber AOM strobes.

The servo's photodiode sees the beam after the fiber AOM, so whenever X0 blocks the
beam the error signal rails and the integrator winds up, then drags the intensity
when the light returns. AWG CH2 drives the servo's rear-panel Int Hold input: TTL
high disconnects the error input and freezes the integrator output (LB1005 manual,
Table 4). The hold engages config.INT_HOLD_LEAD_US before X0 blocks and releases
config.INT_HOLD_RELEASE_US after X0 opens. An opening shorter than the two together
(every 50 ns strobe) never releases it, so through a strobed frame the servo keeps
the level it had locked to with the beam on.

The held integrator creeps by ~2 mV/s on Aux Output (beam on or off, 2026-11-24), against
the locked level's ~3 mV wobble: negligible over a ~66 ms frame, not over a long block.
"""

import numpy as np
from scipy.ndimage import maximum_filter1d

from aqc import config


def int_hold(aom_blocked, sampling_rate=config.AWG_SAMPLING_RATE):
    """CH2 row (0 or 1, relative to config.INT_HOLD_V) for a looped buffer with X0 row aom_blocked."""
    lead = round(config.INT_HOLD_LEAD_US * 1e-6 * sampling_rate)
    release = round(config.INT_HOLD_RELEASE_US * 1e-6 * sampling_rate)
    size = lead + release + 1
    # hold[i] = any(aom_blocked[i - release : i + lead + 1]), wrapping because the buffer loops.
    hold = maximum_filter1d(np.asarray(aom_blocked, bool).view(np.uint8), size,
                            mode="wrap", origin=release - size // 2)
    return hold.astype(np.float32)


def strobe_rows(aom_blocked, sampling_rate=config.AWG_SAMPLING_RATE):
    """X0 and CH2 rows for a looped buffer that should block the beam where aom_blocked is True.

    The card starts every row on the same sample, so a buffer that blocks from its first
    sample darkens the photodiode before the hold engages: the integrator took a ~10 mV
    kick on Aux Output at every start (2026-11-24). The beam is therefore left on for the
    buffer's first config.INT_HOLD_LEAD_US, with CH2 already high. It does so on every
    loop: keep the AODs' RF off there, or the camera not yet exposing.
    """
    x0 = np.array(aom_blocked, bool)
    x0[:round(config.INT_HOLD_LEAD_US * 1e-6 * sampling_rate)] = False
    return x0, int_hold(x0, sampling_rate)
