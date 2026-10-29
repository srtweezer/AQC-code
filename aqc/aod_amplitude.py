"""Flat CH3 power: the per-frequency amplitude scale measured by scripts/calibration/aod_amplitude.py."""

import numpy as np

from aqc import config


def ch3_scale(f_hz):
    """CH3 waveform scale (<= 1) at f_hz (scalar or array) that makes the power the AODs pass flat.

    Holds for the channel amplitude and CH1 frequency it was measured at (stored in
    config.CH3_AMPLITUDE_CAL). Frequencies outside the calibrated band raise: an AOD
    efficiency curve should not be extrapolated.
    """
    cal = np.load(config.CH3_AMPLITUDE_CAL)
    f = np.asarray(f_hz, dtype=float)
    if f.min() < cal["f_hz"][0] or f.max() > cal["f_hz"][-1]:
        raise ValueError(f"{f.min() / 1e6:g}-{f.max() / 1e6:g} MHz is outside the calibrated "
                         f"{cal['f_hz'][0] / 1e6:g}-{cal['f_hz'][-1] / 1e6:g} MHz")
    return np.interp(f, cal["f_hz"], cal["scale"])
