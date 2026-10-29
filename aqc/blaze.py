"""Blazed gratings for the SLM."""

import numpy as np

from aqc import config


def blaze_ramp(m, period_px):
    """Phase ramp of one wave per period_px pixels along pixel index m: 2*pi/N * (m mod N).

    The ramp is wrapped in integers (m mod N) so float error can't push a period start to
    ~2*pi. A negative period reverses the direction (m % N then lies in (N, 0]); 0 gives no ramp.
    """
    if period_px == 0:
        return 0
    return 2 * np.pi / period_px * (m % period_px)


def blaze(period_y_px=0, period_x_px=0):
    """Blazed grating over the whole SLM, signed periods in pixels along y and x (0 = none)."""
    y = np.arange(config.SLM_SHAPE[0])[:, None]
    x = np.arange(config.SLM_SHAPE[1])[None, :]
    return np.zeros(config.SLM_SHAPE) + blaze_ramp(y, period_y_px) + blaze_ramp(x, period_x_px)
