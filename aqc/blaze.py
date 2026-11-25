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


def grating_array(u, v, coef):
    """Phase (SLM_SHAPE) of the superposed gratings for spots at slopes u (n_u,) along x and v (n_v,) along y, waves per px.

    Spot (n, m) is the plane wave coef[n, m] exp(2 pi i (u_m x + v_n y)), with the sign of
    aqc.blaze (a period N is a slope 1 / N); coef (n_v, n_u) is complex, its modulus the
    spot's weight and its argument the spot's phase. The fields are added and only their
    argument is kept, which a phase-only SLM can show. The grid sum factors as
    Y @ coef @ X, so it costs two matrix products instead of n_u * n_v plane waves.
    """
    y = np.arange(config.SLM_SHAPE[0])
    x = np.arange(config.SLM_SHAPE[1])
    Y = np.exp(2j * np.pi * np.outer(y, v))
    X = np.exp(2j * np.pi * np.outer(u, x))
    return np.angle(Y @ coef @ X)
