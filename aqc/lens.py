"""Quadratic (defocus) phase for the SLM."""

import numpy as np

from aqc import config


def rho2():
    """r^2/R^2 over the SLM, r from its centre and R = half the short side (600 px, 4800 um)."""
    h, w = config.SLM_SHAPE
    y, x = np.mgrid[0:h, 0:w]
    return ((y - h / 2) ** 2 + (x - w / 2) ** 2) / (h / 2) ** 2


def defocus(waves):
    """Lens phase 2*pi*waves*r^2/R^2 over the SLM (rho2): `waves` waves at the edges of the short side.

    The steepest gradient, at the corners, is about 0.04 * waves rad/px, so this alone
    stays below the pi/px sampling limit up to ~80 waves; a blaze adds to it.
    """
    return 2 * np.pi * waves * rho2()
