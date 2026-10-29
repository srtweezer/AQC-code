"""Quadratic (defocus) phase for the SLM."""

import numpy as np

from aqc import config


def defocus(waves):
    """Lens phase 2*pi*waves*r^2/R^2 over the SLM, R = half the short side: `waves` waves at its edges.

    The steepest gradient, at the corners, is about 0.04 * waves rad/px, so this alone
    stays below the pi/px sampling limit up to ~80 waves; a blaze adds to it.
    """
    h, w = config.SLM_SHAPE
    y, x = np.mgrid[0:h, 0:w]
    return 2 * np.pi * waves * ((y - h / 2) ** 2 + (x - w / 2) ** 2) / (h / 2) ** 2
