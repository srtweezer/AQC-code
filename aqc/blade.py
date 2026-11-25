"""Blade (knife-edge) centres and widths of spots moved across the pixel grid by known shifts.

See docs/blade_method.md. A spot moved by known sub-pixel shifts sweeps across the fixed
pixel boundaries; at each boundary e the fraction of its light before e is
Phi((e - c - shift) / sigma), so fitting c and sigma over all shifts and boundaries gives
the centre and width free of the pixel-area blur and pixel-phase bias of a Gaussian fit to
one frame. Pixel j covers [j - 0.5, j + 0.5], the convention of aqc.traps.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr

HALF = 6                 # window half-size, px: > 5 sigma of the ~1 px spots, < half the ~28 px pitch


def kept_pixels(shape, centers, reach):
    """Flat indices of the pixels that a window around any of centers (N, 2) (y, x), moved by up to reach px, can touch, and the (H, W) index map of them (pixel number, -1 elsewhere) that windows takes."""
    m = int(np.ceil(reach)) + HALF + 1
    mask = np.zeros(shape, bool)
    for y, x in np.round(centers[np.isfinite(centers).all(axis=1)]).astype(int):
        mask[max(y - m, 0):y + m + 1, max(x - m, 0):x + m + 1] = True
    pixels = np.flatnonzero(mask)
    index = np.full(shape, -1)
    index.flat[pixels] = np.arange(len(pixels))
    return pixels, index


def windows(frames, index, centers):
    """(n, 2 HALF + 1, 2 HALF + 1) windows of frames (n, P) around integer centers (n, 2) (y, x).

    The frames hold only the pixels of the (H, W) map index (pixel number in P, -1 outside).
    """
    d = np.arange(-HALF, HALF + 1)
    idx = index[centers[:, 0, None, None] + d[:, None], centers[:, 1, None, None] + d]
    if (idx < 0).any():
        raise ValueError("a window reaches outside the saved pixels")
    return np.take_along_axis(frames, idx.reshape(len(frames), -1), axis=1).reshape(idx.shape)


def edge_fit(win, corner, shift, guess):
    """Centre (px, camera coordinate at zero shift), Gaussian sigma (px) and background (counts/px) along one axis.

    win: (n, L) the window summed across the axis in n frames, background included;
    corner: (n,) camera coordinate of each window's first pixel; shift: (n,) the spot's
    displacement in each frame (px); guess: starting centre.

    The strobed frames keep a flat background (~10 counts/px, ~30% of a window's light).
    It is one fit parameter b per pixel, shared by all frames: it adds b L per column, so
    the light before boundary k is (total - L^2 b) Phi + b L (k + 1). Subtracting the
    window's border mean instead under-subtracts by ~0.5 counts/px and inflates sigma
    along the row by ~8%; a percentile of the window's pixels is off by ~2 counts/px
    (data/11_23 comparison).
    """
    L = win.shape[1]
    C = np.cumsum(win, axis=1)[:, :-1]
    total = win.sum(axis=1, keepdims=True)
    k = np.arange(1, L)                                                        # columns before each boundary
    z = corner[:, None] + np.arange(L - 1) + 0.5 - shift[:, None]              # boundaries in the unshifted frame

    def residual(q):
        c, sigma, b = q
        return (((total - L * L * b) * ndtr((z - c) / sigma) + b * L * k - C) / total).ravel()

    fit = least_squares(residual, [guess, 1.0, 0.0], bounds=([-np.inf, 0.1, -np.inf], [np.inf, HALF, np.inf]))
    return fit.x


def blade(frames, index, center, shifts):
    """(y0, x0, sigma_y, sigma_x, light) of the spot near center (y, x) at zero shift, all px but light.

    frames (n, P): the spot moved by shifts (n, 2) (y, x) px, pixels as in windows. Each
    window follows the spot to the nearest pixel, so it always holds the whole spot.
    light is the mean window sum (counts) less the fitted background (edge_fit), averaged
    over the y and x fits.
    """
    centers = np.round(center + shifts).astype(int)
    win = windows(frames, index, centers).astype(float)
    corner = centers - HALF
    y0, sy, by = edge_fit(win.sum(axis=2), corner[:, 0], shifts[:, 0], center[0])
    x0, sx, bx = edge_fit(win.sum(axis=1), corner[:, 1], shifts[:, 1], center[1])
    return y0, x0, sy, sx, win.sum(axis=(1, 2)).mean() - win[0].size * (by + bx) / 2
