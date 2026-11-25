"""Per-trap focus position from a global defocus scan, ported from slm_local/zscan_09_15 (zscan.py, zfit.py).

DEFOCUS CONVENTION
    phi(r) = a * rho^2 (aqc.lens.rho2: R = 4800 um, half the 1200-px SLM axis),
    a in radians at rho = 1. dz/da = lambda f^2 / (pi R^2) = 23.52 um per radian
    (lambda = 1.064 um, f = aqc.wgs.FOCAL_UM = 40 mm): z along the beam at the tweezer
    plane the camera sees. The sign of z follows the sign of a. (Before 2026-11-24, with
    f = 7900 um, it was 0.9174 um per radian.)

A trap that is sharpest on the camera when a global lens a = z0 / UM_PER_RAD is added is
brought to the plane of the others by its own lens (z0 - mean z0) / UM_PER_RAD (aqc.wgs.
corrected_waves). Global and per-trap lenses are the same phase on the SLM, so the
correction needs neither the physical direction of z nor its absolute scale.
"""

import numpy as np
from scipy.optimize import curve_fit

from aqc.hardware.slm import send_phase
from aqc.intensity import FULL_SCALE
from aqc.lens import rho2
from aqc.steering import grab
from aqc.traps import find_traps, fit_all, fit_spots
from aqc.wgs import FOCAL_UM, PIXEL_UM, SLM_H, WAVELENGTH_UM

UM_PER_RAD = WAVELENGTH_UM * FOCAL_UM ** 2 / (np.pi * (SLM_H / 2 * PIXEL_UM) ** 2)
FLAGS = "0 ok, 1 unbracketed, 2 too few points, 3 fit failed"


def zscan(cam, phase, zs, average, nv, nh):
    """Centres (nv*nh, 2) at z = 0 and Gaussian fits (len(zs), nv*nh, 5) of every trap at every z in zs (um).

    phase is displayed as given (include the blaze), plus a global lens z / UM_PER_RAD * rho^2
    per step. Traps are found at z = 0 and come out in design order (row-major), and each is
    fitted at its z = 0 centre (aqc.traps.fit_spots columns; NaN where a fit failed). The z
    values are visited in shuffled order so drift shows up as scatter rather than as a
    plausible trend.
    """
    send_phase(phase)
    frame = grab(cam, average)
    pts, r, c = find_traps(frame, nv, nh)
    guess = np.empty((nv, nh, 2))
    guess[r, c] = pts
    pos = fit_all(frame, guess.reshape(-1, 2))
    if not np.all(np.isfinite(pos)):
        raise RuntimeError(f"Gaussian fit failed for {np.isnan(pos[:, 0]).sum()} of {nv * nh} traps at z = 0")
    R2 = rho2()
    fits = np.full((len(zs), nv * nh, 5), np.nan)
    for step, i in enumerate(np.random.default_rng(0).permutation(len(zs))):
        send_phase(phase + zs[i] / UM_PER_RAD * R2)
        frame = grab(cam, average)
        fits[i] = fit_spots(frame, pos)
        sigma = np.sqrt(fits[i, :, 3] * fits[i, :, 4])
        print(f"  [{step + 1:2d}/{len(zs)}] z {zs[i]:+.2f} um: fits {np.isfinite(sigma).sum()}/{nv * nh}, "
              f"sigma {np.nanmean(sigma):.3f} +- {np.nanstd(sigma):.3f} px, "
              f"saturated px {(frame >= FULL_SCALE).sum()}", flush=True)
    return pos, fits


def model(z, s0sq, z0, zr, c):
    """Squared Gaussian-beam width: s0^2 (1 + ((z - z0) / zR)^2), plus a floor c for camera blur and pixel MTF."""
    return s0sq * (1 + ((z - z0) / zr) ** 2) + c


def fit_z0(z, sigma, edge=2, min_points=8):
    """Focus z0 (N,), its 1-sigma error (N,) and a flag (N,) per trap (FLAGS) from widths sigma (len(z), N), px.

    Points where sigma is NaN (failed spot fits) are dropped. A trap is only fitted if its
    width minimum is bracketed by the scan: a minimum within `edge` steps of either end
    would extrapolate a parabola past the data, so it is flagged (z0 records where the
    width bottomed out) and the scan should be widened.
    """
    n = sigma.shape[1]
    z0, err, flag = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, int)
    for m in range(n):
        good = np.isfinite(sigma[:, m])
        if good.sum() < min_points:
            flag[m] = 2
            continue
        zz, ss = z[good], sigma[good, m] ** 2
        imin = int(np.argmin(ss))
        if imin < edge or imin > len(zz) - 1 - edge:
            flag[m] = 1
            z0[m] = zz[imin]
            continue
        try:
            popt, pcov = curve_fit(model, zz, ss, p0=[ss.min(), zz[imin], 1.6, 0.0],
                                   bounds=([0, zz[0], 0.2, -np.inf], [np.inf, zz[-1], 20.0, np.inf]),
                                   maxfev=8000)
        except RuntimeError:          # curve_fit hit maxfev
            flag[m] = 3
            continue
        z0[m], err[m] = popt[1], np.sqrt(abs(pcov[1, 1]))
    return z0, err, flag
