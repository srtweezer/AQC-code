"""Field-dependent aberration correction (Machu et al., arXiv:2512.15967): basis, field positions, model.

A spot at field position r is corrected by adding sum_k c_k(r) Z_k(rho) on the SLM (minus
the paper's Phi). The modes are the paper's Eq. 5 without distortion, in ANSI order, on a
radius of half the short SLM side (as aqc.lens.defocus), in radians at rho = 1, evaluated
over the whole rectangular SLM.

Field positions r = (y, x) are in um in the frame of aqc.wgs.trap_indices, from the 0th
order, with the blaze included: um in the tweezer plane the camera sees (aqc.wgs.FOCAL_UM).
A calibration file records the focal length its field um were computed with (focal_um),
and load refuses one that doesn't match. The 0th order and the half-plane around it are blocked by
a D mirror, so the model is fitted, and only valid, over the patch the reference spots of
scripts/calibration/aberration_scan.py cover.

The model (Eq. 5, with the paper's free centres for astigmatism and curvature) is linear in
its parameters once the centres are folded in (u = A x_a, ...), so it is fitted by linear
least squares. Over an off-axis patch a coma centre is not separable from its constant, so
coma has none. The constants play the role of the paper's on-axis mask phi_0.
"""

import numpy as np
from e2tools.slm.zernike_utils import zernike_polynomial

from aqc.wgs import FOCAL_UM, PIXEL_UM, SLM_H, SLM_W, WAVELENGTH_UM, um_per_ffpix

MODES = [(2, -2), (2, 0), (2, 2), (3, -1), (3, 1)]
MODE_NAMES = ["astigmatism 45", "defocus", "astigmatism 0", "coma y", "coma x"]
R_PX = SLM_H / 2
PARAMS = ["a45_0", "defocus_0", "a0_0", "coma_y_0", "coma_x_0",
          "A", "A_xa", "A_ya", "K", "K_xk", "K_yk", "C"]


def basis():
    """Zernike modes (len(MODES), SLM_H, SLM_W), float32."""
    y, x = np.mgrid[0:SLM_H, 0:SLM_W]
    Y, X = (y - SLM_H / 2) / R_PX, (x - SLM_W / 2) / R_PX
    rho, theta = np.hypot(X, Y), np.arctan2(Y, X)
    return np.stack([zernike_polynomial(n, m, rho, theta) for n, m in MODES]).astype(np.float32)


def blaze_offset_um(period_px):
    """Field shift (um) by which an aqc.blaze period of period_px pixels (signed, 0 = none) moves the first order.

    The same along y and x: the blaze and the trap waves of aqc.wgs.grid_phase share one sign convention.
    """
    return 0.0 if period_px == 0 else WAVELENGTH_UM * FOCAL_UM / (period_px * PIXEL_UM)


def field_um(rr, cc, scal, blaze_y_px, blaze_x_px=0):
    """Field positions (..., 2) as (y, x) um of padded-grid sites (rr, cc), displayed with a blaze."""
    uy, ux = um_per_ffpix(scal)
    return np.stack([(np.asarray(rr) - scal * SLM_H / 2) * uy + blaze_offset_um(blaze_y_px),
                     (np.asarray(cc) - scal * SLM_W / 2) * ux + blaze_offset_um(blaze_x_px)], axis=-1)


def grid_site(r, scal, blaze_y_px, blaze_x_px=0):
    """Nearest padded-grid site (rr, cc) of field positions r (..., 2), (y, x) um; inverse of field_um."""
    uy, ux = um_per_ffpix(scal)
    r = np.asarray(r, float)
    return (np.round(scal * SLM_H / 2 + (r[..., 0] - blaze_offset_um(blaze_y_px)) / uy).astype(int),
            np.round(scal * SLM_W / 2 + (r[..., 1] - blaze_offset_um(blaze_x_px)) / ux).astype(int))


def design(r):
    """Design matrix (N, len(MODES), len(PARAMS)) of the model at field positions r (N, 2), (y, x) um."""
    y, x = np.asarray(r, float).T
    one, zero = np.ones_like(x), np.zeros_like(x)
    e = np.eye(len(MODES))
    rows = [
        # A[(x - xa)^2 - (y - ya)^2] and 2A(x - xa)(y - ya) with u = A xa, v = A ya
        [e[0], 2 * x * y, -2 * y, -2 * x, zero, zero, zero, zero],     # astigmatism 45
        [e[1], zero, zero, zero, x ** 2 + y ** 2, -2 * x, -2 * y, zero],  # defocus: K|r - r_k|^2
        [e[2], x ** 2 - y ** 2, -2 * x, 2 * y, zero, zero, zero, zero],   # astigmatism 0
        [e[3], zero, zero, zero, zero, zero, zero, y],                  # coma y: C y
        [e[4], zero, zero, zero, zero, zero, zero, x],                  # coma x: C x
    ]
    out = np.empty((len(x), len(MODES), len(PARAMS)))
    for k, (const, *terms) in enumerate(rows):
        out[:, k, :len(MODES)] = const * one[:, None]
        out[:, k, len(MODES):] = np.stack(terms, axis=1)
    return out


def model(params, r):
    """Correction coefficients (N, len(MODES)), rad, at field positions r (N, 2), (y, x) um."""
    return design(r) @ np.asarray(params, float)


def fit(r, c):
    """Least-squares params (len(PARAMS),) of the model to measured coefficients c (N, len(MODES))."""
    D = design(r).reshape(-1, len(PARAMS))
    scale = np.linalg.norm(D, axis=0)       # the columns span ~1 to ~1e5 um^2
    p, *_ = np.linalg.lstsq(D / scale, np.asarray(c, float).ravel(), rcond=None)
    return p / scale


def seidel(params):
    """C' (rad/um), A', K' (rad/um^2) and the astigmatism and curvature centres (y, x) um."""
    p = dict(zip(PARAMS, params))
    return {"C": p["C"], "A": p["A"], "K": p["K"],
            "astig_centre": (p["A_ya"] / p["A"], p["A_xa"] / p["A"]),
            "curv_centre": (p["K_yk"] / p["K"], p["K_xk"] / p["K"])}


def load(path):
    """Model params from a calibration file of scripts/calibration/aberration_scan.py."""
    with np.load(path) as f:
        if list(f["param_names"]) != PARAMS:
            raise ValueError(f"{path} has params {list(f['param_names'])}, expected {PARAMS}")
        if "focal_um" not in f or float(f["focal_um"]) != FOCAL_UM:
            raise ValueError(f"{path} has field um for focal length {f['focal_um'] if 'focal_um' in f else 7900.0} um, "
                             f"aqc.wgs.FOCAL_UM is {FOCAL_UM} um")
        return f["params"]


def correction(params, rr, cc, scal, blaze_y_px, blaze_x_px=0):
    """Per-trap coefficients (..., len(MODES)) for run_wgs at padded-grid sites (rr, cc), displayed with that blaze."""
    r = field_um(rr, cc, scal, blaze_y_px, blaze_x_px)
    return model(params, r.reshape(-1, 2)).reshape(*r.shape[:-1], len(MODES))
