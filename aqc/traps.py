"""Finding and fitting the spots of a rectangular tweezer array on the camera."""

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import curve_fit

MIN_SEP_PX = 9           # local-maximum half-window; the camera pitch is ~28 px
# A trap is a local maximum above this fraction of a typical trap's brightness. All
# nv*nh must be found, so a dim trap stops the run instead of being skipped.
SPOT_FRAC = 0.3
MAX_JUMP_PX = 6          # max distance of a fitted trap from its prediction (half a site is ~14 px)
MIN_SNR = 5              # fitted amplitude / fit residual rms


def gauss2d(xy, amp, x0, y0, sx, sy, off):
    x, y = xy
    return amp * np.exp(-((x - x0) ** 2 / (2 * sx ** 2) + (y - y0) ** 2 / (2 * sy ** 2))) + off


def fit_spot(image, cy, cx, box=9, rec=4):
    """Gaussian centre y, x and amplitude (counts) of the spot at (cy, cx), or NaNs if the fit fails or is weak.

    Ported from slm_local/zflat_09_23/fitlib.fit_one: snap to the brightest pixel within
    rec, then fit a (2 box + 1)^2 window around it.
    """
    yi, xi = int(round(cy)), int(round(cx))
    w = image[max(yi - rec, 0):yi + rec + 1, max(xi - rec, 0):xi + rec + 1]
    dy, dx = np.unravel_index(np.argmax(w), w.shape)
    yi, xi = max(yi - rec, 0) + dy, max(xi - rec, 0) + dx
    y0, y1, x0, x1 = yi - box, yi + box + 1, xi - box, xi + box + 1
    if y0 < 0 or x0 < 0 or y1 > image.shape[0] or x1 > image.shape[1]:
        return np.nan, np.nan, np.nan
    p = image[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    off0 = np.percentile(p, 10)
    try:
        popt, _ = curve_fit(lambda xy, *q: gauss2d(xy, *q).ravel(), (xx, yy), p.ravel(),
                            p0=[p.max() - off0, xi, yi, 1.3, 1.3, off0],
                            bounds=([0, x0, y0, 0.3, 0.3, -np.inf], [np.inf, x1, y1, box, box, np.inf]),
                            maxfev=4000)
    except RuntimeError:          # curve_fit hit maxfev
        return np.nan, np.nan, np.nan
    resid = p - gauss2d((xx, yy), *popt)
    if popt[0] < MIN_SNR * np.sqrt(np.mean(resid ** 2)):
        return np.nan, np.nan, np.nan
    return popt[2], popt[1], popt[0]


def fit_spots(image, guess):
    """fit_spot at every (y, x) in guess (N, 2): (N, 3) as y, x, amplitude.

    NaN where a fit fails or lands > MAX_JUMP_PX away.
    """
    out = np.full((len(guess), 3), np.nan)
    for k, (y, x) in enumerate(guess):
        if np.isfinite(y):
            out[k] = fit_spot(image, y, x)
    out[np.linalg.norm(out[:, :2] - guess, axis=1) > MAX_JUMP_PX] = np.nan
    return out


def fit_all(image, guess):
    """Gaussian centres (N, 2) at every (y, x) in guess; see fit_spots."""
    return fit_spots(image, guess)[:, :2]


def find_traps(image, nv, nh):
    """Integer (y, x) of all nv*nh traps (N, 2), with their row and column indices (N,) each."""
    smooth = gaussian_filter(image, 1.5)
    peaks = np.argwhere(smooth == maximum_filter(smooth, size=2 * MIN_SEP_PX + 1))
    values = smooth[peaks[:, 0], peaks[:, 1]]
    typical = np.median(np.sort(values)[::-1][:nv * nh])
    lit = values > SPOT_FRAC * typical
    pts, values = peaks[lit].astype(float), values[lit]
    # Stray spots below the array (ghost orders) are as bright as the dimmest traps at the
    # array ends, but they are isolated, while every trap has a neighbour one pitch away.
    d = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    isolated = nn > 1.5 * np.median(nn)
    if isolated.any():
        print(f"    ignoring {isolated.sum()} isolated spot(s) at (y, x) {pts[isolated].astype(int).tolist()}")
    pts, values = pts[~isolated], values[~isolated]
    # Some ghost spots come in pairs one pitch apart, so they aren't isolated. On a
    # homogenized array every trap is far brighter than them (>0.8 vs <0.4 x typical), so
    # the extras are the dimmest. The row and column checks below catch a wrong pick.
    if len(pts) > nv * nh:
        extra = np.argsort(values)[:len(pts) - nv * nh]
        print(f"    ignoring the {len(extra)} dimmest spot(s) at (y, x) {pts[extra].astype(int).tolist()}")
        pts = np.delete(pts, extra, axis=0)
    if len(pts) != nv * nh:
        raise RuntimeError(f"found {len(pts)} spots above {SPOT_FRAC:g} x typical brightness on the "
                           f"lattice, expected {nv * nh}")

    # The array is several times longer than wide, so its principal axis is the row
    # direction to well under a site over its length; across it the rows are separated
    # by the nv - 1 largest gaps.
    u, v = rotate(pts, principal_angle(pts))
    order = np.argsort(v)
    gaps = np.diff(v[order])
    cuts = np.sort(np.argsort(gaps)[-(nv - 1):])
    if gaps[cuts].min() < 3 * np.delete(gaps, cuts).max():
        raise RuntimeError(f"{len(pts)} spots do not split cleanly into {nv} rows")
    rows = np.empty(len(pts), int)
    rows[order] = np.searchsorted(cuts, np.arange(len(pts)), side="left")

    counts = np.bincount(rows, minlength=nv)
    if np.any(counts != nh):
        raise RuntimeError(f"rows hold {counts.tolist()} spots, expected {nh} each")
    # Columns by order along the row: the pitch on the camera varies across the array,
    # so dividing by a median pitch can put two traps in one column.
    cols = np.empty(len(pts), int)
    for r in range(nv):
        idx = np.flatnonzero(rows == r)
        cols[idx[np.argsort(u[idx])]] = np.arange(nh)
    return pts, rows, cols


def principal_angle(pts):
    """Angle (rad, atan2(y, x) on the camera) of the long axis of a point cloud (N, 2) in (y, x)."""
    d = pts - pts.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov(d[:, 1], d[:, 0]))
    ex, ey = evecs[:, np.argmax(evals)]
    return np.arctan2(ey, ex) if ex >= 0 else np.arctan2(-ey, -ex)


def rotate(pts, phi):
    """Coordinates (u along phi, v across) of (..., 2) (y, x) points."""
    y, x = pts[..., 0], pts[..., 1]
    return x * np.cos(phi) + y * np.sin(phi), -x * np.sin(phi) + y * np.cos(phi)


def row_fit(pos, rows, nv):
    """Common row angle (deg) of centres (N, 2) in rows (N,): one slope, one intercept per row."""
    good = np.all(np.isfinite(pos), axis=1)
    phi0 = principal_angle(pos[good])
    u, v = rotate(pos, phi0)
    X = np.column_stack([u[good], np.eye(nv)[rows[good]]])
    coef, *_ = np.linalg.lstsq(X, v[good], rcond=None)
    return np.degrees(phi0 + np.arctan(coef[0]))
