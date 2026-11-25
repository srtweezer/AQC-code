"""Finding and fitting the spots of a rectangular tweezer array on the camera."""

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

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
    """Gaussian centre y, x, amplitude (counts) and sigma y, x (px) of the spot at (cy, cx), or NaNs if the fit fails or is weak.

    Ported from slm_local/zflat_09_23/fitlib.fit_one: snap to the brightest pixel within
    rec, then fit a (2 box + 1)^2 window around it.
    """
    yi, xi = int(round(cy)), int(round(cx))
    w = image[max(yi - rec, 0):yi + rec + 1, max(xi - rec, 0):xi + rec + 1]
    dy, dx = np.unravel_index(np.argmax(w), w.shape)
    yi, xi = max(yi - rec, 0) + dy, max(xi - rec, 0) + dx
    y0, y1, x0, x1 = yi - box, yi + box + 1, xi - box, xi + box + 1
    if y0 < 0 or x0 < 0 or y1 > image.shape[0] or x1 > image.shape[1]:
        return (np.nan,) * 5
    p = image[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    off0 = np.percentile(p, 10)
    try:
        popt, _ = curve_fit(lambda xy, *q: gauss2d(xy, *q).ravel(), (xx, yy), p.ravel(),
                            p0=[p.max() - off0, xi, yi, 1.3, 1.3, off0],
                            bounds=([0, x0, y0, 0.3, 0.3, -np.inf], [np.inf, x1, y1, box, box, np.inf]),
                            maxfev=4000)
    except RuntimeError:          # curve_fit hit maxfev
        return (np.nan,) * 5
    resid = p - gauss2d((xx, yy), *popt)
    if popt[0] < MIN_SNR * np.sqrt(np.mean(resid ** 2)):
        return (np.nan,) * 5
    return popt[2], popt[1], popt[0], popt[4], popt[3]


def fit_spots(image, guess):
    """fit_spot at every (y, x) in guess (N, 2): (N, 5) as y, x, amplitude, sigma y, sigma x.

    NaN where a fit fails or lands > MAX_JUMP_PX away.
    """
    out = np.full((len(guess), 5), np.nan)
    for k, (y, x) in enumerate(guess):
        if np.isfinite(y):
            out[k] = fit_spot(image, y, x)
    out[np.linalg.norm(out[:, :2] - guess, axis=1) > MAX_JUMP_PX] = np.nan
    return out


def fit_all(image, guess):
    """Gaussian centres (N, 2) at every (y, x) in guess; see fit_spots."""
    return fit_spots(image, guess)[:, :2]


def find_traps(image, nv, nh, spot_frac=SPOT_FRAC):
    """Integer (y, x) of all nv*nh traps (N, 2), with their row and column indices (N,) each.

    A trap is a local maximum above spot_frac of a typical trap's brightness.
    """
    smooth = gaussian_filter(image, 1.5)
    peaks = np.argwhere(smooth == maximum_filter(smooth, size=2 * MIN_SEP_PX + 1))
    values = smooth[peaks[:, 0], peaks[:, 1]]
    typical = np.median(np.sort(values)[::-1][:nv * nh])
    lit = values > spot_frac * typical
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
        raise RuntimeError(f"found {len(pts)} spots above {spot_frac:g} x typical brightness on the "
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


def find_spots(image, n, spot_frac=SPOT_FRAC):
    """Integer (y, x) (n, 2) of the n brightest local maxima: the spots of a pattern of n spots in any layout.

    Raises if the n-th brightest is below spot_frac of a typical spot, i.e. a spot is
    missing and a ghost or noise peak would take its place.
    """
    smooth = gaussian_filter(image, 1.5)
    peaks = np.argwhere(smooth == maximum_filter(smooth, size=2 * MIN_SEP_PX + 1))
    values = smooth[peaks[:, 0], peaks[:, 1]]
    order = np.argsort(values)[::-1]
    typical = np.median(values[order[:n]])
    if values[order[n - 1]] < spot_frac * typical:
        raise RuntimeError(f"the {n}-th brightest spot is {values[order[n - 1]] / typical:.2f} x typical, "
                           f"below {spot_frac:g}: fewer than {n} spots are lit")
    print(f"    {n} spots, dimmest {values[order[n - 1]] / typical:.2f} x typical; "
          f"brightest peak left out {values[order[n]] / typical:.2f} x typical")
    return peaks[order[:n]]


def vertical_design_order(nv, nh):
    """Camera index (nv * nh,) of every design trap of the vertical array, in design (row-major) order.

    find_traps sees the vertical array (nh design rows x nv design columns) as nv camera
    lines of nh traps: line i is design column i, and trap j along it is design row
    nh - 1 - j (checked 2026-11-12, see scripts/calibration/homogenize.py). So
    x[..., order] puts camera-ordered values in design order, and x[np.argsort(order)]
    puts design-ordered values in camera order.
    """
    return np.arange(nv * nh).reshape(nv, nh)[:, ::-1].T.ravel()


def find_central_traps(image, nv, ncols):
    """Integer (y, x) of the central ncols columns of an nv-row array whose ends are cut off (N, 2), with row and column indices (N,) each.

    find_traps needs every trap. When the ends of the array are blocked in the beam path
    they fade out over a column or so, rows end at different columns, and no trap count
    fits. Here each row is indexed by stepping from trap to trap, the rows are registered
    to each other at the middle of what is visible, and only the central ncols columns are
    kept. Every one of those must be found. Column 0 is the first kept column.
    """
    smooth = gaussian_filter(image, 1.5)
    peaks = np.argwhere(smooth == maximum_filter(smooth, size=2 * MIN_SEP_PX + 1))
    values = smooth[peaks[:, 0], peaks[:, 1]]
    pts = peaks[values > SPOT_FRAC * np.median(np.sort(values)[::-1][:nv * ncols])].astype(float)
    d = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    pts = pts[nn < 1.5 * np.median(nn)]      # ghost orders are isolated; traps have a neighbour one pitch away

    u, v = rotate(pts, principal_angle(pts))
    order = np.argsort(v)
    gaps = np.diff(v[order])
    cuts = np.sort(np.argsort(gaps)[-(nv - 1):])
    if gaps[cuts].min() < 3 * np.delete(gaps, cuts).max():
        raise RuntimeError(f"{len(pts)} spots do not split cleanly into {nv} rows")
    rows = np.empty(len(pts), int)
    rows[order] = np.searchsorted(cuts, np.arange(len(pts)), side="left")

    # Along a row, consecutive traps are one pitch apart; the pitch varies by ~1% across
    # the array, so each step is rounded on its own rather than dividing by one pitch.
    idx = [np.flatnonzero(rows == r)[np.argsort(u[rows == r])] for r in range(nv)]
    pitch = np.median(np.concatenate([np.diff(u[i]) for i in idx]))
    local = [np.concatenate([[0], np.cumsum(np.round(np.diff(u[i]) / pitch))]).astype(int) for i in idx]
    # The two AODs are not quite orthogonal, so a column drifts along u from row to row
    # (~2 px per row); over 10 rows that is more than half a pitch.
    shift = np.zeros(nv)
    for r in range(1, nv):
        du = u[idx[r]][:, None] - u[idx[r - 1]][None, :]
        shift[r] = shift[r - 1] + np.median(du[np.arange(len(du)), np.abs(du).argmin(axis=1)])
    shift -= shift.mean()
    mid = np.median(u)
    first = -(ncols // 2)
    keep_pts, keep_rows, keep_cols = [], [], []
    for r in range(nv):
        c = local[r] - local[r][np.argmin(np.abs(u[idx[r]] - (mid + shift[r])))] - first
        sel = (c >= 0) & (c < ncols)
        if not np.array_equal(np.sort(c[sel]), np.arange(ncols)):
            raise RuntimeError(f"row {r} is missing columns {sorted(set(range(ncols)) - set(c[sel]))} "
                               f"of the central {ncols}")
        keep_pts.append(pts[idx[r][sel]])
        keep_rows.append(np.full(sel.sum(), r))
        keep_cols.append(c[sel])
    return np.concatenate(keep_pts), np.concatenate(keep_rows), np.concatenate(keep_cols)


def pair_shifted(p0, p1, nv, nh):
    """Index blocks of the traps of two frames of one nv x nh array that sit on each other.

    p0, p1 (nv*nh, 2) are (y, x) centres in design order (row-major), e.g. the array at two
    CH3 tones a whole number of sites apart. Each trap of p0 takes its nearest neighbour in
    p1, and the most common row/column offset (dr, dc) is kept, so a few wrong neighbours
    don't matter. Returns (dr, dc), the rows and columns of p0 that have a partner, and the
    rows and columns of those partners in p1, as slices of the (nv, nh) grid.
    """
    ok0, ok1 = np.isfinite(p0).all(axis=1), np.isfinite(p1).all(axis=1)       # traps whose fit failed don't vote
    _, idx = cKDTree(p1[ok1]).query(p0[ok0])
    r1, c1 = np.unravel_index(np.flatnonzero(ok1)[idx], (nv, nh))
    r0, c0 = np.indices((nv, nh)).reshape(2, -1)[:, ok0]
    dr = np.bincount(r1 - r0 + nv).argmax() - nv
    dc = np.bincount(c1 - c0 + nh).argmax() - nh
    rs, cs = slice(max(0, -dr), nv - max(0, dr)), slice(max(0, -dc), nh - max(0, dc))
    return (dr, dc), (rs, cs), (slice(rs.start + dr, rs.stop + dr), slice(cs.start + dc, cs.stop + dc))


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


def affine_residual(pos, rows, cols):
    """Offsets (N, 2) of centres pos (N, 2) in (y, x) from their best-fit affine lattice, and the 2x3 map M.

    The lattice is (y, x) = M @ (col, row, 1): translation, rotation, scale and shear of the
    whole array are absorbed, so the residual is only its nonlinear distortion.
    Ported from slm_local/fit_traps.fit_affine.
    """
    A = np.column_stack([cols, rows, np.ones(len(cols))])
    M = np.linalg.lstsq(A, pos, rcond=None)[0].T
    return pos - A @ M.T, M


def row_grid(pos, rows, cols, nv, margin_sites=1):
    """Camera (y, x) coordinates (2, H, W) of an image grid along the array rows, the row angle (rad) and the pitch (px).

    The grid's x runs along the rows (angle from row_fit), 1 camera px per pixel, and it
    covers the centres pos (N, 2) of the traps at (rows, cols) plus margin_sites pitches on
    every side. Sample a frame on it with scipy.ndimage.map_coordinates.
    """
    steps = np.linalg.norm(np.diff(pos[np.lexsort((cols, rows))], axis=0), axis=1)
    pitch = np.nanmedian(steps)      # the few steps from one row's end to the next row's start don't move it
    phi = np.radians(row_fit(pos, rows, nv))
    u, v = rotate(pos, phi)
    m = margin_sites * pitch
    uu, vv = np.meshgrid(np.arange(np.nanmin(u) - m, np.nanmax(u) + m),
                         np.arange(np.nanmin(v) - m, np.nanmax(v) + m))
    # Inverse of rotate: camera (y, x) of each (u, v)
    return np.array([uu * np.sin(phi) + vv * np.cos(phi), uu * np.cos(phi) - vv * np.sin(phi)]), phi, pitch
