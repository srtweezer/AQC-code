"""Pair mismatch of the tweezer array at two CH3 tones one site apart, and the SLM moves that cancel it.

See docs/handoff_calibration.md. Tone A is CH3 at 70 MHz, tone B one site lower. Every
SLM trap appears in both copies of the array, so moving one trap changes two pairs, and
the moves are chained along each row.
"""

import numpy as np

from aqc.traps import pair_shifted


def mismatch(pa, pb, nv, nh):
    """Pair mismatch E (nv, nh, 2) (y, x) and the column step dc of the pairing.

    pa, pb (nv * nh, 2): trap centres at tones A and B in design order. E[r, c] is the
    centre at tone B of the trap that lands on A trap (r, c), i.e. B trap (r, c + dc),
    minus A trap (r, c); NaN where A trap (r, c) has no partner.
    """
    (dr, dc), (rs, cs), (rs1, cs1) = pair_shifted(pa, pb, nv, nh)
    if dr != 0 or abs(dc) != 1:
        raise RuntimeError(f"tone B is not the array one site along the rows: pairing (dr, dc) = ({dr}, {dc})")
    a, b = pa.reshape(nv, nh, 2), pb.reshape(nv, nh, 2)
    e = np.full((nv, nh, 2), np.nan)
    e[rs, cs] = b[rs1, cs1] - a[rs, cs]
    return e, dc


def step_angle_basis(pa, pb, nv, nh, dc):
    """How the pair mismatch E (from mismatch) changes with the tone step and the lattice angle: S, R (nv, nh, 2).

    Lengthening the tone step by a fraction a moves every B trap further along its own
    displacement from A, so E[r, c] gains a S[r, c], S = B trap (r, c + dc) - A trap
    (r, c + dc). Rotating the SLM lattice by theta (rad, atan2(y, x) on the camera) turns
    L, the vector from A trap (r, c) to A trap (r, c + dc), so E gains theta R, R = L
    turned by +90 deg. NaN where E is.
    """
    a, b = pa.reshape(nv, nh, 2), pb.reshape(nv, nh, 2)
    cs = slice(max(0, -dc), nh - max(0, dc))
    cs1 = slice(cs.start + dc, cs.stop + dc)
    s, r = np.full((nv, nh, 2), np.nan), np.full((nv, nh, 2), np.nan)
    s[:, cs] = b[:, cs1] - a[:, cs1]
    lv = a[:, cs1] - a[:, cs]
    r[:, cs] = np.stack([lv[..., 1], -lv[..., 0]], axis=-1)       # (dy, dx) = theta (x, -y)
    return s, r


def fit_step_angle(e, s, r):
    """Fractional step change a and lattice rotation theta (rad) that minimise sum |e + a s + theta r|^2, and that residual.

    e, s, r (..., 2) from mismatch and step_angle_basis, of one shape; stack several tones
    along a leading axis to fit one (a, theta) to all of them.
    """
    ok = np.isfinite(e).all(axis=-1) & np.isfinite(s).all(axis=-1) & np.isfinite(r).all(axis=-1)
    A = np.stack([s[ok].ravel(), r[ok].ravel()], axis=1)
    (a, theta), *_ = np.linalg.lstsq(A, -e[ok].ravel(), rcond=None)
    return a, theta, e + a * s + theta * r


def chain(e, dc):
    """Moves u (nv, nh, 2) of the SLM traps that cancel every pair mismatch e (from mismatch).

    A move u shifts a trap by u in both copies, so pair (r, c) becomes
    e[r, c] + u[r, c + dc] - u[r, c], and u[r, c + dc] = u[r, c] - e[r, c] zeroes it. The
    chain starts at the row end without a partner. A shift of a whole row changes no pair,
    so each row's moves are taken with zero mean, which keeps the array in place.
    """
    nv, nh, _ = e.shape
    cols = range(nh - 1) if dc == 1 else range(nh - 1, 0, -1)
    if np.isnan(e[:, list(cols)]).any():
        raise ValueError("a pair is missing: the chain would carry its error along the rest of the row")
    u = np.zeros_like(e)
    for c in cols:
        u[:, c + dc] = u[:, c] - e[:, c]
    return u - u.mean(axis=1, keepdims=True)


def error(e):
    """Total handoff error eps = sqrt(<|E|^2>) of a pair mismatch e (from mismatch), its parts, and the mean.

    eps^2 = eps_slm^2 + |<E>|^2 exactly: eps_slm, the rms of E - <E>, is the trap-to-trap
    scatter that only the SLM pattern can fix; <E> (y, x) is common to every pair, the
    error of the site step and of the row angle, which the AOD tones fix.
    """
    mean = np.nanmean(e, axis=(0, 1))
    eps_slm = np.sqrt(np.nanmean(np.sum((e - mean) ** 2, axis=-1)))
    return np.hypot(eps_slm, np.linalg.norm(mean)), eps_slm, mean
