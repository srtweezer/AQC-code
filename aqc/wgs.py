"""Weighted Gerchberg-Saxton with support for external (camera) weights.

Copied from slm_local/wgs_local.py, with uniform illumination over the SLM
aperture instead of a Gaussian beam.

Why this exists: static_slm.calculate_phase_mask_first_run has no parameter for
injecting measured trap weights, and its update_tweezer_pattern feedback path is
broken (`if position_x_new != None` on an ndarray raises; np.zeros_like is
called with a scalar pair). Rather than modify anything on Y:, this reproduces
the same geometry locally and exposes the weights.

Geometry is defined directly in padded-FFT pixels so trap positions are known
exactly, instead of going through static_slm's spacing_factor heuristic.
"""
import numpy as np

from aqc import config

try:
    import cupy as cp
    from cupyx.scipy.fft import fft2 as _gfft2, ifft2 as _gifft2
    from cupyx.scipy.fft import fftshift as _gfftshift, ifftshift as _gifftshift
    HAVE_GPU = True
except Exception:                                    # pragma: no cover
    HAVE_GPU = False

from scipy.fft import fft2 as _cfft2, ifft2 as _cifft2
from scipy.fft import fftshift as _cfftshift, ifftshift as _cifftshift

# Optics constants. FOCAL_UM is the effective focal length from the SLM to the tweezer
# plane the camera sees, f1 f3 / f2 = 200 mm x 40 mm / 200 mm (CLAUDE.md), so focal-plane
# um are um on the camera sensor. Before 2026-11-24 it was 7900 um, the value of the old
# experiment's static_slm, and um values from then are 7900 / 40000 of today's.
PIXEL_UM = 8.0
FOCAL_UM = 200_000.0 * 40_000.0 / 200_000.0
WAVELENGTH_UM = 1.064
# Pitch of the tweezer arrays in that plane, 27.4 camera px (27.9 measured). The arrays were
# designed as "10 um" with the old 7900 um focal length; this is the same pitch in today's
# units, so every grid position is unchanged.
PITCH_UM = 10.0 * FOCAL_UM / 7900.0
SLM_H, SLM_W = config.SLM_SHAPE


def um_per_ffpix(scal):
    """Focal-plane micrometres per padded-FFT pixel, (y, x)."""
    return (WAVELENGTH_UM * FOCAL_UM / (scal * SLM_H * PIXEL_UM),
            WAVELENGTH_UM * FOCAL_UM / (scal * SLM_W * PIXEL_UM))


def grid_per_site(pitch_um, scal, angle_deg):
    """2x2 padded-grid px (y, x) per (column, row) step of the design lattice, as trap_indices builds it."""
    uy, ux = um_per_ffpix(scal)
    th = np.radians(angle_deg)
    return pitch_um * np.array([[np.sin(th) / uy, np.cos(th) / uy],
                                [np.cos(th) / ux, -np.sin(th) / ux]])


def trap_indices(n_horiz, n_vert, pitch_um, scal, offset_um=(0.0, 0.0),
                 angle_deg=0.0, offset_px=None):
    """Padded-grid (row, col) integer positions of an n_vert x n_horiz array.

    pitch_um is the true focal-plane pitch we want, converted to pixels per
    axis, so the array is square in physical units even though the padded grid
    has different um/px in x and y.
    """
    uy, ux = um_per_ffpix(scal)
    H, W = scal * SLM_H, scal * SLM_W
    cy, cx = H // 2 + offset_um[0] / uy, W // 2 + offset_um[1] / ux

    # Build the lattice in MICRONS, rotate there, and only then convert to
    # grid pixels. The padded grid has different um/px along x and y, so
    # rotating in pixel space would shear the array instead of turning it.
    off_y = (np.arange(n_vert) - (n_vert - 1) / 2) * pitch_um
    off_x = (np.arange(n_horiz) - (n_horiz - 1) / 2) * pitch_um
    OX, OY = np.meshgrid(off_x, off_y)                 # (n_vert, n_horiz)

    th = np.deg2rad(angle_deg)
    ct, st = np.cos(th), np.sin(th)
    RX = ct * OX - st * OY
    RY = st * OX + ct * OY

    ry = cy + RY / uy
    rx = cx + RX / ux
    if offset_px is not None:
        # Per-trap nudge of the TARGET position, in padded-grid pixels. This is
        # how position feedback steers each trap independently; it is applied
        # before rounding so sub-pixel corrections still accumulate.
        o = np.asarray(offset_px, float)
        if o.shape != (n_vert, n_horiz, 2):
            raise ValueError(f"offset_px must be {(n_vert, n_horiz, 2)}, got {o.shape}")
        ry = ry + o[..., 0]
        rx = rx + o[..., 1]
    rr = np.round(ry).astype(int)
    cc = np.round(rx).astype(int)
    if rr.min() < 0 or cc.min() < 0 or rr.max() >= H or cc.max() >= W:
        raise ValueError(
            f"array does not fit the padded grid: rows {rr.min()}..{rr.max()} "
            f"cols {cc.min()}..{cc.max()} vs grid {H}x{W}. "
            f"Reduce pitch/count or raise scal.")
    return rr, cc


def grid_phase(rr, cc, scal, xp=np):
    """Phase (SLM pixels, N), float32, of the plane waves that put traps at padded-grid sites (rr, cc).

    It is exactly the phase that ifft2 of a delta at (rr, cc) has on the SLM in run_wgs.
    The integer products are wrapped mod H (W) before scaling, so float32 keeps it exact
    to rounding however far the site is from the centre.
    """
    H, W = scal * SLM_H, scal * SLM_W
    j = xp.arange(SLM_H) - SLM_H // 2
    i = xp.arange(SLM_W) - SLM_W // 2
    py = (j[:, None] * (xp.asarray(rr) - H // 2)[None]) % H
    px = (i[:, None] * (xp.asarray(cc) - W // 2)[None]) % W
    ph = (py / H).astype(np.float32)[:, None, :] + (px / W).astype(np.float32)[None, :, :]
    return (2 * np.pi * ph).reshape(SLM_H * SLM_W, -1)


def corrected_waves(rr, cc, scal, correction, lens_rad=None, xp=None):
    """(SLM pixels, traps) complex64 matrix of the corrected waves of traps at padded-grid sites (rr, cc) (N,).

    Trap m's wave is exp(i (k_m . r + sum_k c_mk Z_k + a_m rho^2)): its plane wave, its
    aberration correction (correction (N, len(aqc.aberration.MODES)), rad) and, with
    lens_rad (N,), its own lens a_m rho^2 (aqc.lens.rho2) of Chew et al., arXiv:2407.20699.
    8 bytes per entry, 16.6 GB for 900 traps; built in chunks of traps so the peak memory
    stays near that. On the GPU unless xp is given.
    """
    xp = xp or (cp if HAVE_GPU else np)
    from aqc import aberration        # imported here: it uses this module's optics constants
    from aqc.lens import rho2
    rr, cc = np.ravel(rr), np.ravel(cc)
    n_modes = len(aberration.MODES)
    zb = xp.asarray(aberration.basis().reshape(n_modes, -1))
    coef = xp.asarray(np.asarray(correction, np.float32).reshape(-1, n_modes))
    lens = xp.asarray(np.zeros(rr.size, np.float32) if lens_rad is None else np.ravel(lens_rad).astype(np.float32))
    r2 = xp.asarray(rho2().astype(np.float32).ravel())
    M = xp.empty((SLM_H * SLM_W, rr.size), dtype=np.complex64)
    for s in range(0, rr.size, 64):
        ph = (grid_phase(rr[s:s + 64], cc[s:s + 64], scal, xp) + zb.T @ coef[s:s + 64].T
              + r2[:, None] * lens[None, s:s + 64])
        M[:, s:s + 64] = xp.exp(1j * ph)
    return M


def trap_fields(phase, M):
    """Field (N,) at each trap of SLM phase (any shape, SLM_H * SLM_W pixels): its overlap with the trap's wave in M."""
    xp = cp.get_array_module(M) if HAVE_GPU else np
    return (xp.exp(-1j * xp.asarray(phase).ravel()).astype(M.dtype) @ M).conj()


def refine_amplitudes(spots, M, target, iterations=8):
    """SLM phase arg(M @ S) and the spot field S, with the trap phases of spots frozen and the amplitudes refined.

    Each round scales |S_m| by sqrt(mean r / r_m), r = simulated trap intensity / target, so
    every trap keeps the intensity target (N,) it had before its wave changed. Chew et al.,
    arXiv:2407.20699, Sec. V; ported from slm_local/zflat_09_23/build_zflat.py. Returns
    phase (SLM_H, SLM_W), S (N,) and std/mean of r for each round.
    """
    xp = cp.get_array_module(M) if HAVE_GPU else np
    S = xp.asarray(np.ravel(spots).astype(np.complex64))
    target = xp.asarray(np.ravel(target))
    hist = []
    for it in range(iterations + 1):
        psi = xp.mod(xp.angle(M @ S), 2 * np.pi)
        r = xp.abs(trap_fields(psi, M)) ** 2 / target
        hist.append(float(r.std() / r.mean()))
        if it == iterations:
            break
        S = S * xp.sqrt(r.mean() / r)
    to_np = cp.asnumpy if xp is not np else np.asarray
    return to_np(psi).reshape(SLM_H, SLM_W), to_np(S), hist


def run_wgs(n_horiz, n_vert, pitch_um, weights=None, iterations=30,
            scal=4, use_gpu=True, seed=0, verbose=True,
            single=True, phase0=None, momentum=0.6, momentum_start=10,
            fix_phase_at=0.5, angle_deg=0.0, offset_px=None, correction=None):
    """Weighted GS. `weights` sets the RELATIVE target intensity per trap.

    weights is shaped (n_vert, n_horiz); None means uniform. Larger weight ->
    brighter trap requested. Returns (phase[1200,1920], info dict).

    correction, shaped (n_vert, n_horiz, len(aqc.aberration.MODES)), is each trap's
    Zernike correction in rad (aqc.aberration.correction). With it the solve is the
    corrected WGS of Machu et al., arXiv:2512.15967: trap positions are unchanged.

    single=True runs in float32/complex64: far faster on GPUs with a reduced fp64
    rate, and half the memory; WGS does not need fp64.
    """
    gpu = use_gpu and HAVE_GPU
    rdtype = np.float32 if single else np.float64
    cdtype = np.complex64 if single else np.complex128
    if gpu:
        xp, fft2, ifft2 = cp, _gfft2, _gifft2
        fftshift, ifftshift = _gfftshift, _gifftshift
    else:
        xp, fft2, ifft2 = np, _cfft2, _cifft2
        fftshift, ifftshift = _cfftshift, _cifftshift

    H, W = scal * SLM_H, scal * SLM_W
    rr, cc = trap_indices(n_horiz, n_vert, pitch_um, scal,
                          angle_deg=angle_deg, offset_px=offset_px)

    if weights is None:
        weights = np.ones((n_vert, n_horiz))
    w_all = np.asarray(weights, float).ravel()

    # A trap with zero weight must be DROPPED from the target set, not
    # requested at zero: target_amp would be 0 and the WGS update divides by
    # it, turning the whole phase mask into NaN.
    keep = w_all > 0
    if not keep.any():
        raise ValueError("all trap weights are zero")
    rr_k, cc_k = rr.ravel()[keep], cc.ravel()[keep]
    w = xp.asarray(w_all[keep] / w_all[keep].mean())
    target_amp = xp.sqrt(w)          # amplitude ~ sqrt(intensity)

    # Uniform illumination on the SLM aperture, zero-padded into the grid.
    y0, x0 = (scal - 1) * SLM_H // 2, (scal - 1) * SLM_W // 2

    A_in = xp.zeros((H, W), dtype=rdtype)
    A_in[y0:y0 + SLM_H, x0:x0 + SLM_W] = 1
    aperture = xp.zeros((H, W), dtype=bool)
    aperture[y0:y0 + SLM_H, x0:x0 + SLM_W] = True

    # Warm start: reusing the previous round's phase keeps the speckle
    # realisation stable between feedback rounds, so the measured change
    # reflects the weight update rather than a fresh random solution.
    if phase0 is not None:
        psi = xp.zeros((H, W), dtype=rdtype)
        psi[y0:y0 + SLM_H, x0:x0 + SLM_W] = xp.asarray(
            np.asarray(phase0, dtype=rdtype))
    else:
        rng = np.random.default_rng(seed)
        psi = xp.asarray(rng.uniform(0, 2 * np.pi, (H, W)).astype(rdtype))
    psi = psi * aperture

    if correction is None:
        # fftshift/ifftshift are pure index rotations. Rotating the arrays once up
        # front and the trap indices with them lets the loop use plain fft2/ifft2,
        # avoiding four full-grid (36.9 Mpx) copies per iteration.
        idx = (xp.asarray((rr_k + H // 2) % H),
               xp.asarray((cc_k + W // 2) % W))
        A_in = xp.roll(A_in, (-(H // 2), -(W // 2)), axis=(0, 1))
        aperture = xp.roll(aperture, (-(H // 2), -(W // 2)), axis=(0, 1))
        psi = xp.roll(psi, (-(H // 2), -(W // 2)), axis=(0, 1))

        def forward(psi):
            return fft2((A_in * xp.exp(1j * psi)).astype(cdtype))[idx]

        def backward(spots):
            far = xp.zeros((H, W), dtype=cdtype)
            far[idx] = spots
            return xp.angle(ifft2(far)) * aperture
    else:
        # Each trap's plane wave carries its own correction phase (Machu et al.,
        # Eq. 4), and the field at a trap is the overlap with that corrected wave
        # (Eq. 3). No FFT applies a different phase per trap, so both passes become
        # products with the (SLM pixels x traps) matrix of corrected waves.
        n_modes = np.asarray(correction).shape[-1]
        M = corrected_waves(rr_k, cc_k, scal, np.asarray(correction).reshape(-1, n_modes)[keep], xp=xp)
        psi = psi[y0:y0 + SLM_H, x0:x0 + SLM_W].ravel()

        def forward(psi):
            return trap_fields(psi, M)

        def backward(spots):
            return xp.angle(M @ spots)

    # Plain WGS converges slowly on a large array filling most of the field
    # (~200 iterations to reach 1e-3). Polyak momentum on the log-gain, as in
    # the lab generator's "hybrid" scheme, gets there far sooner. It is only
    # switched on after a few plain iterations, because accelerating the first
    # wild steps destabilises the solution.
    g = xp.ones(rr_k.size, dtype=rdtype)     # WGS gain per trap
    log_g_prev = xp.zeros(rr_k.size, dtype=rdtype)
    hist = []
    # Freeze the far-field phase at the trap sites partway through. Letting the
    # trap phases float for the whole run lets the solution wander -- the
    # speckle realisation keeps changing and the last iterations fight each
    # other. Fixing them and solving only for amplitude afterwards converges
    # much further, and makes the result stable between feedback rounds.
    fix_at = int(fix_phase_at * iterations) if fix_phase_at else 0
    fixed_phasor = None
    for it in range(1, iterations + 1):
        V = forward(psi)                      # field at the traps
        amp = xp.abs(V)
        amp = xp.where(amp == 0, 1e-12, amp)

        # weighted GS update: push each trap toward its share of the mean
        rel = amp / target_amp
        log_g = xp.log(g) + xp.log(rel.mean() / rel)
        if momentum and it > momentum_start:
            log_g = log_g + momentum * (xp.log(g) - log_g_prev)
        log_g_prev = xp.log(g)
        # Clamp the per-trap gain. Very small target weights make rel huge and
        # exp(log_g) overflow in float32, which turns the whole mask into NaN.
        g = xp.exp(xp.clip(log_g, -20.0, 20.0))
        err = float(xp.std(rel / rel.mean()))
        hist.append(err)
        if verbose:
            print(f"  iter {it:3d}  non-uniformity = {err:.5f}", flush=True)

        if fix_at and it == fix_at:
            fixed_phasor = (V / amp).astype(cdtype)

        phasor = fixed_phasor if fixed_phasor is not None else (V / amp)
        spots = (g * target_amp).astype(cdtype) * phasor
        psi = backward(spots)

    if correction is None:
        psi = xp.roll(psi, (H // 2, W // 2), axis=(0, 1))
        phase = psi[y0:y0 + SLM_H, x0:x0 + SLM_W]
    else:
        phase = psi.reshape(SLM_H, SLM_W)
    phase = xp.mod(phase, 2 * xp.pi)
    if gpu:
        phase = cp.asnumpy(phase)
        g = cp.asnumpy(g)
        spots = cp.asnumpy(spots)
    # The spot field the final phase is arg(ifft) of, per trap in (n_vert,
    # n_horiz) order (0 for dropped traps). Lets a caller rebuild or extend
    # the hologram with the trap phases frozen, e.g. per-trap lenses.
    spot_full = np.zeros(w_all.size, dtype=np.complex128)
    spot_full[keep] = np.asarray(spots)
    return np.asarray(phase), {"history": hist, "gain": g,
                               "spots": spot_full.reshape(n_vert, n_horiz),
                               "rows": rr, "cols": cc, "scal": scal,
                               "pitch_um": pitch_um, "correction": correction}
