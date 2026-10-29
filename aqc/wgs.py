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

# Optics constants, matching upload_pattern / static_slm.
PIXEL_UM = 8.0
FOCAL_UM = 7900.0
WAVELENGTH_UM = 1.064
SLM_H, SLM_W = config.SLM_SHAPE


def um_per_ffpix(scal):
    """Focal-plane micrometres per padded-FFT pixel, (y, x)."""
    return (WAVELENGTH_UM * FOCAL_UM / (scal * SLM_H * PIXEL_UM),
            WAVELENGTH_UM * FOCAL_UM / (scal * SLM_W * PIXEL_UM))


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


def run_wgs(n_horiz, n_vert, pitch_um, weights=None, iterations=30,
            scal=4, use_gpu=True, seed=0, verbose=True,
            single=True, phase0=None, momentum=0.6, momentum_start=10,
            fix_phase_at=0.5, angle_deg=0.0, offset_px=None):
    """Weighted GS. `weights` sets the RELATIVE target intensity per trap.

    weights is shaped (n_vert, n_horiz); None means uniform. Larger weight ->
    brighter trap requested. Returns (phase[1200,1920], info dict).

    single=True runs in float32/complex64. The RTX A1000 executes fp64 at 1/32
    rate, so this is far faster and halves memory; WGS does not need fp64.
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
    idx = (xp.asarray(rr_k), xp.asarray(cc_k))

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

    # fftshift/ifftshift are pure index rotations. Rotating the arrays once up
    # front and the trap indices with them lets the loop use plain fft2/ifft2,
    # avoiding four full-grid (36.9 Mpx) copies per iteration.
    idx = (xp.asarray((rr_k + H // 2) % H),
           xp.asarray((cc_k + W // 2) % W))
    A_in = xp.roll(A_in, (-(H // 2), -(W // 2)), axis=(0, 1))
    aperture = xp.roll(aperture, (-(H // 2), -(W // 2)), axis=(0, 1))
    psi = xp.roll(psi, (-(H // 2), -(W // 2)), axis=(0, 1))

    # Plain WGS converges slowly on a large array filling most of the field
    # (~200 iterations to reach 1e-3). Polyak momentum on the log-gain, as in
    # the lab generator's "hybrid" scheme, gets there far sooner. It is only
    # switched on after a few plain iterations, because accelerating the first
    # wild steps destabilises the solution.
    g = xp.ones(idx[0].size, dtype=rdtype)   # WGS gain per trap
    log_g_prev = xp.zeros(idx[0].size, dtype=rdtype)
    hist = []
    # Freeze the far-field phase at the trap sites partway through. Letting the
    # trap phases float for the whole run lets the solution wander -- the
    # speckle realisation keeps changing and the last iterations fight each
    # other. Fixing them and solving only for amplitude afterwards converges
    # much further, and makes the result stable between feedback rounds.
    fix_at = int(fix_phase_at * iterations) if fix_phase_at else 0
    fixed_phasor = None
    for it in range(1, iterations + 1):
        far = fft2((A_in * xp.exp(1j * psi)).astype(cdtype))
        amp = xp.abs(far[idx])
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
            fixed_phasor = (far[idx] / amp).astype(cdtype)

        newfar = xp.zeros_like(far)
        phasor = fixed_phasor if fixed_phasor is not None else (far[idx] / amp)
        spots = (g * target_amp).astype(cdtype) * phasor
        newfar[idx] = spots
        back = ifft2(newfar)
        psi = xp.angle(back) * aperture

    psi = xp.roll(psi, (H // 2, W // 2), axis=(0, 1))
    phase = psi[y0:y0 + SLM_H, x0:x0 + SLM_W]
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
                               "pitch_um": pitch_um}
