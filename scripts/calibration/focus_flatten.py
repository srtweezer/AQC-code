"""Flatten the focal plane of the tweezer array with a per-trap lens, by z-scan feedback.

Ported from slm_local/zscan_09_15 (zscan.py, zfit.py) and slm_local/zflat_09_23
(build_zflat.py, build_round.py, plot_rounds.py), which took uniform90_best from 0.41 um
to 0.08 um focus std in two rounds on 2026-09-23 (with f = 7.9 mm; 10.5 and 2.1 um in
today's aqc.focus units). Method: Chew et al., "Ultra-precise
holographic optical tweezers array", arXiv:2407.20699, Sec. V.

Every round scans a global lens a rho^2 over the displayed pattern, z from --zmin to
--zmax (aqc.focus convention, 23.52 um/rad), visiting the z values in shuffled order so
drift shows up as scatter rather than as a plausible trend. At each z every trap's
Gaussian width is fitted on the camera, at the positions found at z = 0, and each trap's
focus z0 is fitted to its sigma^2(z) (aqc.focus.fit_z0). The next pattern gives trap m
its own lens (z0_m - mean z0) / UM_PER_RAD on top of the lenses it already has, times
--gain.

The base pattern is a corrected WGS solve (homogenize.py --correct), and is re-solved
here from its weights (<pattern>_weights.npy), --offsets and --theta; the script stops if
that does not reproduce the file. Its spot field S_m = f_m exp(i theta_m) is kept: the
trap phases theta_m stay frozen, the lenses go on each trap's corrected wave
(aqc.wgs.corrected_waves), and only the amplitudes f_m are refined so every trap keeps
the simulated intensity it had in the base pattern, whose weights were homogenized
against them (aqc.wgs.refine_amplitudes). Re-running the full solve with the lenses
would change the speckle and lose the homogenization.

Camera trap (row, col) is design trap (row, col), as checked for homogenize.py on
2026-10-29, so a measured z0 goes to the same design trap.

With --vertical the base is the 90 rows x 10 columns array (homogenize.py --vertical
--correct). The camera finds it as 10 lines of 90 traps, and each measured z0 is put back
in design order (aqc.traps.vertical_design_order) before it becomes a lens. --offsets is
not available with --vertical.

--blaze-y / --blaze-x set the blaze the pattern is displayed with (default 8 px along y);
the aberration correction is evaluated at the field positions that blaze puts the traps at.

Setup: the SLM beam imaged onto the Basler (free-run), AODs on CH1/CH3 at 70 MHz and
1.6 V peak. Nothing else may hold the AWG (stop aod_tone.py first). Pick an exposure that
keeps the brightest trap below saturation in focus.

Saves the best round's phase (radians, no blaze; display it with slm_upload.py and the
same blaze) to aqc/calibration/phase-patterns/<timestamp>_<base name>_zflat.npy (vertical/
with --vertical), its
hologram state (each trap's grid position, lens, aberration correction, spot field and
target intensity; the input of handoff_calibration.py) to <timestamp>_<base name>_zflat_state.npz,
and data/<MM_DD>/<timestamp>_focus_flatten.png. Leaves the best pattern on the SLM and the
AWG stopped.

Usage:
    python scripts/calibration/focus_flatten.py \\
        aqc/calibration/phase-patterns/20261030_110541_wgs_cwgs_pos_homogenized.npy \\
        --offsets aqc/calibration/20261030_110252_position_offsets.npz --exposure-us 150
    python scripts/calibration/focus_flatten.py <pattern>.npy --exposure-us 150 --rounds 3 --gain 0.8
    python scripts/calibration/focus_flatten.py \\
        aqc/calibration/phase-patterns/vertical/<timestamp>_wgs_90x10_cwgs_homogenized.npy \\
        --exposure-us 150 --vertical --theta -1.1877 --blaze-x -8 --blaze-y 0
"""

import argparse
from datetime import datetime
from pathlib import Path

import cupy as cp
import e2style as e2
import matplotlib.pyplot as plt
import numpy as np

from aqc import aberration, config
from aqc.blaze import blaze
from aqc.data import save_plot
from aqc.focus import UM_PER_RAD, fit_z0, zscan
from aqc.hardware.awg import aods
from aqc.hardware.camera import Camera
from aqc.hardware.slm import send_phase
from aqc.steering import F_CENTER, play
from aqc.traps import vertical_design_order
from aqc.wgs import PITCH_UM, corrected_waves, refine_amplitudes, run_wgs, trap_fields, trap_indices

PHASE_DIR = Path(__file__).resolve().parents[2] / "aqc" / "calibration" / "phase-patterns"
# On the camera the array is NV lines of NH traps along its long axis: rows by default,
# columns with --vertical.
NV, NH = 10, 90
SCAL = 4
AMPLITUDE = 1.6          # V peak, the CW drive of aod_tone.py
REFINE = 8               # amplitude-refinement rounds per pattern, as build_zflat.py

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("pattern", type=Path, help="base pattern (.npy from homogenize.py --correct, with its _weights.npy)")
parser.add_argument("--exposure-us", type=float, required=True, help="camera exposure, us")
parser.add_argument("--offsets", type=Path, help="per-trap target offsets the base was solved with (.npz)")
parser.add_argument("--theta", type=float, default=3.1910, help="WGS design angle of the base, deg")
parser.add_argument("--rounds", type=int, default=2, help="lens rounds after the base scan")
parser.add_argument("--gain", type=float, default=1.0, help="fraction of the measured focus error corrected per round")
parser.add_argument("--zmin", type=float, default=-64.0, help="scan start, um")
parser.add_argument("--zmax", type=float, default=64.0, help="scan end, um")
parser.add_argument("--zstep", type=float, default=2.5, help="scan step, um")
parser.add_argument("--average", type=int, default=16, help="frames averaged per z")
parser.add_argument("--vertical", action="store_true", help="90 rows x 10 columns array")
parser.add_argument("--blaze-y", type=int, default=8, help="blaze period along y, px (0 = none)")
parser.add_argument("--blaze-x", type=int, default=0, help="blaze period along x, px (0 = none)")
args = parser.parse_args()
if args.vertical and args.offsets:
    raise ValueError("--offsets was calibrated on the horizontal array; not with --vertical")

N_HORIZ, N_VERT = (NV, NH) if args.vertical else (NH, NV)
# Camera index of every design trap
ORDER = vertical_design_order(NV, NH) if args.vertical else np.arange(NV * NH)

offsets = None
if args.offsets is not None:
    cal = np.load(args.offsets)
    if not np.isclose(cal["theta"], args.theta):
        raise ValueError(f"{args.offsets.name} was measured at theta {float(cal['theta'])}, not {args.theta}")
    offsets = cal["offset_px"]
rr, cc = trap_indices(N_HORIZ, N_VERT, PITCH_UM, SCAL, angle_deg=args.theta)
correction = aberration.correction(aberration.load(config.ABERRATION_CAL), rr, cc, SCAL, args.blaze_y, args.blaze_x)
base, info = run_wgs(N_HORIZ, N_VERT, PITCH_UM, weights=np.load(args.pattern.with_name(f"{args.pattern.stem}_weights.npy")),
                     iterations=120, scal=SCAL, momentum=0.6, fix_phase_at=0.5, angle_deg=args.theta, seed=0,
                     verbose=False, correction=correction, offset_px=offsets)
diff = np.abs(np.angle(np.exp(1j * (base - np.load(args.pattern)))))
if diff.max() > 1e-3:
    raise RuntimeError(f"re-solve differs from {args.pattern.name} by up to {diff.max():.3g} rad; "
                       f"check --offsets, --theta and config.ABERRATION_CAL")
S0, rows, cols = info["spots"].ravel(), info["rows"], info["cols"]
I_base = np.abs(trap_fields(base, corrected_waves(rows, cols, SCAL, correction))) ** 2

zs = np.arange(args.zmin, args.zmax + 1e-9, args.zstep)


def scan(cam, phase):
    """Gaussian width sigma (len(zs), NV * NH), px, of every design trap at every z; NaN where a fit failed."""
    _, fits = zscan(cam, phase + blaze(args.blaze_y, args.blaze_x), zs, args.average, NV, NH)
    return np.sqrt(fits[..., 3] * fits[..., 4])[:, ORDER]


def report(label, z0, err, flag):
    ok = flag == 0
    zc = z0[ok] - z0[ok].mean()
    print(f"{label}: fitted {ok.sum()}/{NV * NH} ({(flag == 1).sum()} unbracketed: widen the scan; "
          f"{(flag == 2).sum()} too few points; {(flag == 3).sum()} fit failed), z0 mean {z0[ok].mean():+.3f} um, "
          f"std {zc.std():.3f} um, p2p {np.ptp(zc):.2f} um, fit error {np.nanmean(err[ok]):.4f} um", flush=True)


dz, S, phase = np.zeros(NV * NH), S0, base
log = []
with aods(amplitudes=(AMPLITUDE, AMPLITUDE)) as awg, Camera(args.exposure_us, hardware_trigger=False) as cam:
    play(awg, F_CENTER)
    for rnd in range(args.rounds + 1):
        label = "base" if rnd == 0 else f"round {rnd}"
        if rnd > 0:
            bad = log[-1]["flag"] != 0
            if bad.any():
                raise RuntimeError(f"{bad.sum()} traps unfitted in {log[-1]['label']}; refusing to guess their lens")
            z0 = log[-1]["z0"]
            dz = dz + args.gain * (z0 - z0.mean())
            phase, S, hist = refine_amplitudes(S, corrected_waves(rows, cols, SCAL, correction, dz / UM_PER_RAD),
                                               I_base, REFINE)
            print(f"{label}: lens range {dz.min():+.3f}..{dz.max():+.3f} um; simulated intensity / base "
                  f"std/mean {hist[0]:.4f} -> {hist[-1]:.4f} after {REFINE} refinements", flush=True)
        z0, err, flag = fit_z0(zs, scan(cam, phase))
        report(label, z0, err, flag)
        log.append(dict(label=label, phase=phase, z0=z0, err=err, flag=flag, dz=dz, S=S))
    std = [np.nanstd(m["z0"][m["flag"] == 0]) for m in log]
    best = int(np.argmin(std))
    send_phase(log[best]["phase"] + blaze(args.blaze_y, args.blaze_x))

stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
if best > 0:
    path = (PHASE_DIR / "vertical" if args.vertical else PHASE_DIR) / f"{stamp}_{args.pattern.stem[16:]}_zflat.npy"
    np.save(path, log[best]["phase"].astype(np.float32))
    np.savez(path.with_name(f"{path.stem}_state.npz"), rows=rows.ravel().astype(float), cols=cols.ravel().astype(float),
             lens_rad=log[best]["dz"] / UM_PER_RAD, correction=np.reshape(correction, (NV * NH, -1)),
             spots=log[best]["S"], target=cp.asnumpy(I_base), scal=SCAL, blaze_y_px=args.blaze_y, blaze_x_px=args.blaze_x,
             source=f"focus_flatten.py {log[best]['label']} of {args.pattern.name}")
    print(f"Best: {log[best]['label']} at focus std {std[best]:.3f} um (base {std[0]:.3f} um). Saved {path} "
          f"and its _state.npz")
else:
    print(f"No round beat the base ({std[0]:.3f} um focus std); nothing saved")

# Figure laid out as slm_local/zflat_09_23/plot_rounds.py.
e2.apply()
Z = [np.where(m["flag"] == 0, m["z0"], np.nan).reshape(N_VERT, N_HORIZ) for m in log]
# Profiles run along the long axis: per column (mean over rows), or per row with --vertical
along, across = ("design row", "column") if args.vertical else ("design column", "row")
long_axis = 0 if args.vertical else 1
Z = [z - np.nanmean(z) for z in Z]
colors = list(e2.OKABE_ITO.values())[1:]
fit_err = np.nanmean([np.nanmean(m["err"][m["flag"] == 0]) for m in log])
lim = np.nanmax(np.abs(Z[0]))
fig = plt.figure(figsize=(180 * e2.MM, 130 * e2.MM), layout="constrained")
gs = fig.add_gridspec(3, len(log))
for i, (z, m) in enumerate(zip(Z, log)):
    ax = fig.add_subplot(gs[0, i])
    im = ax.imshow(z, aspect="auto", cmap="RdBu_r", origin="lower", vmin=-lim, vmax=lim, interpolation="nearest")
    ax.set_title(f"{m['label']}: std {np.nanstd(z):.3f} µm, p2p {np.nanmax(z) - np.nanmin(z):.2f} µm")
    ax.set_xlabel("design column")
    ax.set_ylabel("design row")
    fig.colorbar(im, ax=ax, label="z0 − mean (µm)")
ax = fig.add_subplot(gs[1, :])
for z, m, color in zip(Z, log, colors):
    ax.plot(np.nanmean(z, axis=1 - long_axis), lw=1.5, color=color, label=m["label"])
ax.axhline(0, color=e2.GREY.edge, lw=0.8)
ax.set_xlabel(along)
ax.set_ylabel(f"z0 − mean, mean over {across}s (µm)")
ax.legend()
ax.set_title(f"focus vs {along}")
ax = fig.add_subplot(gs[2, :-1])
bins = np.linspace(-lim, lim, 81)
for z, m, color in zip(Z, log, colors):
    ax.hist(z.ravel()[np.isfinite(z.ravel())], bins=bins, histtype="step", lw=1.5, color=color, label=m["label"])
ax.set_xlabel("z0 − mean (µm)")
ax.set_ylabel("traps")
ax.legend()
ax = fig.add_subplot(gs[2, -1])
ax.plot(range(len(log)), std, "o-", lw=1.5, color=colors[0])
ax.axhline(fit_err, color=e2.OKABE_ITO["vermilion"], ls="--", label="fit error per trap")
# A gain-1 round corrects the previous measurement's noise too, so its residual is the
# fit error of two independent measurements.
ax.axhline(np.sqrt(2) * fit_err, color=e2.OKABE_ITO["orange"], ls=":", label="gain-1 floor (est.)")
ax.set_xticks(range(len(log)), [m["label"] for m in log])
ax.set_ylim(0, None)
ax.set_ylabel("focus std (µm)")
ax.legend()
ax.set_title("convergence")
fig.suptitle(f"Per-trap focus flattening, {F_CENTER / 1e6:g} MHz, {datetime.now():%Y-%m-%d} "
             f"(local lenses, arXiv:2407.20699)\nbase {args.pattern.name}, gain {args.gain:g}, "
             f"blaze y {args.blaze_y:+d} x {args.blaze_x:+d} px, exposure {args.exposure_us:g} us, "
             f"{args.average} frames per z")
save_plot(fig, "focus_flatten", stamp)
