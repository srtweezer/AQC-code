"""Align the tweezer rows with the direction CH3 steers the array, closing a loop on the WGS design angle.

The 10x90 array (10 um pitch) is solved with aqc.wgs at design angle theta and displayed
with an 8 px y blaze. On the camera, the row direction is fitted on a frame with both
channels at 70 MHz. CH3 is then swept around 70 MHz (CH1 held) while every trap is
tracked, and each trap's displacement is fitted linearly in frequency. The misalignment
alpha is the median angle from the rows to the trap steering directions. theta is
stepped by alpha / GAIN until |alpha| < --tol twice in a row at the same theta.

Both angles are measured on the camera. The trap plane is imaged onto it by a near-linear
map (rotation, unequal x/y magnification), which keeps parallel lines parallel, so
alpha = 0 on the camera means alpha = 0 in the trap plane, even though the camera row
angle itself (~20.7 deg) has nothing to do with theta (~1.7 deg).

Before every swept point a 70 MHz reference frame is taken and the displacement is
measured against it, so slow drift of the whole array cancels.

Setup: the SLM beam imaged onto the Basler (free-run), AODs on CH1/CH3 at 1.6 V peak.
Nothing else may hold the AWG (stop aod_tone.py first). Pick an exposure that keeps the
brightest trap below saturation at 70 MHz.

Saves data/<timestamp>_align_rows.png and, on convergence,
aqc/calibration/phase-patterns/<timestamp>_wgs_aligned.npy (phase in radians, no blaze;
display it with slm_upload.py --blaze-y 8). Leaves the last pattern on the SLM and the
AWG stopped.

Usage:
    python scripts/calibration/align_rows_to_aod.py --exposure-us 200
    python scripts/calibration/align_rows_to_aod.py --exposure-us 200 --theta0 1.9
"""

import argparse
from datetime import datetime
from pathlib import Path

import e2style as e2
import matplotlib.pyplot as plt
import numpy as np

from aqc.blaze import blaze
from aqc.hardware.awg import aods
from aqc.hardware.camera import Camera
from aqc.hardware.slm import send_phase
from aqc.steering import F_CENTER, grab, play, slopes, sweep_ch3
from aqc.traps import find_traps, fit_all, row_fit
from aqc.wgs import run_wgs

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PHASE_DIR = Path(__file__).resolve().parents[2] / "aqc" / "calibration" / "phase-patterns"
FULL_SCALE = 4095        # Mono12
NV, NH, PITCH_UM = 10, 90, 10.0
BLAZE_Y_PX = 8
AMPLITUDE = 1.6          # V peak, the CW drive of aod_tone.py
# Camera row angle per degree of design angle, measured 2026-10-22 with
# andrii-test/calibration/align_rows_to_aod.py. Its sign says the imaging is not
# mirrored; its size only sets how fast the loop converges.
GAIN = 0.983

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--exposure-us", type=float, required=True, help="camera exposure, us")
parser.add_argument("--theta0", type=float, default=1.6899, help="starting WGS design angle, deg")
# One measurement scatters by ~0.03 deg (whole-array jitter between each swept frame and
# its reference), so a tighter tolerance only makes the loop chase noise.
parser.add_argument("--tol", type=float, default=0.04, help="|alpha| tolerance, deg")
parser.add_argument("--max-iter", type=int, default=6, help="max measurements")
parser.add_argument("--f-span", type=float, default=8.0, help="CH3 sweep half-span around 70 MHz, MHz")
parser.add_argument("--f-step", type=float, default=0.5, help="CH3 sweep step, MHz")
parser.add_argument("--average", type=int, default=8, help="frames averaged per grab")
args = parser.parse_args()


def measure(awg, cam, offs):
    """Row angle, per-trap steering and misalignment for the pattern on the SLM."""
    play(awg, F_CENTER)
    ref = grab(cam, args.average)
    peaks, rows, cols = find_traps(ref, NV, NH)
    p0 = fit_all(ref, peaks)
    if not np.all(np.isfinite(p0)):
        raise RuntimeError(f"Gaussian fit failed for {np.isnan(p0[:, 0]).sum()} of {NV * NH} traps "
                           f"on the 70 MHz frame (peak {ref.max():.0f} / {FULL_SCALE})")
    row_deg = row_fit(p0, rows, NV)
    d = sweep_ch3(awg, cam, p0, offs, args.average)
    S = slopes(offs, d)                    # (y, x) px/MHz
    ok = np.isfinite(S[:, 0])
    steer_deg = np.degrees(np.arctan2(S[:, 0], S[:, 1]))
    alpha = (steer_deg - row_deg + 90) % 180 - 90
    alpha_map = np.full((NV, NH), np.nan)
    alpha_map[rows, cols] = alpha
    return dict(ref=ref, peak=ref.max(), row_deg=row_deg, alpha_map=alpha_map,
                pos=p0, start=p0 + d[0], end=p0 + d[-1], rows=rows, cols=cols, ok=ok,
                alpha=float(np.nanmedian(alpha)), alpha_sem=float(1.2533 * np.nanstd(alpha) / np.sqrt(ok.sum())),
                slope=np.nanmedian(S, axis=0), n=int(ok.sum()))


offs = np.round(np.arange(-args.f_span, args.f_span + args.f_step / 2, args.f_step), 6)
log = []
theta, hits = args.theta0, 0
with aods(amplitudes=(AMPLITUDE, AMPLITUDE)) as awg, Camera(args.exposure_us, hardware_trigger=False) as cam:
    for _ in range(args.max_iter):
        phase, _ = run_wgs(NH, NV, PITCH_UM, iterations=120, scal=4, momentum=0.6, fix_phase_at=0.5,
                           angle_deg=theta, seed=0, verbose=False)
        send_phase(phase + blaze(BLAZE_Y_PX, 0))
        print(f"design angle {theta:+.4f} deg:")
        m = measure(awg, cam, offs)
        ref = m.pop("ref")            # keep only the last frame, for the plot
        m["theta"] = theta
        log.append(m)
        print(f"    peak {m['peak']:.0f} / {FULL_SCALE}; rows {m['row_deg']:.4f} deg on camera, "
              f"steering {np.degrees(np.arctan2(*m['slope'])):.4f} deg "
              f"({np.hypot(*m['slope']):.2f} px/MHz); alpha {m['alpha']:+.4f} +- {m['alpha_sem']:.4f} deg "
              f"over {m['n']} traps")
        if abs(m["alpha"]) < args.tol:
            hits += 1
            if hits == 2:
                break
        else:
            hits = 0
            theta += m["alpha"] / GAIN

converged = hits == 2
stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
DATA_DIR.mkdir(exist_ok=True)
if converged:
    PHASE_DIR.mkdir(exist_ok=True)
    path = PHASE_DIR / f"{stamp}_wgs_aligned.npy"
    np.save(path, phase.astype(np.float32))
    print(f"Aligned at design angle {theta:.4f} deg. Saved {path}")
else:
    print(f"Not converged within {args.max_iter} measurements (last alpha {log[-1]['alpha']:+.4f} deg); "
          f"phase not saved")

last = log[-1]
e2.apply()
fig = plt.figure(figsize=(180 * e2.MM, 200 * e2.MM), layout="constrained")
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.3, 1])
ax = fig.add_subplot(gs[0, 0])
ax.errorbar(range(len(log)), [m["alpha"] for m in log], [m["alpha_sem"] for m in log], fmt="o-", ms=3,
            color=e2.OKABE_ITO["blue"])
for i, m in enumerate(log):
    ax.annotate(f"{m['theta']:.4f}", (i, m["alpha"]), fontsize=6, textcoords="offset points", xytext=(3, 3))
ax.axhspan(-args.tol, args.tol, color=e2.GREY.face, alpha=0.5, lw=0)
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("measurement (labels: design angle, deg)")
ax.set_ylabel("alpha = steering - rows (deg)")
ax.set_title("misalignment per measurement")
ax = fig.add_subplot(gs[0, 1])
im = ax.imshow(last["alpha_map"], cmap="coolwarm", vmin=-0.5, vmax=0.5, aspect="auto", interpolation="nearest")
fig.colorbar(im, ax=ax, label="alpha (deg)")
ax.set_xlabel("trap along row")
ax.set_ylabel("row")
ax.set_title(f"last measurement, per trap: median {last['alpha']:+.4f} deg")
# Where each trap sits at both ends of the CH3 sweep, over the 70 MHz frame. Aligned
# traps slide along their row lines; a tilt of alpha moves the 78 MHz positions off
# them by tan(alpha) * travel (0.04 deg over 40 px is 0.03 px, so read alpha above).
f_lo, f_hi = F_CENTER / 1e6 + offs[0], F_CENTER / 1e6 + offs[-1]
phi = np.radians(last["row_deg"])
pos, start, end, rows, cols = last["pos"], last["start"], last["end"], last["rows"], last["cols"]


def draw(ax, sel, title, arrows):
    ax.imshow(ref, cmap="gray", vmin=0, vmax=np.percentile(ref[ref > 0], 99.9))
    u = pos[:, 1] * np.cos(phi) + pos[:, 0] * np.sin(phi)
    reach = np.nanmax(np.abs(np.r_[start, end] - np.r_[pos, pos])) + 10
    for r in np.unique(rows[sel]):
        c = pos[rows == r].mean(axis=0)
        uc = u[rows == r].mean()
        t = np.array([u[sel & (rows == r)].min() - uc - reach, u[sel & (rows == r)].max() - uc + reach])
        ax.plot(c[1] + t * np.cos(phi), c[0] + t * np.sin(phi), color=e2.GREY.face, lw=0.4)
    ms, mew = (4, 0.8) if arrows else (1.5, 0.4)
    ax.plot(start[sel, 1], start[sel, 0], "o", ms=ms, mew=mew, mfc="none", color=e2.OKABE_ITO["blue"],
            label=f"CH3 = {f_lo:g} MHz")
    ax.plot(end[sel, 1], end[sel, 0], "o", ms=ms, mew=mew, mfc="none", color=e2.OKABE_ITO["vermilion"],
            label=f"CH3 = {f_hi:g} MHz")
    if arrows:
        for k in np.flatnonzero(sel):
            ax.annotate("", (end[k, 1], end[k, 0]), (start[k, 1], start[k, 0]),
                        arrowprops=dict(arrowstyle="->", color=e2.OKABE_ITO["orange"], lw=0.6))
    lo = np.nanmin(np.r_[start[sel], end[sel]], axis=0) - 25
    hi = np.nanmax(np.r_[start[sel], end[sel]], axis=0) + 25
    ax.set_xlim(lo[1], hi[1])
    ax.set_ylim(hi[0], lo[0])
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)


draw(fig.add_subplot(gs[1, :]), last["ok"],
     f"design {last['theta']:.4f} deg: every trap at both ends of the CH3 sweep, over the 70 MHz frame; "
     f"grey = fitted rows ({last['row_deg']:.4f} deg)", arrows=False)
# One trap per row: a trap travels ~3 sites, so neighbours in a row would overlap.
zoom = last["ok"] & (cols == 0) & (np.abs(rows - NV // 2) <= 2)
draw(fig.add_subplot(gs[2, :]), zoom,
     f"zoom: first trap of rows {NV // 2 - 2}-{NV // 2 + 2}, each moving {f_lo:g} -> {f_hi:g} MHz along its row",
     arrows=True)
fig.suptitle(f"Rows vs CH3 steering: {'converged' if converged else 'NOT converged'} at design angle "
             f"{last['theta']:.4f} deg; exposure {args.exposure_us:g} us")
path = DATA_DIR / f"{stamp}_align_rows.png"
e2.save(fig, path, dpi=300, transparent=False)
