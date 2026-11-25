"""Measure the field-dependent aberrations of the SLM path with single spots, and fit Machu's model.

Follows Machu et al., arXiv:2512.15967, Sec. III: aberrations are measured at single spots
on a reference grid and fitted by the field-dependent Seidel model of aqc.aberration. The
paper's phase retrieval is replaced by a sensorless scan. For each spot, each of the five
Zernike modes (astigmatism 45/0, defocus, coma y/x) is stepped in turn while the spot's
peak is measured on the camera. A parabola fitted to log(peak) gives the best coefficient:
by the Marechal approximation the peak falls as exp(-residual phase variance), which is
quadratic in each coefficient. The modes are not quite orthogonal over the rectangular,
Gaussian-lit SLM, so the scan makes --passes passes, each over half the previous span.
Each spot starts from zero. Starting from the neighbour's result (as up to 2026-11-12)
let defocus and astigmatism drift to ~-2 rad along the vertical array's field, into a
valley where the "corrected" peak came out below the uncorrected one; from zero the same
centre spot gained 1.25x with coefficients below 0.6 rad.

The grid is in field coordinates (um, aqc.aberration), with the blaze included. The 0th
order and the half-plane around it are blocked by the blade, so the grid stays on the
passing side and covers the array: y 314-1018 um, x +-2263 um for the 10x90 at 3.19 deg
with the 8 px y blaze; y +-2258 um, x -937..-390 um for the vertical 90x10 at -1.19 deg
with --blaze-x -8 --blaze-y 0 (a 8 px blaze moves the field by 665 um). Every reference
spot is first shown uncorrected and located; a blocked or clipped spot stops the run.

Setup: the SLM beam imaged onto the Basler (free-run), AODs on CH1/CH3 at 70 MHz and 1.6 V
peak. Nothing else may hold the AWG (stop aod_tone.py first). A single spot carries the whole
beam, and the spots at the ends of the grid are ~5x dimmer than the centre (the SLM's
diffraction efficiency), so each spot gets its own exposure: --exposure-us scaled so its
brightest pixel in the uncorrected locate frame is TARGET_FRAC of full scale, leaving room
for the correction to raise it. --exposure-us itself must not saturate any spot (30 us
with the path power of 2026-10-29). Peaks are compared in counts/us. When even the 12 us
minimum exposure saturates a spot, --amplitude lowers the drive of both AODs for the whole
scan (2026-11-12: 1.6 V saturated, so the scan ran at a lower drive). The aberrations are
those of the SLM path, so the drive only scales the light.

Saves the fitted model with the measured coefficients to
aqc/calibration/<timestamp>_aberration.npz, points config.ABERRATION_CAL at it, and saves
data/<MM_DD>/<timestamp>_aberration_scan.png. Leaves the last spot on the SLM and the AWG stopped.

Usage:
    python scripts/calibration/aberration_scan.py --exposure-us 30
    python scripts/calibration/aberration_scan.py --exposure-us 30 --nx 9 --ny 5 --steps 7
    python scripts/calibration/aberration_scan.py --exposure-us 30 --y-um 405 1320
    python scripts/calibration/aberration_scan.py --exposure-us 30 --blaze-x -8 --blaze-y 0 \\
        --nx 3 --ny 7 --x-um -910 -405 --y-um -2230 2230
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

import e2style as e2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from aqc import aberration, config
from aqc.blaze import blaze
from aqc.data import save_plot
from aqc.hardware.awg import aods
from aqc.hardware.camera import Camera
from aqc.hardware.slm import send_phase
from aqc.intensity import FULL_SCALE, peaks
from aqc.steering import F_CENTER, grab, play
from aqc.traps import fit_spot
from aqc.wgs import FOCAL_UM, SLM_H, SLM_W, grid_phase

CAL_DIR = Path(config.__file__).parent / "calibration"
SCAL = 4
MIN_PEAK_FRAC = 0.2      # a located spot dimmer than this x the median is blocked or clipped
TARGET_FRAC = 0.35       # brightest pixel of an uncorrected spot, fraction of full scale

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--exposure-us", type=float, required=True,
                    help="exposure of the locate frames, us; each spot's scan exposure is scaled from it")
parser.add_argument("--nx", type=int, default=7, help="reference spots along x")
parser.add_argument("--ny", type=int, default=3, help="reference spots along y")
parser.add_argument("--x-um", type=float, nargs=2, default=[-2280, 2280], help="field x range, um")
parser.add_argument("--y-um", type=float, nargs=2, default=[305, 1320], help="field y range, um (blaze included)")
parser.add_argument("--steps", type=int, default=5, help="coefficient values per mode scan")
parser.add_argument("--span-rad", type=float, default=1.5, help="half-range of the first pass, rad")
parser.add_argument("--passes", type=int, default=2, help="passes over the five modes")
parser.add_argument("--average", type=int, default=4, help="frames averaged per measurement")
parser.add_argument("--amplitude", type=float, default=1.6, help="AOD drive on CH1 and CH3, V peak")
parser.add_argument("--blaze-y", type=int, default=8, help="blaze period along y, px (0 = none)")
parser.add_argument("--blaze-x", type=int, default=0, help="blaze period along x, px (0 = none)")
args = parser.parse_args()

ys = np.linspace(*args.y_um, args.ny)
xs = np.linspace(*args.x_um, args.nx)
grid = [(i, j) for i in range(args.ny) for j in (range(args.nx) if i % 2 == 0 else reversed(range(args.nx)))]
rr, cc = aberration.grid_site([(ys[i], xs[j]) for i, j in grid], SCAL, args.blaze_y, args.blaze_x)
r = aberration.field_um(rr, cc, SCAL, args.blaze_y, args.blaze_x)
row = np.array([i for i, _ in grid])
n_modes = len(aberration.MODES)
Z = aberration.basis()
base = blaze(args.blaze_y, args.blaze_x)


def spot_phase(q, c):
    return grid_phase([rr[q]], [cc[q]], SCAL).reshape(SLM_H, SLM_W) + np.tensordot(c, Z, 1) + base


def measure(cam, phase, pos):
    """Peak (counts/us) of the spot near pos (y, x) px for this phase, and its re-fitted centre."""
    send_phase(phase)
    frame = grab(cam, args.average)
    y, x = fit_spot(frame, *pos)[:2]
    if not np.isfinite(y):
        raise RuntimeError(f"lost the spot near (y, x) = ({pos[0]:.0f}, {pos[1]:.0f}) px")
    return peaks(frame, np.array([[y, x]]))[0] / cam.exposure_us, (y, x)


c_meas = np.zeros((len(grid), n_modes))
peak0, peak1 = np.zeros(len(grid)), np.zeros(len(grid))
with aods(amplitudes=(args.amplitude, args.amplitude)) as awg, Camera(args.exposure_us, hardware_trigger=False) as cam:
    play(awg, F_CENTER)
    pos = np.zeros((len(grid), 2))
    exposure = np.zeros(len(grid))
    for q in range(len(grid)):
        send_phase(spot_phase(q, np.zeros(n_modes)))
        frame = grab(cam, args.average)
        pos[q] = np.unravel_index(np.argmax(gaussian_filter(frame, 1.5)), frame.shape)
        peak0[q] = peaks(frame, pos[q:q + 1])[0]
        y, x = pos[q].astype(int)
        raw = frame[y - 8:y + 9, x - 8:x + 9].max()
        exposure[q] = max(args.exposure_us * TARGET_FRAC * FULL_SCALE / raw, config.CAMERA_MIN_EXPOSURE_US)
    weak = peak0 < MIN_PEAK_FRAC * np.median(peak0)
    print("Reference spots (field y, x um -> camera y, x px, peak, scan exposure):")
    for q in range(len(grid)):
        print(f"  {q:2d}: ({r[q, 0]:6.1f}, {r[q, 1]:7.1f}) -> ({pos[q, 0]:5.0f}, {pos[q, 1]:5.0f}), "
              f"{peak0[q]:6.0f}, {exposure[q]:5.1f} us{'  WEAK' if weak[q] else ''}")
    if weak.any():
        raise RuntimeError(f"{weak.sum()} reference spot(s) below {MIN_PEAK_FRAC:g} x the median peak: "
                           f"blocked by the blade or clipped; move --y-um/--x-um")

    for q in range(len(grid)):
        c = np.zeros(n_modes)
        cam.set_exposure(exposure[q])
        peak0[q], p = measure(cam, spot_phase(q, np.zeros(n_modes)), pos[q])
        at_edge = np.zeros(n_modes, bool)     # best value at the end of the last pass's range
        for n in range(args.passes):
            span = args.span_rad / 2 ** n
            for k in range(n_modes):
                vals = c[k] + np.linspace(-span, span, args.steps)
                pk = []
                for v in vals:
                    trial = c.copy()
                    trial[k] = v
                    peak, p = measure(cam, spot_phase(q, trial), p)
                    pk.append(peak)
                a, b, _ = np.polyfit(vals, np.log(pk), 2)
                c[k] = np.clip(-b / (2 * a), vals[0], vals[-1]) if a < 0 else vals[np.argmax(pk)]
                at_edge[k] = c[k] in (vals[0], vals[-1])
        c_meas[q] = c
        peak1[q], p = measure(cam, spot_phase(q, c), p)
        print(f"  spot {q:2d} ({r[q, 0]:6.1f}, {r[q, 1]:7.1f}) um: c = "
              f"[{', '.join(f'{v:+.2f}' + ('*' if e else '') for v, e in zip(c, at_edge))}] rad, "
              f"peak {peak0[q]:.1f} -> {peak1[q]:.1f} counts/us", flush=True)
        if at_edge.any():
            print(f"    * at the edge of the last scan range: raise --span-rad or --passes")

params = aberration.fit(r, c_meas)
c_fit = aberration.model(params, r)
s = aberration.seidel(params)
print(f"C' = {s['C'] * 1e3:.3f} mrad/um, A' = {s['A'] * 1e6:.3f} urad/um^2, K' = {s['K'] * 1e6:.3f} urad/um^2; "
      f"astigmatism centre (y, x) = ({s['astig_centre'][0]:.0f}, {s['astig_centre'][1]:.0f}) um, "
      f"curvature centre ({s['curv_centre'][0]:.0f}, {s['curv_centre'][1]:.0f}) um")
print("Residual rms per mode (rad):", ", ".join(f"{n} {v:.3f}" for n, v in
                                             zip(aberration.MODE_NAMES, (c_meas - c_fit).std(axis=0))))

stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
cal_path = CAL_DIR / f"{stamp}_aberration.npz"
np.savez(cal_path, params=params, param_names=aberration.PARAMS, r_um=r, focal_um=FOCAL_UM, c_meas=c_meas,
         peak_uncorrected=peak0, peak_corrected=peak1, amplitude_v=args.amplitude, blaze_y_px=args.blaze_y, blaze_x_px=args.blaze_x, exposure_us=args.exposure_us)
config_path = Path(config.__file__)
line = f'ABERRATION_CAL = Path(__file__).parent / "calibration" / "{cal_path.name}"  # {datetime.now():%Y-%m-%d}'
text, n = re.subn(r"^ABERRATION_CAL = .*$", line, config_path.read_text(), flags=re.M)
if n != 1:
    raise RuntimeError(f"expected one ABERRATION_CAL line in {config_path}, found {n}")
config_path.write_text(text)
print(f"Saved {cal_path}; config.ABERRATION_CAL points at it")

e2.apply()
colors = list(e2.OKABE_ITO.values())[1:]
fig, axs = plt.subplots(2, 3, figsize=(180 * e2.MM, 110 * e2.MM), layout="constrained", sharex=True)
x_line = np.linspace(*args.x_um, 200)
for k, ax in enumerate(axs.flat[:n_modes]):
    for i in range(args.ny):
        on = row == i
        color = colors[i % len(colors)]
        ax.plot(r[on, 1], c_meas[on, k], "o", ms=3, color=color, label=f"y {r[on, 0].mean():.0f} um")
        line_r = np.column_stack([np.full_like(x_line, r[on, 0].mean()), x_line])
        ax.plot(x_line, aberration.model(params, line_r)[:, k], "-", lw=1, color=color)
    ax.set_title(f"{aberration.MODE_NAMES[k]}: residual rms {(c_meas - c_fit)[:, k].std():.2f} rad")
    ax.set_ylabel("correction, rad")
axs[0, 0].legend()
ax = axs.flat[n_modes]
for i in range(args.ny):
    on = row == i
    ax.plot(r[on, 1], peak1[on] / peak0[on], "o-", ms=3, color=colors[i % len(colors)])
ax.axhline(1, color="k", lw=0.5)
ax.set_ylabel("peak corrected / uncorrected")
ax.set_title("single-spot peak gain")
for ax in axs[1]:
    ax.set_xlabel("field x, um")
fig.suptitle(f"C' {s['C'] * 1e3:.3f} mrad/um, A' {s['A'] * 1e6:.3f} urad/um$^2$, K' {s['K'] * 1e6:.3f} urad/um$^2$; "
             f"{args.ny}x{args.nx} spots, {args.passes} passes of {args.steps} steps, "
             f"blaze y {args.blaze_y:+d} x {args.blaze_x:+d} px, AODs {args.amplitude:g} V")
save_plot(fig, "aberration_scan", stamp)
