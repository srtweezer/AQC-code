"""Step the SLM through blaze periods, image the first diffraction order for each, and test the grating-to-camera map.

A phase gradient k = 2 pi / N rad per SLM pixel (pitch p) deflects the beam by
sin(theta) = lambda / (N p). The f1 = 200 mm lens maps the angle to f1 theta in the blade
plane, and the f2 = 200 mm lens with the f3 = 40 mm objective image that plane onto the
camera at magnification f3 / f2, so the first order sits at

    X_cam = X_0 + a / N,   |a| = (f1 f3 / f2) lambda / p = 40 mm * 1.064 um / 8 um = 5.32 mm

(2876 camera px), with X_0 the 0th order (here also deflected by the AODs at their
current tones). The direction of a on the camera is set by the mirrors and the camera
rotation. The sweep steps the period N along --axis, with a fixed --other-px period on
the other axis, fits each first order's Gaussian centre in the first exposure's frame, and
fits X_cam linearly against 1/N; the residuals show any nonlinearity (tan vs sin, lens
distortion).

Setup: SLM beam on the Basler with the zero order blocked, so the first order is
the brightest spot in each frame (found after smoothing, so a hot pixel can't win).
Free-run exposure: the AWG and AOM are left as they are, so the beam has to be on, e.g.
with aod_tone.py running (--amplitude 0.25 keeps a single spot below saturation). Each
blaze is imaged at every exposure in --exposure-us. The SLM is left on the last blaze.

Saves data/<MM_DD>/<timestamp>_blaze_sweep.png (one row per exposure, one column per
period, crops on a shared Mono12 colour scale) and <timestamp>_blaze_sweep_position.png
(first-order position along and across a against 1/N, with the prediction).

Usage:
    python scripts/aod_tone.py --amplitude 0.25        # in another terminal
    python scripts/characterization/blaze_sweep.py --exposure-us 12 100
    python scripts/characterization/blaze_sweep.py --exposure-us 12 --periods 4 8 16 32 --crop 50
    python scripts/characterization/blaze_sweep.py --exposure-us 50 --axis x \\
        --periods -4 -5 -6 -8 -10 -12 -16 -24 -32
    python scripts/characterization/blaze_sweep.py --exposure-us 50 --axis y --other-px -8 \\
        --periods -4 -6 -8 -12 -16 -32 32 16 12 8 6 4
"""

import argparse

import e2style as e2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from aqc import config
from aqc.blaze import blaze
from aqc.data import save_plot
from aqc.hardware.camera import Camera
from aqc.hardware.slm import send_phase
from aqc.traps import fit_spot
from aqc.wgs import PIXEL_UM, WAVELENGTH_UM

FULL_SCALE = 4095   # Mono12
SMOOTH_PX = 5       # so a hot pixel can't win the peak search
MIN_PEAK_FRAC = 0.1  # a first order dimmer than this times the median peak is taken as blocked
F1_MM, F2_MM, F3_MM = 200.0, 200.0, 40.0
A_PRED_PX = F1_MM * F3_MM / F2_MM * 1e3 * WAVELENGTH_UM / PIXEL_UM / config.CAMERA_PIXEL_UM

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--exposure-us", type=float, nargs="+", required=True, help="camera exposures, us")
parser.add_argument("--axis", choices=["y", "x"], default="y", help="SLM axis the periods step along")
parser.add_argument("--periods", type=int, nargs="+", default=[4, 8, 16, 32], help="blaze periods along --axis, px (signed)")
parser.add_argument("--other-px", type=int, default=0, help="fixed blaze period on the other axis, px (signed, 0 = none)")
parser.add_argument("--crop", type=int, default=32, help="crop half-width, camera px")
args = parser.parse_args()


def peak(image):
    return np.unravel_index(np.argmax(gaussian_filter(image, SMOOTH_PX)), image.shape)


def pattern(period):
    if args.axis == "y":
        return blaze(period_y_px=period, period_x_px=args.other_px)
    return blaze(period_y_px=args.other_px, period_x_px=period)


# frames[exposure index][period index]
frames = [[None] * len(args.periods) for _ in args.exposure_us]
with Camera(args.exposure_us[0], hardware_trigger=False) as cam:
    for j, period in enumerate(args.periods):
        send_phase(pattern(period))
        for i, exposure in enumerate(args.exposure_us):
            cam.set_exposure(exposure)
            frames[i][j] = cam.grab()

other = "x" if args.axis == "y" else "y"
label = lambda period: f"blaze {args.axis} {period:+d}" + (f", {other} {args.other_px:+d}" if args.other_px else "") + " px"

e2.apply()
fig, axes = plt.subplots(len(args.exposure_us), len(args.periods), squeeze=False, layout="constrained",
                         figsize=(45 * e2.MM * len(args.periods), 42 * e2.MM * len(args.exposure_us)))
for i, exposure in enumerate(args.exposure_us):
    for j, period in enumerate(args.periods):
        frame = frames[i][j]
        h, w = frame.shape
        y, x = peak(frame.astype(float))
        y0 = min(max(y - args.crop, 0), h - 2 * args.crop)
        x0 = min(max(x - args.crop, 0), w - 2 * args.crop)
        crop = frame[y0:y0 + 2 * args.crop, x0:x0 + 2 * args.crop]
        n_sat = int((crop >= FULL_SCALE).sum())
        print(f"{exposure:g} us, {label(period)}: first order at x {x}, y {y}, "
              f"peak {crop.max()}, {n_sat} saturated px")
        ax = axes[i, j]
        im = ax.imshow(crop, cmap="inferno", vmin=0, vmax=FULL_SCALE,
                       extent=(x0, x0 + crop.shape[1], y0 + crop.shape[0], y0))
        ax.set_title(f"{exposure:g} us, {label(period)}\npeak {crop.max()}, {n_sat} sat. px")
fig.colorbar(im, ax=axes, label="counts", shrink=0.8)
save_plot(fig, "blaze_sweep")

inv_n = 1.0 / np.array(args.periods, float)
pos = np.array([fit_spot(f.astype(float), *peak(f.astype(float)))[:2] for f in frames[0]])   # (n, 2) y, x
# A first order behind the D mirror leaves only background, whose fitted "spot" is noise.
peaks = np.array([f.max() for f in frames[0]], float)
ok = np.all(np.isfinite(pos), axis=1) & (peaks > MIN_PEAK_FRAC * np.median(peaks))
(a, x_0), *_ = np.linalg.lstsq(np.column_stack([inv_n[ok], np.ones(ok.sum())]), pos[ok], rcond=None)
resid = pos - (inv_n[:, None] * a + x_0)
a_len = np.hypot(*a)
u = a / a_len
along, across = (pos - x_0) @ u, (pos - x_0) @ np.array([-u[1], u[0]])
print(f"\n{ok.sum()}/{len(ok)} first orders fitted ({args.exposure_us[0]:g} us frames)")
for period, p, r, k in zip(args.periods, pos, resid, ok):
    print(f"  {args.axis} {period:+4d} px: y {p[0]:8.2f}  x {p[1]:8.2f}   residual y {r[0]:+6.2f}  x {r[1]:+6.2f} px" + ("" if k else "  (blocked, not fitted)"))
print(f"a = (y {a[0]:+.1f}, x {a[1]:+.1f}) px, |a| = {a_len:.1f} px, predicted {A_PRED_PX:.1f} px "
      f"(ratio {a_len / A_PRED_PX:.4f}); direction {np.degrees(np.arctan2(a[0], a[1])):+.2f} deg (atan2(y, x)); "
      f"0th order at y {x_0[0]:.1f}, x {x_0[1]:.1f}; residual rms {np.sqrt(np.mean(resid[ok] ** 2)):.2f} px")

fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, layout="constrained", figsize=(90 * e2.MM, 90 * e2.MM))
grid = np.linspace(min(0, inv_n.min()), max(0, inv_n.max()), 50)
ax0.plot(grid, a_len * grid, color=e2.OKABE_ITO["blue"], label=f"fit, |a| = {a_len:.1f} px")
ax0.plot(grid, A_PRED_PX * grid, "--", color=e2.GREY.edge, label=f"predicted, {A_PRED_PX:.1f} px")
ax0.plot(inv_n[ok], along[ok], "o", color=e2.OKABE_ITO["vermilion"])
ax0.set_ylabel("along a, px")
ax0.legend()
ax1.plot(inv_n[ok], along[ok] - a_len * inv_n[ok], "o", color=e2.OKABE_ITO["vermilion"], label="along a")
ax1.plot(inv_n[ok], across[ok], "s", color=e2.OKABE_ITO["green"], label="across a")
ax1.axhline(0, color=e2.GREY.edge, lw=0.5)
ax1.set_xlabel(f"1 / N{args.axis}, 1/px")
ax1.set_ylabel("residual, px")
ax1.legend()
ax0.set_title(f"first order vs blaze {args.axis}" + (f" ({other} {args.other_px:+d} px)" if args.other_px else ""))
save_plot(fig, "blaze_sweep_position")
