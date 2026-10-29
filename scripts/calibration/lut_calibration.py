"""Measure the SLM phase response (grey level -> phase) on the Basler.

Diffractive method, as in andrii-test/calibration/lut_calibration.py (which made
the current config.SLM_PHASE_RESPONSE) and slm_local/lut_calib.py before it: a
binary grating along y, alternating a reference grey REF with a swept grey g,
fills a small centred patch. For a phase difference dphi(g) between the two levels
the first order goes as sin^2(dphi/2), so sweeping g and recording the first order
gives the phase response.

Setup: SLM beam on the Basler with the zero order blocked; the beam has to be on
(e.g. aod_tone.py running; the AWG and AOM are left as they are). Only the first
order is used: it is normalised to itself, each half period between successive
extrema to its own end levels, so the zero order is never needed.

  * The patch covers 1.56% of the SLM so the first order fits the camera's range
    (a full-aperture grating saturates at the 12 us minimum exposure). The rest
    carries a steep blaze along x, orthogonal to the grating, so its light lands
    away from the calibration orders.
  * The patch is in RAW grey levels (the response being measured). The blaze is
    built in radians and mapped through the current phase response.
  * The first order is located by differencing a flat patch (REF/REF, no first
    order) against REF/128, the exposure is set on it, and grey 128 is repeated
    at the start and end of the sweep to check the source is stable.

The result is valid only for the controller LUT slm-server has loaded. Writes
data/<stamp>_phase_response.npz, in the same format as config.SLM_PHASE_RESPONSE.
It is not installed: to use it, copy it to aqc/calibration/ and point
config.SLM_PHASE_RESPONSE at it. Also writes data/<stamp>_lut_calibration.png.
Leaves the last calibration grating on the SLM.

Usage:
    python scripts/calibration/lut_calibration.py
    python scripts/calibration/lut_calibration.py --step 4 --average 8
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import e2style as e2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from aqc import config
from aqc.blaze import blaze
from aqc.hardware.camera import Camera
from aqc.hardware.slm import phase_to_gray, send_gray

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FULL_SCALE = 4095            # Mono12
GRATING_PERIOD_PX = 8
PATCH_FRAC = 0.125           # linear fraction of the SLM covered by the grating
BLAZE_PERIOD_PX = 4          # blaze along x around the patch
MID_GRAY = 128               # near the first-order maximum; used to locate, expose and check drift


def calibration_pattern(ref, level):
    """Raw-grey binary grating (along y) on a centred patch, calibrated blaze (along x) elsewhere."""
    h, w = config.SLM_SHAPE
    out = phase_to_gray(blaze(period_x_px=BLAZE_PERIOD_PX))
    ph, pw = int(h * PATCH_FRAC), int(w * PATCH_FRAC)
    y0, x0 = (h - ph) // 2, (w - pw) // 2
    y = np.arange(y0, y0 + ph)[:, None]
    out[y0:y0 + ph, x0:x0 + pw] = np.where(y % GRATING_PERIOD_PX < GRATING_PERIOD_PX // 2, ref, level)
    return out


def measure(cam, ref, level, n):
    """Mean of n frames with grey `level` against `ref` on the patch."""
    send_gray(calibration_pattern(ref, level))
    return np.mean([cam.grab() for _ in range(n)], axis=0)


def roi(img, center, r):
    return img[max(center[0] - r, 0):center[0] + r + 1, max(center[1] - r, 0):center[1] + r + 1]


def auto_expose(cam, image_at, center, r, target):
    """Set the exposure that puts the ROI peak near `target` of full scale; return it."""
    exposure = cam.exposure_us
    for _ in range(12):
        peak = roi(image_at(), center, r).max()
        fraction = peak / FULL_SCALE
        print(f"  [expose] {exposure:9.1f} us -> first-order peak {peak:.0f} ({fraction:.0%})")
        if peak >= FULL_SCALE:
            if exposure <= config.CAMERA_MIN_EXPOSURE_US:
                raise RuntimeError("first order saturated at the camera's minimum exposure: "
                                   "reduce PATCH_FRAC or attenuate optically")
            exposure = max(exposure * 0.4, config.CAMERA_MIN_EXPOSURE_US)
        elif abs(fraction - target) < 0.1:
            return exposure
        else:
            exposure = max(exposure * target / max(fraction, 1e-3), config.CAMERA_MIN_EXPOSURE_US)
        cam.set_exposure(exposure)
    print(f"  [expose] did not converge, using {exposure:.1f} us")
    return exposure


def turning_points(y, hysteresis=0.1):
    """Indices of the alternating maxima/minima of y, starting with a maximum
    (the sweep starts at dphi = 0). An extremum only counts once y has moved
    back by `hysteresis` of its full range, so noise can't register as one."""
    span = y.max() - y.min()
    points, ext, rising = [], 0, True
    for i in range(1, len(y)):
        if (y[i] > y[ext]) if rising else (y[i] < y[ext]):
            ext = i
        elif abs(y[ext] - y[i]) > hysteresis * span:
            points.append(ext)
            ext, rising = i, not rising
    return points


def phase_from_first_order(intensity):
    """dphi(g) from I1 ~ sin^2(dphi/2), unwrapped through its extrema.

    Each half period (between successive extrema, dphi = k*pi .. (k+1)*pi) is
    normalised to its own end levels, so a minimum that doesn't reach zero
    (finite contrast) doesn't put a jump in the curve."""
    tps = turning_points(intensity)
    bounds = [0] + tps + [len(intensity) - 1]
    levels = [intensity[0]] + [intensity[t] for t in tps]
    dphi = np.empty(len(intensity))
    for k in range(len(bounds) - 1):
        a, b = bounds[k], bounds[k + 1]
        start = levels[k]
        # The last half period is open-ended: its far level is the previous
        # extremum of the same kind (or the overall max if there is none).
        end = levels[k + 1] if k + 1 < len(levels) else (levels[k - 1] if k else intensity.max())
        rising = k % 2 == 0
        lo, hi = (start, end) if rising else (end, start)
        norm = np.clip((intensity[a:b + 1] - lo) / (hi - lo), 0.0, 1.0)
        w = 2 * np.arcsin(np.sqrt(norm))
        dphi[a:b + 1] = k * np.pi + (w if rising else np.pi - w)
    # The response must be single valued.
    dphi = np.maximum.accumulate(dphi)
    return dphi - dphi[0], tps


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--ref", type=int, default=0, help="reference grey level")
parser.add_argument("--step", type=int, default=2, help="grey-level step of the sweep")
parser.add_argument("--average", type=int, default=4, help="frames averaged per level")
parser.add_argument("--roi", type=int, default=40, help="half width of the first-order ROI, camera px")
parser.add_argument("--target", type=float, default=0.6, help="first-order peak at grey 128, fraction of full scale")
args = parser.parse_args()

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
grays = np.arange(0, 256, args.step)
with Camera(1000.0, hardware_trigger=False) as cam:
    im_flat = measure(cam, args.ref, args.ref, 3)
    im_mid = measure(cam, args.ref, MID_GRAY, 3)
    center = tuple(int(v) for v in np.unravel_index(np.argmax(gaussian_filter(im_mid - im_flat, 3.0)),
                                                    im_mid.shape))
    contrast = roi(im_mid, center, args.roi).sum() / max(roi(im_flat, center, args.roi).sum(), 1)
    print(f"Calibration first order at x {center[1]}, y {center[0]}; ROI sum {MID_GRAY} vs flat: {contrast:.0f}x")

    exposure = auto_expose(cam, lambda: measure(cam, args.ref, MID_GRAY, 2), center, args.roi, args.target)

    drift_start = roi(measure(cam, args.ref, MID_GRAY, args.average), center, args.roi).sum()
    sums, peaks, t0 = [], [], time.time()
    for k, g in enumerate(grays):
        r1 = roi(measure(cam, args.ref, g, args.average), center, args.roi)
        sums.append(r1.sum())
        peaks.append(r1.max())
        if k % 16 == 0:
            print(f"  g={g:3d}  I1={sums[-1]:.4e}  peak={peaks[-1]:.0f}")
    end = measure(cam, args.ref, MID_GRAY, args.average)
drift = abs(roi(end, center, args.roi).sum() - drift_start) / max(drift_start, 1)
print(f"Swept {len(grays)} levels in {time.time() - t0:.0f} s, source drift {drift:.2%}")
if drift > 0.05:
    print("  ** WARNING: >5% drift; the self-normalisation is unsafe **")

intensity, peaks = np.array(sums), np.array(peaks)
dphi, tps = phase_from_first_order(intensity)
span = dphi[-1]
print(f"Measured phase response: span {span / np.pi:.3f} pi "
      f"({'reaches' if span >= 2 * np.pi else 'SHORT of'} 2pi), turning points at grey {[int(grays[t]) for t in tps]}")
if not tps:
    print("  ** no maximum found: the response never reaches pi, so the normalisation is a guess **")
if span >= 2 * np.pi:
    print(f"  2pi reached at grey {np.interp(2 * np.pi, dphi, grays):.1f}")
n_sat = int((peaks >= FULL_SCALE).sum())
if n_sat:
    print(f"  ** {n_sat}/{len(grays)} levels saturated in the ROI **")

DATA_DIR.mkdir(exist_ok=True)
zoom = roi(end, center, 3 * args.roi)
response = DATA_DIR / f"{stamp}_phase_response.npz"
np.savez(response, gray=grays.astype(np.float64), dphi=dphi, I1_sum=intensity, I1_peak=peaks,
         exposure=exposure, full_scale=FULL_SCALE, center=np.array(center), roi=args.roi, zoom=zoom,
         drift=drift, ref=args.ref, period=GRATING_PERIOD_PX, frac=PATCH_FRAC)
print(f"Saved {response} (NOT installed; config.SLM_PHASE_RESPONSE is still {config.SLM_PHASE_RESPONSE.name})")

current = np.load(config.SLM_PHASE_RESPONSE)
c, r, hz = center, args.roi, zoom.shape[0] // 2
e2.apply()
fig, ax = plt.subplots(2, 2, figsize=(180 * e2.MM, 130 * e2.MM), layout="constrained")
m = ax[0, 0].imshow(zoom, cmap="inferno", extent=[c[1] - hz, c[1] + hz, c[0] + hz, c[0] - hz])
ax[0, 0].add_patch(plt.Rectangle((c[1] - r, c[0] - r), 2 * r, 2 * r, fill=False, ec=e2.OKABE_ITO["sky"]))
fig.colorbar(m, ax=ax[0, 0], label="counts")
ax[0, 0].set_title(f"first order at x {c[1]}, y {c[0]}, g = {MID_GRAY}, {exposure:.0f} us (box = ROI)")
ax[0, 1].plot(grays, intensity / intensity.max(), ".-", ms=2, color=e2.OKABE_ITO["blue"])
ax[0, 1].plot(grays[tps], intensity[tps] / intensity.max(), "o", ms=4, mfc="none", color=e2.OKABE_ITO["vermilion"],
             label="turning points")
ax[0, 1].set_xlabel("grey level")
ax[0, 1].set_ylabel("first order (ROI sum), normalised")
ax[0, 1].set_title(f"measured $\\sin^2(\\Delta\\phi/2)$   (source drift {drift:.2%})")
ax[0, 1].legend()
ax[0, 1].grid(alpha=0.3)
ax[1, 0].plot(grays, dphi / np.pi, ".-", ms=2, color=e2.OKABE_ITO["blue"], label="measured")
ax[1, 0].plot(current["gray"], current["dphi"] / np.pi, "--", color=e2.OKABE_ITO["orange"],
             label=f"current ({config.SLM_PHASE_RESPONSE.name})")
ax[1, 0].axhline(2, color="k", lw=0.5, ls=":")
ax[1, 0].set_xlabel("grey level")
ax[1, 0].set_ylabel("$\\Delta\\phi$ / $\\pi$")
ax[1, 0].set_title("phase response")
ax[1, 0].legend()
ax[1, 0].grid(alpha=0.3)
ax[1, 1].plot(grays, (dphi - np.interp(grays, current["gray"], current["dphi"])) / np.pi, ".-", ms=2,
              color=e2.OKABE_ITO["blue"])
ax[1, 1].set_xlabel("grey level")
ax[1, 1].set_ylabel("measured - current [$\\pi$ rad]")
ax[1, 1].set_title("difference from the current phase response")
ax[1, 1].grid(alpha=0.3)
fig.suptitle(f"SLM phase response {stamp}")
plot = DATA_DIR / f"{stamp}_lut_calibration.png"
e2.save(fig, plot, dpi=300, transparent=False)
