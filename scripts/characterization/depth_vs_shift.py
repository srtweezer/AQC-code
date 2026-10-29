"""Tweezer depth across the array with CH3 shifted by whole sites.

The pattern is displayed with an 8 px y blaze. CH1 is held at 70 MHz and CH3 plays
70 MHz + shift * config.CH3_SITE_STEP_MHZ for each --shifts value. Every trap is fitted
with a 2-D Gaussian; depth = Gaussian amplitude / exposure (counts/us). With
--smoothed-peak, depth is instead the smoothed maximum at the fitted centre
(aqc.intensity.peaks, the metric homogenize.py equalizes) / exposure. It reads wider
spots relatively higher than the Gaussian amplitude does. The exposure is
fixed for all shifts, so depths compare directly. A trap keeps its SLM index after the
shift: it is seeded at its 70 MHz centre moved by the steering, measured with a 0.5 MHz
probe step.

With --flat, every CH3 tone (including the 70 MHz reference and the probe) is scaled by
the calibrated aqc.aod_amplitude.ch3_scale(f), so the AODs pass the same power at every
shift.

Setup: the SLM beam imaged onto the Basler (free-run), AODs on CH1/CH3 at 1.6 V peak.
Nothing else may hold the AWG (stop aod_tone.py first). Pick an exposure that keeps the
brightest trap below saturation at every shift.

Saves data/<timestamp>_depth_vs_shift.png. Leaves the pattern on the SLM and the AWG
stopped.

Usage:
    python scripts/characterization/depth_vs_shift.py aqc/calibration/phase-patterns/20261028_182104_wgs_aligned.npy --exposure-us 100
    python scripts/characterization/depth_vs_shift.py <pattern.npy> --exposure-us 100 --shifts -2 -1 0 1 2
    python scripts/characterization/depth_vs_shift.py <pattern.npy> --exposure-us 200 --flat
    python scripts/characterization/depth_vs_shift.py <pattern.npy> --exposure-us 170 --flat --smoothed-peak
"""

import argparse
from datetime import datetime
from pathlib import Path

import e2style as e2
import matplotlib.pyplot as plt
import numpy as np

from aqc import config
from aqc.blaze import blaze
from aqc.hardware.awg import aods
from aqc.hardware.camera import Camera
from aqc.hardware.slm import send_phase
from aqc.aod_amplitude import ch3_scale
from aqc.intensity import peaks as smoothed_peaks
from aqc.steering import F_CENTER, grab, play
from aqc.traps import find_traps, fit_all, fit_spots

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FULL_SCALE = 4095        # Mono12
NV, NH = 10, 90
BLAZE_Y_PX = 8
AMPLITUDE = 1.6          # V peak, the CW drive of aod_tone.py
PROBE_MHZ = 0.5          # small CH3 step to measure the steering (~2.5 px, well inside a fit window)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("pattern", type=Path, help="phase map (.npy, radians, no blaze)")
parser.add_argument("--exposure-us", type=float, required=True, help="camera exposure, us")
parser.add_argument("--shifts", type=int, nargs="+", default=[-1, 0, 1], help="CH3 shifts, sites")
parser.add_argument("--average", type=int, default=16, help="frames averaged per image")
parser.add_argument("--flat", action="store_true", help="scale CH3 by aqc.aod_amplitude.ch3_scale (flat power)")
parser.add_argument("--smoothed-peak", action="store_true",
                    help="depth = smoothed maximum (aqc.intensity.peaks) instead of Gaussian amplitude")
args = parser.parse_args()


def scale(f_hz):
    return float(ch3_scale(f_hz)) if args.flat else 1.0


send_phase(np.load(args.pattern) + blaze(BLAZE_Y_PX, 0))
runs = []
with aods(amplitudes=(AMPLITUDE, AMPLITUDE)) as awg, Camera(args.exposure_us, hardware_trigger=False) as cam:
    play(awg, F_CENTER, scale(F_CENTER))
    ref = grab(cam, args.average)
    peaks, rows, cols = find_traps(ref, NV, NH)
    p0 = fit_all(ref, peaks)
    if not np.all(np.isfinite(p0)):
        raise RuntimeError(f"Gaussian fit failed for {np.isnan(p0[:, 0]).sum()} of {NV * NH} traps at 70 MHz")
    play(awg, F_CENTER + PROBE_MHZ * 1e6, scale(F_CENTER + PROBE_MHZ * 1e6))
    steer = np.nanmedian(fit_all(grab(cam, args.average), p0) - p0, axis=0) / PROBE_MHZ   # (y, x) px/MHz
    for shift in args.shifts:
        df = shift * config.CH3_SITE_STEP_MHZ
        f3 = play(awg, F_CENTER + df * 1e6, scale(F_CENTER + df * 1e6))
        img = grab(cam, args.average)
        fits = fit_spots(img, p0 + df * steer)
        depth = np.full((NV, NH), np.nan)
        if args.smoothed_peak:
            ok = np.all(np.isfinite(fits), axis=1)
            depth[rows[ok], cols[ok]] = smoothed_peaks(img, fits[ok, :2]) / args.exposure_us
        else:
            depth[rows, cols] = fits[:, 2] / args.exposure_us
        n_sat = int((img >= FULL_SCALE).sum())
        print(f"CH3 {f3 / 1e6:.4f} MHz ({shift:+d} site): frame max {img.max():.0f}, {n_sat} saturated px; "
              f"{np.isfinite(depth).sum()} traps fitted; depth mean {np.nanmean(depth):.2f} counts/us, "
              f"std/mean {np.nanstd(depth) / np.nanmean(depth):.1%}")
        runs.append(dict(shift=shift, f3=f3, depth=depth))

ref_depth = next((r["depth"] for r in runs if r["shift"] == 0), None)
vmin, vmax = np.nanpercentile(np.concatenate([r["depth"].ravel() for r in runs]), [1, 99])
e2.apply()
fig = plt.figure(figsize=(180 * e2.MM, (42 * len(runs) + 60) * e2.MM), layout="constrained")
gs = fig.add_gridspec(len(runs) + 1, 1, height_ratios=[1] * len(runs) + [1.4])
for i, r in enumerate(runs):
    ax = fig.add_subplot(gs[i])
    im = ax.imshow(r["depth"], cmap="viridis", aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, label="counts/µs", extend="both")
    rel = f", mean {np.nanmean(r['depth']) / np.nanmean(ref_depth):.3f} × 70 MHz" if ref_depth is not None else ""
    ax.set_title(f"CH3 {r['f3'] / 1e6:.4f} MHz ({r['shift']:+d} site): mean {np.nanmean(r['depth']):.2f} counts/µs, "
                 f"std/mean {np.nanstd(r['depth']) / np.nanmean(r['depth']):.1%}{rel}")
    ax.set_ylabel("row")
ax.set_xlabel("column")
ax = fig.add_subplot(gs[-1])
for r, color in zip(runs, [e2.OKABE_ITO[c] for c in ("blue", "vermilion", "green", "orange", "purple", "sky")]):
    ax.plot(np.arange(NH), np.nanmean(r["depth"], axis=0), "o-", ms=2, lw=0.8, color=color,
            label=f"CH3 {r['f3'] / 1e6:.4f} MHz ({r['shift']:+d} site)")
ax.set_ylim(bottom=0)
ax.set_xlabel("column (same SLM trap in every frame)")
ax.set_ylabel("depth, mean over rows (counts/µs)")
ax.legend(frameon=True, facecolor="white", framealpha=0.9)
metric = "smoothed peak" if args.smoothed_peak else "Gaussian amplitude"
fig.suptitle(f"Tweezer depth ({metric} / exposure), {args.pattern.name}\nCH1 70 MHz, "
             f"{'CH3 flattened (aqc.aod_amplitude), ' if args.flat else ''}"
             f"exposure {args.exposure_us:g} µs, {args.average}-frame average")
DATA_DIR.mkdir(exist_ok=True)
e2.save(fig, DATA_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_depth_vs_shift.png", dpi=300, transparent=False)
