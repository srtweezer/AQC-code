"""Overlap of the tweezer array at two CH3 tones one site apart: per-trap displacement and spot size.

A CH3 step of config.CH3_SITE_STEP_MHZ moves the array by one site along its rows. Each
trap at 70 MHz should then be replaced by the neighbouring SLM trap at the same place and
with the same size. Every trap is fitted with a 2-D Gaussian (aqc.traps.fit_spots) at
70 MHz, at 70 MHz + --shift sites, and at 70 MHz again. The pairs come from nearest
neighbours (aqc.traps.pair_shifted). The second 70 MHz frame, compared with the first,
gives the fit noise and drift that the tone comparison has to beat.

Everything is in camera coordinates: displacement = centre at the shifted tone - centre
of the 70 MHz trap it lands on, along camera x and y, and sizes are the Gaussian sigmas
along camera x and y. The rows run at ~17 deg to camera x (2026-11-02), so x is mostly
along them. Lengths are um on the sensor (px * config.CAMERA_PIXEL_UM); the trap-plane
length is that divided by the imaging magnification. The maps show the displacement minus
its array mean; the mean is the residual of the site step plus drift.

CH3 is always scaled by aqc.aod_amplitude.ch3_scale, so the AODs pass the same power at both
tones.

With --vertical the pattern is the 90 rows x 10 columns array and CH1 is the transport
channel: the second tone is CH1 at 70 MHz + --shift * config.CH1_SITE_STEP_MHZ, CH1 is
flattened by aqc.aod_amplitude.ch1_scale, CH3 is held at 70 MHz unscaled, and the pattern
gets an 8 px blaze along -x and none along y (docs/phase_pattern_calibration.md). The
camera finds the array as 10 lines of 90 traps; the maps' rows and columns are those lines
and the traps along them.

Setup: the SLM beam imaged onto the Basler (free-run), AODs on CH1/CH3 at 1.6 V peak, CH1 at
70 MHz. Nothing else may hold the AWG (stop aod_tone.py first). Pick an exposure that keeps
the brightest trap below saturation.

Saves data/<MM_DD>[/<--subdir>]/<timestamp>_tone_overlap.png. Leaves the pattern on the SLM and the AWG
stopped.

Usage:
    python scripts/characterization/tone_overlap.py \\
        aqc/calibration/phase-patterns/20261101_184357_wgs_cwgs_pos_homogenized_zflat.npy --exposure-us 300
    python scripts/characterization/tone_overlap.py <pattern>.npy --exposure-us 300 --shift 1 --average 64
    python scripts/characterization/tone_overlap.py \\
        aqc/calibration/phase-patterns/vertical/20261112_234338_handoff_homogenized.npy \\
        --exposure-us 250 --vertical --shift 1 --subdir handoff_error
"""

import argparse
from datetime import datetime
from pathlib import Path

import e2style as e2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from aqc import config
from aqc.blaze import blaze
from aqc.data import save_plot
from aqc.hardware.awg import aods
from aqc.hardware.camera import Camera
from aqc.hardware.slm import send_phase
from aqc.intensity import FULL_SCALE
from aqc.steering import F_CENTER, fit_array, play_transport
from aqc.traps import pair_shifted

NV, NH = 10, 90
PX_UM = config.CAMERA_PIXEL_UM
AMPLITUDE = 1.6          # V peak, the CW drive of aod_tone.py

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("pattern", type=Path, help="phase map (.npy, radians, no blaze)")
parser.add_argument("--exposure-us", type=float, required=True, help="camera exposure, us")
parser.add_argument("--shift", type=int, default=-1, help="second tone: transport channel at 70 MHz + shift * site step")
parser.add_argument("--average", type=int, default=32, help="frames averaged per tone")
parser.add_argument("--vertical", action="store_true", help="90 rows x 10 columns array, transported with CH1")
parser.add_argument("--subdir", help="save the plot in data/<MM_DD>/<subdir>/")
args = parser.parse_args()

CH = 1 if args.vertical else 3                                 # transport channel
STEP_MHZ = config.CH1_SITE_STEP_MHZ if args.vertical else config.CH3_SITE_STEP_MHZ
send_phase(np.load(args.pattern) + (blaze(0, -8) if args.vertical else blaze(8, 0)))
runs = []
with aods(amplitudes=(AMPLITUDE, AMPLITUDE)) as awg, Camera(args.exposure_us, hardware_trigger=False) as cam:
    for f in (F_CENTER, F_CENTER + args.shift * STEP_MHZ * 1e6, F_CENTER):
        f = play_transport(awg, f, args.vertical)
        frame, fits = fit_array(cam, NV, NH, args.average)
        print(f"CH{CH} {f / 1e6:.4f} MHz: fitted {np.isfinite(fits[:, 0]).sum()}/{NV * NH}, frame max "
              f"{frame.max():.0f}, {(frame >= FULL_SCALE).sum()} saturated px", flush=True)
        runs.append(dict(f=f, fits=fits))

ref, sh, rep = runs
(dr, dc), (rs, cs), (rs1, cs1) = pair_shifted(ref["fits"][:, :2], sh["fits"][:, :2], NV, NH)


def compare(a, b):
    """Displacement and sigma differences along camera x and y, um on the sensor, of fits b - a ((NV, NH, 5) blocks)."""
    d = (b - a) * PX_UM
    return dict(dx=d[..., 1], dy=d[..., 0], dsx=d[..., 4], dsy=d[..., 3])


F0, F1, F2 = (run["fits"].reshape(NV, NH, 5) for run in runs)
T = compare(F0[rs, cs], F1[rs1, cs1])     # tone step
R = compare(F0[rs, cs], F2[rs, cs])       # 70 MHz repeat, same block of traps
cols = np.arange(cs.start, cs.stop)
f0, f1 = ref["f"] / 1e6, sh["f"] / 1e6
print(f"pairing: 70 MHz (r, c) <-> shifted (r{dr:+d}, c{dc:+d}); lengths in um on the sensor ({PX_UM} um/px)")
for k, label in (("dx", "x"), ("dy", "y"), ("dsx", "sigma x"), ("dsy", "sigma y")):
    print(f"{label:8s}: tone step mean {np.nanmean(T[k]):+.3f} um, std {np.nanstd(T[k]):.3f} um | "
          f"70 MHz repeat mean {np.nanmean(R[k]):+.3f} um, std {np.nanstd(R[k]):.3f} um")

e2.apply()
fig = plt.figure(figsize=(180 * e2.MM, 210 * e2.MM), layout="constrained")
gs = fig.add_gridspec(5, 2)
extent = (cs.start - 0.5, cs.stop - 0.5, rs.start - 0.5, rs.stop - 0.5)
maps = (("dx", "displacement along camera x"), ("dy", "displacement along camera y"))
lim = max(np.nanpercentile(np.abs(T[k] - np.nanmean(T[k])), 99) for k, _ in maps)
for i, (k, label) in enumerate(maps):
    ax = fig.add_subplot(gs[i, :])
    im = ax.imshow(T[k] - np.nanmean(T[k]), aspect="auto", cmap="RdBu_r", origin="lower", vmin=-lim, vmax=lim,
                   extent=extent, interpolation="nearest")
    fig.colorbar(im, ax=ax, label="− mean (µm)")
    ax.set_title(f"{label}, {f1:.2f} − {f0:.2f} MHz: mean {np.nanmean(T[k]):+.3f} µm, std {np.nanstd(T[k]):.3f} µm "
                 f"(70 MHz repeat: std {np.nanstd(R[k]):.3f} µm)")
    ax.set_xlabel(f"{f0:.2f} MHz column")
    ax.set_ylabel("row")

ds = np.sqrt(F1[rs1, cs1, 3] * F1[rs1, cs1, 4]) - np.sqrt(F0[rs, cs, 3] * F0[rs, cs, 4])
ds_rep = np.sqrt(F2[rs, cs, 3] * F2[rs, cs, 4]) - np.sqrt(F0[rs, cs, 3] * F0[rs, cs, 4])
ds, ds_rep = ds * PX_UM, ds_rep * PX_UM
lim = np.nanpercentile(np.abs(ds), 99)
ax = fig.add_subplot(gs[2, :])
im = ax.imshow(ds, aspect="auto", cmap="PuOr_r", origin="lower", vmin=-lim, vmax=lim, extent=extent,
               interpolation="nearest")
fig.colorbar(im, ax=ax, label="Δσ (µm)")
ax.set_title(f"spot size √(σx σy), {f1:.2f} − {f0:.2f} MHz: mean {np.nanmean(ds):+.3f} µm, std {np.nanstd(ds):.3f} µm "
             f"(70 MHz repeat: mean {np.nanmean(ds_rep):+.3f} µm, std {np.nanstd(ds_rep):.3f} µm)")
ax.set_xlabel(f"{f0:.2f} MHz column")
ax.set_ylabel("row")

colors = e2.OKABE_ITO
ax = fig.add_subplot(gs[3, 0])
for k, label, color in (("dx", "x", colors["blue"]), ("dy", "y", colors["vermilion"])):
    ax.plot(cols, np.nanmean(T[k], axis=0), lw=1.2, color=color, label=f"{label}, tone step")
    ax.plot(cols, np.nanmean(R[k], axis=0), ":", lw=1, color=color, label=f"{label}, 70 MHz repeat")
ax.axhline(0, color=e2.GREY.edge, lw=0.8)
ax.set_xlabel(f"{f0:.2f} MHz column")
ax.set_ylabel("displacement, mean over rows (µm)")
ax.set_title("displacement vs column")
ax.legend()

ax = fig.add_subplot(gs[3, 1])
for k, label, color in (("dsx", "σx", colors["blue"]), ("dsy", "σy", colors["vermilion"])):
    ax.plot(cols, np.nanmean(T[k], axis=0), lw=1.2, color=color, label=f"{label}, tone step")
    ax.plot(cols, np.nanmean(R[k], axis=0), ":", lw=1, color=color, label=f"{label}, 70 MHz repeat")
ax.axhline(0, color=e2.GREY.edge, lw=0.8)
ax.set_xlabel(f"{f0:.2f} MHz column")
ax.set_ylabel("Δσ, mean over rows (µm)")
ax.set_title(f"spot size change vs column ({f1:.2f} − {f0:.2f} MHz)")
ax.legend()

for j, (idx, label) in enumerate(((4, "σx"), (3, "σy"))):
    ax = fig.add_subplot(gs[4, j])
    x, y = F0[rs, cs, idx] * PX_UM, F1[rs1, cs1, idx] * PX_UM
    ok = np.isfinite(x) & np.isfinite(y)
    ax.plot(x[ok], y[ok], ".", ms=2, alpha=0.5, color=colors["blue"])
    lo, hi = min(x[ok].min(), y[ok].min()), max(x[ok].max(), y[ok].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y = x")
    ax.set_xlabel(f"{label} at {f0:.2f} MHz (µm)")
    ax.set_ylabel(f"{label} at {f1:.2f} MHz (µm)")
    ax.set_title(f"{label} per pair, r = {pearsonr(x[ok], y[ok])[0]:+.2f}")
    ax.legend()

fig.suptitle(f"Array overlap at two tones, {args.pattern.name}, {datetime.now():%Y-%m-%d}\n"
             f"CH{CH} {f0:.4f} MHz (row r, col c) vs {f1:.4f} MHz (row r{dr:+d}, col c{dc:+d}), CH{CH} flattened "
             f"(aqc.aod_amplitude)\nexposure {args.exposure_us:g} µs, {args.average}-frame average; "
             f"camera x/y, µm on the sensor ({PX_UM} µm/px)")
save_plot(fig, "tone_overlap", subdir=args.subdir)
