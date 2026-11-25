"""Focus and peak amplitude of one SLM pattern at two CH3 tones one site apart, compared trap by trap.

Ported from slm_local/zflat_09_23/plot_diff_u64z.py (2026-09-23, 69.94 vs 64 MHz). A
CH3 step of config.CH3_SITE_STEP_MHZ moves the whole array by one site along its rows, so
each trap at 70 MHz lands on the place where a neighbouring SLM trap sat at the shifted
tone. A handoff between the tones needs those two to match. Both tones run a full z-scan
of a global lens (aqc.focus.zscan), and every trap's focus z0 comes from its width
(aqc.focus.fit_z0). The pairs come from nearest neighbours between the two sets of
measured positions; the row/column offset is not assumed. A pair counts when its two
spots are within MAX_SEP_UM on the camera sensor (px * config.CAMERA_PIXEL_UM).

dz = z0(70 MHz) - z0(shifted). The amplitude is the Gaussian peak at the scan step
nearest that tone's mean focus, normalised to the array mean of that tone. The AODs'
power at each tone therefore cancels, and only the spread across the array remains.
With --flat, CH3 is scaled by the calibrated aqc.aod_amplitude.ch3_scale(f) at both tones,
so the AODs pass the same power and the absolute depths match too.

Setup: the SLM beam imaged onto the Basler (free-run), AODs on CH1/CH3 at 1.6 V peak, CH1
at 70 MHz. Nothing else may hold the AWG (stop aod_tone.py first). Pick an exposure that
keeps the brightest trap below saturation in focus at both tones.

Saves data/<MM_DD>/<timestamp>_tone_differential.png. Leaves the pattern on the SLM (no
lens) and the AWG stopped.

Usage:
    python scripts/characterization/tone_differential.py \\
        aqc/calibration/phase-patterns/20261101_184357_wgs_cwgs_pos_homogenized_zflat.npy --exposure-us 150
    python scripts/characterization/tone_differential.py <pattern>.npy --exposure-us 150 --shift 1
    python scripts/characterization/tone_differential.py <pattern>.npy --exposure-us 150 --flat
"""

import argparse
from datetime import datetime
from pathlib import Path

import e2style as e2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from aqc import config
from aqc.aod_amplitude import ch3_scale
from aqc.blaze import blaze
from aqc.data import save_plot
from aqc.focus import fit_z0, zscan
from aqc.hardware.awg import aods
from aqc.hardware.camera import Camera
from aqc.hardware.slm import send_phase
from aqc.steering import F_CENTER, play
from aqc.traps import pair_shifted

NV, NH = 10, 90
BLAZE_Y_PX = 8
AMPLITUDE = 1.6          # V peak, the CW drive of aod_tone.py
MAX_SEP_UM = 2.6          # on the sensor, 1.4 px

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("pattern", type=Path, help="phase map (.npy, radians, no blaze)")
parser.add_argument("--exposure-us", type=float, required=True, help="camera exposure, us")
parser.add_argument("--shift", type=int, default=-1, help="second tone: CH3 = 70 MHz + shift * site step")
parser.add_argument("--zmin", type=float, default=-64.0, help="scan start, um")
parser.add_argument("--zmax", type=float, default=64.0, help="scan end, um")
parser.add_argument("--zstep", type=float, default=2.5, help="scan step, um")
parser.add_argument("--average", type=int, default=16, help="frames averaged per z")
parser.add_argument("--flat", action="store_true", help="scale CH3 by aqc.aod_amplitude.ch3_scale (flat power)")
args = parser.parse_args()

phase = np.load(args.pattern) + blaze(BLAZE_Y_PX, 0)
zs = np.arange(args.zmin, args.zmax + 1e-9, args.zstep)
scans = []
with aods(amplitudes=(AMPLITUDE, AMPLITUDE)) as awg, Camera(args.exposure_us, hardware_trigger=False) as cam:
    for f in (F_CENTER, F_CENTER + args.shift * config.CH3_SITE_STEP_MHZ * 1e6):
        f3 = play(awg, f, float(ch3_scale(f)) if args.flat else 1.0)
        print(f"CH3 {f3 / 1e6:.4f} MHz", flush=True)
        pos, fits = zscan(cam, phase, zs, args.average, NV, NH)
        scans.append(dict(f3=f3, pos=pos, fits=fits))
    send_phase(phase)

for s in scans:
    z0, err, flag = fit_z0(zs, np.sqrt(s["fits"][..., 3] * s["fits"][..., 4]))
    ok = flag == 0
    i_focus = int(np.argmin(np.abs(zs - z0[ok].mean())))
    amp = s["fits"][i_focus, :, 2]
    s.update(z0=np.where(ok, z0, np.nan).reshape(NV, NH), err=err.reshape(NV, NH),
             amp=(amp / np.nanmean(amp)).reshape(NV, NH), z_amp=zs[i_focus])
    print(f"CH3 {s['f3'] / 1e6:.4f} MHz: z0 fitted {ok.sum()}/{NV * NH} ({(flag == 1).sum()} unbracketed), "
          f"mean {z0[ok].mean():+.3f} um, std {z0[ok].std():.3f} um")
ref, sh = scans

P0, P1 = ref["pos"].reshape(NV, NH, 2), sh["pos"].reshape(NV, NH, 2)
(dr, dc), (rs, cs), (rs1, cs1) = pair_shifted(ref["pos"], sh["pos"], NV, NH)
sep = np.linalg.norm(P0[rs, cs] - P1[rs1, cs1], axis=2) * config.CAMERA_PIXEL_UM
paired = sep < MAX_SEP_UM
dz = np.where(paired, ref["z0"][rs, cs] - sh["z0"][rs1, cs1], np.nan)
ratio = np.where(paired, ref["amp"][rs, cs] / sh["amp"][rs1, cs1], np.nan)
err = np.hypot(ref["err"][rs, cs], sh["err"][rs1, cs1])
cols = np.arange(cs.start, cs.stop)
ok = np.isfinite(dz)
print(f"pairing: 70 MHz (r, c) <-> shifted (r{dr:+d}, c{dc:+d}); separation median {np.median(sep):.3f} um, "
      f"{paired.sum()}/{paired.size} within {MAX_SEP_UM} um (on the sensor)")
print(f"dz mean {np.nanmean(dz):+.3f} um, std {np.nanstd(dz):.3f} um (pair fit error {np.nanmedian(err):.3f} um); "
      f"amplitude ratio std {np.nanstd(ratio):.3f}, min {np.nanmin(ratio):.2f}, max {np.nanmax(ratio):.2f}")

f0, f1 = ref["f3"] / 1e6, sh["f3"] / 1e6
e2.apply()
fig = plt.figure(figsize=(180 * e2.MM, 190 * e2.MM), layout="constrained")
gs = fig.add_gridspec(4, 2)
extent = (cs.start - 0.5, cs.stop - 0.5, rs.start - 0.5, rs.stop - 0.5)
lim = np.nanpercentile(np.abs(dz), 99)
ax = fig.add_subplot(gs[0, :])
im = ax.imshow(dz, aspect="auto", cmap="PuOr_r", origin="lower", vmin=-lim, vmax=lim, extent=extent,
               interpolation="nearest")
fig.colorbar(im, ax=ax, label="dz (µm)")
ax.set_title(f"focus dz = z0({f0:.2f}) − z0({f1:.2f}): {ok.sum()} pairs, mean {np.nanmean(dz):+.3f} µm, "
             f"std {np.nanstd(dz):.3f} µm (pair fit error {np.nanmedian(err):.3f} µm)")
ax.set_xlabel(f"{f0:.2f} MHz design column")
ax.set_ylabel("row")

lim = np.nanpercentile(np.abs(ratio - 1), 99)
ax = fig.add_subplot(gs[1, :])
im = ax.imshow(ratio, aspect="auto", cmap="RdBu_r", origin="lower", vmin=1 - lim, vmax=1 + lim, extent=extent,
               interpolation="nearest")
fig.colorbar(im, ax=ax, label=f"A({f0:.2f}) / A({f1:.2f})")
ax.set_title(f"peak amplitude ratio at each tone's mean focus (z = {ref['z_amp']:+.1f} / {sh['z_amp']:+.1f} µm), "
             f"each normalised to its array mean: std {np.nanstd(ratio):.3f}, min {np.nanmin(ratio):.2f}, "
             f"max {np.nanmax(ratio):.2f}")
ax.set_xlabel(f"{f0:.2f} MHz design column")
ax.set_ylabel("row")

ax = fig.add_subplot(gs[2, 0])
for row in dz:
    ax.plot(cols, row, lw=0.5, alpha=0.4)
ax.plot(cols, np.nanmean(dz, axis=0), color="k", lw=1.5, label="mean over rows")
ax.axhline(np.nanmean(dz), color=e2.OKABE_ITO["vermilion"], ls="--", label=f"global {np.nanmean(dz):+.3f} µm")
ax.axhline(0, color=e2.GREY.edge, lw=0.8)
ax.set_xlabel(f"{f0:.2f} MHz column")
ax.set_ylabel("dz (µm)")
ax.set_title("focus differential vs column")
ax.legend()

ax = fig.add_subplot(gs[2, 1])
for row in ratio:
    ax.plot(cols, row, lw=0.5, alpha=0.4)
ax.plot(cols, np.nanmean(ratio, axis=0), color="k", lw=1.5, label="mean over rows")
ax.axhline(1, color=e2.GREY.edge, lw=0.8)
ax.set_xlabel(f"{f0:.2f} MHz column")
ax.set_ylabel(f"A({f0:.2f}) / A({f1:.2f})")
ax.set_title("amplitude ratio vs column")
ax.legend()

ax = fig.add_subplot(gs[3, 0])
x, y = sh["z0"][rs1, cs1][ok], ref["z0"][rs, cs][ok]
ax.plot(x, y, ".", ms=2, alpha=0.5, color=e2.OKABE_ITO["blue"])
lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y = x")
ax.set_xlabel(f"z0 at {f1:.2f} MHz (µm)")
ax.set_ylabel(f"z0 at {f0:.2f} MHz (µm)")
ax.set_title(f"per-pair focus, r = {pearsonr(x, y)[0]:+.2f}")
ax.legend()

ax = fig.add_subplot(gs[3, 1])
both = ok & np.isfinite(ratio)
ax.plot(dz[both], ratio[both], ".", ms=2, alpha=0.5, color=e2.OKABE_ITO["blue"])
ax.set_xlabel("dz (µm)")
ax.set_ylabel(f"A({f0:.2f}) / A({f1:.2f})")
ax.set_title(f"amplitude ratio vs focus differential, r = {pearsonr(dz[both], ratio[both])[0]:+.2f}")

fig.suptitle(f"Tone differential for {args.pattern.name}, overlapping traps, {datetime.now():%Y-%m-%d}\n"
             f"CH3 {f0:.4f} MHz (row r, col c) vs {f1:.4f} MHz (row r{dr:+d}, col c{dc:+d}), "
             f"{'CH3 flattened (aqc.aod_amplitude), ' if args.flat else ''}"
             f"exposure {args.exposure_us:g} µs, {args.average} frames per z")
save_plot(fig, "tone_differential")
