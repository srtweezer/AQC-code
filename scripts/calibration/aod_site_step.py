"""Measure the CH3 frequency step that moves the tweezer array by exactly one site along its rows.

The pattern (an aligned phase from align_rows_to_aod.py, rows parallel to the CH3
steering) is displayed with an 8 px y blaze. CH1 is held at 70 MHz and CH3 is swept
around 70 MHz while every trap is tracked (aqc.steering.sweep_ch3). For trap (r, c) the
one-site step is the frequency change that carries it, along the rows, to the 70 MHz
position of its neighbour: the measured spacing divided by the trap's fitted steering
(px/MHz). The spacing varies across the array, so these spread; the reported step fits
all traps at once (least squares). Both lengths are measured at the same place on the
camera, so its magnification cancels.

The step is then checked directly: CH3 = 70 +- step, every trap fitted and compared with
the 70 MHz position of the neighbour it should land on (taken just before), and the
step corrected by the median overshoot along the rows.

Setup: the SLM beam imaged onto the Basler (free-run), AODs on CH1/CH3 at 1.6 V peak.
Nothing else may hold the AWG (stop aod_tone.py first).

Writes the step into aqc/config.py (CH3_SITE_STEP_MHZ) and saves
data/<timestamp>_aod_site_step.png. Leaves the pattern on the SLM and the AWG stopped.

Usage:
    python scripts/calibration/aod_site_step.py aqc/calibration/phase-patterns/20261028_182104_wgs_aligned.npy --exposure-us 100
"""

import argparse
import re
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
from aqc.steering import F_CENTER, grab, play, slopes, sweep_ch3
from aqc.traps import find_traps, fit_all, row_fit

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FULL_SCALE = 4095        # Mono12
NV, NH = 10, 90
BLAZE_Y_PX = 8
AMPLITUDE = 1.6          # V peak, the CW drive of aod_tone.py

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("pattern", type=Path, help="phase map (.npy, radians, no blaze)")
parser.add_argument("--exposure-us", type=float, required=True, help="camera exposure, us")
parser.add_argument("--f-span", type=float, default=8.0, help="CH3 sweep half-span around 70 MHz, MHz")
parser.add_argument("--f-step", type=float, default=0.5, help="CH3 sweep step, MHz")
parser.add_argument("--average", type=int, default=8, help="frames averaged per grab in the sweep")
parser.add_argument("--check-average", type=int, default=16, help="frames averaged per grab in the check")
parser.add_argument("--repeats", type=int, default=3, help="checks per direction")
args = parser.parse_args()


def grid(values, rows, cols):
    """Per-trap values (N, ...) arranged as (NV, NH, ...), NaN where missing."""
    out = np.full((NV, NH) + values.shape[1:], np.nan)
    out[rows, cols] = values
    return out


def neighbour(G, shift):
    """G (NV, NH, ...) moved so entry (r, c) holds (r, c + shift); NaN past the row ends."""
    out = np.roll(G, -shift, axis=1)
    if shift > 0:
        out[:, -shift:] = np.nan
    else:
        out[:, :-shift] = np.nan
    return out


send_phase(np.load(args.pattern) + blaze(BLAZE_Y_PX, 0))
offs = np.round(np.arange(-args.f_span, args.f_span + args.f_step / 2, args.f_step), 6)
with aods(amplitudes=(AMPLITUDE, AMPLITUDE)) as awg, Camera(args.exposure_us, hardware_trigger=False) as cam:
    play(awg, F_CENTER)
    ref = grab(cam, args.average)
    peaks, rows, cols = find_traps(ref, NV, NH)
    p0 = fit_all(ref, peaks)
    if not np.all(np.isfinite(p0)):
        raise RuntimeError(f"Gaussian fit failed for {np.isnan(p0[:, 0]).sum()} of {NV * NH} traps "
                           f"on the 70 MHz frame (peak {ref.max():.0f} / {FULL_SCALE})")
    phi = np.radians(row_fit(p0, rows, NV))
    u = np.array([np.sin(phi), np.cos(phi)])            # (y, x) unit vector along the rows
    S = slopes(offs, sweep_ch3(awg, cam, p0, offs, args.average))

    P = grid(p0, rows, cols)
    speed = grid(np.abs(S @ u), rows, cols)             # px/MHz along the rows
    # Raising CH3 moves the traps towards column c + sgn.
    sgn = int(np.sign(np.nanmedian(S @ u)))
    dist = {s: np.abs((neighbour(P, s * sgn) - P) @ u) for s in (+1, -1)}   # px to the neighbour
    step_map = {s: dist[s] / speed for s in (+1, -1)}                       # MHz per site, per trap
    step = {}
    for s in (+1, -1):
        ok = np.isfinite(step_map[s])
        step[s] = float(np.sum(dist[s][ok] * speed[ok]) / np.sum(speed[ok] ** 2))
        print(f"sweep: {'+' if s > 0 else '-'}1 site = {step[s]:.4f} MHz (per-trap median "
              f"{np.nanmedian(step_map[s]):.4f}, spread {np.nanstd(step_map[s]):.4f} MHz, {ok.sum()} traps)")

    S_seed = np.where(np.isfinite(S), S, np.nanmedian(S, axis=0))
    check = {}
    for s in (+1, -1):
        overs, acrosses = [], []
        for _ in range(args.repeats):
            play(awg, F_CENTER)
            r70 = grab(cam, args.check_average)
            at70 = fit_all(r70, p0)
            f3 = play(awg, F_CENTER + s * step[s] * 1e6)
            img = grab(cam, args.check_average)
            moved = fit_all(img, p0 + s * step[s] * S_seed)
            res = grid(moved, rows, cols) - neighbour(grid(at70, rows, cols), s * sgn)
            direction = s * sgn * u                     # unit vector of the motion
            overs.append(res @ direction)
            acrosses.append(res @ np.array([-direction[1], direction[0]]))
        played = s * (f3 - F_CENTER) / 1e6
        over, across = np.mean(overs, axis=0), np.mean(acrosses, axis=0)
        corrected = played - np.nanmedian(over) / np.nanmedian(speed)
        check[s] = dict(played=played, over=over, across=across, corrected=corrected,
                        at70=at70, moved=moved, frame=r70)
        print(f"check at CH3 = {f3 / 1e6:.6f} MHz ({'+' if s > 0 else '-'}{played:.4f} MHz): overshoot along "
              f"the rows median {np.nanmedian(over):+.3f} px (repeats "
              f"{', '.join(f'{np.nanmedian(o):+.3f}' for o in overs)}), rms {np.sqrt(np.nanmean(over ** 2)):.3f} px; "
              f"across rms {np.sqrt(np.nanmean(across ** 2)):.3f} px -> corrected step {corrected:.4f} MHz")

site = 0.5 * (check[1]["corrected"] + check[-1]["corrected"])
print(f"One site = {site:.4f} MHz (+ {check[1]['corrected']:.4f}, - {check[-1]['corrected']:.4f}); "
      f"median spacing {np.nanmedian(dist[1]):.2f} px, steering {np.nanmedian(speed):.3f} px/MHz")

config_path = Path(config.__file__)
line = f"CH3_SITE_STEP_MHZ = {site:.4f}        # {datetime.now():%Y-%m-%d}, {args.pattern.name}"
text, n = re.subn(r"^CH3_SITE_STEP_MHZ = .*$", line, config_path.read_text(), flags=re.M)
if n != 1:
    raise RuntimeError(f"expected one CH3_SITE_STEP_MHZ line in {config_path}, found {n}")
config_path.write_text(text)
print(f"Wrote CH3_SITE_STEP_MHZ = {site:.4f} to {config_path}")

e2.apply()
fig = plt.figure(figsize=(180 * e2.MM, 150 * e2.MM), layout="constrained")
gs = fig.add_gridspec(2, 2)
colors = {+1: e2.OKABE_ITO["blue"], -1: e2.OKABE_ITO["vermilion"]}
ax = fig.add_subplot(gs[0, 0])
for s in (+1, -1):
    has = np.isfinite(step_map[s]).any(axis=0)          # the end column has no neighbour that way
    m, x = step_map[s][:, has], np.arange(NH)[has]
    ax.fill_between(x, np.nanmin(m, axis=0), np.nanmax(m, axis=0), color=colors[s], alpha=0.2, lw=0)
    ax.plot(x, np.nanmedian(m, axis=0), color=colors[s], lw=0.8,
            label=f"{'+' if s > 0 else '-'}1 site, checked: {check[s]['corrected']:.4f} MHz")
ax.axhline(site, color="k", lw=0.5, ls="--")
ax.set_xlabel("trap along row")
ax.set_ylabel("one-site step (MHz)")
ax.set_title("from the sweep: median over rows, band = min-max")
ax.legend(frameon=True, facecolor="white", framealpha=0.9)
ax = fig.add_subplot(gs[0, 1])
bins = np.linspace(-1, 1, 41)
for s in (+1, -1):
    ax.hist(check[s]["over"][np.isfinite(check[s]["over"])], bins, histtype="step", color=colors[s],
            label=f"{'+' if s > 0 else '-'}{check[s]['played']:.4f} MHz, along")
    ax.hist(check[s]["across"][np.isfinite(check[s]["across"])], bins, histtype="step", color=colors[s], ls=":",
            label=f"{'+' if s > 0 else '-'}{check[s]['played']:.4f} MHz, across")
ax.set_xlabel("trap position - neighbour's 70 MHz position (px)")
ax.set_ylabel("traps")
ax.set_title("direct check at 70 MHz +- step")
ax.legend(frameon=True, facecolor="white", framealpha=0.9)
ax = fig.add_subplot(gs[1, :])
c = check[1]
frame = c["frame"]
ax.imshow(frame, cmap="gray", vmin=0, vmax=np.percentile(frame, 99.99))
sel = (rows == NV // 2) & (cols < 8)
ax.plot(c["at70"][sel, 1], c["at70"][sel, 0], "o", ms=6, mew=0.8, mfc="none", color=colors[1],
        label="70 MHz")
ax.plot(c["moved"][sel, 1], c["moved"][sel, 0], "+", ms=6, mew=0.8, color=colors[-1],
        label=f"CH3 = 70 + {c['played']:.4f} MHz")
lo = np.nanmin(np.r_[c["at70"][sel], c["moved"][sel]], axis=0) - 20
hi = np.nanmax(np.r_[c["at70"][sel], c["moved"][sel]], axis=0) + 20
ax.set_xlim(lo[1], hi[1])
ax.set_ylim(hi[0], lo[0])
ax.set_title(f"row {NV // 2}, traps 0-7: after one site up, each + should sit on a neighbouring o")
ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
fig.suptitle(f"CH3 one-site step: {site:.4f} MHz ({args.pattern.name}, exposure {args.exposure_us:g} us)")
DATA_DIR.mkdir(exist_ok=True)
e2.save(fig, DATA_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_aod_site_step.png", dpi=300, transparent=False)
