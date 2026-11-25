"""Trap shape along and across the array rows versus the Shepard cycle period.

For each period T in --periods-us, the continuous Shepard handoff (aqc.shepard with no
chirp hold: two tones crossfade all cycle and move the array one site per T) is strobed
at --n-frames points of one cycle. The tones get random phases, so the beat between the
crossfading tones averages out over the cycles of each frame instead of distorting the
spots. Every tone chirps at spacing / T, and a chirp across the AOD's acoustic aperture is
a cylindrical lens along CH3, i.e. along the rows, of focal length v^2 / (lambda r), with
v the acoustic velocity: at short periods it widens the traps along the rows only.

Each strobed frame, and the reference (CH3 static at 70 MHz; leak frame subtracted, which
also removes room light), is resampled onto the array rows (aqc.movie.on_rows), so a
Gaussian with axes along that grid (aqc.traps.fit_spot) gives sigma along and across the
rows. In a frame the two tones' copies of the array lie on one lattice, shifted along the
rows from the reference by some delta (mod pitch); delta is the shift that puts the most
light on the traps, and the traps of columns 2 .. NH - 3 are fitted there. The end
columns are left out: there one tone's spot has no partner from the other.

Setup: the SLM pattern displayed, its beam imaged onto the Basler, AWG X0 -> fiber AOM,
X2 -> Basler Line1, CH2 -> LB1005-S Int Hold. Nothing else may hold the AWG or the camera.
Keep room light off the camera: the frames expose for tens of ms.

Saves data/<MM_DD>/<timestamp>_handoff_spot_shape.npz: per period, frame and fitted trap
(row, column 2 .. NH - 3) the fit (y, x, amplitude, sigma_y, sigma_x) on the row grid
(x along the rows, y across, camera px), the same for the reference, the strobe times
and shifts, and the run settings. Then data/<MM_DD>/<timestamp>_handoff_spot_shape.png.
--replot remakes the figure from a saved .npz without the hardware, overwriting its PNG.
--add-to <npz> measures only --periods-us and saves them merged with that earlier run
(sorted by period) as a new .npz; its settings must match. Leaves the AWG stopped.

Usage:
    python scripts/characterization/handoff_spot_shape.py
    python scripts/characterization/handoff_spot_shape.py --periods-us 5 10 20 55 --n-frames 5
    python scripts/characterization/handoff_spot_shape.py --replot data/11_04/<stamp>_handoff_spot_shape.npz
    python scripts/characterization/handoff_spot_shape.py --periods-us 60 70 85 100 \\
        --add-to data/11_04/<stamp>_handoff_spot_shape.npz
"""

import argparse
from datetime import datetime
from pathlib import Path

import e2style as e2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter, ScalarFormatter
from scipy.ndimage import gaussian_filter, map_coordinates

from aqc.data import data_path, save_plot
from aqc.traps import fit_spot

NV, NH = 10, 90
FIT_COLS = slice(2, NH - 2)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--periods-us", type=float, nargs="+", default=[5, 6, 7, 8, 10, 12, 15, 20, 30, 40, 55],
                    help="Shepard cycle periods, us")
parser.add_argument("--n-frames", type=int, default=10, help="strobe positions per cycle")
parser.add_argument("--aom-pulse-us", type=float, default=0.05,
                    help="AOM strobe width, us (the array moves one 50.6 um site / T: 0.5 um in 50 ns at 5 us)")
parser.add_argument("--n-cycles", type=int, default=500, help="cycles per camera frame")
parser.add_argument("--replot", type=Path, help="remake the figure from this saved .npz")
parser.add_argument("--add-to", type=Path, help="saved .npz to merge the new periods into (same strobe settings)")
args = parser.parse_args()


def measure():
    """Strobe and fit every period; returns the dict that is saved."""
    # Imported only here: aqc.shepard loads libspcm, which --replot must not.
    from aqc.movie import on_rows
    from aqc.shepard import spacing, strobe

    pulse = args.aom_pulse_us * 1e-6
    out = {k: [] for k in ("fits", "ref_fits", "offsets_us", "shift_px", "pitch_px")}
    for T_us in args.periods_us:
        T = T_us * 1e-6
        offsets = np.linspace(pulse / 2, T - pulse / 2, args.n_frames)
        print(f"--- T = {T_us:g} us")
        ref, frames, played = strobe(offsets, pulse, args.n_cycles, 1, t_chirp=0.0, t_cycle=T, random_phases=True)
        ref = ref - played["leak"]
        movie, u, v, rows, cols, pitch = on_rows(ref, np.concatenate([ref[None], frames]), NV, NH)
        del frames
        fit = (cols >= FIT_COLS.start) & (cols < FIT_COLS.stop)
        shifts = np.arange(0, pitch, 0.25)

        def fit_frame(img, shift):
            res = np.full((NV, NH, 5), np.nan)
            for k in np.flatnonzero(fit):
                res[rows[k], cols[k]] = fit_spot(img, v[k], u[k] + shift)
            return res[:, FIT_COLS]

        def best_shift(img):
            smooth = gaussian_filter(img, 1.5)
            light = [map_coordinates(smooth, [v[fit], u[fit] + s], order=1).sum() for s in shifts]
            return shifts[int(np.argmax(light))]

        frame_shifts = np.array([best_shift(img) for img in movie[1:]])
        out["ref_fits"].append(fit_frame(movie[0], 0.0))
        out["fits"].append(np.array([fit_frame(img, s) for img, s in zip(movie[1:], frame_shifts)]))
        out["offsets_us"].append(offsets * 1e6)
        out["shift_px"].append(frame_shifts)
        out["pitch_px"].append(pitch)
        f = out["fits"][-1]
        print(f"sigma along {np.nanmedian(f[..., 4]):.2f} px, across {np.nanmedian(f[..., 3]):.2f} px "
              f"(static {np.nanmedian(out['ref_fits'][-1][..., 4]):.2f}, "
              f"{np.nanmedian(out['ref_fits'][-1][..., 3]):.2f}); {np.isnan(f[..., 0]).sum()} failed fits")
    out = {k: np.array(val) for k, val in out.items()}
    out.update(periods_us=np.array(args.periods_us), spacing_hz=spacing(1), aom_pulse_us=args.aom_pulse_us,
               n_cycles=args.n_cycles, fit_cols=np.arange(NH)[FIT_COLS], random_phases=True)
    return out


def plot(d, stamp):
    """Median and quartiles over the traps of each trap's sigma averaged over the cycle, per period."""
    T = d["periods_us"]
    e2.apply()
    c = e2.OKABE_ITO
    fig, ax = plt.subplots(figsize=(110 * e2.MM, 80 * e2.MM), layout="constrained")
    for i, color, label in ((4, c["blue"], "along the rows"), (3, c["vermilion"], "across the rows")):
        q = np.nanpercentile(np.nanmean(d["fits"][..., i], axis=1).reshape(len(T), -1), [25, 50, 75], axis=1)
        ax.errorbar(T, q[1], [q[1] - q[0], q[2] - q[1]], fmt="o-", ms=3, lw=0.8, capsize=2, color=color, label=label)
        ax.axhline(np.nanmedian(d["ref_fits"][..., i]), color=color, ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50, 100])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Shepard cycle T (µs)")
    ax.set_ylabel("σ (camera px)")
    ax.legend(title="dashed: CH3 static", frameon=True, facecolor="white")
    ax.set_title(f"Trap shape vs continuous Shepard period, columns {d['fit_cols'][0]}-{d['fit_cols'][-1]}\n"
                 f"median, quartiles over traps; {d['fits'].shape[1]} strobes per cycle, "
                 f"{d['aom_pulse_us'] * 1e3:g} ns x {int(d['n_cycles'])} cycles, random tone phases")
    save_plot(fig, "handoff_spot_shape", stamp)


if args.replot:
    data = dict(np.load(args.replot))
    stamp = args.replot.name.removesuffix("_handoff_spot_shape.npz")
else:
    if args.add_to:
        old = dict(np.load(args.add_to))
        if (old["fits"].shape[1], float(old["aom_pulse_us"]), int(old["n_cycles"])) != \
                (args.n_frames, args.aom_pulse_us, args.n_cycles):
            raise ValueError(f"{args.add_to.name} was taken with {old['fits'].shape[1]} frames, "
                             f"{float(old['aom_pulse_us']):g} us strobe, {int(old['n_cycles'])} cycles: use the same")
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    data = measure()
    if args.add_to:
        order = np.argsort(np.concatenate([old["periods_us"], data["periods_us"]]))
        for k in ("periods_us", "fits", "ref_fits", "offsets_us", "shift_px", "pitch_px"):
            data[k] = np.concatenate([old[k], data[k]])[order]
        data["added_to"] = str(args.add_to)
    path = data_path("handoff_spot_shape.npz", stamp)
    np.savez(path, **data)
    print(f"wrote {path}")
plot(data, stamp)
