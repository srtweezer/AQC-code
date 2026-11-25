"""Spectra of the trap trace an atom rides through the Shepard handoff, from measured trajectories.

Reads the .npz of handoff_trajectory.py (one tweezer followed along a row every ~0.93 us
for ~4.7 ms, in camera px) through aqc.trajectory.load, which merges the duplicate frame at
each loop wrap and converts px to m by taking the mean waist as 1 um. Each channel is put on a uniform grid at the
frame spacing by the same cubic spline handoff_heating.py drives the atoms with. Its
one-sided PSD is a Hann-windowed periodogram (density scaling) of e, the record minus its
sample mean: S(f) per Hz, normalized so that int_0^inf S(f) df is the variance of e
(resolution 1 / record ~ 0.2 kHz, Nyquist ~ 0.54 MHz):
  * trap centre along / across the row, relative to uniform transport (um^2 / Hz);
  * waists w = 2 sigma along / across, relative to their mean;
  * window light I and depth g = I / (w_x w_y), the peak intensity, relative to their mean.
--traces picks the panels (default all six). Dotted lines mark the harmonics of the handoff
cycle, measured as the sites traversed per second. Several trajectories (e.g. one row of
two runs) are overlaid and then labelled by their run folder. The frame-to-frame fit jitter
is in the record and sets a white floor.

--poly-degree N instead takes e as the record minus its best degree-N polynomial in time, which
removes the slow bow (the static position errors along the row, crossed once in ~4 ms) and
leaves the jitter. N = 6 follows features no faster than ~1 kHz and leaves the power above
5 kHz unchanged; a quadratic still leaves most of the bow in x.

With --acf, each trace gets a column: e against time, its autocovariance
C(tau_k) = (1/N) sum_i e_i e_(i+k) (the 1/N estimator, the Fourier pair of the unwindowed
periodogram, C(tau) = int S(f) cos(2 pi f tau) df), for lags up to half the record,
the magnitude of its DFT (no window) scaled by 2/N, so that a sinusoid of amplitude A at a
bin frequency peaks at A, and its PSD. Dotted lines on C mark multiples of the handoff period.

Saves data/<MM_DD>/<timestamp>_handoff_spectrum_row<r>.png.

Usage:
    python scripts/characterization/handoff_spectrum.py \\
        data/11_05/randomize_phase_blade_fit/20261105_172634_continuous_random_handoff_trajectory_blade_row5.npz \\
        data/11_06/blade_run_handoff_corrected/20261106_181824_continuous_random_handoff_trajectory_blade_row5.npz
    python scripts/characterization/handoff_spectrum.py <npz> --traces x y
    python scripts/characterization/handoff_spectrum.py <npz> --traces x y --acf
    python scripts/characterization/handoff_spectrum.py <npz> --traces x y --acf --poly-degree 6
"""

import argparse
from pathlib import Path

import e2style as e2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import correlate, periodogram

from aqc import trajectory
from aqc.data import save_plot

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("trajectory", type=Path, nargs="+", help="<stamp>_handoff_trajectory[_blade]_row<r>.npz")
PANELS = {"x": "$S_x(f)$ (µm²/Hz)", "y": "$S_y(f)$ (µm²/Hz)",
          "wx": "waist along, rel. (1/Hz)", "wy": "waist across, rel. (1/Hz)",
          "I": "window light I, rel. (1/Hz)", "g": "depth I/(w$_x$w$_y$), rel. (1/Hz)"}
SYMBOLS = {"x": ("x", "µm"), "y": ("y", "µm"), "wx": ("w_x", "rel."), "wy": ("w_y", "rel."),
           "I": ("I", "rel."), "g": ("g", "rel.")}
parser.add_argument("--traces", nargs="+", choices=PANELS, default=list(PANELS), help="panels to plot")
parser.add_argument("--acf", action="store_true",
                    help="one column per trace: the trace, its autocovariance, |DFT| and PSD")
parser.add_argument("--poly-degree", type=int, default=0,
                    help="subtract the best polynomial of this degree in time (0: the mean)")
args = parser.parse_args()
colors = [e2.OKABE_ITO[k] for k in ("blue", "vermilion", "green", "purple")]
n_traces = len(args.traces)

e2.apply()
if args.acf:
    fig, grid = plt.subplots(4, n_traces, sharex="row", squeeze=False,
                             figsize=(90 * n_traces * e2.MM, 200 * e2.MM), layout="constrained")
    trace_axes, acf_axes, dft_axes, psd_axes = grid
else:
    nrows = -(-n_traces // 2)
    fig, grid = plt.subplots(nrows, 2, sharex=True, squeeze=False, figsize=(180 * e2.MM, 50 * nrows * e2.MM),
                             layout="constrained")
    psd_axes = grid.flat[:n_traces]
    for ax in grid.flat[n_traces:]:
        ax.remove()
rows = set()
for path, color in zip(args.trajectory, colors):
    row, t, site, trace, _ = trajectory.load(path)
    rows.add(row)
    f_cycle = abs(np.polyfit(t, site, 1)[0])            # one site per handoff cycle
    dt = np.median(np.diff(t))
    ts = np.arange(t[0], t[-1], dt)
    x = {k: CubicSpline(t, a)(ts) for k, a in trace.items()}
    x["g"] = x["I"] / (x["wx"] * x["wy"])
    x["x"], x["y"] = x["x"] * 1e6, x["y"] * 1e6         # um
    for k in ("wx", "wy", "I", "g"):
        x[k] = x[k] / x[k].mean()
    label = f"{path.parent.name}, row {row}"
    legend_label = label if len(args.trajectory) > 1 else None
    style = dict(color=color, lw=0.5, alpha=0.8, label=legend_label)
    rms = []
    for i, k in enumerate(args.traces):
        e = x[k] - np.polyval(np.polyfit(ts - ts.mean(), x[k], args.poly_degree), ts - ts.mean())
        f, p = periodogram(e, fs=1 / dt, window="hann", detrend="constant", return_onesided=True,
                           scaling="density")
        psd_axes[i].semilogy(f[1:] / 1e3, p[1:], **style)
        psd_axes[i].set_ylabel(PANELS[k])
        above = f > 1e3
        rms.append(f"{k} {e.std():.3g} ({np.sqrt(np.sum(p[above]) * f[1]):.3g})")
        if args.acf:
            name, unit = SYMBOLS[k]
            fit = f"p_{{{args.poly_degree}}}(t)" if args.poly_degree else f"\\langle {name} \\rangle"
            trace_axes[i].plot((ts - ts[0]) * 1e3, e, **style)
            trace_axes[i].set_ylabel(f"${name}(t) - {fit}$ ({unit})")
            lags = len(e) // 2
            c = correlate(e, e, mode="full", method="fft")[len(e) - 1:][:lags] / len(e)
            acf_axes[i].plot(np.arange(lags) * dt * 1e6, c, **style)
            acf_axes[i].set_ylabel(f"$C_{{{name}}}(\\tau)$ ({unit}²)")
            amp = 2 * np.abs(np.fft.rfft(e)) / len(e)
            dft_axes[i].semilogy(np.fft.rfftfreq(len(e), dt)[1:] / 1e3, amp[1:], **style)
            dft_axes[i].set_ylabel(f"$2|\\tilde{{{name}}}(f)|/N$ ({unit})")
    print(f"{label}: handoff {f_cycle / 1e3:.2f} kHz; rms (above 1 kHz), um or relative: {', '.join(rms)}")

for ax in (*psd_axes, *(dft_axes if args.acf else ())):
    for n in range(1, int(0.5 / dt / f_cycle) + 1):
        ax.axvline(n * f_cycle / 1e3, color="0.6", lw=0.4, ls=":", zorder=0,
                   label=f"handoff harmonics, n × {f_cycle / 1e3:.2f} kHz" if n == 1 else None)
    ax.set_xlabel("f (kHz)")
psd_axes[0].legend(fontsize="small")
if args.acf:
    for ax in acf_axes:
        for n in range(1, int(0.5 * len(ts) * dt * f_cycle) + 1):
            ax.axvline(n / f_cycle * 1e6, color="0.6", lw=0.4, ls=":", zorder=0,
                       label=f"handoff period, n × {1e6 / f_cycle:.1f} µs" if n == 1 else None)
        ax.axhline(0, color="0.6", lw=0.4, zorder=0)
        ax.set_xlabel("τ (µs)")
    acf_axes[0].legend(fontsize="small")
    for ax in trace_axes:
        ax.set_xlabel("t (ms)")
save_plot(fig, f"handoff_spectrum_row{'-'.join(map(str, sorted(rows)))}")
