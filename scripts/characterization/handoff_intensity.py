"""Integrated intensity of one tweezer, followed as it moves, through a Shepard handoff.

Strobes the array through the aqc.shepard handoff (--n-sites per 55 us cycle) at
--n-frames points, by default over exactly one AWG cycle (chirp start to reset end). The
beam lags the RF by the AOD's acoustic transit (fitted here, 23.5 us on 2026-10-30), so
the handoff the camera sees falls inside the chirp in AWG time. Strobe times wrap modulo
the cycle, so --window-us may run past it. The AOM leak frame is subtracted; a reference
frame with CH3 static at 70 MHz locates the traps.

Each strobed frame is the reference array shifted along its rows by
s * pitch * (t - t_mid) / T_CYCLE: s is the direction of rising CH3 frequency, t_mid the
strobe time at which a tone is at 70 MHz. Crossfading tones sit one pitch apart, so the
light in boxes on every trap depends only on that shift mod pitch, whatever the fade; s
and t_mid are fitted from it, and the handoff band in the plot follows from t_mid, not
from the intensity. A box moving at that rate then integrates the counts of tweezer
--tweezer. Through the handoff its tone fades out while the next tone's spot, from the
neighbouring SLM trap, fades in at the same place, so the box holds the trap an atom
riding along sees; the two spots beat at the tone spacing.

Over the handoff the counts are fitted with I0 (1 + c sin(pi (t - t_shift) / T_RESET)
cos(2 pi f_beat t + phi)), t_shift the handoff start from the motion fit: the cross term
of a cos-faded and a sin-faded field. f_beat starts from the strongest Fourier component
below the sampling Nyquist frequency, so it only measures the beat when the strobe step
resolves it (under 88 ns for the 5.67 MHz spacing); coarser scans return an alias. c is
also given divided by sinc(f_beat x strobe width), the contrast the strobe averages away.
With --random-phases every CH3 tone gets a random phase (aqc.shepard.waveform), so the
beat phase differs from cycle to cycle and averages out over the cycles of each frame.

Setup: the SLM pattern displayed, its beam imaged onto the Basler, AWG X0 -> fiber AOM,
X2 -> Basler Line1, CH2 -> LB1005-S Int Hold. Nothing else may hold the AWG or the camera.

Saves data/<MM_DD>/<timestamp>_handoff_intensity.png. Leaves the AWG stopped.

Usage:
    python scripts/characterization/handoff_intensity.py
    python scripts/characterization/handoff_intensity.py --tweezer 2 30 --n-frames 120
    python scripts/characterization/handoff_intensity.py --window-us -35 -5
    python scripts/characterization/handoff_intensity.py --n-frames 60 --random-phases
    python scripts/characterization/handoff_intensity.py --window-us -33 -11 --n-frames 440 --aom-pulse-us 0.05
"""

import argparse

import e2style as e2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from aqc.data import save_plot
from aqc.movie import on_rows
from aqc.shepard import FS, SKIP_CYCLES, T_CHIRP, T_CYCLE, T_RESET, spacing, strobe

NV, NH = 10, 90
US = 1e6

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--n-frames", type=int, default=240, help="strobe positions across the window")
parser.add_argument("--aom-pulse-us", type=float, default=0.05, help="AOM strobe width, us")
parser.add_argument("--n-cycles", type=int, default=500, help="chirp + reset cycles per camera frame")
parser.add_argument("--window-us", type=float, nargs=2, default=[-T_CHIRP * US, T_RESET * US],
                    help="strobe window relative to the reset start, AWG time, us (default: the AWG cycle)")
parser.add_argument("--n-sites", type=int, default=1, help="sites handed off per cycle")
parser.add_argument("--tweezer", type=int, nargs=2, default=[NV // 2, NH // 2], metavar=("ROW", "COL"),
                    help="tweezer to follow")
parser.add_argument("--random-phases", action="store_true",
                    help="random phase per CH3 tone, so the beat averages out over a frame's cycles")
args = parser.parse_args()

if args.window_us[1] - args.window_us[0] > T_CYCLE * US + 1e-6:
    raise ValueError(f"--window-us spans more than one cycle ({T_CYCLE * US:g} us)")
pulse = args.aom_pulse_us * 1e-6
rel_us = np.linspace(args.window_us[0] + args.aom_pulse_us / 2, args.window_us[1] - args.aom_pulse_us / 2,
                     args.n_frames)    # strobe centres from the reset start
ref, frames, _ = strobe(T_CHIRP + rel_us / US, pulse, args.n_cycles, args.n_sites, random_phases=args.random_phases)
movie, u, v, rows, cols, pitch = on_rows(ref, frames, NV, NH)

# Counts in a box of half width `half` along each row, from cumulative row profiles so the
# box can sit at fractional positions.
half = pitch / 3
hb = round(half)
row_v = [np.nanmedian(v[rows == r]) for r in range(NV)]
profile = np.stack([movie[:, round(rv) - hb:round(rv) + hb + 1].sum(axis=1) for rv in row_v], axis=1)
cum = np.concatenate([np.zeros(profile.shape[:2] + (1,)), np.cumsum(profile, axis=-1)], axis=-1)
edges = np.arange(cum.shape[-1]) - 0.5


def box(f, r, x):
    """Counts of frame f in the box centred on grid position x (px along row r)."""
    return np.interp(x + half, edges, cum[f, r]) - np.interp(x - half, edges, cum[f, r])


good = np.isfinite(u)
shifts = np.linspace(0, pitch, 200, endpoint=False)
light = np.array([sum(box(f, r, u[good & (rows == r)][:, None] + shifts).sum(axis=0) for r in range(NV))
                  for f in range(len(frames))])
t_grid = np.arange(rel_us[0], rel_us[0] + T_CYCLE * US, 0.05)
score = {(s, tm): sum(np.interp(s * pitch * (t - tm) / (T_CYCLE * US) % pitch, shifts, row_light, period=pitch)
                      for t, row_light in zip(rel_us, light))
         for s in (1, -1) for tm in t_grid}
s, t_mid = max(score, key=score.get)
transit = (t_mid + T_CHIRP * US / 2) % (T_CYCLE * US)
print(f"Fitted: tone at 70 MHz at the beam {t_mid:+.2f} us from reset start (acoustic transit {transit:.2f} us "
      f"mod {T_CYCLE * US:g} us), motion along {'+' if s > 0 else '-'}u")

tr, tc = args.tweezer
k = np.flatnonzero((rows == tr) & (cols == tc))[0]
counts = np.array([box(f, tr, u[k] + s * pitch * (t - t_mid) / (T_CYCLE * US)) for f, t in enumerate(rel_us)])
# The tone at 70 MHz at t_mid starts fading out T_CHIRP / 2 later; take the handoff nearest the window centre.
h0 = t_mid + T_CHIRP * US / 2
h0 -= T_CYCLE * US * np.round((h0 + T_RESET * US / 2 - rel_us.mean()) / (T_CYCLE * US))


def handoff(t, i0, c, f_beat, phi):
    """Old spot cos, new spot sin of pi (t - h0) / (2 T_RESET) in field: their interference term."""
    return i0 * (1 + c * np.sin(np.pi * (t - h0) / (T_RESET * US)) * np.cos(2 * np.pi * f_beat * t + phi))


# The beat is only measured if the strobe step resolves it: the start frequency is the
# strongest envelope-weighted Fourier component below Nyquist, not the expected spacing.
inside = (rel_us >= h0) & (rel_us <= h0 + T_RESET * US)
t_in, i_in = rel_us[inside], counts[inside]
env = np.sin(np.pi * (t_in - h0) / (T_RESET * US))
nyquist = 0.5 / np.median(np.diff(rel_us))      # MHz
f_scan = np.arange(0.05, nyquist, 0.005)
ft = np.exp(-2j * np.pi * f_scan[:, None] * t_in) @ ((i_in - i_in.mean()) * env)
j = np.argmax(np.abs(ft))
p0 = [i_in.mean(), 2 * np.abs(ft[j]) / (env ** 2).sum() / i_in.mean(), f_scan[j], -np.angle(ft[j])]
popt, pcov = curve_fit(handoff, t_in, i_in, p0=p0)
perr = np.sqrt(np.diag(pcov))
i0, c, f_beat, phi = popt
# The strobe averages the beat over its width, scaling the contrast by sinc(f_beat x pulse).
c_true = c / np.sinc(f_beat * args.aom_pulse_us)
print(f"Handoff fit: f_beat {f_beat:.4f} +- {perr[2]:.4f} MHz (Nyquist {nyquist:.2f} MHz; tone spacing "
      f"{spacing(args.n_sites) / 1e6:.4f} MHz), c {c:.3f} +- {perr[1]:.3f} ({c_true:.3f} strobe-corrected), "
      f"I0 {i0:.0f} counts")

e2.apply()
fig, ax = plt.subplots(figsize=(150 * e2.MM, 80 * e2.MM), layout="constrained")
ax.axvspan(h0, h0 + T_RESET * US, color=e2.OKABE_ITO["orange"], alpha=0.2, lw=0,
           label="handoff (Shepard reset, from the fitted motion)")
ax.plot(rel_us, counts, "o-", ms=2, lw=0.8, color=e2.OKABE_ITO["blue"], label="measured")
t_fit = np.linspace(h0, h0 + T_RESET * US, 4000)
ax.plot(t_fit, handoff(t_fit, *popt), lw=0.6, color=e2.OKABE_ITO["vermilion"],
        label=f"fit: f_beat {f_beat:.4f} ± {perr[2]:.4f} MHz, c {c:.2f} ({c_true:.2f} strobe-corrected)")
ax.axhline(0, color=e2.GREY.edge, lw=0.5)
ax.set_xlabel("strobe time from reset start, AWG time (µs)")
ax.set_ylabel(f"integrated counts ({2 * half:.0f} px box, moving)")
ax.legend(frameon=True, facecolor="white", framealpha=0.9)
ax.set_title(f"Tweezer row {tr}, col {tc} through the Shepard handoff: {args.n_sites} site per "
             f"{T_CYCLE * US:g} µs, {max(1, round(pulse * FS)) / FS * 1e9:.0f} ns strobe x "
             f"{args.n_cycles - SKIP_CYCLES} cycles, leak subtracted{', random tone phases' if args.random_phases else ''}\n"
             f"fitted acoustic transit {transit:.1f} µs (mod {T_CYCLE * US:g} µs); tone spacing "
             f"{spacing(args.n_sites) / 1e6:.4f} MHz, sampling Nyquist {nyquist:.2f} MHz")
save_plot(fig, "handoff_intensity")
