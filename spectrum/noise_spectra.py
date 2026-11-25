"""Intensity and position noise power spectra of one tweezer carried through the Shepard handoff.

Reads the trajectory .npz of scripts/characterization/handoff_trajectory.py (one atom
followed along a row) and computes the one-sided power spectra of Savard et al., PRA 56,
R1095 (1997), Eqs. (8)-(9), normalised so that the integral over nu >= 0 is the variance:

    S_eps(nu)  fractional intensity, eps = I / <I> - 1, I the Gaussian peak amplitude (1/Hz)
    S_x(nu)    trap-centre position along and across the row, in the trap plane (um^2/Hz)

The trajectory loops strobe frames that each average many cycles, so random
cycle-to-cycle laser noise is averaged out: the spectra show the periodic handoff
modulation (lines at multiples of 1/cycle) and site-to-site differences between SLM
traps. Strobes are evenly spaced within a cycle but the wrap to the next cycle is short,
so the trace is interpolated onto a uniform grid of cycle / (frames - 1). Camera um are
converted to the trap plane with the transport speed: one cycle moves one 10 um site.
The beat between the two crossfading tones (the tone spacing, MHz) is far above Nyquist
and appears aliased; its alias frequency is marked.

Marked: nu_r, nu_z (position) and 2 nu_r, 2 nu_z (intensity, parametric) for a trap of
depth --depth-mk and waist --waist-um at 1064 nm.

Prints the heating times of Savard Eqs. (12) and (16) at these frequencies for atoms at
--temperature-uk, from both the spectral floor and the nearest handoff line taken as resonant.

Then combines the two (Gehm et al., PRA 58, 3914 (1998)): <dE/dt> = Gamma_eps <E> + P_x, with
Gamma_eps = pi^2 nu_tr^2 S_eps(2 nu_tr) and P_x = 4 pi^4 M nu_tr^4 S_x(nu_tr) (S per Hz), so

    <E(t)> = (E_0 + P_x / Gamma_eps) exp(Gamma_eps t) - P_x / Gamma_eps,

and plots it for the two radial axes from E_0 = k_B * --e0-uk, with each S read off the
spectrum at 2 nu_r and nu_r.

Usage:
    python heating/noise_spectra.py data/11_02/20261102_151300_continuous_handoff_trajectory_row4.npz
"""

import argparse
from pathlib import Path

import e2style as e2
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import k as KB, physical_constants
from scipy.signal import periodogram

from aqc.data import save_plot

M_CS = 132.905 * physical_constants["atomic mass constant"][0]
WAVELENGTH_UM = 1.064
PITCH_UM = 10.0

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("trajectory", type=Path, help="<stamp>_handoff_trajectory_row<r>.npz")
parser.add_argument("--depth-mk", type=float, default=1.0)
parser.add_argument("--waist-um", type=float, default=0.73)
parser.add_argument("--temperature-uk", type=float, default=50.0, help="atom temperature, for <x^2>")
parser.add_argument("--e0-uk", type=float, default=10.0, help="initial energy per axis / k_B, for <E(t)>")
args = parser.parse_args()

d = np.load(args.trajectory)
run = np.load(str(d["run"]))
T = float(run["t_cycle_s"]) * 1e6                       # us
n_frames = len(run["rel_us"])
t, ok = d["t_us"], np.isfinite(d["amp"])

# Trap-plane um per camera um: one cycle carries the atom one site
sensor_um_per_site = np.polyfit(t[ok], d["u_um"][ok], 1)[0] * T
plane = PITCH_UM / abs(sensor_um_per_site)

dt = T / (n_frames - 1)
grid = np.arange(t[ok][0], t[ok][-1], dt)
fs = 1e6 / dt                                           # Hz
amp = np.interp(grid, t[ok], d["amp"][ok])
eps = amp / amp.mean() - 1
x_along = np.interp(grid, t[ok], d["residual_um"][ok]) * plane
x_across = np.interp(grid, t[ok], d["across_um"][ok]) * plane

nu, S_eps = periodogram(eps, fs, window="hann", scaling="density")
_, S_along = periodogram(x_along, fs, window="hann", scaling="density")
_, S_across = periodogram(x_across, fs, window="hann", scaling="density")
dnu = nu[1]
print(f"{len(grid)} samples at {dt:.4f} us (Nyquist {fs / 2e3:.0f} kHz, resolution {dnu:.0f} Hz); "
      f"trap plane {plane * 1.85:.4f} um/px")
print(f"eps rms {eps.std():.3f} (sqrt int S {np.sqrt(S_eps.sum() * dnu):.3f}); "
      f"along rms {x_along.std() * 1e3:.1f} nm, across rms {x_across.std() * 1e3:.1f} nm")

U0 = args.depth_mk * 1e-3 * KB
w0 = args.waist_um * 1e-6
zR = np.pi * w0 ** 2 / (WAVELENGTH_UM * 1e-6)
nu_r = np.sqrt(4 * U0 / (M_CS * w0 ** 2)) / (2 * np.pi)
nu_z = np.sqrt(2 * U0 / (M_CS * zR ** 2)) / (2 * np.pi)
print(f"nu_r {nu_r / 1e3:.1f} kHz, nu_z {nu_z / 1e3:.1f} kHz")

# Heating times, Savard Eqs. (12) and (16). The spectra are handoff lines on a floor, so
# each S is taken two ways: the floor (median over one harmonic spacing around the
# frequency, i.e. the broadband part) and the nearest handoff line taken as resonant, its
# power spread over the interaction-limited width 1 / (trace duration).
f_cyc = 1e6 / T
width = 1 / (len(grid) * dt * 1e-6)


def floor_and_line(S, f):
    band = np.abs(nu - f) < f_cyc / 2
    n = round(f / f_cyc)
    line = np.abs(nu - n * f_cyc) < 3 * dnu
    return np.median(S[band]), S[line].sum() * dnu / width, n, f - n * f_cyc


x2 = KB * args.temperature_uk * 1e-6 / (M_CS * (2 * np.pi * nu_r) ** 2) * 1e12     # um^2
print(f"\nheating times at {args.temperature_uk:g} uK (x_rms {np.sqrt(x2) * 1e3:.0f} nm radial); "
      f"line width {width:.0f} Hz")
print(f"{'':28s}{'freq':>9s}{'S floor':>11s}{'T floor':>10s}   nearest line{'S line':>11s}{'T line':>10s}")
for label, S, f, kind in (("T_I radial (2 nu_r)", S_eps, 2 * nu_r, "I"), ("T_I axial (2 nu_z)", S_eps, 2 * nu_z, "I"),
                          ("T_x along row (nu_r)", S_along, nu_r, "x"), ("T_x across row (nu_r)", S_across, nu_r, "x")):
    s_floor, s_line, n, det = floor_and_line(S, f)
    nu_tr = f / 2 if kind == "I" else f
    times = [1 / (np.pi ** 2 * nu_tr ** 2 * s * (1 if kind == "I" else 1 / x2)) for s in (s_floor, s_line)]
    print(f"{label:28s}{f / 1e3:7.1f}kHz{s_floor:11.2e}{times[0]:9.3g}s   n={n:2d} {det / 1e3:+5.2f}kHz"
          f"{s_line:11.2e}{times[1]:9.3g}s")

# Through the crossfade two tones one site apart light the same spot and interfere at their
# RF difference; the strobes sample that beat once per frame, so it aliases.
spacing = float(run["spacing_hz"])
strobe_dt = np.diff(run["rel_us"])[0] * 1e-6
phase = (spacing * strobe_dt) % 1
beat_alias = min(phase, 1 - phase) / strobe_dt
print(f"{spacing / 1e6:.4f} MHz tone beat aliases to {beat_alias / 1e3:.1f} kHz")

e2.apply()
c = e2.OKABE_ITO
fig, (a1, a2) = plt.subplots(2, 1, figsize=(180 * e2.MM, 130 * e2.MM), sharex=True, layout="constrained")
a1.semilogy(nu[1:] / 1e3, S_eps[1:], lw=0.6, color=c["green"])
a1.set_ylabel(r"$S_\epsilon$ (1/Hz)")
a1.set_title(f"fractional intensity (Gaussian peak amplitude), rms {eps.std():.3f}")
a2.semilogy(nu[1:] / 1e3, S_along[1:], lw=0.6, color=c["blue"], label=f"along the row, rms {x_along.std() * 1e3:.0f} nm")
a2.semilogy(nu[1:] / 1e3, S_across[1:], lw=0.6, color=c["vermilion"], alpha=0.8,
            label=f"across the row, rms {x_across.std() * 1e3:.0f} nm")
a2.set_ylabel(r"$S_x$ (µm$^2$/Hz, trap plane)")
a2.set_title("trap-centre position relative to uniform transport")
a2.legend(loc="upper right")
for ax, factor, tag in ((a1, 2, "2"), (a2, 1, "")):
    for f, name in ((nu_z, "z"), (nu_r, "r")):
        ax.axvline(factor * f / 1e3, color=e2.GREY.edge, lw=0.8, ls="--")
        ax.text(factor * f / 1e3 + 4, 0.96, rf"${tag}\nu_{name}$", transform=ax.get_xaxis_transform(),
                ha="left", va="top")
    ax.axvline(beat_alias / 1e3, color=c["purple"], lw=0.8, ls=":")
    ax.text(beat_alias / 1e3 + 4, 0.04, f"{spacing / 1e6:.2f} MHz beat, aliased", color=c["purple"],
            transform=ax.get_xaxis_transform(), ha="left", va="bottom")
a2.set_xlabel(f"frequency (kHz); lines at multiples of 1/{T:g} µs = {1e3 / T:.2f} kHz")
fig.suptitle(f"Noise spectra through the handoff, {args.trajectory.name}\n"
             f"marks: {args.depth_mk:g} mK, $w_0$ = {args.waist_um:g} µm: "
             rf"$\nu_r$ = {nu_r / 1e3:.0f} kHz, $\nu_z$ = {nu_z / 1e3:.0f} kHz")
stamp = args.trajectory.name.split("_handoff_trajectory")[0]
save_plot(fig, args.trajectory.name.removeprefix(stamp + "_").removesuffix(".npz") + "_spectra", stamp)

# Combined intensity and position heating along the two radial axes, from the spectra at
# 2 nu_r and nu_r. Gamma in 1/s, P_x / k_B in K/s.
E0 = args.e0_uk * 1e-6
t_end = len(grid) * dt * 1e-6
s_eps = np.interp(2 * nu_r, nu, S_eps)
gamma = np.pi ** 2 * nu_r ** 2 * s_eps
print(f"\n<E(t)> from E_0 = {args.e0_uk:g} uK per axis, over the {t_end * 1e3:.2f} ms transport")
print(f"S_eps(2 nu_r) {s_eps:.3g} /Hz: Gamma_eps {gamma:.3g} /s (1/Gamma {1e6 / gamma:.3g} us)")
axes = []
for name, S in (("along the row", S_along), ("across the row", S_across)):
    s_x = np.interp(nu_r, nu, S)
    power = 4 * np.pi ** 4 * M_CS * nu_r ** 4 * s_x * 1e-12 / KB
    axes.append((name, power))
    print(f"{name:15s} S_x(nu_r) {s_x:.3g} um^2/Hz: P_x / k_B {power * 1e6:.3g} uK/s")


def energy(t, p):
    return (E0 + p / gamma) * np.exp(gamma * t) - p / gamma


fig, ax = plt.subplots(figsize=(120 * e2.MM, 80 * e2.MM), layout="constrained")
t_plot = np.geomspace(1e-8, t_end, 400)
for (name, p), color in zip(axes, (c["blue"], c["vermilion"])):
    with np.errstate(over="ignore"):
        ax.loglog(t_plot * 1e6, energy(t_plot, p) * 1e6, color=color, label=f"radial, {name}")
ax.axhline(args.depth_mk * 1e3, color=e2.GREY.edge, lw=0.8, ls=":")
ax.text(t_plot[0] * 1e6 * 1.3, args.depth_mk * 1e3 * 1.15, f"depth {args.depth_mk:g} mK", va="bottom")
ax.set_ylim(args.e0_uk / 2, args.depth_mk * 1e3 * 30)
ax.set_xlim(t_plot[0] * 1e6, t_end * 1e6)
ax.set_xlabel("time (µs)")
ax.set_ylabel(r"$\langle E\rangle / k_B$ (µK)")
ax.set_title(rf"$\langle E(t)\rangle$, $E_0$ = {args.e0_uk:g} µK, {args.depth_mk:g} mK, $w_0$ = {args.waist_um:g} µm")
ax.legend(loc="upper left")
save_plot(fig, args.trajectory.name.removeprefix(stamp + "_").removesuffix(".npz") + f"_energy_{args.depth_mk * 1e3:g}uK", stamp)
