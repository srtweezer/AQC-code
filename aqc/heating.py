"""Classical simulation of atoms carried through the Shepard handoff, driven by a measured trajectory.

Takes the .npz of handoff_trajectory.py (one tweezer followed along a row: centre, blade
waists and window light every strobe) and integrates Newton's equations for a
thermal ensemble of Cs atoms in the trap that trace describes:

    U(x, y, z, t) = -U0(t) / sqrt(a b) * exp(-2 dx^2 / (w_x^2 a) - 2 dy^2 / (w_y^2 b)),
    a = 1 + z^2 / z_Rx^2,  b = 1 + z^2 / z_Ry^2,  z_Rx = pi w_x^2 / lambda,  z_Ry = pi w_y^2 / lambda,

x along the row, y across it, z along the beam, dx = x - x_c(t), dy = y - y_c(t).
  * length scale: camera px are converted to m by taking the trajectory's mean waist
    (2 sigma, along and across) as 1 um (aqc.trajectory.WAIST_UM).
  * centre (x_c, y_c): the measured position relative to uniform transport. The
    simulation runs in the frame moving with the transport, so the atom starts already
    moving with the array (no start or stop kick).
  * waists w_x(t), w_y(t) along and across the row, as measured. These are
    the imaged spots, so they include the imaging resolution.
  * depth U0(t) = U0 * g(t) / <g>, g = I / (w_x w_y): I is the window light (the power),
    so a wider spot is also a shallower one. U0, the mean depth, is not measured; with the
    waists fixed it sets the trap frequencies.
The trap's axial position is not measured and is held at z = 0.

The atoms start thermal (harmonic approximation of the trap at t = 0) and are integrated
by velocity Verlet. The energy in the trap frame, E = 1/2 M |v - dr_c/dt|^2 + U + U0(t), is
3 k_B T at the start; an atom with E > U0(t) is unbound and counted as lost from then on,
and <E> is over the atoms still held. harmonic replaces U by its harmonic approximation
about the centre (docs/heating_simulation.md, "Harmonic approximation"),
    U_h = -U0(t) + 2 U0(t) (dx^2 / w_x^2 + dy^2 / w_y^2) + U0(t) z^2 (1 / z_Rx^2 + 1 / z_Ry^2) / 2,
in the force and in E; U_h has no rim, so no atom is lost. freeze holds channels at their mean: position
(centre), x or y (one coordinate of the centre), intensity (I), width (w_x, w_y, which also
enter the depth).

site_us replays the same trace (the same shapes against the sites traversed) at another
transport speed: its time axis is stretched so that one site takes site_us.

The trace (aqc.trajectory.load, which merges the duplicate frame at each loop wrap) is
spline-interpolated. The fit jitter between frames is still in the trace and drives the
atoms like real trap noise; this simulation does not separate the two. For blade traces it
is ~0.003 px in position, ~0.006 px in waist and ~0.3% in intensity (docs/heating_simulation.md).
"""

import numpy as np
from scipy.constants import k as KB, physical_constants
from scipy.interpolate import CubicSpline

from aqc import trajectory
from aqc.wgs import WAVELENGTH_UM

M = 132.905 * physical_constants["atomic mass constant"][0]
LAM = WAVELENGTH_UM * 1e-6
RECORD_US = 1.0          # spacing of the recorded energies


def simulate(path, depth_mk, temperature_uk=10.0, n_atoms=2000, dt_ns=20.0, freeze=(), site_us=None, harmonic=False):
    """Simulate one trajectory .npz; returns a dict.

    row; sites traversed, <E>/3k_B (uK) and lost fraction every RECORD_US; pitch (m);
    summary, a one-line printout.
    """
    depth = depth_mk * 1e-3 * KB
    dt = dt_ns * 1e-9
    n = n_atoms
    row, t, site, trace, pitch = trajectory.load(path)
    if site_us is not None:
        s_per_site = abs(np.polyfit(site, t, 1)[0])
        t = t[0] + (t - t[0]) * site_us * 1e-6 / s_per_site
    for name, keys in (("position", ("x", "y")), ("x", ("x",)), ("y", ("y",)), ("intensity", ("I",)),
                       ("width", ("wx", "wy"))):
        if name in freeze:
            for k in keys:
                trace[k] = np.full_like(trace[k], trace[k].mean())

    ts = np.arange(t[0], t[-1], dt)
    spl = {k: CubicSpline(t, a) for k, a in trace.items()}
    cx, cy, I, wx, wy = (spl[k](ts) for k in ("x", "y", "I", "wx", "wy"))
    vcx, vcy = spl["x"](ts, 1), spl["y"](ts, 1)
    g = I / (wx * wy)
    eps = g / g.mean()
    zrx, zry = np.pi * wx ** 2 / LAM, np.pi * wy ** 2 / LAM
    del I, g

    def gaussian(x, y, z, k):
        """Acceleration (m/s^2) of atoms at (x, y, z) and their potential energy (J), at step k."""
        dx, dy = x - cx[k], y - cy[k]
        a, b = 1 + (z / zrx[k]) ** 2, 1 + (z / zry[k]) ** 2
        wx2, wy2 = wx[k] ** 2, wy[k] ** 2
        U = -depth * eps[k] / np.sqrt(a * b) * np.exp(-2 * dx ** 2 / (wx2 * a) - 2 * dy ** 2 / (wy2 * b))
        da, db = 2 * z / zrx[k] ** 2, 2 * z / zry[k] ** 2
        dlnU_dz = -0.5 * da / a - 0.5 * db / b + 2 * dx ** 2 * da / (wx2 * a ** 2) + 2 * dy ** 2 * db / (wy2 * b ** 2)
        return U * 4 * dx / (wx2 * a) / M, U * 4 * dy / (wy2 * b) / M, -U * dlnU_dz / M, U

    def parabola(x, y, z, k):
        """As gaussian, in the harmonic approximation of the trap at step k."""
        dx, dy = x - cx[k], y - cy[k]
        U0k = depth * eps[k]
        kx, ky, kz = 4 * U0k / wx[k] ** 2, 4 * U0k / wy[k] ** 2, U0k * (1 / zrx[k] ** 2 + 1 / zry[k] ** 2)
        U = -U0k + 0.5 * (kx * dx ** 2 + ky * dy ** 2 + kz * z ** 2)
        return -kx * dx / M, -ky * dy / M, -kz * z / M, U

    force = parabola if harmonic else gaussian

    # Harmonic frequencies at the mean trap, for the printout
    w_x, w_y = wx.mean(), wy.mean()
    nu = np.sqrt(np.array([4 * depth / w_x ** 2, 4 * depth / w_y ** 2,
                           depth * LAM ** 2 / np.pi ** 2 * (1 / w_x ** 4 + 1 / w_y ** 4)]) / M) / (2 * np.pi)

    # Thermal start in the harmonic approximation of the trap at t = 0
    rng = np.random.default_rng(0)
    kT = temperature_uk * 1e-6 * KB
    U0_start = depth * eps[0]
    ks0 = np.array([4 * U0_start / wx[0] ** 2, 4 * U0_start / wy[0] ** 2,
                    U0_start * (1 / zrx[0] ** 2 + 1 / zry[0] ** 2)])[:, None]
    pos = rng.normal(size=(3, n)) * np.sqrt(kT / ks0)
    pos[0] += cx[0]
    pos[1] += cy[0]
    vel = rng.normal(size=(3, n)) * np.sqrt(kT / M)
    vel[0] += vcx[0]
    vel[1] += vcy[0]

    every = max(1, round(RECORD_US * 1e-6 / dt))
    rec = np.arange(0, len(ts), every)
    energy = np.full(len(rec), np.nan)
    lost_frac = np.zeros(len(rec))
    lost = np.zeros(n, bool)
    acc = np.array(force(*pos, 0)[:3])
    j = 0
    for k in range(len(ts)):
        if k == rec[j]:
            U = force(*pos, k)[3]
            E = 0.5 * M * ((vel[0] - vcx[k]) ** 2 + (vel[1] - vcy[k]) ** 2 + vel[2] ** 2) + U + depth * eps[k]
            if not harmonic:      # a parabola has no rim to leave over
                lost |= E > depth * eps[k]
            energy[j] = E[~lost].mean() / (3 * KB) * 1e6 if (~lost).any() else np.nan     # uK
            lost_frac[j] = lost.mean()
            j = min(j + 1, len(rec) - 1)
        if k == len(ts) - 1:
            break
        pos += vel * dt + 0.5 * acc * dt ** 2
        new = np.array(force(*pos, k + 1)[:3])
        vel += 0.5 * (acc + new) * dt
        acc = new

    # Unsigned, so the plot reads forward in time whichever way the row moves
    sites = np.abs(np.interp(ts[rec], t, site) - np.interp(ts[0], t, site))
    summary = (f"row {row}: {len(ts)} steps of {dt_ns:g} ns, {sites[-1]:.1f} sites of {pitch * 1e6:.2f} um; waists "
               f"{w_x * 1e6:.3f} um along, {w_y * 1e6:.3f} um across; depth modulation rms {eps.std() * 100:.1f}%; "
               f"nu {nu[0] / 1e3:.1f} along, {nu[1] / 1e3:.1f} across, {nu[2] / 1e3:.1f} axial kHz; "
               f"<E>/3k_B {energy[0]:.2f} -> {energy[-1]:.2f} uK (change {energy[-1] - energy[0]:+.2f} uK); "
               + ("harmonic, no loss" if harmonic else f"lost {lost_frac[-1] * 100:.2f}%"))
    return dict(row=row, sites=sites, energy=energy, lost_frac=lost_frac, pitch=pitch, summary=summary)
