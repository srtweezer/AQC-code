# Heating simulation of the Shepard handoff

Lengths in µm, scaled from camera px by taking the mean waist as 1 µm; energies as $E/k_B$. Script:
`scripts/characterization/handoff_heating.py`.

## Why

The strobed trajectory (`handoff_trajectory.py`) gives the trap one atom rides through
the handoff: its centre, width and brightness every 0.93 µs for 4.7 ms. Instead of
reading heating rates off noise spectra, the simulation puts a thermal ensemble of Cs
atoms in that trap and integrates their motion, so it captures:
- resonances with the handoff harmonics,
- the anharmonic Gaussian potential,
- the combined effect of all channels.

## Input

One `…_trajectory[_blade]_row<r>.npz` (keys in `DATA_TAKING.md`), read by
`aqc.trajectory.load`:

| Key | Use |
|---|---|
| `t_us` | time |
| `x_residual_cam_um`, `y_residual_cam_um` ÷ `CAMERA_PIXEL_UM` | trap centre along / across the row, relative to uniform transport, camera px |
| `waist_x_cam_um`, `waist_y_cam_um` ÷ `CAMERA_PIXEL_UM` | spot waist along / across the row, camera px |
| `x_cam_um`, `site` | site pitch in camera px (slope of position against site) |
| `amp` | window light $I(t)$, i.e. the trap's power |

Each strobe cycle ends 0.1 µs (one strobe width) before the next starts. So the last
frame of one loop and the first frame of the next show the same moment of the handoff.
That pair is averaged into one point. Left as two independently fitted points 0.1 µs
apart, the spline would turn their fit difference into a velocity kick. The trace is
then interpolated with a cubic spline, because linear interpolation would kink every
0.93 µs.

## Trap model

An elliptical Gaussian beam, $x$ along the row, $y$ across it, $z$ along the beam:

$$
U(x, y, z, t) = -\frac{U_0(t)}{\sqrt{ab}}\,
\exp\!\left(-\frac{2\,\delta x^2}{w_x^2\,a} - \frac{2\,\delta y^2}{w_y^2\,b}\right)
$$

$$
a = 1 + \frac{z^2}{z_{R,x}^2}, \qquad b = 1 + \frac{z^2}{z_{R,y}^2}, \qquad
z_{R,x} = \frac{\pi w_x^2}{\lambda}, \qquad z_{R,y} = \frac{\pi w_y^2}{\lambda}, \qquad \lambda = 1.064\ \mu\text{m}
$$

$$
\delta x = x - x_c(t), \qquad \delta y = y - y_c(t)
$$

Its gradient, which gives the force $-\nabla U$:

$$
\frac{\partial U}{\partial x} = -\frac{4\,\delta x}{w_x^2\,a}\,U, \qquad
\frac{\partial U}{\partial y} = -\frac{4\,\delta y}{w_y^2\,b}\,U
$$

$$
\frac{\partial U}{\partial z} = z\,U\left[\frac{1}{z_{R,x}^2\,a}\left(\frac{4\,\delta x^2}{w_x^2\,a} - 1\right)
+ \frac{1}{z_{R,y}^2\,b}\left(\frac{4\,\delta y^2}{w_y^2\,b} - 1\right)\right]
$$

- **Centre** $(x_c, y_c)$: the measured residuals. The simulation runs in the frame
  moving with the transport (one pitch per handoff cycle), so the atom starts already
  moving with the array. There is no start or stop kick.
- **Waists:** $w_x(t)$ and $w_y(t)$ along and across the row, as measured, in camera px.
- **Length scale:** px are converted to µm by setting the mean waist,
  $\langle (w_x + w_y)/2 \rangle$ over the trace, to 1 µm (`aqc.trajectory.WAIST_UM`).
  The rig's own scale (pitch `aqc.wgs.PITCH_UM` = 50.6 µm on the camera) is not used:
  the simulation models tweezers of a 1 µm waist, as in the experiment, not the rig's
  ~3.6 µm waists. For row 0 of the 2026-11-05 blade run this gives waists of 0.90 µm along
  and 1.10 µm across and a pitch of 13.0 µm.
- **Depth:** depth follows peak intensity, i.e. power over area:

  $$
  U_0(t) = U_0\,\frac{g(t)}{\langle g\rangle}, \qquad g = \frac{I}{w_x w_y}
  $$

  A wider spot is also a shallower one.
- **$U_0$**, the mean depth, is not measured. It is set by `--depth-mk`, default
  0.2 mK. With the waists fixed it sets the trap frequencies (see Harmonic approximation).
- **Axial position** is not measured, so the trap is held at $z = 0$.

Depth enters in three places:
- the force, where the trap frequency is $\omega^2 \propto U_0(t)/w(t)^2$, so intensity
  and width changes give parametric heating;
- the energy reference $U_0(t)$;
- the loss threshold (none under `--harmonic`).

## Harmonic approximation

Expanding $U$ to second order about the trap centre, using
$1/\sqrt{ab} \approx 1 - \tfrac{z^2}{2}\left(1/z_{R,x}^2 + 1/z_{R,y}^2\right)$:

$$
U_h(x, y, z, t) = -U_0(t) + \tfrac12 m\left(\omega_x^2\,\delta x^2 + \omega_y^2\,\delta y^2 + \omega_z^2\,z^2\right)
$$

$$
\omega_x = \sqrt{\frac{4U_0(t)}{m w_x^2}}, \qquad
\omega_y = \sqrt{\frac{4U_0(t)}{m w_y^2}}, \qquad
\omega_z = \sqrt{\frac{U_0(t)}{m}\left(\frac{1}{z_{R,x}^2} + \frac{1}{z_{R,y}^2}\right)}
= \sqrt{\frac{U_0(t)\,\lambda^2}{m\,\pi^2}\left(\frac{1}{w_x^4} + \frac{1}{w_y^4}\right)}
$$

The force is linear, $-m\omega_i^2\,\delta r_i$ along each axis, and the frequencies
follow the trap in time through $U_0(t)$, $w_x(t)$ and $w_y(t)$. The approximation drops:
- the $\delta x^2 z^2$ and $\delta y^2 z^2$ cross terms, which couple radial and axial
  motion (the $\delta x^2$, $\delta y^2$ terms in $\partial U/\partial z$ above);
- all higher orders, which make the Gaussian softer than the parabola away from the
  centre and the oscillation frequency fall with energy.

$U_h$ is unbounded, so in it no atom can leave the trap. $E > U_0(t)$ (Simulation, step 3)
would only mark an energy above the nominal depth.

The simulation always uses $U_h$ for:
- the initial thermal state (Simulation, step 1), with $U_h$ at $t = 0$;
- the trap frequencies it prints, with $U_h$ of the mean trap. For row 0 of the
  2026-11-05 blade run at 0.2 mK these are 39, 33 and 9 kHz (88, 73 and 20 kHz at 1 mK).

By default the dynamics, the energy and the loss use the full Gaussian $U$. `--harmonic`
uses $U_h$ for the dynamics and the energy as well, and counts no loss: only the heating
is computed. Comparing the two runs separates heating that comes from the anharmonicity
(hot atoms whose frequency falls out of resonance, and the radial–axial coupling) from
heating a parabola would also give. Because hot atoms are kept rather than dropped,
$\langle E\rangle$ of a harmonic run is also not biased down by loss. For row 0 at
0.2 mK (500 atoms, every channel live) $\langle E\rangle/3k_B$ rises by 5.6 µK in the
Gaussian, with 0.6% lost, and by 15.7 µK in the harmonic trap.

## Simulation

1. **Start:** $N$ atoms (2000 by default), thermal at $T$ (10 µK) in $U_h$ at $t = 0$
   (Harmonic approximation), independently along each axis $i$:
   - positions $\sim \mathcal{N}\!\left(r_{c,i},\ k_BT/m\omega_i^2\right)$, with $r_{c,z} = 0$,
   - velocities $\sim \mathcal{N}\!\left(\dot r_{c,i},\ k_BT/m\right)$, with $\dot r_{c,z} = 0$.

   This is the thermal (Boltzmann) distribution:

   $$
   p(\mathbf r, \mathbf v) \propto \exp\!\left(-\frac{\tfrac12 m v^2 + U(\mathbf r)}{k_BT}\right)
   $$

   - **Velocities:** the kinetic term makes each velocity component Gaussian with
     variance $k_BT/m$ (Maxwell–Boltzmann), in any potential.
   - **Positions:** near the bottom $U \approx U_h$, so each position is Gaussian with
     variance $k_BT/m\omega_i^2$.

   The harmonic approximation is good enough here. At 10 µK and 0.2 mK,
   $k_BT/U_0 = 5\%$, so the atoms stay where the Gaussian potential matches a parabola
   to a few percent. The exact distribution, $\propto e^{-U(\mathbf r)/k_BT}$, is only
   slightly wider, because a Gaussian well is softer than a parabola. Starting from the
   harmonic one gives at most a tiny initial breathing of the ensemble, not heating.
   For much hotter starts, sample the exact distribution instead (rejection sampling).
   With `--harmonic` this start is the exact thermal state.

   The starting $\langle E\rangle$ is the mean of $N$ random draws. In $U_h$ each atom's
   $E$ is a sum of six quadratic terms, so $\langle E\rangle$ scatters around $3k_BT$ by
   $1/\sqrt{3N}$: 1.3% for 2000 atoms, 2.6% for 500.

   In the Gaussian it also sits systematically below $3k_BT$, because the draws are
   harmonic but their energy is taken in the softer $U$. To fourth order in the
   displacements (radial, axial and radial–axial cross terms),
   $\langle U - U_h\rangle \approx -\tfrac52\,(k_BT)^2/U_0$ (radial $\tfrac34$, axial
   $\tfrac34$, cross 1), i.e. $-\tfrac56\,k_BT/U_0$ in $\langle E\rangle/3k_BT$: $-0.42$ µK at 10 µK and 0.2 mK (row 0 of the 2026-11-05
   blade run, checked by Monte Carlo). This is most of the 9.57 µK start of that row at
   2000 atoms, which is 3σ from 10 µK. It is not heating: it is already there at $t = 0$,
   and the static-trap control keeps it constant. With `--harmonic` only the
   statistical scatter remains (9.89 µK at 500 atoms, 0.4σ).

   At 10 µK and 0.2 mK, $k_BT/\hbar\omega_x \approx 5$, so classical dynamics is a rough
   approximation near the ground state. It is still fine for heating rates.
2. **Integrate** $m\ddot{\mathbf r} = -\nabla U(\mathbf r, t)$ ($-\nabla U_h$ with `--harmonic`) by velocity Verlet:
   - 20 ns steps, about 1300 per trap period at 0.2 mK, about 235 000 steps in all;
   - all atoms at once.

   Velocity Verlet advances position and velocity together over a step $\Delta t$:

   $$
   \mathbf r_{n+1} = \mathbf r_n + \mathbf v_n\,\Delta t + \tfrac12\,\mathbf a_n\,\Delta t^2, \qquad
   \mathbf a_{n+1} = \mathbf F(\mathbf r_{n+1}, t_{n+1})/m, \qquad
   \mathbf v_{n+1} = \mathbf v_n + \tfrac12\,(\mathbf a_n + \mathbf a_{n+1})\,\Delta t
   $$

   The position update is a Taylor step. The velocity update uses the average of the old
   and new accelerations. Each step needs only one new force evaluation, because
   $\mathbf a_{n+1}$ is reused as $\mathbf a_n$ in the next step. The method is:
   - **second order:** the error per unit time scales as $\Delta t^2$;
   - **symplectic:** it preserves phase-space volume, so in a static potential the
     energy error stays bounded and oscillates instead of drifting. A method like
     4th-order Runge–Kutta is more accurate per step, but its energy slowly drifts over
     many periods, which would look like heating;
   - **time-reversible.**

   The symplectic property is what makes the all-frozen control (see Controls) a check
   of the integrator. The force depends on time because the trap moves and breathes, so
   $\mathbf F(\mathbf r_{n+1}, t_{n+1})$ is evaluated with the trap at the next step.
3. **Energy in the trap frame:**

   $$
   E = \tfrac12 m\,\lvert \mathbf v - \dot{\mathbf r}_c \rvert^2 + U(\mathbf r, t) + U_0(t)
   $$

   with $U_h$ in place of $U$ under `--harmonic`.

   This is $3k_BT$ at the start: in $U_h$ each axis holds $\tfrac12 k_BT$
   kinetic and $\tfrac12 k_BT$ potential energy. So $\langle E\rangle/3k_B$ reads as a
   temperature, or as mean energy per axis once the atoms are no longer thermal. An
   atom with $E > U_0(t)$ is unbound and counted as lost from then on. $\langle E\rangle$
   is averaged over the atoms still held. Under `--harmonic` no atom is counted as lost
   (Harmonic approximation), and $\langle E\rangle$ is over all atoms.
4. **Output:** the per-row curves go to an `.npz`; `plot_handoff_heating.py` draws
   two panels from it against the number of sites traversed ($|\text{site} - \text{site}_0|$
   from the trajectory, so time runs left to right whichever way the row moves), with the distance travelled (sites × pitch, on the same scale) on the
   top axis:
   - $\langle E\rangle/3k_B$, the mean energy per axis. It equals $T$ at the start, but
     is only a temperature while the three axes share the energy equally: the
     position noise drives x and y only, and the axes have different trap frequencies;
   - the fraction of atoms lost, zero under `--harmonic`.

## Controls

`--freeze position intensity width` holds any of the channels at their mean; `x` or `y`
in place of `position` holds one coordinate of the centre.

- **All frozen** (static trap): $\langle E\rangle$ must not change. This checks the
  integrator.
- **One channel driving at a time:** attributes the heating.

## Caveats

- **Fit jitter acts as real shaking.** The position trace still holds the frame-to-frame
  fit jitter. For blade traces (shared fitted background, row 4 of the 2026-11-13 50 µs
  and 20 µs runs) it is about 0.003 px, or 1.4 nm at the 1 µm mean-waist scale, from
  the second difference of consecutive strobe frames; the waists jitter by about
  0.006 px and the intensity by about 0.3%. Bootstrapping each point over the 181 scan
  runs gives 0.005 px: the extra part is common to neighbouring frames (per-run errors
  such as the drift correction), so it is slow along the trace rather than white. The
  simulated atoms feel the jitter as real trap noise,
  and heating from broadband position noise grows as $\omega^4 S_x(\omega)$, i.e. as
  $U_0^2$ with depth. Simulated heating is an upper bound until a noise-only
  control (the smoothed trace plus white noise at the jitter level) gives the part that
  is measurement noise.
- **Cycle-to-cycle noise is missing.** Each frame averages 500 cycles, so the trace
  holds only what repeats every cycle and differs from site to site.
- **The 5.67 MHz beat is missing.** It is averaged out of the frames by the random
  phases, and it sits far above the trap frequencies.
- **The length scale is an assumption.** All µm values, and through them $\nu$ at a
  given $U_0$, follow from the mean waist being 1 µm. The imaged spots also include the
  imaging blur, so the true waist is a smaller fraction of the pitch than measured.

## Usage

```
python scripts/characterization/handoff_heating.py \
    data/11_05/randomize_phase_blade_fit/20261105_172634_continuous_random_handoff_trajectory_blade_row0.npz
python scripts/characterization/handoff_heating.py data/11_05/randomize_phase_blade_fit/*_trajectory_blade_row*.npz   # every row, averaged
python scripts/characterization/handoff_heating.py <trajectory>.npz --depth-mk 1 --n-atoms 5000
python scripts/characterization/handoff_heating.py <trajectory>.npz --freeze intensity width   # position only
python scripts/characterization/handoff_heating.py <trajectory>.npz --harmonic   # parabolic trap, heating only
python scripts/characterization/plot_handoff_heating.py data/<MM_DD>/<stamp>_handoff_heating_<run stamp>_rows0-9.npz
```

Several trajectories are simulated one by one and the curves averaged over them (thin
lines: each row). Saves `data/<MM_DD>/<timestamp>_handoff_heating_<run stamp>_row<r>_<U0>mK[_harmonic][_freeze-…].npz`,
or `…_rows<r0>-<r1>…` for several rows: per row the sites traversed, $\langle E\rangle/3k_B$
and the lost fraction (zero under `--harmonic`) every 1 µs, plus the pitch and the simulation parameters.
`plot_handoff_heating.py` writes the figure next to it as `<same stem>.png`, so the figure
can be changed without rerunning the simulation.
