"""Classical simulation of atoms carried through the Shepard handoff, driven by measured trajectories.

Reads the .npz of handoff_trajectory.py (one tweezer followed along a row, centre, blade
waists and window light every ~0.93 us for ~4.7 ms) and integrates Newton's
equations for a thermal ensemble of Cs atoms in the trap that trace describes
(aqc.heating.simulate, which documents the model): mean depth --depth-mk, atoms starting
thermal at --temperature-uk, velocity Verlet steps of --dt-ns. --freeze holds channels at
their mean: position (centre), x or y (one coordinate of the centre), intensity (I), width
(w_x, w_y, which also enter the depth). Freezing position, intensity and width is a static
trap, where <E> must stay put. --harmonic runs the atoms in the harmonic approximation of the
trap instead of the Gaussian; it has no rim, so only the heating is computed and no atom is lost.

Given several trajectories (e.g. every row of one run), each is simulated on its own and
the curves are averaged over them: <E> is the mean of each row's <E> over its held atoms,
the loss the mean of each row's lost fraction.

Saves data/<MM_DD>/<timestamp>_handoff_heating_<run stamp>_row<r>_<U0>mK[_harmonic][_freeze-...].npz (rows<r0>-<r1>
for several rows): per row, the sites traversed (|site - site at t = 0| from the trajectory),
<E>/3k_B (uK) and the lost fraction (zero with --harmonic) every 1 us, plus the pitch and the simulation
parameters. This script only simulates and saves; plot_handoff_heating.py makes the figure
from the .npz.

Usage:
    python scripts/characterization/handoff_heating.py \\
        data/11_05/randomize_phase_blade_fit/20261105_172634_continuous_random_handoff_trajectory_blade_row0.npz
    python scripts/characterization/handoff_heating.py data/11_05/randomize_phase_blade_fit/*_trajectory_blade_row*.npz
    python scripts/characterization/handoff_heating.py <trajectory>.npz --depth-mk 1 --n-atoms 5000
    python scripts/characterization/handoff_heating.py <trajectory>.npz --freeze position intensity width
    python scripts/characterization/handoff_heating.py <trajectory>.npz --freeze y intensity width
    python scripts/characterization/handoff_heating.py <trajectory>.npz --harmonic
"""

import argparse
from pathlib import Path

import numpy as np

from aqc.data import data_path
from aqc.heating import simulate

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("trajectory", type=Path, nargs="+", help="<stamp>_handoff_trajectory[_blade]_row<r>.npz, one per row")
parser.add_argument("--depth-mk", type=float, default=0.2, help="mean trap depth U0 / k_B, mK")
parser.add_argument("--temperature-uk", type=float, default=10.0, help="initial temperature, uK")
parser.add_argument("--n-atoms", type=int, default=2000)
parser.add_argument("--dt-ns", type=float, default=20.0, help="integration step, ns")
parser.add_argument("--freeze", nargs="*", default=[], choices=("position", "x", "y", "intensity", "width"),
                    help="channels held at their mean")
parser.add_argument("--harmonic", action="store_true", help="harmonic approximation of the trap, no atom loss")
args = parser.parse_args()
n = args.n_atoms

runs = []
for p in args.trajectory:
    r = simulate(p, args.depth_mk, args.temperature_uk, n, args.dt_ns, args.freeze, harmonic=args.harmonic)
    print(r["summary"], flush=True)
    runs.append((r["row"], r["sites"], r["energy"], r["lost_frac"], r["pitch"]))
rows = [r[0] for r in runs]
# Rows start within a frame of each other, so their records line up to < 1 us; trim to the shortest.
m = min(len(r[1]) for r in runs)
pitch_um = np.mean([r[4] for r in runs]) * 1e6
sites, energy, lost_frac = (np.array([r[i][:m] for r in runs]) for i in (1, 2, 3))
dE = energy[:, -1] - energy[:, 0]
print(f"U0 {args.depth_mk:g} mK{', harmonic' if args.harmonic else ''}, frozen: {', '.join(args.freeze) or 'none'}; {len(rows)} rows: "
      f"<E>/3k_B {energy.mean(0)[0]:.2f} -> {energy.mean(0)[-1]:.2f} uK, change {dE.mean():+.2f} uK "
      f"(row spread {dE.std():.2f}, range {dE.min():+.2f} to {dE.max():+.2f})"
      + ("" if args.harmonic else f"; lost {lost_frac.mean(0)[-1] * 100:.2f}%"))

stamp = args.trajectory[0].name[:15]     # <YYYYmmdd_HHMMSS> of the strobe run
name = f"handoff_heating_{stamp}_" + (f"row{rows[0]}" if len(rows) == 1 else f"rows{min(rows)}-{max(rows)}") + f"_{args.depth_mk:g}mK" + ("_harmonic" if args.harmonic else "")
path = data_path(name + (f"_freeze-{'-'.join(sorted(args.freeze))}" if args.freeze else "") + ".npz")
np.savez(path, rows=rows, sites=sites, energy_uk=energy, lost_frac=lost_frac, pitch_um=pitch_um, run_stamp=stamp,
         depth_mk=args.depth_mk, harmonic=args.harmonic, temperature_uk=args.temperature_uk, n_atoms=n, dt_ns=args.dt_ns,
         freeze=np.array(args.freeze, dtype=str), trajectories=np.array([str(p) for p in args.trajectory]))
print(f"wrote {path}")
