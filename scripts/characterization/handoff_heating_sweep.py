"""Atom loss of the handoff simulation against transport speed and trap depth, from one measured run.

Replays the trajectories of one strobe run (the .npz of handoff_trajectory.py, one per row)
through aqc.heating.simulate on a grid of time per site (--site-us: the same trace, its
time axis stretched so that one site takes that long) and mean trap depth (--depth-mk), with
every channel (position, intensity, width) live. The loss of a cell is the fraction of atoms
lost by the end of the row, averaged over the rows. Each simulation is independent, so they
run in parallel (--workers).

Saves data/<MM_DD>/<stamp>_handoff_heating_sweep_<run stamp>.npz (run stamp of the strobe run
folder the trajectories are in; the grid, the loss and
<E>/3k_B change of every row in every cell, the simulation parameters, the trajectories)
and a heatmap of the row-mean loss, <stamp>_handoff_heating_sweep_<run stamp>.png. Needs no
hardware.

--compare A.npz B.npz ... plots saved sweeps instead (no simulation): one heatmap per sweep
on a common 0-100% scale, titled by --titles, and with exactly two sweeps a third panel, the
difference B - A in percentage points. Saves
data/<MM_DD>/<stamp>_handoff_heating_sweep_compare_<run stamp A>_<run stamp B>[...].png.

Usage:
    python scripts/characterization/handoff_heating_sweep.py \\
        data/11_13/20261113_003551_shepard_continuous_50us_1site_blade/analysis/*_handoff_trajectory_blade_row?.npz
    python scripts/characterization/handoff_heating_sweep.py <trajectories> --site-us 10 20 50 --depth-mk 0.2 0.5
    python scripts/characterization/handoff_heating_sweep.py --compare <sweep A>.npz <sweep B>.npz \
        --titles "handoff-calibrated" "straightened"
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import e2style as e2
import matplotlib.pyplot as plt
import numpy as np

from aqc.data import data_path, save_plot
from aqc.heating import simulate


def heatmap(ax, values, site_us, depth_mk, **kw):
    """values (depth x site) as a heatmap with the grid values as tick labels."""
    im = ax.imshow(values, origin="lower", aspect="auto", **kw)
    ax.set_xticks(range(len(site_us)), [f"{s:g}" for s in site_us])
    ax.set_yticks(range(len(depth_mk)), [f"{d:g}" for d in depth_mk])
    ax.set_xlabel("Time per site (µs)")
    return im


def compare(paths, titles):
    sweeps = [np.load(p) for p in paths]
    for d in sweeps[1:]:
        if not (np.array_equal(d["site_us"], sweeps[0]["site_us"]) and np.array_equal(d["depth_mk"], sweeps[0]["depth_mk"])):
            raise ValueError("--compare needs sweeps on the same grid")
    site_us, depth_mk = sweeps[0]["site_us"], sweeps[0]["depth_mk"]
    loss = [100 * d["lost_frac"].mean(2) for d in sweeps]
    n = len(sweeps) + (len(sweeps) == 2)
    e2.apply()
    plt.rcParams.update({"axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14, "axes.titlesize": 15})
    fig, axes = plt.subplots(1, n, sharey=True, figsize=(130 * e2.MM * n, 100 * e2.MM), layout="constrained")
    for ax, values, title in zip(axes, loss, titles or [Path(str(d["trajectories"][0])).parents[1].name[:15]
                                                        for d in sweeps]):
        im = heatmap(ax, values, site_us, depth_mk, cmap="viridis", vmin=0, vmax=100)
        ax.set_title(title)
    fig.colorbar(im, ax=axes[:len(sweeps)], label="$P_\\mathrm{atom\\ loss}$ over the row (%)")
    if n > len(sweeps):
        diff = loss[1] - loss[0]
        lim = max(np.abs(diff).max(), 1)
        im = heatmap(axes[-1], diff, site_us, depth_mk, cmap="RdBu_r", vmin=-lim, vmax=lim)
        axes[-1].set_title("Difference (right − left)")
        fig.colorbar(im, ax=axes[-1], label="Δ$P_\\mathrm{atom\\ loss}$ (points)")
    axes[0].set_ylabel("Trap depth $U_0/k_B$ (mK)")
    stamps = "_".join(Path(str(d["trajectories"][0])).parents[1].name[:15] for d in sweeps)
    save_plot(fig, f"handoff_heating_sweep_compare_{stamps}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trajectory", type=Path, nargs="*", help="<stamp>_handoff_trajectory[_blade]_row<r>.npz, one per row")
    parser.add_argument("--site-us", type=float, nargs="+", default=[5, 10, 15, 20, 30, 40, 50, 70, 100],
                        help="time per site, us")
    parser.add_argument("--depth-mk", type=float, nargs="+", default=[0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0],
                        help="mean trap depth U0 / k_B, mK")
    parser.add_argument("--temperature-uk", type=float, default=10.0, help="initial temperature, uK")
    parser.add_argument("--n-atoms", type=int, default=2000)
    parser.add_argument("--dt-ns", type=float, default=20.0, help="integration step, ns")
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--compare", type=Path, nargs="+", help="saved sweep .npz files to plot side by side")
    parser.add_argument("--titles", nargs="+", help="with --compare: one title per sweep")
    args = parser.parse_args()
    if args.compare:
        compare(args.compare, args.titles)
        raise SystemExit

    site_us, depth_mk = np.array(args.site_us), np.array(args.depth_mk)
    jobs = [(i, j, p) for i in range(len(depth_mk)) for j in range(len(site_us)) for p in args.trajectory]
    lost = np.full((len(depth_mk), len(site_us), len(args.trajectory)), np.nan)
    dE = np.full_like(lost, np.nan)
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(simulate, p, depth_mk[i], args.temperature_uk, args.n_atoms, args.dt_ns, (), site_us[j]):
                   (i, j, args.trajectory.index(p)) for i, j, p in jobs}
        for done, (f, (i, j, r)) in enumerate(futures.items(), 1):
            out = f.result()
            lost[i, j, r] = out["lost_frac"][-1]
            dE[i, j, r] = out["energy"][-1] - out["energy"][0]
            if done % len(args.trajectory) == 0:
                print(f"[{done}/{len(jobs)}] {depth_mk[i]:g} mK, {site_us[j]:g} us/site: "
                      f"lost {np.mean(lost[i, j]) * 100:.1f}%", flush=True)

    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    run_stamp = args.trajectory[0].parents[1].name[:15]      # the strobe run (<run>/analysis/<npz>)
    name = f"handoff_heating_sweep_{run_stamp}"
    np.savez(data_path(f"{name}.npz", stamp), site_us=site_us, depth_mk=depth_mk, lost_frac=lost, energy_change_uk=dE,
             temperature_uk=args.temperature_uk, n_atoms=args.n_atoms, dt_ns=args.dt_ns,
             trajectories=np.array([str(p) for p in args.trajectory]))

    e2.apply()
    plt.rcParams.update({"axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14})
    loss = 100 * lost.mean(2)
    fig, ax = plt.subplots(figsize=(150 * e2.MM, 100 * e2.MM), layout="constrained")
    im = heatmap(ax, loss, site_us, depth_mk, cmap="viridis", vmin=0, vmax=max(loss.max(), 1))
    ax.set_ylabel("Trap depth $U_0/k_B$ (mK)")
    ax.set_title(f"Trace of run {run_stamp}, {len(args.trajectory)} rows, {args.temperature_uk:g} µK start",
                 fontsize=13, loc="left")
    fig.colorbar(im, ax=ax, label="$P_\\mathrm{atom\\ loss}$ over the row (%)")
    save_plot(fig, name, stamp)
