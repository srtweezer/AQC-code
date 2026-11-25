"""Run outputs: data/<MM_DD>/<stamp>_<name>, one folder per working day.

An investigation that spans several scripts can group its outputs in a subfolder,
data/<MM_DD>/<subdir>/<stamp>_<name> (e.g. subdir "aod_steering").

The stamp keeps the full date so a plot and the calibration file written by the same run
(which share it) can be matched by name alone.
"""

from datetime import datetime
from pathlib import Path

import e2style as e2

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def data_path(name, stamp=None, subdir=None):
    """Path for data/<MM_DD>[/<subdir>]/<stamp>_<name>, creating the folder. `stamp` starts %Y%m%d_%H%M%S, default now."""
    stamp = stamp or f"{datetime.now():%Y%m%d_%H%M%S}"
    folder = DATA_DIR / f"{stamp[4:6]}_{stamp[6:8]}"
    if subdir:
        folder = folder / subdir
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{stamp}_{name}"


def save_plot(fig, name, stamp=None, subdir=None):
    """Save `fig` as data/<MM_DD>[/<subdir>]/<stamp>_<name>.png and return the path."""
    path = data_path(f"{name}.png", stamp, subdir)
    e2.save(fig, path, dpi=300, transparent=False)
    return path
