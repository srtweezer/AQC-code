"""Data-taking runs: one self-contained folder per run, laid out as DATA_TAKING.md describes.

    data/<MM_DD>/<stamp>_<experiment>/
        metadata.json    run, setup, calibrations, parameters, arrays
        raw/<name>.npy   one array per file, never changed after the run
        setup/           copies of the phase pattern and of every calibration file the run read
        analysis/        derived results, each <stamp>_<name>.* with a <stamp>_<name>.json

A script opens a Run as a context manager: the folder and metadata.json (status "running")
are written at once, every record call rewrites metadata.json, and the exit sets the end
time and status "complete", or "failed" with the error, which is then re-raised.
"""

import hashlib
import json
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

from aqc import config
from aqc.data import data_path

REPO = Path(__file__).resolve().parents[1]


def _git(*cmd):
    return subprocess.run(["git", *cmd], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _plain(v):
    """v as JSON-ready Python values (numpy scalars and arrays, paths, nested dicts and lists)."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, Path):
        return str(v)
    return v


def provenance():
    """Script, command line, git commit and whether the working tree differs from it."""
    return dict(script=Path(sys.argv[0]).resolve().relative_to(REPO).as_posix(), argv=sys.argv[1:],
                git_commit=_git("rev-parse", "HEAD"), git_dirty=bool(_git("status", "--porcelain")))


class Run:
    """One data-taking run in data/<MM_DD>/<stamp>_<experiment>/."""

    def __init__(self, experiment, args=None, note=None, stamp=None, subdir=None):
        self.stamp = stamp or f"{datetime.now():%Y%m%d_%H%M%S}"
        self.path = data_path(experiment, self.stamp, subdir)
        self.meta = dict(
            run=dict(stamp=self.stamp, experiment=experiment, **provenance(), note=note,
                     start=datetime.now().isoformat(timespec="seconds"), end=None, status="running"),
            setup={}, calibrations={}, parameters=_plain(vars(args)) if args is not None else {}, arrays={})

    def __enter__(self):
        (self.path / "raw").mkdir(parents=True)
        (self.path / "setup").mkdir()
        self._write()
        print(f"run folder {self.path}")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.meta["run"]["end"] = datetime.now().isoformat(timespec="seconds")
        self.meta["run"]["status"] = "complete" if exc_type is None else "failed"
        if exc_type is not None:
            self.meta["run"]["error"] = "".join(traceback.format_exception_only(exc_type, exc)).strip()
        self._write()
        return False

    def _write(self):
        tmp = self.path / "metadata.json.tmp"
        tmp.write_text(json.dumps(self.meta, indent=2) + "\n")
        tmp.replace(self.path / "metadata.json")

    def parameters(self, **values):
        """Record derived parameters (values the script computed, beyond its arguments)."""
        self.meta["parameters"].update(_plain(values))
        self._write()

    def setup(self, pattern=None, **values):
        """Record the setup; pattern (a phase .npy) is copied to setup/pattern.npy."""
        if pattern is not None:
            pattern = Path(pattern)
            shutil.copy2(pattern, self.path / "setup" / "pattern.npy")
            values["pattern"] = dict(file=pattern.name, sha256=_sha256(pattern), copy="setup/pattern.npy")
        self.meta["setup"].update(_plain(values))
        self._write()

    def calibration(self, name, keys=()):
        """Record config.<name>: a value as is, or a calibration file copied to setup/ with its hash and keys' values."""
        value = getattr(config, name)
        if isinstance(value, Path):
            shutil.copy2(value, self.path / "setup" / value.name)
            entry = dict(file=value.name, sha256=_sha256(value), copy=f"setup/{value.name}")
            if keys:
                with np.load(value) as f:
                    entry["values"] = {k: _plain(f[k]) for k in keys}
        else:
            entry = dict(value=_plain(value))
        self.meta["calibrations"][name] = entry
        self._write()
        return value

    def _array(self, name, shape, dtype, axes, units, note):
        if len(axes) != len(shape):
            raise ValueError(f"{name}: {len(axes)} axes for shape {shape}")
        self.meta["arrays"][name] = dict(file=f"raw/{name}.npy", shape=list(shape), dtype=np.dtype(dtype).name,
                                         axes=list(axes), units=units, note=note)
        self._write()

    def save(self, name, array, axes, units, note=None):
        """Save array as raw/<name>.npy and describe it: one axis name per dimension, units."""
        array = np.asarray(array)
        np.save(self.path / "raw" / f"{name}.npy", array)
        self._array(name, array.shape, array.dtype, axes, units, note)

    def memmap(self, name, shape, dtype, axes, units, note=None):
        """A writable raw/<name>.npy of this shape, for arrays filled as the run goes."""
        out = np.lib.format.open_memmap(self.path / "raw" / f"{name}.npy", "w+", dtype, tuple(shape))
        self._array(name, shape, dtype, axes, units, note)
        return out

    def figure(self, fig, name):
        """Save a figure the run itself makes (e.g. of the waveform it played) to analysis/<stamp>_<name>.png."""
        import e2style as e2
        path = analysis_dir(self.path) / f"{self.stamp}_{name}.png"
        e2.save(fig, path, dpi=300, transparent=False)
        return path


def load(folder):
    """metadata (dict) and every raw array of a run folder, memory-mapped read-only."""
    folder = Path(folder)
    meta = json.loads((folder / "metadata.json").read_text())
    arrays = {name: np.load(folder / a["file"], mmap_mode="r") for name, a in meta["arrays"].items()}
    return meta, arrays


def analysis_dir(folder):
    out = Path(folder) / "analysis"
    out.mkdir(exist_ok=True)
    return out


def analysis(folder, name, args=None):
    """Stem analysis/<stamp>_<name> for a derived result of the run in folder, with its <stem>.json written.

    The JSON records the analysis script, command line, git commit and arguments; the
    caller saves its outputs as <stem>.npz, <stem>.png, ...
    """
    stem = analysis_dir(folder) / f"{datetime.now():%Y%m%d_%H%M%S}_{name}"
    info = dict(**provenance(), time=datetime.now().isoformat(timespec="seconds"),
                arguments=_plain(vars(args)) if args is not None else {})
    stem.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
    return stem
