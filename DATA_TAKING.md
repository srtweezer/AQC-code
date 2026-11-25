
# Data taking

# Plan

We need to collect the data of different variants of shepard tones. I want you to run the following experiments, and collect data using the blade method and also randomize phases of shepard tones to eliminate beating.

1. Continuous shepard, 50 microseconds, one site handoff. 
2. Sudden switch between tones, 50 microseconds, one site handoff.
3. Continuous shepard, 50 microseconds, two site handoff. 

4. Continuous shepard, 20 microseconds, one site handoff. 
5. Sudden switch between tones, 20 microseconds, one site handoff.
6. Continuous shepard, 20 microseconds, two site handoff. 

Then do position feedback to straighten the array and homogenize it one more time. Then take data for 1 and 4 with the straightened array.

# Info
A data-taking run is a measurement whose raw arrays are kept for later analysis (e.g.
`handoff_strobe.py --save` or `--blade`). Each run is one self-contained folder, written
through `aqc.run`. Quick measurements that only produce a plot (snapshots,
`depth_vs_shift.py`, calibration plots) keep the flat `data/<MM_DD>/<stamp>_<name>.png`.

## Layout

```
data/<MM_DD>/<stamp>_<experiment>/
  metadata.json    what was measured, how, and with which calibrations
  raw/<name>.npy   one array per file, memory-mappable
  setup/           copies of the phase pattern and of every calibration file the run read
  analysis/        derived results: <stamp>_<name>.npz / .png / .gif, each with <stamp>_<name>.json
```

`metadata.json` has five sections:

| Section | Contents |
|---|---|
| `run` | stamp, experiment, script, command line, git commit and whether the tree had uncommitted changes, start and end time, status (`running`, `complete`, `failed` with the error), `note` |
| `setup` | the displayed pattern (file name, sha256, copy), blaze, array geometry, camera and AWG settings |
| `calibrations` | every `config` entry the run used: a value as is, or a file copied to `setup/` with its sha256 and the values the run relied on (e.g. `J`) |
| `parameters` | every script argument, plus the values the script derived from them |
| `arrays` | for every file in `raw/`: shape, dtype, the name of each axis, units, and a note |

An analysis JSON records the analysis script, its command line, git commit and arguments.

## Rules

- One run per folder; a script never writes into another run's folder, except into its `analysis/`.
- `raw/` and `setup/` are never changed after the run. Derived data and plots go to `analysis/`.
- Calibrations and the pattern are copied into the run, not referenced by path, so the run
  stays readable after `aqc/calibration/` is cleaned up.
- A script displays the pattern it records; don't rely on what is already on the SLM.
- Pass `--note` to say why the run was taken.
- A run that stops with an error is left with status `failed`, not deleted.
- Analysis scripts read runs with `aqc.run.load(folder)` (metadata and memory-mapped arrays)
  and name their outputs with `aqc.run.analysis(folder, name, args)`.

## Writing a run

```python
from aqc.run import Run

with Run("handoff_strobe", args, note=args.note) as run:      # folder + metadata.json (running)
    run.setup(pattern=args.pattern, blaze_y_px=0, blaze_x_px=-8)
    run.calibration("CH1_SITE_STEP_MHZ")
    run.calibration("AOD_STEERING_VERTICAL_CAL", keys=["J"])
    ...
    run.parameters(t_cycle_s=t_cycle)                          # derived values
    run.save("ref", ref, ["y", "x"], "counts")
    stack = run.memmap("blade_frames", shape, np.int16, ["strobe", "run", "pixel"], "counts")
# on exit: end time and status complete, or failed with the error
```

## Scripts using it

- `scripts/characterization/handoff_strobe.py` with `--save` or `--blade` writes a run,
  named `<stamp>_shepard_<continuous|sudden|standard>_<cycle>us_<n>site_<blade|gauss>[_<tag>]`
  (`--tag` adds the last part, e.g. `straight`);
  `scripts/characterization/handoff_trajectory.py <run folder>` analyses one (fits and saves
  an `.npz`), and `scripts/characterization/plot_handoff_trajectory.py <npz>...` plots it.
- Runs before 2026-11-13 (e.g. `data/11_06/blade_run`) are in the old flat `.npz` + `.npy`
  format, which the current scripts don't read.

## Trajectory `.npz` (`handoff_trajectory.py`)

`<run>/analysis/<stamp>_handoff_trajectory[_blade][_quad]_row<r>.npz`, one atom followed
along row `r`, one entry per tracked strobe frame. The keys use two frames:

- camera: `u`, `v` are the sensor's column and row axes, in px;
- row: `x` along the row, `y` across it (rotated from the camera by the row angle), as in
  the plots.

A `_cam_um` key is µm on the sensor (px × `config.CAMERA_PIXEL_UM`), which is the tweezer
plane on this rig. Widths are saved as waists w = 2σ (1/e² intensity radius).

| Key | Shape | Meaning |
|---|---|---|
| `method` | scalar | `gaussian` or `blade`, the fit used (absent in some files before 2026-11-06) |
| `run` | scalar | path of the strobe run |
| `row` | scalar | row number |
| `transit_us` | scalar | acoustic transit, subtracted from AWG time |
| `waist_median_cam_um` | scalar | ½ (median `waist_x_cam_um` + median `waist_y_cam_um`) (absent before 2026-11-06) |
| `t_us` | per frame | AWG time − transit |
| `loop`, `frame` | per frame | handoff cycle and strobe frame index |
| `site` | per frame | fractional site index along the row |
| `u_px`, `v_px` | per frame | spot centre, camera axes |
| `waist_u_px`, `waist_v_px` | per frame | spot waist along the camera axes |
| `amp` | per frame | Gaussian peak amplitude, or the blade window light (counts) |
| `x_cam_um`, `y_cam_um` | per frame | position along / across the row |
| `x_residual_cam_um` | per frame | x − (x₀ + v t): `x` minus uniform transport (minus its best quadratic in time with `--quadratic`) |
| `y_residual_cam_um` | per frame | y − y_row(x): `y` minus the row's line (or quadratic) |
| `waist_x_cam_um`, `waist_y_cam_um` | per frame | spot waist along / across the row |

Files written before 2026-11-19 used other names (`x_px`/`y_px` for the camera axes,
`u_um`/`v_um` along/across the row, σ instead of w, and `_trap_um` keys scaled to a 10 µm
pitch); they were rewritten to these keys on 2026-11-19.
`aqc.trajectory.load`, which `handoff_heating.py` and `handoff_spectrum.py` use, reads the
residuals and waists and converts to m by taking the mean waist as 1 µm
(`docs/heating_simulation.md`).

## Runs

2026-11-13, vertical 90x10 array, CH1 transport, random tone phases, blade 0.9 MHz / 20 kHz,
all 10 lines analysed (`handoff_trajectory.py --blade --row 0..9`). Runs 1–6 on
`vertical/20261112_234338_handoff_homogenized.npy` (0.6% std/mean); runs 7–8 on the
straightened `vertical/20261113_103928_wgs_90x10_cwgs_pos_homogenized.npy` (6.7%; position
offsets `20261113_103652_position_offsets.npz`, no focus lenses or handoff moves).

| Plan | Folder (`data/11_13/`) |
|---|---|
| test | `20261113_001458_shepard_continuous_50us_1site_gauss_test` (Gaussian fits, format test) |
| 1 | `20261113_003551_shepard_continuous_50us_1site_blade` |
| 2 | `20261113_020430_shepard_sudden_50us_1site_blade` |
| 3 | `20261113_032958_shepard_continuous_50us_2site_blade` |
| 4 | `20261113_045724_shepard_continuous_20us_1site_blade` |
| 5 | `20261113_061232_shepard_sudden_20us_1site_blade` |
| 6 | `20261113_072655_shepard_continuous_20us_2site_blade` |
| 1, straightened | `20261113_103930_shepard_continuous_50us_1site_blade_straight` |
| 4, straightened | `20261113_120714_shepard_continuous_20us_1site_blade_straight` |

These folders were renamed by hand from `<stamp>_handoff_strobe`, so their `metadata.json`
says experiment `handoff_strobe` and the analysis JSONs of plans 1–6 record the old path.

Straightening (trap-plane rms over the 10 lines, about a straight line in time): plan 1
along 0.184 → 0.070 µm, across 0.423 → 0.056 µm; plan 4 along 0.189 → 0.073 µm, across
0.428 → 0.056 µm. It is all the bow: with each line's quadratic removed (`--quadratic`,
line 4) all four runs are at 0.075–0.079 µm along and 0.056–0.060 µm across.
