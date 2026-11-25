# Phase pattern calibration

How to build the SLM tweezer pattern from scratch. All scripts are in `scripts/calibration/`.

Every step uses the same setup: the SLM beam on the Basler (free-run), AODs on CH1/CH3 at
70 MHz and 1.6 V peak, and the pattern displayed with an 8 px y blaze (the vertical array: see below). Nothing else may hold
the AWG (stop `aod_tone.py` first). Each script stops the AWG when it exits. The exposures
below are the ones used last time; pick one that keeps the brightest trap below saturation.

## Rig calibrations

These don't depend on the pattern. Run them first, and again only when the hardware changes.

| # | Script | What it measures | Output | Last run |
|---|---|---|---|---|
| R1 | `lut_calibration.py` | SLM grey level → phase | `data/<day>/<stamp>_phase_response.npz`. Not installed automatically: copy it to `aqc/calibration/` and point `config.SLM_PHASE_RESPONSE` at it | 10-27 |
| R2 | `align_rows_to_aod.py --exposure-us 200` | WGS design angle θ that makes the rows parallel to the CH3 steering | `phase-patterns/<stamp>_wgs_aligned.npy`. θ converged to **3.191°**, the `--theta` default of every later script | 10-28 |
| R3 | `aod_site_step.py <wgs_aligned>.npy --exposure-us 100` | CH3 step that moves the array by one site | writes `config.CH3_SITE_STEP_MHZ` (5.6708 MHz) | 10-28 |
| R4 | `aod_amplitude.py --exposure-us 105` | CH3 amplitude per frequency for flat power | `config.CH3_AMPLITUDE_CAL` | 10-28 |
| R5 | `aberration_scan.py --exposure-us 30` | Field-dependent aberrations from single-spot Zernike scans, fitted to Machu's model | `config.ABERRATION_CAL` | 10-29 |

## Two pipelines

The two pipelines set the trap positions differently, and the two goals conflict (see
[below](#why-two-pipelines)). Each homogenizes once.

### 1. Position feedback: a straight, evenly spaced array on the camera

| # | Script | What it does | Output |
|---|---|---|---|
| P1 | `position_feedback.py --exposure-us 170 --correct` | Moves each trap's WGS target until the array is an affine lattice on the camera (uniform weights) | `aqc/calibration/<stamp>_position_offsets.npz` |
| P2 | `homogenize.py --exposure-us 170 --correct --offsets <P1>.npz` | Corrected WGS on the moved lattice, weights equalized by camera feedback | `<stamp>_wgs_cwgs_pos_homogenized.npy` + `_weights.npy` |
| P3 | `focus_flatten.py <P2>.npy --offsets <P1>.npz --exposure-us 150` | Per-trap lens from z-scans, so every trap focuses in one plane | `<stamp>_..._zflat.npy` + `_state.npz` |

### 2. Handoff calibration: every CH3 one-site step lands each trap on its neighbour

| # | Script | What it does | Output |
|---|---|---|---|
| H1 | `homogenize.py --exposure-us 100 --correct` | Corrected WGS on the plain lattice, weights equalized by camera feedback | `<stamp>_wgs_cwgs_homogenized.npy` + `_weights.npy` |
| H2 | `focus_flatten.py <H1>.npy --exposure-us 150` | Per-trap lens from z-scans | `<stamp>_..._zflat.npy` + `_state.npz` |
| H3 | `aod_steering.py <H2>.npy --exposure-us 100` | 2×2 steering matrix J, camera px per MHz, for the blade method | `config.AOD_STEERING_CAL` |
| H4 | `handoff_calibration.py <H2>_state.npz --exposure-us 250` | Moves the traps so every tone-B trap lands on its tone-A neighbour (docs/handoff_calibration.md) | `<stamp>_handoff.npy` + `_state.npz` |

`focus_flatten.py` and `handoff_calibration.py` keep every trap's simulated intensity when
they add lenses or move traps (`aqc.wgs.refine_amplitudes`), so the homogenization of P2/H1
carries through.

Neither pipeline has been run end to end in this form. Check that the final uniformity and,
for pipeline 2, the handoff convergence are as good as in the run below.

## How the current pattern was made

`aqc/calibration/phase-patterns/20261106_175439_handoff.npy` came from a mix of the two
pipelines. It homogenized twice and ran position feedback before the handoff calibration:

| Date | Script | Output |
|---|---|---|
| 10-29 | `homogenize.py --correct` | `20261029_173833_wgs_cwgs_homogenized.npy` |
| 10-30 | `position_feedback.py --correct --weights <10-29 weights>` | `20261030_110252_position_offsets.npz` |
| 10-30 | `homogenize.py --correct --offsets <10-30 offsets>` | `20261030_110541_wgs_cwgs_pos_homogenized.npy` |
| 11-01 | `focus_flatten.py --offsets <10-30 offsets>` | `20261101_184357_wgs_cwgs_pos_homogenized_zflat.npy` |
| 11-04 | `aod_steering.py <zflat>.npy` | `20261104_175646_aod_steering.npz` |
| 11-06 | one-off script, since deleted | `20261101_184357_..._zflat_state.npz`, recovered from the `.npy` (focus_flatten didn't save it then) |
| 11-06 | `handoff_calibration.py <zflat state>` | `20261106_175439_handoff.npy` + `_state.npz` (round 2 of 2) |

The handoff run used tone A = CH3 70 MHz and tone B = CH3 64.3292 MHz (70 − one site,
snapped to 64.328125 MHz), CH1 at 70 MHz. The blade scan stepped CH3 and then CH1 over
±0.9 MHz in 25 kHz steps around each tone, gain 1, 4 frames per shift.

## The vertical 90x10 array

After the realignment of 2026-11-12, the vertical array (90 rows x 10 columns, transported
along its long axis by CH1) went through pipeline 2. Every script takes `--vertical`, and
the pattern is displayed with an 8 px blaze along −x and none along y
(`--blaze-x -8 --blaze-y 0`). With +8 px along y, a ghost lattice reached the camera. The
camera finds the array as 10 lines of 90 traps; `aqc.traps.vertical_design_order` maps
them to design rows and columns.

| # | Command (all in `scripts/calibration/`, plus `--vertical --blaze-x -8 --blaze-y 0`) | Output |
|---|---|---|
| V1 | `align_rows_to_aod.py --theta0 3.191 --exposure-us 150` | θ = −1.1877° |
| V2 | `aod_amplitude.py --channel 1 --exposure-us 105 --iterations 7` (no `--vertical`: own defocused spot) | `config.CH1_AMPLITUDE_CAL` |
| V3 | `aberration_scan.py --exposure-us 50 --amplitude 0.25 --passes 3 --nx 3 --ny 7 --x-um -180 -80 --y-um -440 440` | `config.ABERRATION_CAL` |
| V4 | `homogenize.py --correct --theta -1.1877 --exposure-us 150` | `vertical/<stamp>_wgs_90x10_cwgs_homogenized.npy` + weights |
| V5 | `focus_flatten.py <V4>.npy --theta -1.1877 --exposure-us 150` | `vertical/<stamp>_..._zflat.npy` + `_state.npz` |
| V6 | `aod_steering.py <V5>.npy --exposure-us 150` | `config.AOD_STEERING_VERTICAL_CAL` |
| V7 | `aod_site_step.py <V5>.npy --exposure-us 150` | `config.CH1_SITE_STEP_MHZ` |
| V8 | `handoff_calibration.py <V5>_state.npz --exposure-us 250` | `vertical/<stamp>_handoff.npy` + `_state.npz` |
| V9 | `homogenize.py --state <V8>_state.npz --exposure-us 150` | `vertical/<stamp>_handoff_homogenized.npy` + `_state.npz` |

V9 is needed because the handoff moves made the trap powers less even (14% std/mean of the
peaks), although they held every trap's simulated intensity. It refines only the trap
amplitudes, so the lenses and the handoff moves stay: 0.6% std/mean, with the pair
mismatch at 0.08 px.

The aberration scan covers the array's field: y ±446 µm, x −185..−77 µm, the blaze moving
the field by 131 µm. A single reference spot carries the whole beam, so the scan drives the
AODs at 0.25 V instead of 1.6 V. Results of the 11-12 run are in `TODO.md`.

## Why two pipelines

- **Position feedback** makes the array straight and evenly spaced *on the camera*.
- **Handoff calibration** needs it evenly spaced *in beam angle*, because a CH3 step adds
  the same angle to every trap. The plain WGS lattice is evenly spaced in grid px, which is
  angle, so it already nearly satisfies this.
- If the map from angle to camera position isn't linear, the two goals differ. In the run
  above they did:
  - Round 0 of position feedback (`data/10_30/20261030_110252_position_feedback.png`): the
    plain lattice's deviation from affine is largest at the row ends (about 2.7 camera px)
    and near zero around columns 18 and 68 in every row. That is a parabola along the rows
    with its best line removed.
  - The position-feedback offsets in x are −8.1 grid px at the row ends, the same in every
    row (94% of the variation along a row is quadratic). In y they are −0.9 grid px at
    row 0 falling to −0.2 at row 9.
  - The handoff moves are +10.2 grid px in x. Trap by trap they anticorrelate with the
    offsets: correlation −0.98 in x (slope −1.22) and −0.68 in y. After each row's
    quadratic is removed, the correlation is still −0.77 in x.
  - Net, the handoff pattern is close to the plain lattice again: +1.95 grid px quadratic
    in x.
- Where the non-linearity comes from (optics distortion, camera tilt, other) is not known.
  Nor is it known whether the plane the atoms see shares it.

## The hologram state file

`handoff_calibration.py` doesn't re-solve WGS, which would change the speckle and lose the
homogenization and the lenses. It rebuilds the phase from its parts,
arg(Σₘ Sₘ exp(i[kₘ·r + correctionₘ + lensₘ ρ²])), and moves traps by changing kₘ. Those
parts are the `_state.npz`: `rows`, `cols` (grid positions), `correction`, `lens_rad`,
`spots` (Sₘ), `target` (simulated intensities, held when the amplitudes are refined),
`scal`, `blaze_y_px`. `focus_flatten.py` writes it next to its `.npy` (since 2026-11-09;
not yet run with that change).

## Physical units

The hologram is defined in WGS grid px. Every µm value in these scripts comes from
`aqc.wgs.um_per_ffpix` with `FOCAL_UM` the effective focal length from the SLM to the
tweezer plane the camera sees: f₁f₃/f₂ = 200 mm × 40 mm / 200 mm = 40 mm for the optics in
CLAUDE.md (SLM → 200 mm → 0th-order blade → 200 mm → 40 mm objective → camera), with
`PIXEL_UM = 8` and λ = 1.064 µm. A grid px is then 1.108 µm along y and 0.693 µm along x,
and focal-plane µm are µm on the camera sensor. At the blade (f = 200 mm) every length is
5× larger.

The array pitch is `aqc.wgs.PITCH_UM` = 50.63 µm, 27.4 camera px (the camera measures
27.9 px × 1.85 µm = 51.6 µm). Focus z is 23.52 µm per radian of lens (`aqc.focus`). An 8 px
blaze moves the field by 665 µm.

Until 2026-11-24 `FOCAL_UM` was 7900 µm, the value of the old experiment's `static_slm`,
and the pitch was written as "10 µm". Every grid position is the same under both (the
pitch was rescaled with the focal length). µm values from before then (field positions
of the aberration scan, focus z, position offsets in µm) are 7.9/40 of today's (lengths)
or (7.9/40)² (focus z). The two aberration calibrations in `aqc/calibration/` were
rescaled to today's units (their `focal_um` key), and `aqc.aberration.load` refuses a
file without it. The copies inside data runs (`setup/`) are as they were taken.
