# Handoff calibration

Moves the SLM traps so that each Shepard handoff lands an atom exactly on the next trap.
Script: `scripts/calibration/handoff_calibration.py`, logic in `aqc/handoff.py`.
Positions are in camera px. Rows run along the transport and hold 90 traps each; column
c is the c-th trap of a row.

## Idea

CH3 one site lower (70 MHz − `config.CH3_SITE_STEP_MHZ`) shifts the whole array by one
site, so each trap at the low tone should sit on its neighbour at 70 MHz. Both copies come
from one hologram, so moving a trap moves it in both copies. Each pair is therefore
coupled to the next one along the row, and the corrections are built up column by column.

## 1. Measure the pair mismatch

Locate every trap with the blade method (docs/blade_method.md) at each tone separately:
tone A is CH3 at 70 MHz, tone B is CH3 at 70 MHz − 1 site, CH1 at 70 MHz for both, CH3
amplitude flattened. At every blade shift take a 70/70 MHz reference frame (its
array-mean displacement is the drift), then tone A, then tone B. For every pair, in camera
x and y:

```
E[r, c] = P_B[r, c + dc] − P_A[r, c]
```

B trap (r, c + dc) is the one that lands on A trap (r, c); dc = ±1 comes from
nearest-neighbour pairing (`aqc.traps.pair_shifted`). That gives 89 pairs per row and 890
in all.

The array mean of E is the error of the CH3 site step (along the rows) and of the row
angle against the AOD (across). Neither is a property of individual SLM traps, so it is
taken out before chaining. Chaining it would stretch or shear the whole array.

With `--rotate` the angle part is corrected as a rigid rotation instead: the step error a
and lattice rotation θ are fitted to E (`aqc.handoff.fit_step_angle`: a scales each trap's
own A → B displacement, θ turns the vector to its partner), and every trap also moves by
θ ẑ × (P − P̄), which adds θ ẑ × L to each pair. The step stays with
`config.CH*_SITE_STEP_MHZ` (`aod_site_step.py`). Measure at the pair the Shepard
crossfades through (`--tone-a-mhz` = centre + step / 2): on 2026-11-23 the angle was
−2.1 mrad at the 70 MHz pair and −3.65 mrad around mid-crossfade on the vertical array.

## 2. Corrections, column by column

Shift SLM trap (r, c) by u[r, c]. It moves by u in both copies, so

```
E'[r, c] = E[r, c] + u[r, c + dc] − u[r, c]
```

Start at the row end without a partner and zero each pair in turn:

```
u[r, c + dc] = u[r, c] − E[r, c]
```

The first trap is moved onto its partner. That changes the next pair's mismatch, so the
next trap absorbs both its own E and the previous move, and so on along the row. Rows are
independent. A linear trend of E along the row (the one-site step changing across the
field) becomes a parabolic bow in u.

Strictly, a trap moves by (1 + Δ') u in tone B, where Δ' ≈ 10⁻³ is how the one-site step
changes across the field. Over moves of a few px that is a few mpx, far below the floor,
and the next round absorbs it, so it is left out.

**Gauge.** Shifting a whole row changes no mismatch, so each row has one free constant.
It is chosen to give zero-mean moves in each row, which keeps the array in place.

## 3. Rebuild the hologram

The pattern is kept as a state file (`<pattern>_state.npz`): every trap's WGS grid
position (fractional), its lens, its aberration correction, the spot field and every
trap's target intensity. A move converts u from camera px to grid px through the
measured camera map of the design lattice (round 0's affine fit of the tone A centres).
Each trap's wave then moves to its new position, the trap phases stay frozen and only the
amplitudes are refined, so each trap keeps its simulated intensity
(`aqc.wgs.refine_amplitudes`, as `focus_flatten.py` does for lenses).

This replaces a WGS re-solve, which failed on 2026-11-02:
- WGS rounds targets to whole grid px (about 0.6 camera px in y, 0.4 in x), so per-trap
  corrections of 0.2–0.3 px were either dropped or doubled;
- each re-solve changes the speckle and scatters the traps.

Simulated far fields show fractional moves land at 0.86 (y) and 0.93 (x) of the command,
with about 0.1 grid px scatter.

The state of `20261101_184357_wgs_cwgs_pos_homogenized_zflat.npy` was recovered from the
phase alone (focus_flatten.py didn't save the lenses, and the base no longer re-solves
identically with today's `aqc/wgs.py`): every trap's lens is where its overlap with the
phase peaks, and the rebuilt phase keeps every trap's intensity, with 0.12 rad rms phase
difference elsewhere. Round 0 therefore measures the rebuilt pattern.

## 4. Iterate

Remeasure E on the new pattern. With gain g the mismatch should fall as E' = (1 − g) E.
If it grows by (1 + g) instead, a sign or the pairing is wrong. Each round also checks
that the tone A traps moved as commanded (slope of measured vs commanded move) and prints
the move error, the rms of each trap's measured move minus slope × commanded.

The moves, not the measurement, limit the result: the blade noise on E is ~0.004 px, and
the move error is close to the mismatch left after each round. It shrinks with the moves,
so more rounds keep helping. On 2026-11-24 (vertical array, `--rotate`, pair 72.787 →
67.213 MHz, gain 1), starting from the 2026-11-23 pattern:

| round | max move (px) | move error y, x (px) | pair mismatch rms y, x (px) |
|---|---|---|---|
| 0 | — | — | 0.067, 0.048 |
| 1 | 0.31 | 0.039, 0.027 | 0.054, 0.036 |
| 2 | 0.20 | 0.027, 0.020 | 0.036, 0.027 |
| 3 | 0.20 | 0.025, 0.019 | 0.033, 0.025 |
| 4 | 0.16 | 0.021, 0.015 | 0.027, 0.020 |

Round 0 also showed that the pattern held for ~20 h: the 2026-11-23 run ended at 0.067,
0.047 px.

## Things to watch

- **Errors accumulate along the row.** u is a running sum of E, so measurement noise in u
  grows as √c × the per-pair noise, about 9× at the far end. That wanders the absolute
  positions but leaves every pair corrected. A missing pair breaks the chain for every
  column after it, so the script refuses to chain with one missing.
- **The corrected array is not a regular lattice.** Don't run `position_feedback.py` on it
  afterwards: it fits an affine lattice and would undo the correction.
- **Re-fit positions before re-homogenizing.** `homogenize.py` measures brightness at the
  trap positions, which move.
