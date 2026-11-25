# Blade (knife-edge) method for tweezer position and width

All quantities are in camera pixels unless stated otherwise. Pixel *j* covers
[*j* − 0.5, *j* + 0.5], the convention of `aqc.traps`.

## Why

A tweezer spot is only about 1 camera px wide, so a Gaussian fit to a single frame is
biased by the pixels:

- each pixel's area blurs the spot, so the fitted σ is too large;
- the fitted centre depends on where the spot falls within a pixel (pixel-phase bias).

Moving the spot by known sub-pixel steps across the fixed pixel edges measures its centre
and width free of both effects.

## Step 0: steering matrix J

J maps AOD frequency steps to array shifts on the camera:

```
(δx, δy) = J · (δf_CH3, δf_CH1)          [px/MHz]
```

`scripts/calibration/aod_steering.py` steps each channel over ±2 MHz around 70 MHz (the
other held at 70 MHz), fits all 900 traps at each step, and fits a straight line to their
mean displacement. Averaging over 900 spots at all pixel phases removes the pixel bias from
the mean, so J is good to sub-mpx/MHz. The result is stored in `config.AOD_STEERING_CAL`.

| (2026-11-04) | CH3    | CH1    |
|--------------|--------|--------|
| x            | 4.717  | −1.065 |
| y            | 1.437  | 4.843  |

CH3 moves the array 4.93 px/MHz at 17° to camera x, along the rows; CH1 moves it
4.96 px/MHz, across them.

## Step 1: blade scan

Shift CH3 over ±0.9 MHz in 20 kHz steps (about ±4.4 px in 0.1 px steps) with CH1 at
70 MHz, then CH1 the same way with CH3 unshifted. Take the steps in shuffled order, so
slow drift doesn't masquerade as a shift.

Each step is a full measurement (a "run"), and each run also takes a 70/70 MHz reference
frame to measure drift.

For the strobed Shepard handoff, `handoff_strobe.py --blade 0.9 0.02` shifts every CH3
tone by δf_CH3 and the CH1 tone by δf_CH1. That moves the whole handoff rigidly and leaves
the timing within the cycle unchanged. The scan is 181 runs of 60 strobe frames, takes
about 90 min, and writes about 18 GB.

## Step 2: fit

For each spot (`aqc.blade.blade`):

1. **Known shift per run.** The spot's shift in run *s* is

   ```
   shift_s = J · δf_s + drift_s
   ```

   where drift_s is the array-mean displacement of run *s*'s 70/70 MHz reference from the
   first run's (Gaussian fits; the pixel bias averages out over 900 traps).
2. **Window.** Take a ±6 px window centred on the predicted position, rounded to the
   nearest pixel. Because it follows the spot, the window always holds the whole spot.
3. **Background.** The strobed frames keep a flat background of about 10 counts/px, about
   30% of a window's light. It is fitted, not subtracted: one level b per pixel, shared by
   all runs, is a parameter of step 5.
4. **Edge fractions.** Sum the window (L = 13 px) across one axis. For every pixel edge
   *e* in it, take the fraction of the light before *e*. With k columns before *e*:

   ```
   F_e,s = P_before(e) / P_total
         = [ (P_total − L² b) Φ( (e − x0 − shift_s,x) / σ_x ) + b L k ] / P_total
   ```

   Φ is the normal CDF. Dividing by P_total cancels brightness changes between runs.
5. **Least squares.** Fit x0, σ_x and b jointly over all runs (CH3 and CH1 scans) and all
   edges. Fit y0, σ_y and b the same way with the pixel-row edges.

   On row 4 of the 2026-11-13 runs (comparison plots in data/11_23), the fitted b is
   about 0.5 counts/px above the window's border mean, which the fit used to subtract;
   the border mean made w along the row about 8% larger. A percentile of the window's
   pixels is off by about 2 counts/px (25th below, median above, since the window holds
   light outside the Gaussian core) and moves w by ±15%. A separate b per run gives the
   same result as the shared one. Centres agree to 0.005 px across all of these.

Each spot crosses several edges, which gives independent repeats. dF/dδf is the spot's
line-spread function, measured without assuming a Gaussian.

## From camera axes to row axes

Assume each spot's principal axes lie along (u) and across (v) the rows, at α ≈ 17° to the
camera axes. With c = cos α and s = sin α:

```
σ_x² = c² σ_u² + s² σ_v²,      σ_y² = s² σ_u² + c² σ_v²

σ_u² = (c² σ_x² − s² σ_y²) / (c² − s²)
σ_v² = (c² σ_y² − s² σ_x²) / (c² − s²)
```

If the axes might be tilted, fit instead a pixel-integrated 2-D Gaussian (centre, σ_u, σ_v,
angle) to every pixel of every run, with the shifts given by J.

## Code

| Step                       | Where                                                                 |
|----------------------------|-----------------------------------------------------------------------|
| J                          | `scripts/calibration/aod_steering.py`                                 |
| Scan (strobed Shepard)     | `scripts/characterization/handoff_strobe.py --blade SPAN STEP`        |
| Fit                        | `aqc/blade.py`                                                        |
| Handoff pair, CW tones     | `aqc/pair.py` (site step, row angle, handoff error vs f)              |
| Trajectory from the fits   | `scripts/characterization/handoff_trajectory.py --blade`              |

## Checks (2026-11-05 run)

- The mean shift per MHz measured on the strobed frames matches J to 0.2%.
- The array drifted by up to 0.25 px over the 90 min scan, which the drift term removes.
- The blade σ along the rows is 0.96 px, against 1.05 px from Gaussian fits, as expected
  once the pixel blur is removed.
