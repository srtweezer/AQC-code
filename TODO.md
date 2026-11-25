# TODO
0) Calibrate blade fitting method -- see if the offset changes anything
-- Sigma would come out too small. The fit assumes F_e = Φ((e − c)/σ), the cumulative of a complete Gaussian. Zeroing the dim pixels cuts off the tails, so the profile looks narrower

--The pixel-phase bias would come back. Which pixels survive the cut depends on where the spot sits inside its pixel, and that changes with every shift.

--The accuracy of the fitting method:

| Quantity | Error bar (trap quantity) | Sensor µm (1.85 µm/px) | Statistical only (imaged spot) |
|---|---|---|---|
| Position x (along row) | ±0.08 px | ±0.15 µm | ±0.005 px |
| Position y (across row) | ±0.03 px | ±0.06 µm | ±0.005 px |
| Peak intensity | ±4% | — | ±0.3% |
| Waist w_x, w_y | ±0.4 px (±17%) | ±0.7 µm | ±0.006 px |

Blade method with a fitted shared background; error bars from injection–recovery on row 4 of the
2026-11-13 runs (50 µs and 20 µs cycles, 360 points). "Trap quantity" means the trap centre
(intensity peak), the peak intensity, and the curvature waist w = 2√(I₀/κ). These errors are mostly
systematic and correlated between neighbouring points. The waist bound is conservative: the camera
data do not determine the curvature waist to better than about 15%.

1) Rerun the handoff error plot using blade method 
2) Design a better method to find the angle of the array and the spacing of the tones
3) Rerun the handoff error plot and see how much it improved
4) Do another iteration of the handoff correction
5) Collect more data. 
) Intensity-servo hold (aqc.intensity_lock, CH2 -> LB1005-S Int Hold), 2026-11-24: the hold works
(CH2 4.1 V at the BNC; Aux Output creeps ~2 mV/s while held, locked wobble 4-7 mV). With the 5 us
lead the onset kick dropped from ~10 to ~4 mV but didn't vanish. Open: does the step come with the
beam going dark (error feeding through the hold -> offset during strobes) or is the hold slower than
5 us (longer INT_HOLD_LEAD_US)? Scope at 2 us/div on CH2 rising edge, Aux Output + Error Monitor;
scratch test: 1 s holds with the AOM blocked, strobe_rows(np.ones(N, bool)).
